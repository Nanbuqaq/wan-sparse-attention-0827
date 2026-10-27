"""Isolate weight updates, text visibility and text time without any history.

All eight ranks load slices of the SAME checkpoint. Four rule variants and two
seeds share one FSDP session; different checkpoints require different sessions.
This is an explicitly changed-interface diagnostic, never a streaming score.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import torch
import torch.distributed as dist
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_shared_stage1 import shared_modules
from src.inference.render import generate_latents, decode_and_save


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


_CURRENT_TEXT_CONTEXT_TYPES = {}


def current_text_context(core, context, *, allow_clean_current=False, allow_history=False):
    """Explicit sampling/clean-writer diagnostic; production defaults unchanged."""
    if context.write_bank is not None:
        if not allow_clean_current:
            return context
        if any(s.role == 'noisy' for s in context.splits):
            raise ValueError('A clean-writer intervention must not contain a noisy target')

    context_type = _CURRENT_TEXT_CONTEXT_TYPES.get(core.Context)
    if context_type is None:
        class CurrentTextContext(core.Context):
            def visible_splits(self, query):
                visible = super().visible_splits(query)
                if query.role != 'text':
                    return visible
                role = 'clean' if self.write_bank is not None else 'noisy'
                return tuple(s for s in self.splits if s in visible or
                             (s.role == role and s.block == query.block))
        context_type = _CURRENT_TEXT_CONTEXT_TYPES[core.Context] = CurrentTextContext
    return context_type(context.splits, context.text_reads_history or allow_history,
                        context.cached, context.write_bank, context.modulation_segments)


class CurrentTextAttentionProbe:
    """Delegate QKV/cache arithmetic to the unchanged shared attention processor."""
    def __init__(self, original, core):
        self.original, self.core = original, core
        self.allow_current = False
        self.allow_history = False
        self.allow_clean_current = False
        self.modified_sample_calls = 0
        self.unchanged_commit_calls = 0
        self.modified_commit_calls = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None, is_causal=False):
        if attention_mask.write_bank is not None:
            if self.allow_clean_current:
                attention_mask = current_text_context(self.core, attention_mask, allow_clean_current=True, allow_history=self.allow_history)
                self.modified_commit_calls += 1
            else:
                self.unchanged_commit_calls += 1
        elif self.allow_current:
            attention_mask = current_text_context(self.core, attention_mask, allow_history=self.allow_history)
            self.modified_sample_calls += 1
        return self.original(attn, hidden_states, rotary_emb, attention_mask, is_causal)


def sampling_text_clock(inputs, context, text_time):
    if context.write_bank is not None or text_time != 'native':
        return inputs
    changed = dict(inputs)
    clock = inputs['timestep'][inputs['timestep_indices']].clone()
    clock[inputs['text_indices']] = clock[inputs['video_indices'][0]].clone()
    changed['timestep'], changed['timestep_indices'] = torch.unique(clock, sorted=True, return_inverse=True)
    return changed


def sampled_bank_state(bank):
    """Small endpoint samples, not a full-KV checksum or history re-encoding."""
    ids, values = [], []
    for layer, entries in sorted(bank.snapshot().items()):
        for entry in entries:
            ids.append((layer, entry.block))
            values.append(torch.cat((entry.key[0, 0, 0, :8].float(),
                                     entry.key[0, -1, -1, -8:].float(),
                                     entry.value[0, 0, 0, :8].float(),
                                     entry.value[0, -1, -1, -8:].float())))
    if not values:
        return {}
    samples = torch.stack(values).cpu()
    return {identity: sample.clone() for identity, sample in zip(ids, samples)}


def streaming_text_probe(request_path, request):
    """Bounded rollouts; reuse FixedKVStream and the verified decoder."""
    import subprocess
    import wave
    from h3_shared_kv_video import Observer, block_template
    from h3_shared_stage1 import TransformerCallProxy
    from h3_shared_kv_heads import require_same_request
    from h3_buffered_video_decode import BufferedVideoDecoder
    from h3_frame_writer import FrameWriter
    from sample_demo import unpack_video

    if not request.get('diagnostic_only') or request.get('strategy') != 'A':
        raise ValueError('Explicit A-rule changed-interface diagnosis required')
    if request.get('NFE') != 50 or request.get('history_seconds') != 8.:
        raise ValueError('Do not change NFE or the committed history window')
    policies = request['policies']
    allowed = {('joint_current', 'fixed_clean'), ('readonly_text', 'native'), ('joint_current', 'native'), ('joint_current_history', 'native')}
    original_group = ({(p['visibility'], p['text_time']) for p in policies} == allowed
                      and len(policies) == 4 and not any(p.get('clean_feedback') for p in policies))
    writer_group = (request.get('clean_writer_diagnostic') is True and len(policies) == 1
                    and policies[0]['visibility'] == 'joint_current' and policies[0]['text_time'] == 'native'
                    and policies[0].get('clean_feedback') is True)
    if not (original_group or writer_group):
        raise ValueError('Use the three sampling controls or one explicit clean-writer control')
    execution_contract = ('diagnostic_current_clean_text_feedback_v1' if writer_group
                          else 'diagnostic_current_text_sampling_only_v1')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30), device_id=device)
    writer = None
    try:
        if dist.get_world_size() != 8:
            raise ValueError('One eight-card same-video group required')
        require_same_request(file_sha(request_path))
        code, source = shared_modules()
        session = load_session(request['weights_spec'], device)
        core = code['h3_fixed_kv_stage1']
        probes = []
        for block in session['model'].transformer.transformer_blocks:
            while hasattr(block, '_checkpoint_wrapped_module'):
                block = block._checkpoint_wrapped_module
            probe = CurrentTextAttentionProbe(block.attn.processor, core)
            block.attn.set_processor(probe)
            probes.append(probe)
        prompts = [torch.load(p, map_location='cpu', weights_only=True) for p in request['prompt_paths']]
        for p, text in zip(prompts, request['prompts']):
            if p['prompt'] != text:
                raise ValueError('Encoded text differs from the frozen request')
        reference = None
        if request.get('reference_result'):
            reference = json.loads(Path(request['reference_result']).read_text())
            if file_sha(request['reference_result']) != request['reference_result_sha256']:
                raise ValueError('Reference rollout result changed')
        origin = len(prompts[0]['prompt_embeds'])
        for policy in policies:
            for probe in probes:
                probe.allow_current = policy['visibility'] in ('joint_current', 'joint_current_history')
                probe.allow_history = policy['visibility'] == 'joint_current_history'
                probe.allow_clean_current = policy.get('clean_feedback', False)
                probe.modified_sample_calls = probe.unchanged_commit_calls = 0
                probe.modified_commit_calls = 0
            observer = Observer()

            class SamplingProxy(TransformerCallProxy):
                def __call__(self, *, attention_mask, **inputs):
                    if attention_mask.write_bank is not None and policy.get('clean_feedback'):
                        if inputs['audio_indices'].numel():
                            raise ValueError('No audio may enter the clean writer input')
                        times = inputs['timestep'][inputs['timestep_indices']]
                        if not bool((times == 1.).all()):
                            raise ValueError('Clean writer must keep the fixed clean timestep')
                    inputs = sampling_text_clock(inputs, attention_mask, policy['text_time'])
                    return super().__call__(attention_mask=attention_mask, **inputs)

            proxy = SamplingProxy(session['model'], observer)
            stream = code['h3_fixed_kv_runtime'].FixedKVStream(proxy, 'A')
            generator = torch.Generator(device=device).manual_seed(request['seed'])
            out = Path(os.environ['INFER_OUTPUT_DIR']) / policy['name']
            out.mkdir(exist_ok=True)
            decoder = BufferedVideoDecoder(session['vae'], device) if rank == 0 else None
            writer = FrameWriter(out / 'VIDEO.mp4', 24, encoder_threads=2) if rank == 0 else None
            records, audio_pieces = [], []
            started = time.monotonic()
            if rank == 0:
                (out / 'RESOLVED_CONFIG.json').write_text(json.dumps(dict(
                    request=request, policy=policy, shared_source=source,
                    checkpoint_contract_verified=True, execution_contract=execution_contract,
                    shared_production_mask_source_modified=False,
                    clean_commit_rule_unchanged=not policy.get('clean_feedback', False),
                    changed_interface_experiment=True,
                    model_history_recomputation_allowed=False, formal_memory_quality_claim=False), indent=2)+'\n')
            with torch.inference_mode():
                for row in request['blocks']:
                    before = sampled_bank_state(stream.bank)
                    observer.begin(row['block'])
                    rng = generator.get_state().cpu()
                    require_same_request((row['block'], tensor_sha(rng)))
                    block = block_template(row, prompts[row['prompt_index']], origin)
                    tick = time.monotonic()
                    video, audio, stats = stream.generate(block, generator, device, 50)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - tick
                    timing = observer.timings()
                    after = sampled_bank_state(stream.bank)
                    retained = before.keys() & after.keys()
                    unchanged = all(torch.equal(before[key], after[key]) for key in retained)
                    ok = torch.tensor(int(unchanged), device=device)
                    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                    if not ok.item():
                        raise ValueError('Previously committed retained KV samples changed')
                    if reference is not None:
                        for key in ('video_sha256', 'audio_sha256'):
                            if observer.initial_noise[key] != reference['records'][row['block']][key]:
                                raise ValueError('Actual initial noise differs from retained production control')
                    if rank == 0:
                        latent_count = int(row.get('latent_count', 7 if int(row['block']) == 0 else 5))
                        latent = unpack_video(video, latent_count, 48, 84).to(torch.bfloat16)
                        sound = audio[0].reshape(2, -1, 32).permute(0, 2, 1).to(torch.bfloat16)
                        torch.save(dict(video=latent.cpu(), audio=sound.cpu(), generator_before=rng,
                                        generator_after=generator.get_state().cpu(), row=row,
                                        initial_noise=observer.initial_noise), out / f'chunk{row["block"]:03d}_GENERATED.pt')
                        if session['reshard']:
                            session['vae'].to(device)
                        decoded = decoder.push(latent)
                        if decoded is not None:
                            writer.append(decoded)
                        if session['reshard']:
                            session['vae'].cpu()
                            torch.cuda.empty_cache()
                        audio_pieces.append(sound)
                        record = dict(block=row['block'], prompt_index=row['prompt_index'],
                                      sample_wall_seconds=elapsed, **timing, **observer.initial_noise,
                                      history_blocks=list(stats['read_past_blocks']), KV_bytes_per_rank=stats['KV_bytes'],
                                      old_history_queries_recomputed=0, retained_KV_endpoint_samples_exact=True,
                                      retained_KV_sample_scope='32 endpoint values per layer/block; not full checksum')
                        records.append(record)
                        with (out / 'TIMING.jsonl').open('a') as f:
                            f.write(json.dumps(record)+'\n')
                        print(policy['name'], row['block'], 'COMPLETE', elapsed, flush=True)
                    dist.barrier()
                counts = dict(modified_sample_attention_calls=probes[0].modified_sample_calls,
                              unchanged_commit_attention_calls=probes[0].unchanged_commit_calls,
                              modified_clean_commit_attention_calls=probes[0].modified_commit_calls)
                expected_clean_modified = len(request['blocks']) if policy.get('clean_feedback') else 0
                if (counts['modified_clean_commit_attention_calls'] != expected_clean_modified or
                        counts['unchanged_commit_attention_calls'] != len(request['blocks'])-expected_clean_modified):
                    raise ValueError('Clean-commit lifecycle changed')
                expected_modified = 50 * len(request['blocks']) if policy['visibility'] in ('joint_current', 'joint_current_history') else 0
                if counts['modified_sample_attention_calls'] != expected_modified:
                    raise ValueError('Current-text attention intervention did not execute as declared')
                if rank == 0:
                    if session['reshard']:
                        session['vae'].to(device)
                    tail = decoder.finish()
                    if tail is not None:
                        writer.append(tail)
                    writer.close()
                    if writer.frames != request['actual_frames']:
                        raise ValueError('Diagnostic output duration differs from fixed control')
                    if session['reshard']:
                        session['vae'].cpu()
                        session['audio_vae'].to(device)
                        torch.cuda.empty_cache()
                    avae = session['audio_vae']
                    mean = torch.tensor(avae.config.latents_mean, device=device).view(1, -1, 1)
                    std = torch.tensor(avae.config.latents_std, device=device).view(1, -1, 1)
                    pcm = avae.decode(torch.cat(audio_pieces, 2)*std+mean, return_dict=False)[0].float().permute(1,0,2)[0]
                    wav = out / 'AUDIO.wav'
                    with wave.open(str(wav), 'wb') as f:
                        f.setnchannels(2); f.setsampwidth(2); f.setframerate(32000)
                        f.writeframes((pcm.clamp(-1,1).cpu().T*32767).round().short().contiguous().numpy().tobytes())
                    import imageio_ffmpeg
                    mux = out / 'MUX.mp4'
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v','error','-nostdin','-y',
                                    '-i',str(out/'VIDEO.mp4'),'-i',str(wav),'-c:v','copy','-c:a','aac',
                                    '-threads','2','-af','apad','-t',str(writer.frames/24),str(mux)],
                                   check=True, timeout=180)
                    mux.replace(out/'VIDEO.mp4')
                    result = dict(status='complete_changed_interface_streaming_diagnostic', policy=policy,
                                  actual_frames=writer.frames, FPS=24, NFE=50, records=records,
                                  video=str(out/'VIDEO.mp4'), video_sha256=file_sha(out/'VIDEO.mp4'),
                                  shared_source=source, checkpoint_sha256=request['weights_spec']['checkpoint_sha256'],
                                  request_sha256=file_sha(request_path), **counts,
                                  clean_commit_rule_unchanged=not policy.get('clean_feedback', False),
                                  clean_commit_rule='own_clean_video_text_feedback' if policy.get('clean_feedback') else 'shared_A_writer',
                                  changed_interface_experiment=True, execution_contract=execution_contract,
                                  old_history_queries_recomputed=0,
                                  model_load_seconds=session['load_seconds'],
                                  model_load_accounting='shared across declared rollouts, do not sum per-policy',
                                  wall_seconds=time.monotonic()-started, formal_memory_quality_claim=False)
                    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
                    if session['reshard']:
                        session['audio_vae'].cpu()
                        torch.cuda.empty_cache()
                dist.barrier()
            peaks = [None] * 8
            dist.all_gather_object(peaks, torch.cuda.max_memory_allocated()/2**30)
            if rank == 0:
                (Path(os.environ['INFER_OUTPUT_DIR'])/'RUN_SUMMARY.json').write_text(json.dumps(dict(
                    policies=policies, load_seconds=session['load_seconds'], peak_allocated_gib_by_rank=peaks,
                    formal_memory_quality_claim=False, execution_contract=execution_contract),indent=2)+'\n')
    finally:
        if writer is not None and not writer.closed:
            writer.__exit__(RuntimeError, None, None)
        dist.destroy_process_group()


class RuleProbe:
    def __init__(self, model, core, visibility, text_time):
        self.model, self.core = model, core
        self.visibility, self.text_time = visibility, text_time
        self.calls = 0
        self.first = None

    def __getattr__(self, name):
        try:
            return getattr(self.model, name)
        except AttributeError:
            return getattr(self.model.transformer, name)

    def __call__(self, **inputs):
        inputs = dict(inputs)
        if self.text_time == 'fixed_clean':
            clock = inputs['timestep'][inputs['timestep_indices']].clone()
            clock[inputs['text_indices']] = 1.
            inputs['timestep'], inputs['timestep_indices'] = torch.unique(clock, sorted=True, return_inverse=True)
        if self.visibility == 'readonly_text':
            nt = inputs['text_indices'].numel()
            total = len(inputs['position_ids'])
            ctx = self.core.Context((self.core.Split('text', 0, 0, nt, ()),
                                     self.core.Split('noisy', 0, nt, total, ())))
            inputs['attention_mask'] = self.core.bind(self.model.transformer, inputs, ctx)
        output = self.model(**inputs)
        if self.calls == 0:
            negative = self.model(**{**inputs, 'encoder_hidden_states': torch.zeros_like(inputs['encoder_hidden_states'])})
            self.first = dict(prompt_tensor_sha256=tensor_sha(inputs['encoder_hidden_states']),
                              video_noise_sha256=tensor_sha(inputs['hidden_states']),
                              audio_noise_sha256=tensor_sha(inputs['audio_hidden_states']),
                              positions_sha256=tensor_sha(inputs['position_ids']),
                              actual_text_time=float(inputs['timestep'][inputs['timestep_indices'][inputs['text_indices'][0]]]),
                              video_velocity_sha256=tensor_sha(output[0]),
                              audio_velocity_sha256=tensor_sha(output[1]),
                              zero_text_video_relative_l2=float(torch.linalg.vector_norm(output[0].float()-negative[0].float()) /
                                                               torch.linalg.vector_norm(output[0].float()).clamp_min(1e-12)))
        self.calls += 1
        return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    args = ap.parse_args()
    request_path = Path(args.request)
    request = json.loads(request_path.read_text())
    if request.get('mode') == 'streaming_text_probe':
        streaming_text_probe(request_path, request)
        return
    if request.get('diagnostic_only') is not True or request.get('history_blocks') != 0:
        raise ValueError('Explicit empty-history diagnostic required')
    if request.get('NFE') != 50 or request.get('num_frames') != 124:
        raise ValueError('Use the declared unchanged 124-frame, 50-NFE short protocol')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30), device_id=device)
    try:
        if dist.get_world_size() != 8:
            raise ValueError('One eight-rank model session is required')
        # Prevent accidentally assembling one model from different weight shards.
        identities = [None] * 8
        dist.all_gather_object(identities, file_sha(request_path))
        if len(set(identities)) != 1:
            raise ValueError('All ranks must load the same frozen checkpoint request')
        if request['weights_spec']['kind'] == 'full':
            # Read the large shared checkpoint only once for the hash audit.
            error = [None]
            if rank == 0:
                try:
                    if file_sha(request['weights_spec']['checkpoint']) != request['weights_spec']['checkpoint_sha256']:
                        error[0] = 'Checkpoint file hash differs from the frozen handoff'
                except Exception as exc:
                    error[0] = repr(exc)
            dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise ValueError(error[0])
        code, source = shared_modules()
        core = code['h3_fixed_kv_stage1']
        session = load_session(request['weights_spec'], device, dense=True, diagnostic_native_rules=True)
        visibility, text_time = [('joint', 'native'), ('joint', 'fixed_clean'),
                                  ('readonly_text', 'native'), ('readonly_text', 'fixed_clean')][rank % 4]
        if visibility == 'readonly_text':
            blocks = []
            for block in session['model'].transformer.transformer_blocks:
                while hasattr(block, '_checkpoint_wrapped_module'):
                    block = block._checkpoint_wrapped_module
                blocks.append(block)
            core.install(SimpleNamespace(transformer_blocks=blocks, token_refiner=session['model'].transformer.token_refiner))
        proxy = RuleProbe(session['model'], core, visibility, text_time)
        prompt = torch.load(request['prompt_path'], map_location='cpu', weights_only=True)
        if prompt['prompt'] != request['prompt']:
            raise ValueError('Prompt plaintext does not match its encoding')
        if tensor_sha(prompt['prompt_embeds']) != request['prompt_tensor_sha256']:
            raise ValueError('Prompt tensor changed')
        seed = request['seed'] + rank // 4
        times = []
        started = time.monotonic()
        with torch.inference_mode():
            video, audio = generate_latents(proxy, prompt['prompt_embeds'].to(device), prompt['text_token_tags'],
                                           124, 50, seed, device, step_seconds=times)
            if len(times) != 50 or proxy.calls != 50:
                raise ValueError('Sampler step count changed')
            if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()):
                raise ValueError('Nonfinite generated output')
            if proxy.first['prompt_tensor_sha256'] != request['prompt_tensor_sha256']:
                raise ValueError('Actual model input differs from frozen text tensor')
            out = Path(os.environ['INFER_OUTPUT_DIR'])
            name = f'rank{rank}_{visibility}_{text_time}_seed{seed}'
            archive = out / (name + '_LATENTS.pt')
            torch.save(dict(video=video.cpu(), audio=audio.cpu(), first_NFE=proxy.first), archive)
            path = out / (name + '.mp4')
            session['vae'].to(device)
            session['audio_vae'].to(device)
            decode_and_save(video, audio, session['vae'], session['audio_vae'], str(path), str(device))
        report = dict(status='complete_empty_history_diagnostic', rank=rank, seed=seed,
                      weight_id=request['weight_id'], checkpoint_sha256=request['weights_spec'].get('checkpoint_sha256'),
                      checkpoint_step=session['checkpoint_step'], visibility=visibility, text_time=text_time,
                      current_document_latent_frames=37, frames=124, FPS=24, NFE=50,
                      extra_zero_text_model_evaluations=1, CFG=1., history_blocks=0, prompt_switches=0,
                      first_NFE=proxy.first, video=str(path), video_sha256=file_sha(path),
                      latent_archive=str(archive), latent_archive_sha256=file_sha(archive),
                      request_sha256=file_sha(request_path), shared_source=source['bundle_sha256'],
                      total_seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      diagnostic_only=True, formal_memory_quality_claim=False, changed_interface_experiment=True,
                      frame_window_protocol='native legal 124-frame document; not fixed5 rollout',
                      GPU=torch.cuda.get_device_name(device))
        (out / (name + '_RESULT.json')).write_text(json.dumps(report, indent=2)+'\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
