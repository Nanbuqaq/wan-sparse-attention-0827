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
