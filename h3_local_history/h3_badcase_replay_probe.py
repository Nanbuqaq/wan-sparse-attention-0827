"""Replay seven original H3 chunks; inspect actual current attention, never old queries."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def verify(request):
    if request['schema']!='h3_seven_chunk_exact_replay_v1' or request['history_recomputation_allowed'] is not False:
        raise ValueError('Wrong replay contract')
    for f in request['frozen_files']:
        if sha(f['path'])!=f['sha256']:raise ValueError('Frozen replay input differs: '+f['path'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args()
    path=Path(a.request);request=json.loads(path.read_text())
    if a.preflight_only:
        verify(request);write(path.with_name('PREP_READY.json'),dict(request_sha256=sha(path)));return
    if json.loads(path.with_name('PREP_READY.json').read_text())['request_sha256']!=sha(path):raise ValueError('Request changed')
    # These two modules are the exact published files used by the original video.
    sys.path.insert(0,request['original_source_directory'])
    import torch
    import torch.distributed as dist
    from h3_weights_interface_diagnostic import (stage1b_chunk,stage1b_clock,stage1b_clocks,
        stage1b_context,install_stage1b_processors,tensor_sha,stage1b_decode_long)
    import h3_weights_interface_diagnostic as core
    from h3_shared_loader import load_session
    import h3_shared_loader as loader
    from h3_stage1_clock_probe import VideoKVBank
    if sha(core.__file__)!=request['original_worker_sha256'] or sha(loader.__file__)!=request['original_loader_sha256']:
        raise ValueError('Original numerical module import drift')
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK']);torch.set_num_threads(2);torch.cuda.set_device(local)
    device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=12),device_id=device)
    out=Path(os.environ['INFER_OUTPUT_DIR']);out.mkdir(exist_ok=True)
    started=time.monotonic()
    try:
        if dist.get_world_size()!=8:raise ValueError('Eight FSDP ranks required')
        config=json.loads(Path(request['original_config']).read_text())['request']
        session=load_session(config['weights_spec'],device,dense=True,load_decoders=rank==0,stage1b_current_only=True)
        processors=install_stage1b_processors(session['model'])
        from torch.distributed.tensor import DTensor,Replicate,distribute_tensor
        cw=session['model'].transformer.context_embedder.weight;mesh=cw.device_mesh if isinstance(cw,DTensor) else None
        prompt=torch.load(config['prompt_paths'][0],map_location='cpu',weights_only=True)
        old=json.loads(Path(request['original_result']).read_text())['records']
        generator=torch.Generator(device=device).manual_seed(int(config['seed']))
        video_template=torch.zeros(24,422,48,84,dtype=torch.bfloat16)
        audio_template=torch.zeros(2,32,round(1433*40/24),dtype=torch.bfloat16)
        bank=VideoKVBank(local_frames=9,sink_frames=3);records=[];archives=[];reference_rows=[];processor_rows=[]
        training_reference=None
        if request.get('training_attention_reference'):
            reference_path=request['training_attention_reference']
            spec=importlib.util.spec_from_file_location('frozen_training_attention_reference',reference_path)
            training_reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(training_reference)
        import h3_attention_backend
        actual_attention=h3_attention_backend.flash_attention
        active=dict(chunk=-1,rung=-1,phase='',layer=-1,group_index=0)
        sampled_heads=[0,7,15,31];query_budget=8
        def checked_attention(q,k,v,backend='torch_flash'):
            actual=actual_attention(q,k,v,backend)
            if active['chunk']==6 and active['rung']==0 and active['phase']=='sample' and active['layer'] in (0,24,49):
                # Same real keys/values, selected actual queries; FP32 explicit logits as reference.
                heads=[h for h in sampled_heads if h<q.shape[1]]
                indices=torch.linspace(0,q.shape[2]-1,min(query_budget,q.shape[2]),device=q.device).round().long().unique()
                sq=q[:,heads].index_select(2,indices).float();sk=k[:,heads].float();sv=v[:,heads].float()
                logits=torch.matmul(sq,sk.transpose(-2,-1))*(q.shape[-1]**-.5)
                ref=torch.matmul(logits.softmax(-1),sv);candidate=actual[:,heads].index_select(2,indices).float()
                delta=(candidate-ref).abs()
                reference_rows.append(dict(layer=active['layer'],group_index=active['group_index'],heads=heads,
                    queries=indices.cpu().tolist(),q_shape=list(q.shape),k_shape=list(k.shape),
                    max_abs=float(delta.max()),mean_abs=float(delta.mean()),reference_absmax=float(ref.abs().max()),
                    reference='FP32 explicit scaled qk softmax v; identical real retained K/V',
                    full_attention_output_equivalence_proven=False))
            active['group_index']+=1
            return actual
        h3_attention_backend.flash_attention=checked_attention
        class Observed:
            _attention_backend=None;_parallel_config=None
            def __init__(self,processor):self.processor=processor
            def __call__(self,attn,hidden_states,rotary_emb=None,attention_mask=None,is_causal=False):
                active['layer']=self.processor.layer_id;active['group_index']=0
                actual=self.processor(attn,hidden_states,rotary_emb,attention_mask,is_causal)
                if training_reference is not None and active['chunk']==6 and active['phase']=='sample':
                    from h3_worldattention_causal import H3CausalContext
                    layout=attention_mask.layout
                    ref_layout=training_reference.WorldAttentionCausalLayout(layout.block_ids,layout.video_frames,layout.local_frames,layout.sink_frames)
                    context=H3CausalContext(ref_layout,cached=attention_mask.cached,write_bank=None,
                        modulation_segments=attention_mask.modulation_segments)
                    # Adapt only the old frozen context container to the training API.
                    object.__setattr__(context,'text_scope','current_only')
                    active['phase']='reference'
                    expected=training_reference.H3WorldAttentionCausalAttention(self.processor.layer_id)(
                        attn,hidden_states,rotary_emb,context,is_causal)
                    active['phase']='sample'
                    delta=(actual.float()-expected.float()).abs()
                    processor_rows.append(dict(layer=self.processor.layer_id,rung=active['rung'],
                        max_abs=float(delta.max()),mean_abs=float(delta.mean()),exact=torch.equal(actual,expected),
                        full_current_output_shape=list(actual.shape),same_current_hidden=True,
                        same_immutable_stored_KV=True,old_history_queries_recomputed=0,
                        reference_source_sha256=sha(reference_path)))
                return actual
        for wrapped,processor in zip(session['model'].transformer.transformer_blocks,processors):
            block=wrapped
            while hasattr(block,'_checkpoint_wrapped_module'):block=block._checkpoint_wrapped_module
            block.attn.set_processor(Observed(processor))
        with torch.inference_mode():
            for ci in range(7):
                start=0 if ci==0 else 2+5*ci;end=start+(7 if ci==0 else 5)
                ae=round((5+17*((end-2)//5))*40/24)
                batch=stage1b_chunk(video_template,audio_template,prompt['prompt_embeds'],prompt['text_token_tags'],generator,device,start,end,ae)
                x=batch['inputs'];hashes=[None,None,tensor_sha(x['position_ids']),tensor_sha(x['encoder_hidden_states'])]
                if mesh is not None:x['encoder_hidden_states']=distribute_tensor(x['encoder_hidden_states'],mesh,[Replicate()])
                video=torch.randn(x['hidden_states'].shape,device=device,generator=generator)
                audio=torch.randn(x['audio_hidden_states'].shape,device=device,generator=generator)
                hashes[:2]=[tensor_sha(video),tensor_sha(audio)]
                bank_before={layer:(tensor_sha(e.key),tensor_sha(e.value),tensor_sha(e.frames))
                             for layer,e in bank.snapshot().items()} if ci==6 else None
                rungs=[]
                for rung in range(4):
                    tv,ta=stage1b_clocks(1-rung/4);x['hidden_states'],x['audio_hidden_states']=video,audio;stage1b_clock(batch,tv,ta)
                    active.update(chunk=ci,rung=rung,phase='sample')
                    pv,pa=session['model'].transformer(**x,attention_mask=stage1b_context(batch,ci,start,end,bank))
                    clean_v=(video+(1-tv)*pv.float()).to(torch.bfloat16);clean_a=(audio+(1-ta)*pa.float()).to(torch.bfloat16)
                    if ci==6 and rank==0:
                        torch.save(dict(video=video.cpu(),audio=audio.cpu(),prediction_video=pv.cpu(),prediction_audio=pa.cpu(),
                            clean_video=clean_v.cpu(),clean_audio=clean_a.cpu(),tv=tv,ta=ta,rung=rung),out/f'chunk006_rung{rung}.pt')
                    rungs.append(dict(rung=rung,video_time=tv,audio_time=ta,text_time=tv,
                        clean_video_sha256=tensor_sha(clean_v),clean_audio_sha256=tensor_sha(clean_a)))
                    if rung<3:
                        nv,na=stage1b_clocks(1-(rung+1)/4)
                        ev=torch.randn(clean_v.shape,device=device,generator=generator);ea=torch.randn(clean_a.shape,device=device,generator=generator)
                        hashes.extend([tensor_sha(ev),tensor_sha(ea)])
                        video=(nv*clean_v+(1-nv)*ev).to(torch.bfloat16);audio=(na*clean_a+(1-na)*ea).to(torch.bfloat16)
                    else:video,audio=clean_v,clean_a
                if hashes!=old[ci]['noise_and_position_hashes']:raise ValueError('Original noise/text/position/RNG reproduction failed')
                if ci==6:
                    bank_after={layer:(tensor_sha(e.key),tensor_sha(e.value),tensor_sha(e.frames)) for layer,e in bank.snapshot().items()}
                    if bank_before!=bank_after:raise ValueError('Historical KV mutated during denoising')
                    write(out/f'KV_rank{rank}.json',dict(full_stored_KV_sha_before=bank_before,
                        full_stored_KV_sha_after=bank_after,exact_immutable=True,scope='all stored K/V bytes, every layer on this rank'))
                active.update(phase='commit');stage1b_clock(batch,1.,1.);x['hidden_states'],x['audio_hidden_states']=video.detach(),audio.detach()
                session['model'].transformer(**x,attention_mask=stage1b_context(batch,ci,start,end,bank,bank))
                row=dict(chunk=ci,input_hashes_match=True,video_sha256=tensor_sha(video),audio_sha256=tensor_sha(audio),
                    original_video_sha256=old[ci]['video_sha256'],original_audio_sha256=old[ci]['audio_sha256'],
                    output_exact_original=tensor_sha(video)==old[ci]['video_sha256'] and tensor_sha(audio)==old[ci]['audio_sha256'],
                    rungs=rungs,old_history_queries_recomputed=0,clean_writes=1)
                if rank==0:
                    archive=out/f'chunk{ci:03d}_GENERATED.pt';torch.save(dict(video=video.cpu(),audio=audio.cpu(),chunk=ci),archive);archives.append(str(archive))
                records.append(row);print('REPLAY_CHUNK',ci,row['output_exact_original'],flush=True)
                del batch,x,video,audio,clean_v,clean_a,pv,pa
            write(out/f'ATTENTION_rank{rank}.json',dict(rows=reference_rows,scope='sampled real heads/queries in layers0/24/49, chunk6 first rung',
                history_recomputed=0,production_attention_unchanged=True))
            if training_reference is not None:
                write(out/f'TRAINING_PROCESSOR_rank{rank}.json',dict(rows=processor_rows,
                    all_50_layers_four_rungs=len(processor_rows)==200,
                    all_exact=bool(processor_rows) and all(row['exact'] for row in processor_rows),
                    full_current_output_compared=True,history_recomputed=0,production_output_unchanged=True,
                    context_adapter='frozen H3CausalContext with text_scope=current_only; tensor data unchanged'))
            del bank;torch.cuda.empty_cache();dist.barrier();export_error=[None]
            if rank==0:
                try:
                    media=stage1b_decode_long(session,out,archives,124,device)
                    write(out/'RESULT.json',dict(status='complete',records=records,**media,
                        original_worker_sha256=request['original_worker_sha256'],original_loader_sha256=request['original_loader_sha256'],
                        replay_worker_sha256=sha(__file__),checkpoint_sha256=config['weights_spec']['checkpoint_sha256'],
                        request_sha256=sha(path),history_recomputed=0,old_latents_reencoded=False,
                        original_seven_chunks_exact=all(r['output_exact_original'] for r in records),
                        sampled_attention_reference_only=True,quality_pass=None,wall_seconds=time.monotonic()-started))
                except Exception as exc:export_error[0]=repr(exc)
            dist.broadcast_object_list(export_error,src=0)
            if export_error[0]:raise RuntimeError(export_error[0])
            dist.barrier()
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
