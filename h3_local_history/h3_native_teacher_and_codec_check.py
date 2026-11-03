"""Joint noncausal Dense teacher plus same-latent native/buffered VAE checks.

The native teacher processes all 73 frames together, never a causal prefix.
No new current-block mask or historical KV is installed on it.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist

if 'LOCAL_RANK' in os.environ:torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_teacher_rung_diagnostic import prompt_for_model,stage1b_decode_long,file_sha,tensor_sha


def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def verify_decoder(session,archives,out,device):
 """Compare pre-encoding uint8 pixels, not lossy MP4s, on identical latents."""
 from sample_demo import unpack_video
 from h3_buffered_video_decode import BufferedVideoDecoder
 from h3_frame_writer import FrameWriter
 vae=session['vae'];vae.to(device)
 values=[torch.load(p,map_location='cpu',weights_only=True)['video'] for p in archives]
 packed=torch.cat(values,dim=1);latents=unpack_video(packed[0],22,48,84).to(device)
 mean=torch.tensor(vae.config.latents_mean,device=device).view(1,-1,1,1,1)
 std=torch.tensor(vae.config.latents_std,device=device).view(1,-1,1,1,1)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
  native=vae.decode(latents*std+mean,return_dict=False)[0].cpu()
 decoder=BufferedVideoDecoder(vae,device);parts=[]
 for v in values:
  image=decoder.push(unpack_video(v[0],v.shape[1]//1008,48,84).to(device))
  if image is not None:parts.append(image.cpu())
 tail=decoder.finish()
 if tail is not None:parts.append(tail.cpu())
 buffered=torch.cat(parts,dim=2)
 if native.shape!=buffered.shape or native.shape[2]!=73:raise ValueError('Native/buffered geometry differs')
 delta=(native.float()-buffered.float()).abs()
 pixel_mean=torch.tensor((.485,.456,.406)).view(1,3,1,1,1)
 pixel_std=torch.tensor((.229,.224,.225)).view(1,3,1,1,1)
 def pixels(x):return ((x.float()*pixel_std+pixel_mean).clamp(0,1)*255).round().byte()
 n,b=pixels(native),pixels(buffered)
 stat=dict(raw_float_max_abs=float(delta.max()),raw_float_mean_abs=float(delta.mean()),
           pixels_exact=bool(torch.equal(n,b)),pixel_uint8_max_abs=int((n.int()-b.int()).abs().max()),
           pixel_uint8_mean_abs=float((n.float()-b.float()).abs().mean()),
           framewise_pixel_mean_abs=(n.float()-b.float()).abs().mean((0,1,3,4)).tolist(),
           native_pixels_sha256=tensor_sha(n),buffered_pixels_sha256=tensor_sha(b),
           same_saved_latents=True,frames=73,codec=decoder.last_summary)
 out.mkdir(exist_ok=True,parents=True)
 for name,x in [('native_full_VAE',native),('buffered_VAE',buffered)]:
  with FrameWriter(out/(name+'.mp4'),24,encoder_threads=2) as writer:writer.append(x)
  if writer.frames!=73:raise ValueError('VAE-only video duration changed')
 stat['native_video']=str(out/'native_full_VAE.mp4');stat['buffered_video']=str(out/'buffered_VAE.mp4')
 atomic(out/'CODEC_COMPARISON.json',stat)
 del native,buffered,latents,parts,n,b
 vae.cpu();torch.cuda.empty_cache()
 return stat


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args()
 path=Path(a.request);req=json.loads(path.read_text());rank=int(os.environ['RANK'])
 local=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',local)
 torch.cuda.set_device(local);torch.set_num_threads(2)
 dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=40),device_id=device)
 try:
  if dist.get_world_size()!=8:raise ValueError('Eight ranks required')
  prompt=torch.load(req['prompt_path'],map_location='cpu',weights_only=True)
  if file_sha(req['prompt_path'])!=req['prompt_file_sha256']:raise ValueError('Prompt changed')
  session=load_session(req['teacher_weights_spec'],device,dense=True,load_decoders=rank==0,
                       dense_oracle=True,force_reshard=True)
  class NativeProxy:
   def __getattr__(self,name):return getattr(session['model'].transformer,name)
   def __call__(self,**inputs):
    batch={'inputs':dict(inputs)};prompt_for_model(batch,session['model'])
    return session['model'].transformer(**batch['inputs'])
  noise=torch.load(req['native_noise_path'],map_location='cpu',weights_only=True)
  for k in ['video','audio']:
   if tensor_sha(noise[k])!=req['native_noise_hashes'][k]:raise ValueError('Frozen global noise changed')
  from src.inference.render import generate_latents
  timings=[];begin=time.monotonic()
  with torch.inference_mode():
   video,audio=generate_latents(NativeProxy(),prompt['prompt_embeds'].to(device),prompt['text_token_tags'],
       73,50,req['seed'],device,step_seconds=timings,
       initial_noise_override={'video':noise['video'],'audio':noise['audio']})
  if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()):raise ValueError('Native joint output nonfinite')
  err=[None]
  if rank==0:
   try:
    out=Path(os.environ['INFER_OUTPUT_DIR']);native_out=out/'native_dense_joint50';native_out.mkdir(exist_ok=True,parents=True)
    packed=video.reshape(1,24,22,24,2,42,2).permute(0,2,3,5,1,4,6).reshape(1,22176,96)
    packed_a=audio.permute(0,2,1).reshape(1,244,32)
    archive=native_out/'NATIVE_LATENTS.pt'
    torch.save(dict(video=packed.to(torch.bfloat16).cpu(),audio=packed_a.to(torch.bfloat16).cpu()),archive)
    with torch.inference_mode():media=stage1b_decode_long(session,native_out,[str(archive)],73,device)
    atomic(native_out/'RESULT.json',dict(status='generated_pending_visual_review',**media,
       native_full_document_attention=True,block_autoregressive=False,video_frames=73,NFE=50,
       same_initial_noise_as_retained_causal_control=True,intermediate_noise_grid_differs_from_student4=True,
       KV_bytes=0,teacher_load_seconds=session['load_seconds'],wall_seconds=time.monotonic()-begin,
       per_step_seconds=timings,request_sha256=file_sha(path),source_sha256=file_sha(__file__),quality_verified=False))
    comparisons=[]
    for case in req['codec_controls']+[dict(name='native_joint_new',archives=[str(archive)])]:
     stat=verify_decoder(session,case['archives'],out/'codec_checks'/case['name'],device)
     comparisons.append(dict(name=case['name'],**stat))
    atomic(out/'RUN_SUMMARY.json',dict(status='complete_pending_quality_review',codec_controls=comparisons,
       request_sha256=file_sha(path),code_sha256=file_sha(__file__),native_teacher_has_no_causal_mask=True,
       normal_full_video_latents_generated_jointly=True))
   except Exception as e:err[0]=repr(e)
  dist.broadcast_object_list(err,src=0)
  if err[0]:raise RuntimeError(err[0])
  dist.barrier()
 finally:dist.destroy_process_group()

if __name__=='__main__':main()
