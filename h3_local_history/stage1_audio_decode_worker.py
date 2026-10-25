"""Decode Stage1 generated audio latents and mux with an already assembled video."""
import argparse, hashlib, json, os, subprocess, wave
from pathlib import Path
import torch

def sha(path):
 h=hashlib.sha256();
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--request',required=True); a=ap.parse_args()
 req=json.loads(Path(a.request).read_text()); out=Path(os.environ['INFER_OUTPUT_DIR']); out.mkdir(parents=True,exist_ok=True)
 from diffusers import AutoencoderKLMiniMaxH3Audio
 base=req['base_source']; chunks=[Path(x) for x in req['chunk_paths']]
 audio=[]
 for p in chunks:
  d=torch.load(p,map_location='cpu',weights_only=True); x=d['audio']
  if x.ndim!=3 or x.shape[0]!=1 or x.shape[2]!=32: raise ValueError(f'bad audio shape {p}: {tuple(x.shape)}')
  audio.append(x[0].reshape(2,-1,32).permute(0,2,1).to(torch.bfloat16))
 al=torch.cat(audio,dim=2)
 if not bool(torch.isfinite(al).all()): raise ValueError('audio latent nonfinite')
 device=torch.device('cuda'); av=AutoencoderKLMiniMaxH3Audio.from_pretrained(base,subfolder='audio_vae').to(device).eval()
 mean=torch.tensor(av.config.latents_mean,device=device).view(1,-1,1); std=torch.tensor(av.config.latents_std,device=device).view(1,-1,1)
 with torch.inference_mode(): sound=av.decode(al.to(device)*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
 if not bool(torch.isfinite(sound).all()): raise ValueError('decoded audio nonfinite')
 rate=int(getattr(av.config,'sampling_rate',32000)); frames=int(req['video_frames']); fps=float(req['fps']); desired=round(frames/fps*rate)
 if sound.shape[-1]<desired: sound=torch.nn.functional.pad(sound,(0,desired-sound.shape[-1]))
 sound=sound[:,:desired]
 wav=out/'AUDIO.wav'
 with wave.open(str(wav),'wb') as f:
  f.setnchannels(2);f.setsampwidth(2);f.setframerate(rate);f.writeframes((sound.T*32767).round().short().contiguous().numpy().tobytes())
 video=Path(req['video_path']);
 if not video.is_file(): raise FileNotFoundError(video)
 import imageio_ffmpeg
 ff=imageio_ffmpeg.get_ffmpeg_exe(); mux=out/'VIDEO_AV.mp4'
 subprocess.run([ff,'-v','error','-nostdin','-y','-i',str(video),'-i',str(wav),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-af','apad','-t',str(frames/fps),str(mux)],check=True,timeout=180)
 result=dict(status='complete_stage1_audio_decode_and_mux',chunks=len(chunks),audio_latents=int(al.shape[2]),audio_samples=int(desired),audio_sample_rate=rate,video_frames=frames,fps=fps,audio_latent_finite=True,decoded_audio_finite=True,video_sha256=sha(video),audio_sha256=sha(wav),mux_sha256=sha(mux),video_path=str(video),audio_path=str(wav),mux_path=str(mux),future_audio_context_allowed=False,audio_encoder_lookahead_sec=0.,formal_memory_quality_claim=False)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
