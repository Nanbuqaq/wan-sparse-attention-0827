"""Decode retained H3 badcases twice; never load a DiT or regenerate history."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+chr(10));tmp.replace(path)


def read_latent(item):
    import torch
    if sha(item['path'])!=item['sha256']:raise ValueError('Retained archive changed')
    value=torch.load(item['path'],map_location='cpu',weights_only=True)['video']
    count=item['latent_count']
    if value.ndim==3:
        value=value.reshape(1,count,24,42,24,1,2,2).permute(0,4,1,5,2,6,3,7).reshape(1,24,count,48,84)
    if tuple(value.shape)!=(1,24,count,48,84):raise ValueError('Archive layout differs')
    if not torch.isfinite(value).all():raise ValueError('Nonfinite original latent')
    return value


def run(request,case,out):
    import torch
    import av
    import numpy as np
    from PIL import Image,ImageDraw
    from train_h3_v2_local_history_ddp import bootstrap
    bootstrap()
    from diffusers import AutoencoderKLMiniMaxH3
    from h3_buffered_video_decode import BufferedVideoDecoder
    from h3_frame_writer import FrameWriter
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.set_grad_enabled(False)
    device=torch.device('cuda',0);out.mkdir(parents=True,exist_ok=True)
    chunks=[read_latent(item) for item in case['archives']]
    stats=[dict(chunk=item['chunk'],shape=list(z.shape),mean=float(z.float().mean()),
        std=float(z.float().std()),absmax=float(z.float().abs().max())) for item,z in zip(case['archives'],chunks)]
    latent=torch.cat(chunks,dim=2).to(device)
    vae=AutoencoderKLMiniMaxH3.from_pretrained(request['base_source'],subfolder='vae').to(device).eval()
    mean=torch.tensor(vae.config.latents_mean,device=device).view(1,-1,1,1,1)
    std=torch.tensor(vae.config.latents_std,device=device).view(1,-1,1,1,1)
    tick=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        native=vae.decode(latent*std+mean,return_dict=False)[0].cpu()
    frames=case['expected_frames']
    if native.shape[2]!=frames:raise ValueError('Native decoded frame count differs')
    if not torch.isfinite(native).all():raise ValueError('Native decoded pixels are nonfinite')
    with FrameWriter(out/'native_VAE.mp4',24,encoder_threads=2) as writer:
        for start in range(0,frames,17):writer.append(native[:,:,start:start+17])
    pixel_mean=torch.tensor((.485,.456,.406)).view(1,3,1,1,1)
    pixel_std=torch.tensor((.229,.224,.225)).view(1,3,1,1,1)
    def pixels(x):return ((x.float()*pixel_std+pixel_mean).clamp(0,1)*255).round().byte()
    decoder=BufferedVideoDecoder(vae,device);offset=0;per_frame=[];maxabs=0.;pixelmax=0
    with FrameWriter(out/'buffered_VAE.mp4',24,encoder_threads=2) as writer:
        def consume(value):
            nonlocal offset,maxabs,pixelmax
            if value is None:return
            value=value.cpu();reference=native[:,:,offset:offset+value.shape[2]]
            if not torch.isfinite(value).all():raise ValueError('Buffered decoded pixels are nonfinite')
            delta=(reference.float()-value.float()).abs()
            pd=(pixels(reference).int()-pixels(value).int()).abs()
            maxabs=max(maxabs,float(delta.max()));pixelmax=max(pixelmax,int(pd.max()))
            for j in range(value.shape[2]):
                per_frame.append(dict(frame=offset+j,float_max_abs=float(delta[:,:,j].max()),
                    pixel_max_abs=int(pd[:,:,j].max()),pixel_MAE=float(pd[:,:,j].float().mean())))
            writer.append(value);offset+=value.shape[2]
        for z in chunks:consume(decoder.push(z.to(device)))
        consume(decoder.finish())
    if offset!=frames:raise ValueError('Buffered decoded frame count differs')
    original_MAE=[];original_pts=[];previews=[]
    wanted=set(case['inspect_frames'])
    with av.open(case['original_video']) as container:
        container.streams.video[0].codec_context.thread_count=2
        for i,frame in enumerate(container.decode(video=0)):
            if i>=frames:break
            pts=float(frame.pts*frame.time_base);original_pts.append(pts)
            if abs(pts-i/24)>1e-5:raise ValueError('Original PTS differs')
            reference=pixels(native[:,:,i:i+1])[0,:,0].permute(1,2,0).numpy()
            original=frame.to_ndarray(format='rgb24')
            original_MAE.append(float(np.abs(original.astype(np.float32)-reference.astype(np.float32)).mean()))
            if i in wanted:previews.append((i,Image.fromarray(reference).resize((336,192))))
    if len(original_MAE)!=frames:raise ValueError('Original video is incomplete')
    # Decode an individual real seven-latent window; changed decoder scope is explicit.
    unblended=[]
    for window in case.get('unblended_windows',[]):
        lo=5*window;piece=latent[:,:,lo:lo+7]
        if piece.shape[2]!=7:continue
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            value=vae.decode(piece*std+mean,return_dict=False)[0].cpu()
        path=out/f'unblended_window{window:03d}.mp4'
        with FrameWriter(path,24,encoder_threads=2) as writer:writer.append(value)
        unblended.append(dict(window=window,latent_range=[lo,lo+7],video=str(path),
            changed_decoder_diagnostic=True,DiT_forward_calls=0))
    sheet=Image.new('RGB',(4*336,218*max(1,(len(previews)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
    for k,(i,im) in enumerate(previews):
        x=k%4*336;y=k//4*218;sheet.paste(im,(x,y+26))
        draw.text((x+3,y+3),f'f{i} t={i/24:.3f}s native VAE',fill='black')
    sheet.save(out/'CONTACT.jpg',quality=94)
    summary=dict(status='complete',case=case['id'],checkpoint_sha256=case['checkpoint_sha256'],
        code_sha256=sha(__file__),request_sha256=request['_request_sha256'],
        VAE_asset_files=request['vae_files'],sampling_not_executed=True,
        source_result=case['source_result'],source_result_sha256=case['source_result_sha256'],
        original_video=case['original_video'],original_video_sha256=case['original_video_sha256'],
        native_video=str(out/'native_VAE.mp4'),buffered_video=str(out/'buffered_VAE.mp4'),
        source_latent_archives=case['archives'],native_frames=frames,buffered_frames=offset,
        float_max_abs=maxabs,pixel_max_abs=pixelmax,pixels_exact=pixelmax==0,
        per_frame=per_frame,original_mp4_vs_native_pixel_MAE=original_MAE,
        source_comparison_note='Original is lossy H264. Prefix-end frames can differ because the complete source uses later latent context; compare interior separately.',
        original_PTS=original_pts,codec=decoder.last_summary,latent_statistics=stats,
        unblended_windows=unblended,DiT_forward_calls=0,history_recomputed=0,
        new_generation=False,quality_pass=None,wall_seconds=time.monotonic()-tick,
        hardware=torch.cuda.get_device_name(0),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write(out/'RESULT.json',summary)
    print(json.dumps(dict(case=case['id'],pixels_exact=pixelmax==0,float_max_abs=maxabs)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    parser.add_argument('--lane',type=int);parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();path=Path(args.request);request=json.loads(path.read_text())
    if request['schema']!='h3_badcase_saved_codec_v1':raise ValueError('Unknown request')
    if args.preflight_only:
        for case in request['cases']:
            for key in ['source_result','original_video']:
                if sha(case[key])!=case[key+'_sha256']:raise ValueError('Frozen source changed')
            for item in case['archives']:read_latent(item)
        for item in request['vae_files']:
            if sha(item['path'])!=item['sha256']:raise ValueError('VAE asset changed')
        write(path.with_name('PREP_READY.json'),dict(request_sha256=sha(path),DiT_forward_calls=0))
        return
    if json.loads(path.with_name('PREP_READY.json').read_text())['request_sha256']!=sha(path):raise ValueError('Request changed')
    request['_request_sha256']=sha(path)
    case=request['cases'][args.lane];out=Path(os.environ['INFER_OUTPUT_DIR'])/case['id']
    try:run(request,case,out)
    except Exception:
        out.mkdir(parents=True,exist_ok=True);write(out/'FAILURE.json',dict(error=traceback.format_exc(),new_generation=False));raise

if __name__=='__main__':main()
