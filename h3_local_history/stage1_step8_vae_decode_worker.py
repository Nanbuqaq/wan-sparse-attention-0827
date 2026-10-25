"""Decode Stage1 health latents with the released H3 VAE."""
import argparse, json, os
from pathlib import Path
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args()
    req=json.loads(Path(a.request).read_text());out=Path(os.environ['INFER_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    from diffusers import AutoencoderKLMiniMaxH3
    from h3_buffered_video_decode import BufferedVideoDecoder
    vae=AutoencoderKLMiniMaxH3.from_pretrained(req['base_source'],subfolder='vae').to('cuda').eval()
    decoder=BufferedVideoDecoder(vae,'cuda');latent_shapes=[];finite_all=True
    chunk_paths=list(req.get('chunk_paths', []))
    if not chunk_paths:
        chunk_dir=Path(req['chunk_dir'])
        chunk_paths=sorted(str(p) for p in chunk_dir.glob('chunk*.pt'))
    if not chunk_paths:
        raise ValueError('No chunk paths supplied')
    for i,chunk_path in enumerate(chunk_paths):
        v=torch.load(chunk_path,map_location='cuda',weights_only=True)['video']
        v=v.to(torch.bfloat16)
        if v.ndim!=3 or v.shape[0]!=1 or v.shape[2]!=96: raise ValueError('Expected [1,frames*1008,96] Stage1 rows')
        frames=v.shape[1]//1008
        if frames*1008!=v.shape[1]: raise ValueError('Stage1 row count is not divisible by 1008')
        # Stage1 rows are [1, frames*1008, 96], matching native H3 patch layout.
        rows=v.reshape(1,frames,48//2,84//2,24,1,2,2).permute(0,4,1,5,2,6,3,7).reshape(1,24,frames,48,84)
        latent_shapes.append(list(rows.shape));finite_all = finite_all and bool(torch.isfinite(rows).all())
        value=decoder.push(rows)
        if value is not None:torch.save(value.cpu(),out/f'decoded_part{i:02d}.pt')
    tail=decoder.finish()
    if tail is not None:torch.save(tail.cpu(),out/'decoded_tail.pt')
    report=dict(status='stage1_step8_vae_redecode_complete',chunks=len(chunk_paths),finite_all=finite_all,codec_summary=decoder.last_summary,latent_shapes=latent_shapes)
    (out/'VAE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
