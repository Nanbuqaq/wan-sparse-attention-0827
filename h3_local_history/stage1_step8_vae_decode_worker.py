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
    decoder=BufferedVideoDecoder(vae,'cuda');chunks=[]
    for i in range(4):
        v=torch.load(req['chunk_paths'][i],map_location='cuda',weights_only=True)['video']
        v=v.to(torch.bfloat16)
        if v.ndim==3:v=v.unsqueeze(0)
        # Stage1 rows are [1, N, 96], matching the native H3 patch layout.
        rows=v.reshape(1,v.shape[1],48//2,84//2,24,1,2,2).permute(0,4,1,5,2,6,3,7).reshape(1,24,v.shape[1],48,84)
        chunks.append(rows)
        value=decoder.push(rows)
        if value is not None:torch.save(value.cpu(),out/f'decoded_part{i:02d}.pt')
    tail=decoder.finish()
    if tail is not None:torch.save(tail.cpu(),out/'decoded_tail.pt')
    report=dict(status='stage1_step8_vae_redecode_complete',chunks=4,finite_all=all(bool(torch.isfinite(x).all()) for x in chunks),codec_summary=decoder.last_summary,latent_shapes=[list(x.shape) for x in chunks])
    (out/'VAE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
