"""Compare native and buffered VAE on retained bad/good student latents only."""
import argparse
import json
import os
from pathlib import Path

import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
    request=json.loads(Path(a.request).read_text())
    if request.get('schema')!='h3_saved_codec_controls_v1':raise ValueError('Explicit saved-latent codec request required')
    case=request['cases'][a.index]
    torch.set_num_threads(2);torch.cuda.set_device(0)
    from train_h3_v2_local_history_ddp import bootstrap
    bootstrap()
    from h3_native_teacher_and_codec_check import verify_decoder,atomic,file_sha
    from diffusers import AutoencoderKLMiniMaxH3
    if file_sha(case['source_result'])!=case['source_result_sha256']:raise ValueError('Source result changed')
    paths=[]
    for item in case['archives']:
        if file_sha(item['path'])!=item['sha256']:raise ValueError('Retained latent changed')
        paths.append(item['path'])
    out=Path(os.environ['INFER_OUTPUT_DIR'])/case['id'];out.mkdir(parents=True,exist_ok=True)
    vae=AutoencoderKLMiniMaxH3.from_pretrained(request['base_source'],subfolder='vae').eval()
    with torch.inference_mode():stat=verify_decoder({'vae':vae},paths,out,torch.device('cuda',0))
    atomic(out/'RESULT.json',dict(status='codec_complete',case=case['id'],source_result=case['source_result'],
        source_result_sha256=case['source_result_sha256'],request_sha256=file_sha(a.request),
        no_DiT_forward=True,no_new_generation=True,scope='first four retained chunks; not full long-video pixel parity',**stat))
    print(json.dumps(dict(case=case['id'],pixels_exact=stat['pixels_exact'],raw_float_max_abs=stat['raw_float_max_abs'])),flush=True)


if __name__=='__main__':main()
