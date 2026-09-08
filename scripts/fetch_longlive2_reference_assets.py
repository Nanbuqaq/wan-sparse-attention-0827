#!/usr/bin/env python3
"""Fetch source-locked public native BF16 reference assets into a private root.

No shared environment changes. No base generator download: the complete merged
generator checkpoint will be strictly loaded into its official architecture.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download

BASE='Wan-AI/Wan2.2-TI2V-5B'
BASE_REV='921dbaf3f1674a56f47e83fb80a34bac8a8f203e'
GEN='Efficient-Large-Model/LongLive-2.0-5B'
GEN_REV='8521079b863720a57c1a8d9b19c8d9e6ccb04c0f'
FILES=[
    (GEN,GEN_REV,'model_bf16.pt','checkpoints/model_bf16.pt','ec9063a44ea3c91e8ff55edcdd58dba3f1bcf6ac9091249629cb57fcebe35fd8'),
    (BASE,BASE_REV,'Wan2.2_VAE.pth','wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth','20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'),
    (BASE,BASE_REV,'models_t5_umt5-xxl-enc-bf16.pth','wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth','7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d'),
]+[(BASE,BASE_REV,name,'wan_models/Wan2.2-TI2V-5B/'+name,None) for name in (
    'config.json','google/umt5-xxl/tokenizer.json','google/umt5-xxl/tokenizer_config.json','google/umt5-xxl/special_tokens_map.json')]


def sha256(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(16*1024**2):result.update(block)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reuse-text-encoder',type=Path);args=p.parse_args()
    args.root.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    def fetch(spec):
        repo,revision,name,relative,expected=spec
        dest=args.root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            if name=='models_t5_umt5-xxl-enc-bf16.pth' and args.reuse_text_encoder:
                actual=sha256(args.reuse_text_encoder)
                if actual!=expected:raise ValueError('existing text encoder is not the official weight')
                dest.symlink_to(args.reuse_text_encoder.resolve())
            else:
                path=Path(hf_hub_download(repo_id=repo,revision=revision,filename=name,
                    local_dir=args.root/'download'/repo.split('/')[-1],token=False))
                dest.symlink_to(path.resolve())
        actual=sha256(dest)
        if expected and actual!=expected:raise ValueError('checksum mismatch: '+relative)
        record=dict(repo=repo,revision=revision,file=name,path=str(dest),bytes=dest.stat().st_size,
                    sha256=actual,verified_LFS_checksum=expected is not None)
        print(json.dumps(record),flush=True);return record
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,FILES))
    report=dict(status='pass',files=records,wall_s=time.perf_counter()-started,
        generator_loading='official_from_config_then_strict_complete_merged_checkpoint_no_base_generator',
        code_commit='6b36d20ec6f7958d29d11a704dfa64611a9f2572')
    with (args.root/'assets_manifest.json').open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')


if __name__=='__main__':main()
