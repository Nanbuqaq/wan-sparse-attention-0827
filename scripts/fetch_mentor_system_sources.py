#!/usr/bin/env python3
"""Read-only public source acquisition with hashes; no shared environment edits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import urllib.request


SOURCES={
    'fa4_paper_v1.html':'https://arxiv.org/html/2603.05451v1',
    'fa1_paper_v2.html':'https://arxiv.org/html/2205.14135v2',
    'fa2_paper_v1.html':'https://arxiv.org/html/2307.08691v1',
    'fa3_paper_v2.html':'https://arxiv.org/html/2407.08608v2',
    'nvidia_h200.html':'https://www.nvidia.com/en-us/data-center/h200/',
    'nvidia_4090.html':'https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/',
    'nvidia_ada_guide.html':'https://docs.nvidia.com/cuda/ada-tuning-guide/index.html',
    'nvidia_hopper_guide.html':'https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html',
    'nvidia_blackwell_guide.html':'https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html',
    'nvidia_ada_whitepaper.pdf':'https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf',
    'fa4_release_readme.md':'https://raw.githubusercontent.com/Dao-AILab/flash-attention/a365a1909c081744693177255a23c669c0f208fa/README.md',
    'fa4_release_tree.json':'https://api.github.com/repos/Dao-AILab/flash-attention/git/trees/a365a1909c081744693177255a23c669c0f208fa?recursive=1',
    'fa2_forward.h':'https://raw.githubusercontent.com/Dao-AILab/flash-attention/4f285b354796fb17df8636485b9a04df3ebbb7dc/csrc/flash_attn/src/flash_fwd_kernel.h',
    'flash_current_commit.json':'https://api.github.com/repos/Dao-AILab/flash-attention/commits/main',
    'triton_fused_attention.html':'https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html',
    'flashinfer_attention.html':'https://docs.flashinfer.ai/api/attention.html',
}


def extract_html(path):
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(path.read_bytes(),'html.parser')
    for element in soup(['script','style','nav','footer']):element.decompose()
    for element in soup.find_all('math'):
        element.replace_with(' '+(element.get('alttext') or element.get_text(' ',strip=True))+' ')
    lines=[line.strip() for line in soup.get_text('\n').splitlines() if line.strip()]
    path.with_suffix('.txt').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-map',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def fetch(item):
        name,url=item;row=dict(file=name,url=url,accessed_at_UTC=datetime.now(timezone.utc).isoformat())
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'LongLive-research-source-audit/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response:
                raw=response.read();row.update(resolved_url=response.url,HTTP_status=response.status)
            (args.output/name).write_bytes(raw)
            row.update(status='pass',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            if name.endswith('.html'):extract_html(args.output/name)
        except Exception as error:row.update(status='fail',error=repr(error))
        print(json.dumps(row),flush=True);return row
    source_map=json.loads(args.source_map.read_text()) if args.source_map else SOURCES
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,source_map.items()))
    manifest=dict(scope='public_primary_sources_and_documented_operator_references',sources=rows,
        older_publication_versions_not_replaced_by_current_main=True,failed_fetches_preserved=True)
    (args.output/'source_lock.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
