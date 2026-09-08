#!/usr/bin/env python3
"""Verify local report targets and actual Perfetto parsing before review handoff."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote,urlsplit
import zipfile


class Targets(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('src','href') and value and not urlsplit(value).scheme and not value.startswith('#'):
                self.targets.append(unquote(value.split('#')[0]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--processor',type=Path,required=True)
    p.add_argument('--zip',type=Path,required=True);args=p.parse_args();root=args.review.resolve();checks=[]
    for page in root.glob('*.html'):
        parser=Targets();parser.feed(page.read_text())
        for target in parser.targets:
            if not (page.parent/target).is_file():raise FileNotFoundError(f'{page.name}: {target}')
        checks.append(dict(file=page.name,local_links=len(parser.targets),status='pass'))
    query='SELECT count(*) AS slices FROM slice; SELECT count(*) AS flows FROM flow; SELECT name,severity,value FROM stats WHERE severity="error" AND value>0;'
    def validate(path):
        process=subprocess.run([str(args.processor.resolve()),'query',str(path),query],capture_output=True,text=True,timeout=120)
        if process.returncode:raise RuntimeError(path.name+': '+process.stderr)
        after=process.stdout.split('"name","severity","value"')[-1].strip()
        if after:raise RuntimeError(path.name+': Perfetto ingestion errors '+after)
        with gzip.open(path,'rt') as handle:trace=json.load(handle)
        xs=[e for e in trace['traceEvents'] if e.get('ph')=='X']
        if not xs or any(e['dur']<0 for e in xs):raise ValueError('missing or invalid real spans')
        row=dict(file=str(path.relative_to(root)),status='pass',bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),Perfetto_SQL=process.stdout,
            source_metadata=trace.get('metadata'),all_spans_nonnegative=True)
        print(json.dumps({k:v for k,v in row.items() if k not in ('source_metadata',)}),flush=True);return row
    with ThreadPoolExecutor(max_workers=2) as pool:traces=list(pool.map(validate,sorted((root/'traces').glob('*.gz'))))
    version=subprocess.check_output([str(args.processor.resolve()),'--version'],text=True).strip()
    result=dict(status='pass',processor_version=version,processor_sha256=hashlib.sha256(args.processor.read_bytes()).hexdigest(),
                pages=checks,traces=traces,shared_or_uploaded_to_external_service=False,
                verification_is_native_Trace_Processor_not_claimed_browser_screenshot=True)
    (root/'VALIDATION.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    files={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file()}
    (root/'SHA256SUMS.json').write_text(json.dumps(files,indent=2,ensure_ascii=False)+'\n')
    with zipfile.ZipFile(args.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file():archive.write(path,str(path.relative_to(root)))
    print(json.dumps(dict(status='pass',validated_traces=len(traces),zip=str(args.zip.resolve()),zip_bytes=args.zip.stat().st_size)))


if __name__=='__main__':main()
