#!/usr/bin/env python3
"""InferHub-only worker: prompt encoding then bounded prompt-program rendering."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from h3_demo_program import normalize, materialize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args()
    request=json.loads(Path(a.request).read_text());program=normalize(request['shots'])
    out=Path(os.environ['INFER_OUTPUT_DIR']);code=Path(os.environ['INFER_CODE_DIR'])/'h3_local_history'
    weights=Path(os.environ['INFER_WEIGHTS_DIR']);prompts=out/'prompts';prompts.mkdir(exist_ok=True)
    def state(event,**extra):
        row=dict(event=event,time=time.time(),**extra);print(json.dumps(row),flush=True)
        tmp=out/'DEMO_STATUS.json.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');os.replace(tmp,out/'DEMO_STATUS.json')
    state('preparing_prompts',prompt_count=len(program['prompts']))
    cached=request.get('cached_prompts',{})
    missing=[]
    for i,prompt in enumerate(program['prompts']):
        key=hashlib.sha256(prompt.encode()).hexdigest();rel=cached.get(key)
        if rel:
            import torch
            src=weights/rel;data=torch.load(src,map_location='cpu',weights_only=True)
            assert data['prompt']==prompt and data['prompt_embeds'].shape[1]==5120
            assert bool(torch.isfinite(data['prompt_embeds']).all())
            shutil.copyfile(src,prompts/f'prompt_{i}.pt')
        else:missing.append(i)
    if missing:
        prompt_jsonl=out/'PROMPTS.jsonl';prompt_jsonl.write_text(''.join(json.dumps(dict(prompt=p),ensure_ascii=False)+'\n' for p in program['prompts']))
        env=dict(os.environ)
        env['PYTHONPATH']=':'.join(str(p) for p in [code/'offline_h3_python_20261013',code/'overlay',code/'vdn-minimax-h3',code/'vdn-minimax-h3/diffusers/src'])
        subprocess.run([sys.executable,str(code/'vdn-minimax-h3/src/inference/encode_prompt.py'),'--jsonl',str(prompt_jsonl),
            '--rows',*[str(i) for i in missing],'--out_dir',str(prompts),'--name','prompt',
            '--model_root',str(weights/'selected_h3_holdout_inputs_v1_20261016/native_text_source'),'--device','cuda:0'],env=env,check=True)
    materialize(program,prompts,out/'inputs',int(request.get('seed',20261016)))
    allocated=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert len(allocated) in (2,4),allocated
    waiting=list(request['modes']);running=[];results={};failed=[]
    state('generating',modes=waiting,program=program)
    free=[allocated[i:i+2] for i in range(0,len(allocated),2)]
    while waiting or running:
        while waiting and free:
            mode=waiting.pop(0);devices=free.pop(0);mode_out=out/mode;mode_out.mkdir(exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(devices))
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
            args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                '--log-dir',str(mode_out/'torchrun_logs'),'--tee','3','--local-ranks-filter','0',str(code/'sample_demo.py'),
                '--base-source',str(weights/'h3-base'),'--checkpoint',str(weights/'local_history_step200_compare_20261014/checkpoints/step200.pt'),
                '--cases',str(out/'inputs/GENERATION_CASES.json'),'--source-contracts',str(out/'inputs/SOURCE_CONTRACTS.json'),
                '--out',str(mode_out),'--label',mode,'--nfe','50','--adapter-mode',mode,'--collective-timeout-min','120']
            log=open(mode_out/'console.log','w');proc=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
            running.append((mode,devices,proc,log));state('generating',started=mode,completed=results,active=[r[0] for r in running],waiting=waiting)
        for item in list(running):
            mode,devices,proc,log=item;rc=proc.poll()
            if rc is None:continue
            log.close();running.remove(item);free.append(devices);results[mode]=rc
            if rc:failed.append(mode)
            state('generating',finished=mode,exit_code=rc,completed=results,active=[r[0] for r in running],waiting=waiting)
        if running:time.sleep(2)
    # Decode all video frames and the audio stream before exposing a result.
    import av
    artifacts=[]
    for mode in request['modes']:
        for path in sorted((out/mode).glob('rank*_RESULT.json')):
            result=json.loads(path.read_text());video=Path(result['artifact']['path'])
            with av.open(str(video)) as c:
                stream=c.streams.video[0];frames=list(c.decode(stream));pts=[f.time for f in frames]
                assert len(frames)==program['frames'] and all(a<b for a,b in zip(pts,pts[1:]))
                width,height=frames[0].width,frames[0].height
            with av.open(str(video)) as c:
                assert c.streams.audio;aud=list(c.decode(audio=0));assert aud
                audio_samples=sum(f.samples for f in aud)
            artifacts.append(dict(mode=mode,state=result['state_mode'],rank=result['rank'],
                seed=json.loads(Path(result['resolved_config_path']).read_text())['case']['seed'],
                path=str(video),relative_path=str(video.relative_to(out)),frames=len(frames),width=width,height=height,
                audio_samples=audio_samples,full_decode_pass=True,sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                result_json=str(path),formal_quality_pass=None))
    report=dict(status='complete' if not failed else 'partial_failure',artifacts=artifacts,program=program,
        modes=request['modes'],mode_exit_codes=results,failed_modes=failed,hardware='InferHub H workers only',
        model_semantics=dict(trained='step200 generated-video-history carry and zero-history pair',
            native_init='untrained projection + zero LoRA, same history interface; not stock H3',
            native_stock='released dense H3 weights and native sampler, independent shots without cross-shot history'),
        streaming_audio_claim=False,history_learning_claim=None)
    (out/'DEMO_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    state(report['status'],videos=len(artifacts),failed_modes=failed)
    if failed:raise RuntimeError(f'Failed modes: {failed}')

if __name__=='__main__':main()
