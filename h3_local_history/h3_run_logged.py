"""Run inference with local stdout and bounded, independent shared-log delivery.

The copier imports no tensor libraries, has no visible GPUs, stays in the job's
process group, and never delays CUDA-process exit for an unresponsive sink.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def copy_loop(local,destination,done,uploaded):
    local,destination,done,uploaded=map(Path,[local,destination,done,uploaded])
    last=None
    while True:
        try:
            version=(local.stat().st_size,local.stat().st_mtime_ns)
            final=json.loads(done.read_text()) if done.exists() else None
            if version!=last or final:
                content=local.read_bytes();temporary=destination.with_name(destination.name+f'.tmp.{os.getpid()}')
                temporary.write_bytes(content);temporary.replace(destination);last=version
                if final:
                    receipt=dict(final_uploaded=True,bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),
                                 child_exit_code=final['exit_code'],CUDA_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 copier_has_no_tensor_import=True)
                    status=destination.with_suffix('.transport.json');temp=status.with_suffix('.tmp')
                    temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(status)
                    uploaded.write_text('complete\n');return
        except (OSError,ValueError):
            # A shared-filesystem problem must not propagate into the model.
            pass
        time.sleep(1)


def run(command,output,log_name,final_wait=2.,copier_override=None):
    if not command or not re.fullmatch('[A-Za-z0-9_.-]{1,100}',log_name):raise ValueError('Explicit command and safe log name required')
    if not 0<=final_wait<=10:raise ValueError('Final log delivery wait must be bounded')
    with tempfile.TemporaryDirectory(prefix='h3-job-log-',dir=os.environ.get('TMPDIR','/tmp')) as folder:
        root=Path(folder);local=root/'stdout.log';local.touch();done=root/'done.json';uploaded=root/'uploaded'
        destination=Path(output)/(log_name+'.log')
        argv=copier_override or [sys.executable,str(Path(__file__).resolve()),'--copy-loop',str(local),str(destination),str(done),str(uploaded)]
        copier=subprocess.Popen(argv,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)
        try:
            with local.open('ab',buffering=0) as stream:
                try:
                    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,close_fds=True)
                    code=child.wait()
                except OSError as error:
                    stream.write(('Command launch failed: '+repr(error)+'\n').encode());code=127
            done.write_text(json.dumps(dict(exit_code=code))+'\n')
            end=time.monotonic()+final_wait
            while not uploaded.exists() and time.monotonic()<end:
                time.sleep(.05)
            # Do not leave a detached uploader or join a blocked filesystem
            # writer indefinitely. It belongs to the existing job group.
            if copier.poll() is None:
                copier.terminate()
                try:copier.wait(timeout=.2)
                except subprocess.TimeoutExpired:pass
            return code
        finally:
            if copier.poll() is None:copier.terminate()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--copy-loop',nargs=4)
    parser.add_argument('--log-name');parser.add_argument('--final-wait',type=float,default=2.)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.copy_loop:return copy_loop(*args.copy_loop)
    command=args.command[1:] if args.command and args.command[0]=='--' else args.command
    output=os.environ['INFER_OUTPUT_DIR']
    code=run(command,output,args.log_name,args.final_wait)
    raise SystemExit(code if code>=0 else 128-code)


if __name__=='__main__':main()
