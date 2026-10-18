"""Bound an inference command and its descendants, including detached torch ranks.

No tensor imports or GPU access. Linux child-subreaping keeps detached workers
in this task's ownership when their original parent exits.
"""
import argparse,ctypes,json,os,signal,subprocess,sys,time
from pathlib import Path


def processes():
    rows={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            raw=(path/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
            rows[int(path.name)]=dict(parent=int(parts[1]),start=parts[19],state=parts[0])
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return rows


def descendants(root):
    rows=processes();owned={root}
    while True:
        extra={pid for pid,row in rows.items() if row['parent'] in owned and pid not in owned}
        if not extra:break
        owned.update(extra)
    return {pid:rows[pid] for pid in owned if pid!=root and rows[pid]['state'] not in ['Z','X']}


def send(owned,sig):
    current=processes();sent=[]
    for pid,row in owned.items():
        if current.get(pid,{}).get('start')!=row['start']:continue
        try:os.kill(pid,sig);sent.append(pid)
        except ProcessLookupError:pass
    return sent


def reap():
    while True:
        try:
            pid,_=os.waitpid(-1,os.WNOHANG)
            if not pid:return
        except ChildProcessError:return


def run(command,seconds,grace_seconds,poll_seconds=.1):
    if not command or seconds<=0 or not 0<=grace_seconds<=60:
        raise ValueError('Explicit command, positive deadline and bounded termination grace required')
    if descendants(os.getpid()):
        raise RuntimeError('Use a dedicated supervisor process; do not adopt an existing workload')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0):raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER failed')
    interrupted=[]
    def request_stop(signum,frame):interrupted.append(signum)
    prior={sig:signal.signal(sig,request_stop) for sig in [signal.SIGTERM,signal.SIGINT]}
    started=time.monotonic();child=None;reason=None;sent_term=[];sent_kill=[]
    try:
        child=subprocess.Popen(command,start_new_session=True)
        while True:
            code=child.poll();owned=descendants(os.getpid())
            if interrupted:reason='external_stop';break
            if time.monotonic()-started>=seconds:reason='command_deadline';break
            if code is not None:
                if not owned:
                    return code,dict(reason='command_exited',child_exit_code=code,elapsed_seconds=time.monotonic()-started,
                                     remaining_descendants=[],tensor_library_imported=False)
                reason='command_exited_with_live_descendants';break
            time.sleep(poll_seconds)
        end=time.monotonic()+grace_seconds;seen=set()
        while True:
            owned=descendants(os.getpid());new={pid:row for pid,row in owned.items() if (pid,row['start']) not in seen}
            sent_term+=send(new,signal.SIGTERM);seen.update((pid,row['start']) for pid,row in new.items())
            child.poll()
            if not owned or time.monotonic()>=end:break
            time.sleep(min(poll_seconds,max(0,end-time.monotonic())))
        # Detached sessions are addressed by verified PID identity, not only
        # the original process group. Repeated discovery catches adopted ranks.
        final_end=time.monotonic()+2
        while True:
            owned=descendants(os.getpid())
            if not owned:break
            sent_kill+=send(owned,signal.SIGKILL);child.poll()
            if time.monotonic()>=final_end:break
            time.sleep(poll_seconds)
        child.poll();reap();remaining=descendants(os.getpid())
        code=128+interrupted[0] if interrupted else (124 if reason=='command_deadline' else 70)
        return code,dict(reason=reason,elapsed_seconds=time.monotonic()-started,command_limit_seconds=seconds,
                         termination_grace_seconds=grace_seconds,TERM_pids=sorted(set(sent_term)),KILL_pids=sorted(set(sent_kill)),
                         remaining_descendants=[dict(pid=pid,**row) for pid,row in remaining.items()],
                         detached_sessions_included=True,tensor_library_imported=False)
    finally:
        # A Python exception must not leave the launched worker tree behind.
        send(descendants(os.getpid()),signal.SIGKILL)
        if child is not None:
            try:child.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass
        reap()
        for sig,handler in prior.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--grace-seconds',type=float,default=30);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    code,receipt=run(command,a.seconds,a.grace_seconds)
    print('INFERENCE_BUDGET_RESULT '+json.dumps(receipt),flush=True)
    raise SystemExit(code if code>=0 else 128-code)


if __name__=='__main__':main()
