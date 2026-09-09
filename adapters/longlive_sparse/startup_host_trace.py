"""Nested host startup observation; never infer GPU service from host duration."""
from contextlib import contextmanager
from functools import wraps
import threading
import time


class StartupHostTrace:
    def __init__(self):
        self.origin=time.perf_counter_ns();self.events=[];self.stacks={}

    @contextmanager
    def phase(self,name,metadata=None):
        thread=threading.get_ident();stack=self.stacks.setdefault(thread,[])
        event=dict(id=len(self.events),name=name,thread=thread,parent=stack[-1]['id'] if stack else None,
            start_ns=time.perf_counter_ns()-self.origin,metadata=metadata or {},children_ns=0,status='running')
        self.events.append(event);stack.append(event)
        try:yield event
        except BaseException:
            event['status']='failed';raise
        finally:
            event['duration_ns']=time.perf_counter_ns()-self.origin-event['start_ns']
            event['exclusive_host_ns']=event['duration_ns']-event['children_ns']
            if event['status']=='running':event['status']='pass'
            stack.pop()
            if stack:stack[-1]['children_ns']+=event['duration_ns']

    def wrap(self,name,function,metadata=None):
        @wraps(function)
        def call(*args,**kwargs):
            with self.phase(name,metadata(*args,**kwargs) if metadata else None):
                return function(*args,**kwargs)
        return call

    def aggregate(self):
        rows={}
        for event in self.events:
            row=rows.setdefault(event['name'],dict(calls=0,inclusive_host_s=0.,exclusive_host_s=0.,tensor_payload_bytes=0))
            row['calls']+=1;row['inclusive_host_s']+=event['duration_ns']/1e9
            row['exclusive_host_s']+=event['exclusive_host_ns']/1e9
            row['tensor_payload_bytes']+=event['metadata'].get('tensor_payload_bytes',0)
        return rows

    def chrome(self):
        threads={tid:i+1 for i,tid in enumerate(sorted({r['thread'] for r in self.events}))}
        return dict(traceEvents=[dict(ph='X',name=e['name'],pid=1,tid=threads[e['thread']],
            ts=e['start_ns']/1000,dur=e['duration_ns']/1000,cat='CPU host startup',args=e['metadata']) for e in self.events],
            displayTimeUnit='ms',metadata=dict(host_wall_only=True,GPU_activity_not_captured=True))
