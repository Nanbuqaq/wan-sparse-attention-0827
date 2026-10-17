"""Bounded raw-frame export; preserve native frame order and encoder options."""
import hashlib
import os
from pathlib import Path
import select
import subprocess
import time
import torch


class FrameWriter:
    def __init__(self,path,fps,ffmpeg=None,timeout=120):
        self.path=Path(path);self.fps=fps;self.ffmpeg=ffmpeg;self.timeout=timeout
        self.frames=0;self.bytes=0;self.process=None;self.pending_skip=0
        self.write_seconds=0.;self.finalize_seconds=0.;self.closed=False

    def __enter__(self):return self

    def skip(self,frames):
        if self.pending_skip:raise ValueError('A native startup trim is still pending')
        self.pending_skip=frames

    def append(self,decoded):
        if decoded is None or decoded.shape[2]==0:return
        if self.pending_skip:
            count=min(self.pending_skip,decoded.shape[2]);decoded=decoded[:,:,count:];self.pending_skip-=count
            if decoded.shape[2]==0:return
        if self.closed:raise ValueError('Cannot append after encoder close')
        begin=time.monotonic()
        mean=torch.tensor((.485,.456,.406),device=decoded.device).view(1,3,1,1,1)
        std=torch.tensor((.229,.224,.225),device=decoded.device).view(1,3,1,1,1)
        pixels=((decoded.float()*std+mean).clamp(0,1)*255).round().byte()[0].permute(1,2,3,0).cpu().contiguous()
        height,width=pixels.shape[1:3]
        if self.process is None:
            if self.ffmpeg is None:
                import imageio_ffmpeg
                self.ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
            self.shape=(height,width);self.log=self.path.with_suffix('.encode.log').open('wb')
            command=[self.ffmpeg,'-v','error','-nostdin','-y','-f','rawvideo','-pixel_format','rgb24',
                     '-video_size',f'{width}x{height}','-framerate',str(self.fps),'-i','pipe:0','-an',
                     '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(self.path)]
            self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log,bufsize=0)
            os.set_blocking(self.process.stdin.fileno(),False)
        if (height,width)!=self.shape:raise ValueError('Resolution changed inside a video')
        data=memoryview(pixels.numpy()).cast('B');offset=0;deadline=time.monotonic()+self.timeout
        while offset<len(data):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Video encoder stopped consuming native frames')
            writable=select.select([], [self.process.stdin.fileno()], [], min(remaining,5))[1]
            if not writable:
                if self.process.poll() is not None:raise RuntimeError('Video encoder exited early')
                continue
            try:offset+=os.write(self.process.stdin.fileno(),data[offset:offset+1024*1024])
            except BlockingIOError:continue
        self.frames+=len(pixels);self.bytes+=len(data);self.write_seconds+=time.monotonic()-begin

    def close(self):
        if self.closed:return
        if self.process is None:raise ValueError('No native frames were exported')
        if self.pending_skip:raise ValueError('Native startup trim consumed all output')
        started=time.monotonic();self.process.stdin.close()
        code=self.process.wait(timeout=self.timeout);self.log.close();self.closed=True
        self.finalize_seconds=time.monotonic()-started
        if code:raise RuntimeError('Video encoder failed: '+self.path.with_suffix('.encode.log').read_text()[-2000:])

    def artifact(self):
        self.close();h=hashlib.sha256()
        with self.path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        return dict(path=str(self.path),frames=self.frames,fps=self.fps,sha256=h.hexdigest(),
                    export_buffer='one native decoded part; complete sequence is not retained',
                    encoder_write_seconds=self.write_seconds,encoder_finalize_seconds=self.finalize_seconds)

    def __exit__(self,typ,value,traceback):
        if typ is None:
            try:self.close()
            except Exception:
                if self.process and self.process.poll() is None:self.process.kill();self.process.wait(timeout=10)
                raise
        elif self.process and self.process.poll() is None:
            try:self.process.stdin.close()
            except Exception:pass
            self.process.kill();self.process.wait(timeout=10)
            self.log.close()
