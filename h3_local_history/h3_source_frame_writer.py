"""Bounded VFR export with explicit source presentation times; CPU media only."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time


def timeline(source_pts, source_start_sec, source_end_sec):
    origin,end=float(source_start_sec),float(source_end_sec)
    if not math.isfinite(origin) or not math.isfinite(end) or any(not math.isfinite(t) for t in source_pts):
        raise ValueError('Finite source timestamps required')
    base=Fraction(1,1_000_000)
    ticks=[round((t-origin)/base) for t in source_pts]
    final=round((end-origin)/base)
    if not ticks or ticks[0]<0 or any(a>=b for a,b in zip(ticks,ticks[1:])) or final<=ticks[-1]:
        raise ValueError('Source PTS/end must remain increasing at the declared microsecond timebase')
    return ticks,dict(zip(ticks,[b-a for a,b in zip(ticks,ticks[1:]+[final])]))


class _CodecWriter:
    def __init__(self, path, source_pts, source_start_sec, source_end_sec):
        import av
        self.av = av
        self.path = Path(path)
        self.temporary = self.path.with_suffix('.partial.mp4')
        self.origin = float(source_start_sec)
        self.end = float(source_end_sec)
        self.source_pts = list(source_pts)
        self.time_base = Fraction(1, 1_000_000)
        self.ticks,self.durations=timeline(source_pts,self.origin,self.end)
        self.frames = 0
        self.container = None
        self.closed = False
        self.write_seconds = 0.
        self.finalize_seconds = 0.

    def __enter__(self):
        return self

    def _mux(self, packets):
        for packet in packets:
            # No B frames: packet PTS and duration use the codec timebase.
            if packet.time_base != self.time_base or packet.pts not in self.durations:
                raise ValueError('Encoder changed declared source presentation timestamps')
            packet.duration = self.durations[packet.pts]
            self.container.mux(packet)

    def append_pixels(self, pixels):
        if self.closed:
            raise ValueError('Writer is closed')
        if pixels.ndim != 4 or pixels.shape[-1] != 3 or str(pixels.dtype) != 'uint8':
            raise ValueError('Expected uint8 [T,H,W,3] pixels')
        if self.frames + len(pixels) > len(self.ticks):
            raise ValueError('Padded or undeclared source frames reached delivery')
        start = time.monotonic()
        for array in pixels:
            if self.container is None:
                self.container = self.av.open(str(self.temporary), 'w', format='mp4',
                                              options={'video_track_timescale': '1000000'})
                self.stream = self.container.add_stream('libx264', rate=24)
                self.stream.width, self.stream.height = array.shape[1], array.shape[0]
                self.stream.pix_fmt = 'yuv420p'
                self.stream.time_base = self.time_base
                self.stream.codec_context.time_base = self.time_base
                self.stream.codec_context.max_b_frames = 0
                self.stream.options = {'preset': 'fast', 'crf': '18', 'bf': '0'}
                self.shape = array.shape
            if array.shape != self.shape:
                raise ValueError('Frame dimensions changed')
            frame = self.av.VideoFrame.from_ndarray(array, format='rgb24')
            frame.pts, frame.time_base = self.ticks[self.frames], self.time_base
            frame.duration = self.durations[frame.pts]
            self._mux(self.stream.encode(frame))
            self.frames += 1
        self.write_seconds += time.monotonic() - start

    def append(self, decoded):
        if decoded is None or decoded.shape[2] == 0:
            return
        import torch
        mean = torch.tensor((.485, .456, .406), device=decoded.device).view(1,3,1,1,1)
        std = torch.tensor((.229, .224, .225), device=decoded.device).view(1,3,1,1,1)
        pixels = ((decoded.float()*std+mean).clamp(0,1)*255).round().byte()
        self.append_pixels(pixels[0].permute(1,2,3,0).cpu().contiguous().numpy())

    def close(self):
        if self.closed:
            return
        if self.frames != len(self.ticks) or self.container is None:
            raise ValueError('Incomplete source frame delivery')
        start = time.monotonic()
        self._mux(self.stream.encode())
        self.container.close()
        self.container = None
        # Decode the entire file and compare integer PTS, including nonzero starts.
        with self.av.open(str(self.temporary)) as video:
            stream = video.streams.video[0]
            actual = [round(float(f.pts*f.time_base)/self.time_base) for f in video.decode(stream)]
            duration = round(float(stream.duration*stream.time_base)/self.time_base)
        if actual != self.ticks or abs(duration - sum(self.durations.values())) > 1:
            raise ValueError('Encoded VFR frame count, timestamps or last-frame duration differ')
        self.temporary.replace(self.path)
        self.closed = True
        self.finalize_seconds = time.monotonic() - start

    def artifact(self):
        self.close()
        h = hashlib.sha256()
        with self.path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(block)
        return dict(path=str(self.path), sha256=h.hexdigest(), frames=self.frames, fps=None,
                    source_origin_sec=self.origin, source_end_sec=self.end,
                    delivery_clock='source PTS minus declared source origin; microsecond timebase',
                    maximum_pts_quantization_sec=.0000005, full_decode_pts_verified=True,
                    encoder_write_seconds=self.write_seconds, encoder_finalize_seconds=self.finalize_seconds)

    def __exit__(self, typ, value, traceback):
        try:
            if typ is None:
                self.close()
        finally:
            if self.container is not None:
                self.container.close()
                self.container = None


class SourceWindowTrim:
    """Drop only explicitly declared model padding, preserving each valid frame."""
    def __init__(self, valid_frames, native_frames):
        if not 0 < valid_frames <= native_frames:
            raise ValueError('Invalid source/native frame counts')
        self.valid, self.native = valid_frames, native_frames
        self.seen, self.delivered = 0, 0

    def push(self, decoded):
        if decoded is None:
            return None
        count = decoded.shape[2]
        if self.seen + count > self.native:
            raise ValueError('Decoder exceeded declared native padding')
        keep = max(0, min(count, self.valid-self.seen))
        self.seen += count
        self.delivered += keep
        return decoded[:, :, :keep]

    def finish(self):
        if (self.seen, self.delivered) != (self.native, self.valid):
            raise ValueError('Incomplete native decode or valid source frames were lost')


class SourceFrameWriter:
    """GPU process only sends pixels; codec, shared IO, hash and decode are bounded."""
    def __init__(self,path,source_pts,source_start_sec,source_end_sec,timeout=120,*,process_factory=None):
        self.path=Path(path);self.source_pts=list(source_pts)
        self.origin,self.end=source_start_sec,source_end_sec
        self.ticks,_=timeline(self.source_pts,self.origin,self.end)
        if not math.isfinite(timeout) or timeout<=0:
            raise ValueError('Finite positive encoder timeout required')
        self.timeout=timeout;self.factory=process_factory or subprocess.Popen
        self.frames=0;self.process=None;self.closed=False;self.aborted=False
        self.pipe_seconds=0.;self.finalize_seconds=0.;self.receipt=None
        self.local=None;self.log=None

    def __enter__(self):return self

    def _start(self,shape):
        self.shape=shape;self.local=tempfile.TemporaryDirectory(prefix='h3-source-vfr-')
        folder=Path(self.local.name);self.log=(folder/'encoder.log').open('w+b')
        request=folder/'request.json'
        request.write_text(json.dumps(dict(path=str(self.path),source_pts=self.source_pts,
            source_start_sec=self.origin,source_end_sec=self.end,shape=list(shape)))+'\n')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
        self.process=self.factory([sys.executable,str(Path(__file__).resolve()),'--encode-stream',str(request)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0,env=env)
        os.set_blocking(self.process.stdin.fileno(),False)

    def _log_tail(self):
        if self.log is None:return ''
        self.log.flush();self.log.seek(0,2);self.log.seek(max(0,self.log.tell()-2500))
        return self.log.read().decode(errors='replace')

    def _cleanup(self):
        if self.log is not None:self.log.close();self.log=None
        if self.local is not None:self.local.cleanup();self.local=None

    def abort(self):
        self.aborted=True
        if self.process is not None:
            if self.process.poll() is None:self.process.kill()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
            for stream in [self.process.stdin,self.process.stdout]:
                if stream is not None:
                    try:stream.close()
                    except OSError:pass
        self._cleanup()

    def append_pixels(self,pixels):
        if self.closed or self.aborted:raise ValueError('Writer is closed or aborted')
        if pixels.ndim!=4 or pixels.shape[-1]!=3 or str(pixels.dtype)!='uint8':
            raise ValueError('Expected uint8 [T,H,W,3] pixels')
        if self.frames+len(pixels)>len(self.ticks):
            raise ValueError('Padded or undeclared source frames reached delivery')
        if not len(pixels):return
        if self.process is None:self._start(tuple(pixels.shape[1:]))
        if tuple(pixels.shape[1:])!=self.shape:raise ValueError('Frame dimensions changed')
        import numpy as np
        data=memoryview(np.ascontiguousarray(pixels)).cast('B');offset=0
        started=time.monotonic();deadline=started+self.timeout
        try:
            while offset<len(data):
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('VFR encoder stopped consuming frames')
                if self.process.poll() is not None:raise RuntimeError('VFR encoder exited: '+self._log_tail())
                ready=select.select([],[self.process.stdin.fileno()],[],min(remaining,.5))[1]
                if ready:
                    try:offset+=os.write(self.process.stdin.fileno(),data[offset:offset+1024*1024])
                    except BlockingIOError:continue
        except BaseException:
            self.abort();raise
        self.frames+=len(pixels);self.pipe_seconds+=time.monotonic()-started

    def append(self,decoded):
        if decoded is None or decoded.shape[2]==0:return
        import torch
        mean=torch.tensor((.485,.456,.406),device=decoded.device).view(1,3,1,1,1)
        std=torch.tensor((.229,.224,.225),device=decoded.device).view(1,3,1,1,1)
        pixels=((decoded.float()*std+mean).clamp(0,1)*255).round().byte()
        self.append_pixels(pixels[0].permute(1,2,3,0).cpu().contiguous().numpy())

    def close(self):
        if self.closed:return
        if self.aborted:raise ValueError('Writer aborted')
        if self.frames!=len(self.ticks) or self.process is None:
            self.abort();raise ValueError('Incomplete source frame delivery')
        started=time.monotonic()
        try:
            self.process.stdin.close()
            self.process.wait(timeout=self.timeout)
            if self.process.returncode:raise RuntimeError('VFR encoder failed: '+self._log_tail())
            result=json.loads(self.process.stdout.read().decode())
            if (result['path']!=str(self.path) or result['frames']!=self.frames or
                result.get('full_decode_pts_verified') is not True or
                result['source_origin_sec']!=self.origin or result['source_end_sec']!=self.end):
                raise ValueError('VFR child receipt differs from declared output')
            self.process.stdout.close();self.receipt=result;self.closed=True
            self.finalize_seconds=time.monotonic()-started
            self._cleanup()
        except subprocess.TimeoutExpired as error:
            self.abort();raise TimeoutError('VFR encoding/hash/full-decode finalization timed out') from error
        except BaseException:
            self.abort();raise

    def artifact(self):
        self.close()
        return dict(self.receipt,encoder_parent_pipe_seconds=self.pipe_seconds,
            encoder_parent_finalize_seconds=self.finalize_seconds,encoder_subprocess_timeout_sec=self.timeout,
            encoder_subprocess_gpu_visible=False)

    def __exit__(self,typ,value,traceback):
        if typ is None:self.close()
        else:self.abort()


def encode_stream(request):
    import numpy as np
    value=json.loads(Path(request).read_text());shape=tuple(value.pop('shape'));size=math.prod(shape)
    with _CodecWriter(**value) as writer:
        for _ in writer.ticks:
            data=bytearray()
            while len(data)<size:
                part=sys.stdin.buffer.read(size-len(data))
                if not part:raise ValueError('Truncated VFR RGB stream')
                data.extend(part)
            writer.append_pixels(np.frombuffer(data,dtype=np.uint8).reshape((1,)+shape))
        if sys.stdin.buffer.read(1):raise ValueError('Undeclared extra VFR pixels')
        result=writer.artifact()
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--encode-stream':raise SystemExit('Use --encode-stream request.json')
    encode_stream(sys.argv[2])
