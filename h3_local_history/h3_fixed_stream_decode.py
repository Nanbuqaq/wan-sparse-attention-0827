"""Production H3 codec facade; real buffering replaces synthetic left prefixes."""
from contextlib import nullcontext
import torch
from h3_buffered_video_decode import BufferedVideoDecoder


class FixedChunkVideoDecoder(BufferedVideoDecoder):
    # Name retained for existing samplers and profiler hooks. DiT chunk size
    # is not constrained by this decoder; a not-yet-ready push returnsNone.
    @torch.inference_mode()
    def decode(self,z):
        context=torch.autocast('cuda',dtype=torch.float16) if self.device.type=='cuda' else nullcontext()
        with context:
            return self.vae.decode(z.to(self.device)*self.std+self.mean,return_dict=False)[0]

    @torch.inference_mode()
    def push(self,chunk):
        value=super().push(chunk)
        if value is not None:return value
        # Existing writers accept a zero-frame tensor. This preserves their
        # tensor interface while waiting for genuine startup context.
        ratio=getattr(self.vae,'spatial_compression_ratio',1)
        channels=getattr(self.vae.config,'out_channels',3)
        return chunk.new_empty((1,channels,0,chunk.shape[3]*ratio,chunk.shape[4]*ratio))
