"""Stateful native Wan2.2 decode, retaining original per-latent decoder calls.

No weights or third-party code are modified. conv2 is time-pointwise (1x1x1);
chunking can still change CUDA numerical execution and must pass a real gate.
"""
import torch


class NativeVAEStream:
    def __init__(self,model,scale,unpatchify):
        if tuple(model.conv2.kernel_size)!=(1,1,1):
            raise ValueError('streaming adapter requires time-pointwise conv2')
        self.model,self.scale,self.unpatchify=model,scale,unpatchify
        self.active=False;self.closed=False;self.failed=False;self.frames=0;self.signature=None
        model.clear_cache()

    def reset(self):
        if self.active:raise RuntimeError('cannot reset an active chunk')
        self.model.clear_cache();self.closed=False;self.failed=False;self.frames=0;self.signature=None

    def finish(self):
        if self.active:raise RuntimeError('must consume the active chunk before finish')
        self.model.clear_cache();self.closed=True

    def iter_decode(self,z):
        """Consume BCTHW latents and yield BCTHW pixels, never a growing video."""
        if self.closed or self.failed or self.active:raise RuntimeError('decoder is closed, failed, or already active')
        if z.ndim!=5 or z.shape[0]!=1 or z.shape[1]!=self.model.z_dim or z.shape[2]<1:
            raise ValueError('expected nonempty batch1 native BCTHW latents')
        signature=(z.shape[0],z.shape[1],z.shape[3],z.shape[4],z.dtype,z.device)
        if self.signature is not None and signature!=self.signature:raise ValueError('geometry/dtype/device changed within one video')
        self.signature=signature;self.active=True;completed=False
        try:
            with torch.inference_mode():
                if isinstance(self.scale[0],torch.Tensor):
                    normalized=z/self.scale[1].view(1,self.model.z_dim,1,1,1)+self.scale[0].view(1,self.model.z_dim,1,1,1)
                else:normalized=z/self.scale[1]+self.scale[0]
                x=self.model.conv2(normalized)
            for index in range(x.shape[2]):
                with torch.inference_mode():
                    self.model._conv_idx=[0]
                    kwargs=dict(feat_cache=self.model._feat_map,feat_idx=self.model._conv_idx)
                    if self.frames==0:kwargs['first_chunk']=True
                    output=self.model.decoder(x[:,:,index:index+1],**kwargs)
                    self.frames+=1
                    pixels=self.unpatchify(output,patch_size=2)
                yield pixels
            completed=True
        finally:
            self.active=False
            if not completed:
                self.failed=True;self.model.clear_cache()

    def decode_chunk(self,z):
        # Only this caller-selected chunk is joined, never the whole growing video.
        return torch.cat(list(self.iter_decode(z)),dim=2)
