"""Same FP64-complex Wan RoPE with direct final-dtype output storage.

The pinned upstream concatenates an often-empty suffix and then stacks FP64
results before the final cast. This implementation keeps its arithmetic and
casts directly into the final buffer; it does not reduce arithmetic precision.
"""
import torch
from .profiling import profiled


@profiled('rope/dense_direct_output')
def direct_output_causal_rope(x, grid_sizes, freqs, start_frame=0, relative_frame_indices=None):
    if x.ndim != 4 or x.shape[-1] % 2 or grid_sizes.shape != (x.shape[0],3):
        raise ValueError('matching dense [B,T,H,D] geometry required')
    batch,tokens,heads,dim=x.shape
    c=dim//2
    temporal,height,width=freqs.split([c-2*(c//3),c//3,c//3],dim=1)
    output=torch.empty(x.shape,dtype=x.dtype,device=x.device)
    for b,(frames,h,w) in enumerate(grid_sizes.tolist()):
        length=frames*h*w
        if min(frames,h,w)<1 or length>tokens:
            raise ValueError('invalid dense frame geometry')
        ids=(relative_frame_indices.long() if relative_frame_indices is not None
             else torch.arange(start_frame,start_frame+frames,device=freqs.device))
        if ids.numel()!=frames:
            raise ValueError('relative frame count mismatch')
        rotation=torch.cat((temporal[ids].view(frames,1,1,-1).expand(frames,h,w,-1),
            height[:h].view(1,h,1,-1).expand(frames,h,w,-1),
            width[:w].view(1,1,w,-1).expand(frames,h,w,-1)),dim=-1).reshape(length,1,-1)
        source=torch.view_as_complex(x[b,:length].to(torch.float64).reshape(length,heads,-1,2))
        rotated=torch.view_as_real(source*rotation).flatten(2)
        output[b,:length]=rotated.to(x.dtype)
        if length<tokens:
            output[b,length:]=x[b,length:]
    return output
