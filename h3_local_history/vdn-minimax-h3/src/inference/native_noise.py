"""Native-layout initial noise, with an explicit deterministic replay option."""
import torch

def request_initial_noise(video_shape,audio_shape,generator,device,noise_override=None):
    if noise_override is None:
        return (torch.randn(video_shape,generator=generator,device=device,dtype=torch.float32),
                torch.randn(audio_shape,generator=generator,device=device,dtype=torch.float32))
    if set(noise_override)!={'video','audio'}:raise ValueError('Replay requires video and audio noise tensors')
    values=[]
    for key,shape in [('video',video_shape),('audio',audio_shape)]:
        value=noise_override[key]
        if tuple(value.shape)!=tuple(shape) or value.dtype!=torch.float32:raise ValueError('Native replay noise shape/dtype mismatch: '+key)
        if not torch.isfinite(value).all():raise ValueError('Nonfinite native replay noise: '+key)
        values.append(value.detach().to(device=device).clone())
    return tuple(values)
