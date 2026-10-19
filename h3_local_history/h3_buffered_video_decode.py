"""Bounded H3 output buffer, independent of DiT generation chunk sizes.

The emitted sequence follows the released VAE's whole-video decode. Only its
native right-end padding is allowed; no invented left context or cut repair.
Call finish at an independent codec-shot boundary, never at a prompt change
inside a continuously encoded sequence.
"""
from contextlib import nullcontext
import torch


class BufferedVideoDecoder:
    def __init__(self,vae,device):
        self.vae=vae;self.device=torch.device(device)
        if (vae.tokens_chunk_size,vae.token_overlap,vae.frame_overlap,vae.config.token_drop,
            vae.temporal_compression_ratio,vae.config.clip_length)!=(5,2,5,3,4,17):
            raise ValueError('Unvalidated H3 codec geometry')
        self.mean=torch.tensor(vae.config.latents_mean,device=self.device).view(1,-1,1,1,1)
        self.std=torch.tensor(vae.config.latents_std,device=self.device).view(1,-1,1,1,1)
        self.last_summary=None;self._reset()

    def _reset(self):
        self.buffer=None;self.previous_two=None;self.pending=None
        self.total_latents=0;self.emitted_frames=0;self.decode_calls=0

    def _decode(self,window):
        if window.shape[2]!=7:raise ValueError('Decode window must contain7 real-or-native-tail-padded latents')
        context=torch.autocast('cuda',dtype=torch.float16) if self.device.type=='cuda' else nullcontext()
        with context:
            decoded=self.vae.decode(window*self.std+self.mean,return_dict=False)[0]
        self.decode_calls+=1
        if decoded.shape[2]!=22:raise ValueError('Native7-latent decode must return22 frames')
        return decoded

    def _join(self,parts):
        if not parts:return None
        value=parts[0] if len(parts)==1 else torch.cat(parts,dim=2)
        self.emitted_frames+=value.shape[2]
        return value

    @torch.inference_mode()
    def push(self,chunk):
        if chunk.ndim!=5 or chunk.shape[0]!=1 or chunk.shape[1]!=self.mean.shape[1] or chunk.shape[2]<1:
            raise ValueError('Expected a nonempty [1,C,new_latents,H,W] tensor')
        chunk=chunk.detach().to(self.device)
        if self.buffer is not None and (chunk.shape[1],*chunk.shape[3:])!=(self.buffer.shape[1],*self.buffer.shape[3:]):
            raise ValueError('Codec canvas changed without finishing the previous shot')
        self.total_latents+=chunk.shape[2]
        self.buffer=chunk if self.buffer is None else torch.cat([self.buffer,chunk],dim=2)
        parts=[]
        while self.buffer.shape[2]>=(7 if self.previous_two is None else 5):
            take=7 if self.previous_two is None else 5
            fresh=self.buffer[:,:,:take];self.buffer=self.buffer[:,:,take:]
            window=fresh if self.previous_two is None else torch.cat([self.previous_two,fresh],dim=2)
            decoded=self._decode(window)
            if self.pending is not None:decoded=self.vae._blend(self.pending,decoded,5,dim=-3)
            parts.append(decoded[:,:,:-5]);self.pending=decoded[:,:,-5:].clone()
            self.previous_two=window[:,:,-2:].clone()
        # A view of the remaining tokens must not retain an entire input video.
        self.buffer=self.buffer.clone() if self.buffer.shape[2] else None
        return self._join(parts)

    @torch.inference_mode()
    def finish(self):
        if not self.total_latents:return None
        if self.total_latents<3:
            raise ValueError('Released video VAE requires at least3 latents; image decoding is a separate interface')
        parts=[];padding=0;trim=0
        if self.buffer is not None:
            fresh=self.buffer
            needed=7 if self.previous_two is None else 5
            padding=needed-fresh.shape[2]
            padded=torch.cat([fresh,fresh[:,:,-1:].repeat(1,1,padding,1,1)],dim=2)
            window=padded if self.previous_two is None else torch.cat([self.previous_two,padded],dim=2)
            decoded=self._decode(window)
            if self.pending is not None:decoded=self.vae._blend(self.pending,decoded,5,dim=-3)
            # Exactly the released _decode() right-padding removal formula.
            trim=sum(1 if (self.total_latents+k)%5==0 else 4 for k in range(padding))
            parts.append(decoded[:,:,:-trim] if trim else decoded)
        elif self.pending is not None:
            parts.append(self.pending)
        value=self._join(parts)
        self.last_summary=dict(input_latents=self.total_latents,output_frames=self.emitted_frames,
            VAE_calls=self.decode_calls,codec_tail_padding_latents=padding,codec_tail_trim_frames=trim,
            invented_left_context=False,held_pixel_frames=5,retained_latent_context=2,
            generation_chunk_size_fixed_by_decoder=False)
        self._reset()
        return value

    def flush(self):
        return self.finish()
