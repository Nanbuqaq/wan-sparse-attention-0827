"""Native continuous H3 decode; defer five pixels, flush separately at hard cuts."""
import torch


class FixedChunkVideoDecoder:
    def __init__(self, vae, device):
        self.vae=vae;self.device=device;self.previous_two=None;self.pending=None
        self.mean=torch.tensor(vae.config.latents_mean,device=device).view(1,-1,1,1,1)
        self.std=torch.tensor(vae.config.latents_std,device=device).view(1,-1,1,1,1)
        assert vae.tokens_chunk_size==5 and vae.token_overlap==2 and vae.frame_overlap==5

    @torch.inference_mode()
    def decode(self,z):
        with torch.autocast('cuda',dtype=torch.float16):
            return self.vae.decode(z.to(self.device)*self.std+self.mean,return_dict=False)[0]

    @torch.inference_mode()
    def push(self,chunk):
        if chunk.ndim!=5 or chunk.shape[0]!=1 or chunk.shape[2] not in (5,7):
            raise ValueError('Require [1,24,5/7,H,W] normalized native latents.')
        if self.previous_two is None:
            if chunk.shape[2]==7:
                decoded=self.decode(chunk)
            else:
                # New shot: no old-scene decoder prefix. Replicate only current
                # generated latents, then discard the synthetic five pixels.
                prefix=chunk[:,:,:1].repeat(1,1,2,1,1)
                decoded=self.decode(torch.cat((prefix,chunk),dim=2))[:,:,5:]
        else:
            if chunk.shape[2]!=5:raise ValueError('Only source initialization uses seven latents.')
            decoded=self.decode(torch.cat((self.previous_two,chunk),dim=2))
            decoded=self.vae._blend(self.pending,decoded,5,dim=-3)
        self.previous_two=chunk[:,:,-2:].clone()
        self.pending=decoded[:,:,-5:].clone()
        return decoded[:,:,:-5]

    def flush(self):
        pending=self.pending
        self.pending=None;self.previous_two=None
        return pending
