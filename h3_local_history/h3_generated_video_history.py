"""Bounded active history, separate from the optional output archive."""
import torch
from h3_fixed_chunk_contract import history_ids


class GeneratedVideoHistory:
    def __init__(self,source,h_max_sec=8.):
        self.source=source;self.h_max_sec=h_max_sec;self.frames={}

    def put(self,start,chunk):
        if chunk.ndim!=4 or chunk.shape[0]!=24:raise ValueError('Require [24,T,H,W]')
        for offset in range(chunk.shape[1]):
            self.frames[start+offset]=chunk[:,offset].detach().cpu().clone()

    def select(self,start):
        ids=sorted(set(self.frames)&set(history_ids(self.source,start,self.h_max_sec)))
        values=torch.stack([self.frames[i] for i in ids],dim=1) if ids else None
        return ids,values

    def evict(self,end):
        allowed=set(history_ids(self.source,end,self.h_max_sec));removed=[]
        for i in list(self.frames):
            if i not in allowed:del self.frames[i];removed.append(i)
        return sorted(removed)

    def reset(self):
        self.frames.clear()

    @property
    def bytes(self):
        return sum(x.numel()*x.element_size() for x in self.frames.values())


def wrong_past_video(peer_video,peer_chunks,current_chunk_number,needed_frames):
    """Only prior peer chunks; later saved outputs never enter the control."""
    ids=[i for r in peer_chunks[:current_chunk_number]
         for i in range(r['video_latent_start'],r['video_latent_end'])]
    if needed_frames>len(ids):raise ValueError('Peer has insufficient generated past.')
    ids=ids[-needed_frames:] if needed_frames else []
    return peer_video.index_select(1,torch.tensor(ids,dtype=torch.long)),ids
