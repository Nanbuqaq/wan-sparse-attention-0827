"""Fixed eight-frame CPU bank with explicit mixed-version physical slots.

Used for the one-update diagnostic, not general entity tracking. Every pending
raw write goes directly into existing slots; no sixteen-frame candidate pool.
"""
import torch


def selected_version_frames(policy,old,new):
    if policy not in ('latest8','old4_new4','uniform8'):
        raise ValueError('unknown frozen version policy')
    if len(old)!=8 or len(new)!=8:raise ValueError('two complete committed clean8 versions required')
    if max(x['frame'] for x in old)>=min(x['frame'] for x in new):raise ValueError('versions must be chronological')
    if policy=='latest8':return list(new)
    if policy=='old4_new4':return old[-4:]+new[-4:]
    # Uniform over the same sixteen candidate frames; this is explicitly not
    # uniform coverage over unseen/intermediate full-scene history.
    joined=old+new
    return [joined[i] for i in (0,2,4,6,9,11,13,15)]


class EightFrameBank:
    def __init__(self,frame_tokens,capacity_bytes,role_halves=None):
        if role_halves not in (None,'oldk_newv','newk_oldv'):raise ValueError('unknown K/V role halves')
        self.role_halves=role_halves
        self.frame_tokens=frame_tokens;self.capacity_bytes=capacity_bytes
        self.kv=[];self.records=[];self.raw_bytes=0;self.D2H_bytes=0;self.updates=0

    def update(self,caches,records,policy):
        if len(records)!=8:raise ValueError('only completed clean8 inputs')
        if self.updates>=2:raise RuntimeError('one-update version diagnostic already has v0 and v1')
        ft=self.frame_tokens
        if self.role_halves is not None:
            # Bounded role-aware pack: keep only the trailing four frames of each
            # update (v1[4:] become old4, v2[4:] become new4) and only the tensor
            # the role actually reads: K of the K-version, V of the V-version.
            if policy!='old4_new4':raise ValueError('role halves require the frozen old4_new4 policy')
            k_is_old=self.role_halves=='oldk_newv'
            take_k=(k_is_old and self.updates==0) or (not k_is_old and self.updates==1)
            if not self.kv:
                needed=sum(4*ft*c['k'].shape[0]*c['k'].shape[2]*c['k'].shape[3]*(c['k'].element_size()+c['v'].element_size()) for c in caches)
                if needed>self.capacity_bytes:raise RuntimeError('role-half allocation exceeds registered budget')
                self.kv=[(torch.empty_like(c['k'][:,:4*ft],device='cpu'),torch.empty_like(c['v'][:,:4*ft],device='cpu')) for c in caches]
                self.raw_bytes=needed
            for offset,r in enumerate(records[4:]):
                for cache,(key,value) in zip(caches,self.kv):
                    end=int(cache['local_end_index']);start=end-8*ft+(4+offset)*ft
                    source,target=(cache['k'],key) if take_k else (cache['v'],value)
                    target[:,offset*ft:(offset+1)*ft].copy_(source[:,start:start+ft])
                    self.D2H_bytes+=ft*target.shape[0]*target.shape[2]*target.shape[3]*target.element_size()
            self.records=list(records) if not self.records else self.records[4:]+list(records[4:])
            self.updates+=1
            return
        if not self.kv:
            needed=sum(8*ft*c['k'].shape[0]*c['k'].shape[2]*c['k'].shape[3]*(c['k'].element_size()+c['v'].element_size()) for c in caches)
            if needed>self.capacity_bytes:raise RuntimeError('eight-frame raw allocation exceeds registered budget')
            self.kv=[(torch.empty_like(c['k'][:,:8*ft],device='cpu'),torch.empty_like(c['v'][:,:8*ft],device='cpu')) for c in caches]
            self.raw_bytes=needed;selected=records
        else:selected=selected_version_frames(policy,self.records,records)
        old_slots={r['frame']:slot for slot,r in enumerate(self.records)}
        kept={old_slots[r['frame']] for r in selected if r['frame'] in old_slots}
        free=iter(slot for slot in range(8) if slot not in kept)
        assignments={r['frame']:old_slots[r['frame']] if r['frame'] in old_slots else next(free) for r in selected}
        new_by_frame={r['frame']:i for i,r in enumerate(records)}
        physical=[None]*8
        for r in selected:
            slot=assignments[r['frame']];physical[slot]=dict(r)
            if r['frame'] in old_slots:continue
            offset=new_by_frame[r['frame']]
            for cache,(key,value) in zip(caches,self.kv):
                end=int(cache['local_end_index']);start=end-8*ft+offset*ft
                for source,target in ((cache['k'],key),(cache['v'],value)):
                    target[:,slot*ft:(slot+1)*ft].copy_(source[:,start:start+ft])
                    self.D2H_bytes+=ft*target.shape[0]*target.shape[2]*target.shape[3]*target.element_size()
        self.records=physical;self.updates+=1

    def ordered_slots(self):
        if len(self.records)!=8 or any(r is None for r in self.records):raise RuntimeError('uncommitted version bank')
        return sorted(range(8),key=lambda i:self.records[i]['frame'])

    def audit(self):
        storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for pair in self.kv for t in pair}
        owned=sum(storages.values())
        if owned!=self.raw_bytes:raise RuntimeError('version bank has hidden/noncompact raw storage')
        return dict(capacity_bytes=self.capacity_bytes,unique_owned_raw_bytes=owned,D2H_bytes=self.D2H_bytes,
            pending_additional_raw_bytes=0,records=self.records,updates=self.updates,
            uniform_scope='same v0clean8 + v1clean8 candidate set, not whole-history oracle')
