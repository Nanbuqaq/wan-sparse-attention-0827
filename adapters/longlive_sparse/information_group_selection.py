"""Bounded per-head diversity on original Block64/48 history.

Membership uses only committed V means. Within a group, the j-th block in
current relevance order receives relevance/j. Physical groups are the simple
control. An exact two-cost solver makes all arms execute the same token budget.
"""
import torch
import torch.nn.functional as F


class ExactTwoCostBudget:
    def __init__(self,costs,budget,device):
        if not costs or not set(costs)<={48,64} or budget<0:
            raise ValueError('only original48/64 groups are registered')
        self.budget=budget;self.blocks=len(costs)
        ids48=[i for i,c in enumerate(costs) if c==48]
        ids64=[i for i,c in enumerate(costs) if c==64]
        options=[(n,(budget-48*n)//64) for n in range(len(ids48)+1)
                 if budget-48*n>=0 and (budget-48*n)%64==0 and (budget-48*n)//64<=len(ids64)]
        if not options:raise ValueError('exact resource point is infeasible')
        self.ids48=torch.tensor(ids48,device=device,dtype=torch.long)
        self.ids64=torch.tensor(ids64,device=device,dtype=torch.long)
        self.n48=torch.tensor([x[0] for x in options],device=device,dtype=torch.long)
        self.n64=torch.tensor([x[1] for x in options],device=device,dtype=torch.long)
        self.costs=torch.tensor(costs,device=device,dtype=torch.long)

    def select(self,scores):
        if scores.ndim!=2 or scores.shape[-1]!=self.blocks:raise ValueError('head/block geometry differs')
        s48=scores[:,self.ids48];s64=scores[:,self.ids64]
        a48=s48.argsort(dim=-1,descending=True,stable=True);a64=s64.argsort(dim=-1,descending=True,stable=True)
        zero=torch.zeros((scores.shape[0],1),device=scores.device,dtype=scores.dtype)
        p48=torch.cat([zero,s48.gather(1,a48).cumsum(1)],1)
        p64=torch.cat([zero,s64.gather(1,a64).cumsum(1)],1)
        choice=(p48[:,self.n48]+p64[:,self.n64]).argmax(1)
        n48=self.n48[choice];n64=self.n64[choice]
        chosen=torch.zeros_like(scores,dtype=torch.bool)
        chosen.scatter_(1,self.ids48[a48],torch.arange(len(self.ids48),device=scores.device)[None]<n48[:,None])
        chosen.scatter_(1,self.ids64[a64],torch.arange(len(self.ids64),device=scores.device)[None]<n64[:,None])
        return chosen


def rank_discount(scores,groups,number):
    if groups.shape!=scores.shape:raise ValueError('head-specific membership required')
    order=scores.argsort(dim=-1,descending=True,stable=True)
    sorted_groups=groups.gather(1,order)
    one_hot=F.one_hot(sorted_groups,num_classes=number).to(torch.int32)
    rank=one_hot.cumsum(1,dtype=torch.int32).gather(2,sorted_groups[...,None]).squeeze(-1)
    restored=torch.empty_like(rank).scatter(1,order,rank)
    return scores/restored.float()


class IncrementalValueGroups:
    """Keep surviving assignments; assign only newly committed blocks.

    Centers are refreshed from surviving members after eviction. Empty groups
    use fixed evenly spaced new-block seeds. No Lloyd iterations or full-history
    reassignment; no query/output is used for membership.
    """
    def __init__(self,maximum=16):
        if maximum!=16:raise ValueError('first pilot fixes16 groups')
        self.maximum=maximum;self.states={};self.updates=self.hits=0
        self.new_blocks=self.retained_blocks=self.expired_blocks=0

    def update(self,layer,keys,values,counts):
        if not keys or not 0<=layer<30 or len(keys)>512 or len(keys)!=values.shape[0]:raise ValueError('bounded layer/block geometry differs')
        key=tuple(keys);old=self.states.get(layer)
        if old is not None and old['keys']==key:self.hits+=1;return old['groups']
        if len(set(key))!=len(key):raise ValueError('duplicate content/version block identity')
        heads=values.shape[1];number=min(self.maximum,len(key))
        weights=counts.float();center=(values.float()*weights[:,None,None]).sum(0)/weights.sum()
        x=F.normalize((values.float()-center[None]).permute(1,0,2),dim=-1,eps=1e-6)
        old_indices={} if old is None else {k:i for i,k in enumerate(old['keys'])}
        retained=[i for i,k in enumerate(key) if k in old_indices]
        new=[i for i,k in enumerate(key) if k not in old_indices]
        groups=torch.empty((heads,len(key)),device=values.device,dtype=torch.long)
        if retained and old['number']==number:
            take=torch.tensor(retained,device=values.device)
            prior=torch.tensor([old_indices[key[i]] for i in retained],device=values.device)
            ids=old['groups'][:,prior];groups[:,take]=ids
            membership=F.one_hot(ids,num_classes=number).float()
            sums=membership.transpose(1,2)@(x[:,take]*weights[take][None,:,None])
            mass=(membership*weights[take][None,:,None]).sum(1)
            centers=F.normalize(sums,dim=-1,eps=1e-6)
        else:
            new=list(range(len(key)));retained=[]
            centers=torch.zeros((heads,number,x.shape[-1]),device=x.device)
            mass=torch.zeros((heads,number),device=x.device)
        if new:
            new_ids=torch.tensor(new,device=x.device)
            seed_ids=torch.linspace(0,len(new)-1,number,device=x.device).round().long()
            seed=x[:,new_ids[seed_ids]]
            centers=torch.where((mass>0)[...,None],centers,seed)
            assigned=(x[:,new_ids]@centers.transpose(1,2)).argmax(-1)
            groups[:,new_ids]=assigned
        self.updates+=1;self.new_blocks+=len(new);self.retained_blocks+=len(retained)
        self.expired_blocks+=0 if old is None else len(set(old['keys'])-set(key))
        self.states[layer]=dict(keys=key,groups=groups,number=number)
        return groups

    def audit(self):
        return dict(max_groups=self.maximum,updates=self.updates,hits=self.hits,
            new_blocks_assigned=self.new_blocks,retained_assignments=self.retained_blocks,expired_blocks=self.expired_blocks,
            assignment_GPU_bytes=sum(x['groups'].numel()*x['groups'].element_size() for x in self.states.values()),
            last_group_sizes={str(layer):F.one_hot(s['groups'],num_classes=s['number']).sum(1).cpu().tolist()
                              for layer,s in self.states.items()},
            descriptor='count-weighted eligible-pool-centered committed V mean, unit normalized',
            centroid_update='surviving assignments only; new blocks assigned once; empty centers seeded deterministically',
            uses_current_Q_or_output_for_membership=False)


class InformationGroupSelector:
    def __init__(self,kind):
        if kind not in ('flat_exact','physical16','value16'):raise ValueError('unknown information-group pilot')
        self.kind=kind;self.budgets={};self.value_groups=IncrementalValueGroups()
        self.key_cache={}

    def keys_for_owners(self,layer,owners,blocks_per_frame):
        signature=(tuple(owners),blocks_per_frame);old=self.key_cache.get(layer)
        if old is None or old[0]!=signature:
            old=(signature,tuple((owner,b) for owner in owners for b in range(blocks_per_frame)))
            self.key_cache[layer]=old
        return old[1]

    def select(self,a,costs,budget,values,keys,layer):
        raw=a.sum(1);key=(tuple(costs),budget,str(a.device));solver=self.budgets.get(key)
        if solver is None:
            if len(self.budgets)>=32:raise RuntimeError('budget geometry bound32 exceeded')
            solver=ExactTwoCostBudget(costs,budget,a.device);self.budgets[key]=solver
        groups=None
        if self.kind=='physical16':
            number=min(16,len(costs))
            groups=(torch.arange(len(costs),device=a.device)*number//len(costs))[None].expand(a.shape[0],-1)
        elif self.kind=='value16':
            number=min(16,len(costs))
            groups=self.value_groups.update(layer,keys,values,solver.costs)
        scores=raw if groups is None else rank_discount(raw,groups,number)
        chosen=solver.select(scores)
        coverage=(a*chosen[:,None]).sum(-1)
        used=torch.full((a.shape[0],),budget,device=a.device,dtype=torch.long)
        return chosen,coverage,used,groups

    def audit(self):
        return dict(kind=self.kind,exact_48_64_token_budget=True,
            rank_rule='unchanged relevance' if self.kind=='flat_exact' else 'relevance divided by1-based rank within fixed group',
            budget_geometry_GPU_bytes=sum(t.numel()*t.element_size() for b in self.budgets.values()
                for t in (b.ids48,b.ids64,b.n48,b.n64,b.costs)),
            groups=self.value_groups.audit() if self.kind=='value16' else None,
            keeps_original_raw_KV=True,teacher_or_future_inputs=False,
            speedup_over_old_novelty_not_a_claim=True)
