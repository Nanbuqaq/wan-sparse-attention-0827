"""Independent sampled-query FP64 global-deletion/value-removal diagnostics."""
import math
import time
import torch
from .immutable_source_reader import ImmutableSourceReader
from .layer_stream_source_reader import LayerStreamSourceReader


def counterfactual_statistics(q,k,v,reference,global_tokens,head_chunk=4):
    if q.shape!=reference.shape or k.shape!=v.shape or q.shape[0]!=1 or not 0<global_tokens<k.shape[1]:
        raise ValueError('matched batch-one attention and a nonempty global/rest partition required')
    heads=q.shape[2];dim=q.shape[3];queries=q.shape[1];keys=k.shape[1]
    # Bounds logical live payloads, not opaque BLAS workspaces.
    bound=2*head_chunk*keys*dim*8+4*head_chunk*queries*keys*8+16*head_chunk*queries*dim*8
    if bound>768*1024**2:raise ValueError('FP64 diagnostic logical temporary bound exceeded')
    metrics=[];masses=[];log_gains=[]
    for first in range(0,heads,head_chunk):
        last=min(heads,first+head_chunk)
        qq=q[0,:,first:last].transpose(0,1).double()
        kk=k[0,:,first:last].transpose(0,1).double()
        vv=v[0,:,first:last].transpose(0,1).double()
        ref=reference[0,:,first:last].transpose(0,1).double()
        scores=(qq@kk.transpose(-1,-2))/math.sqrt(dim)
        probability=scores.softmax(-1)
        full=probability@vv
        zero_v=probability[:,:,global_tokens:]@vv[:,global_tokens:]
        # Delete/re-normalize directly; do not validate via a candidate proxy.
        drop=scores[:,:,global_tokens:].softmax(-1)@vv[:,global_tokens:]
        zero_scores=scores.clone();zero_scores[:,:,:global_tokens]=0
        zero_kv=zero_scores.softmax(-1)[:,:,global_tokens:]@vv[:,global_tokens:]
        metrics.append(torch.stack([(full-ref).square().sum(),ref.square().sum(),(full-ref).abs().max(),
            full.square().sum(),(zero_v-full).square().sum(),(drop-full).square().sum(),(zero_kv-full).square().sum(),
            vv[:,:global_tokens].square().sum(),vv[:,global_tokens:].square().sum()]))
        masses.append(probability[:,:,:global_tokens].sum(-1))
        log_gains.append(scores.logsumexp(-1)-scores[:,:,global_tokens:].logsumexp(-1))
    m=torch.stack(metrics).cpu();p=torch.cat(masses).cpu();gain=torch.cat(log_gains).cpu()
    values=m.sum(0).tolist();reference_max=float(m[:,2].max())
    relative=math.sqrt(values[0]/max(values[1],1e-30));denominator=max(values[3],1e-30)
    return dict(reference_relative_L2=relative,reference_max_abs=reference_max,
        independent_reference_passed=relative<=.01 and reference_max<=.02,
        zero_global_V_relative_L2=math.sqrt(values[4]/denominator),
        delete_global_KV_relative_L2=math.sqrt(values[5]/denominator),
        zero_global_KV_relative_L2=math.sqrt(values[6]/denominator),
        global_V_RMS=math.sqrt(values[7]/(heads*global_tokens*dim)),
        other_V_RMS=math.sqrt(values[8]/(heads*(keys-global_tokens)*dim)),
        global_mass_mean=float(p.mean()),global_mass_median=float(p.median()),
        global_mass_p90=float(p.quantile(.9)),global_mass_max=float(p.max()),
        global_mass_by_head_query=p.tolist(),
        deletion_log_normalizer_gain_median=float(gain.median()),
        deletion_log_normalizer_gain_p90=float(gain.quantile(.9)),
        diagnostic_QK_pairs=heads*queries*keys,
        diagnostic_PV_pairs=heads*queries*(keys+3*(keys-global_tokens)),
        logical_temporary_bound_bytes=bound,FP64_teacher=True,
        deletion_and_zeroing_computed_directly=True,teacher_not_used_for_routing=True)


class GlobalNormalizerProbeMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if (self.source_policy!='full_once' or self.context_policy!='anchor_transition'
                or self.source_order!='after_global' or self.snapshot_window!='latest8'):
            raise ValueError('normalizer probe requires the qualified unchanged raw read graph')
        self.normalizer_records=[]

    def dispatch(self,layer,original,q,k,v,**kwargs):
        frame=self.active_start//self.frame_tokens;step=self.phase_counts[frame]-1
        wanted=(self.return_start is not None and frame in (self.return_start,self.return_start+8)
            and step in (0,4) and layer in (0,9,19,29))
        if not wanted:return super().dispatch(layer,original,q,k,v,**kwargs)
        def observe(qq,kk,vv):
            out=original(qq,kk,vv)
            if out.shape!=qq.shape:raise RuntimeError('native attention output geometry changed')
            began=time.perf_counter();events=[torch.cuda.Event(enable_timing=True) for _ in range(2)];events[0].record()
            ids=torch.linspace(0,qq.shape[1]-1,64,device=qq.device).round().long()
            sampled_q=qq.index_select(1,ids);sampled_actual=out.index_select(1,ids);g=kwargs['global_sink_tokens']
            stats=counterfactual_statistics(sampled_q,kk,vv,sampled_actual,g)
            # Native direct counterfactuals are independent of the FP64 mass
            # explanation and of any candidate deletion-error formula.
            native_full=original(sampled_q,kk,vv)
            zero_values=vv.clone();zero_values[:,:g].zero_()
            native_zero_v=original(sampled_q,kk,zero_values)
            native_drop=original(sampled_q,kk[:,g:],vv[:,g:])
            zero_keys=kk.clone();zero_keys[:,:g].zero_()
            native_zero_kv=original(sampled_q,zero_keys,zero_values)
            denominator=native_full.float().square().sum().clamp_min(1e-30)
            ref_denominator=sampled_actual.float().square().sum().clamp_min(1e-30)
            native_metrics=torch.stack([((native_full.float()-sampled_actual.float()).square().sum()/ref_denominator).sqrt(),
                (native_full.float()-sampled_actual.float()).abs().max(),
                ((native_zero_v.float()-native_full.float()).square().sum()/denominator).sqrt(),
                ((native_drop.float()-native_full.float()).square().sum()/denominator).sqrt(),
                ((native_zero_kv.float()-native_full.float()).square().sum()/denominator).sqrt()]).cpu().tolist()
            extra=qq.shape[2]*64*(4*kk.shape[1]-g)
            stats.update(native_sample_reference_relative_L2=native_metrics[0],native_sample_reference_max_abs=native_metrics[1],
                native_sample_reference_passed=native_metrics[0]<=.01 and native_metrics[1]<=.02,
                native_zero_global_V_relative_L2=native_metrics[2],native_delete_global_KV_relative_L2=native_metrics[3],
                native_zero_global_KV_relative_L2=native_metrics[4],native_teacher_FA2_calls=4)
            stats['diagnostic_QK_pairs']+=extra;stats['diagnostic_PV_pairs']+=extra
            events[1].record();events[1].synchronize()
            self.normalizer_records.append(dict(frame=frame,step=step,layer=layer,
                sampled_queries=64,full_Q_not_captured=True,Q=qq.shape[1],K=kk.shape[1],
                global_tokens=kwargs['global_sink_tokens'],source_allowed=self.source_is_allowed(layer,frame),
                probe_host_s=time.perf_counter()-began,probe_stream_ms=events[0].elapsed_time(events[1]),**stats))
            if len(self.normalizer_records)>16:raise RuntimeError('single-return diagnostic capacity exceeded')
            return out
        return super().dispatch(layer,observe,q,k,v,**kwargs)

    def audit(self):
        result=super().audit()
        if len(self.normalizer_records)!=16:raise RuntimeError('registered normalizer samples were not all reached')
        result['one_attention_dispatch_per_layer']=False
        result['FA2_calls_when_source_allowed']=None
        result['global_normalizer_probe']=dict(records=self.normalizer_records,production_FA2_calls_per_dispatch=1,
            extra_diagnostic_FA2_calls=sum(r['native_teacher_FA2_calls'] for r in self.normalizer_records),
            diagnostics_only_no_online_selection=True,source_and_attention_outputs_unmodified=True,
            extra_QK_pairs=sum(r['diagnostic_QK_pairs'] for r in self.normalizer_records),
            extra_PV_pairs=sum(r['diagnostic_PV_pairs'] for r in self.normalizer_records),
            timing_scope='teacher computation is inside the parent attention span; not production latency')
        return result


class GlobalNormalizerResidentProbe(GlobalNormalizerProbeMixin,ImmutableSourceReader):pass
class GlobalNormalizerStreamProbe(GlobalNormalizerProbeMixin,LayerStreamSourceReader):pass
