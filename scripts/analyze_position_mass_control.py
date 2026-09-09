#!/usr/bin/env python3
"""Offline-only: is temporal rebinding just giving the source more total mass?

The per-query/head matched bias is an oracle diagnostic, NOT an online policy.
Fixed original Q/V at every sampled phase; do not equate later rows with a new
generation trajectory. Reuse full verified captures, no GPU or video generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_temporal_rephase import rephase_temporal_keys
from scripts.analyze_native_attention_teacher import decompose,output_error,summary
from scripts.analyze_retimed_attention import load_verified


def match_group_mass(log_z,outputs,new_log_z,new_source_output,source=1):
    changed_z=log_z.clone();changed_z[...,source]=new_log_z
    bias=new_log_z-log_z[...,source]
    biased_z=log_z.clone();biased_z[...,source]+=bias
    target_mass=changed_z.softmax(-1);matched_mass=biased_z.softmax(-1)
    changed_outputs=outputs.clone();changed_outputs[...,source,:]=new_source_output
    return dict(bias=bias,mass=target_mass,matched_mass=matched_mass,
        retimed=(target_mass[...,None]*changed_outputs).sum(-2),
        mass_only=(matched_mass[...,None]*outputs).sum(-2))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--actual-comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);report=dict(status='running',rows=[])
    try:
        _,records,capture_sha,summary_sha=load_verified(args.capture)
        witness=json.loads(args.actual_comparison.read_text())
        assert witness['status']=='pass' and witness['sources']['original_capture_sha256']==capture_sha
        for key in sorted(records):
            record=records[key];base_report,base=decompose(record,['initial','source','away','current'])
            assert base_report['FP32_replay_gate']
            width=8*record['frame_tokens'];q=record['q'][0].permute(1,0,2).float().contiguous()
            source_k=record['k'][:,width:2*width];v=record['v'][0,width:2*width].permute(1,0,2).float().contiguous()
            old_k=source_k[0].permute(1,0,2).float().contiguous()
            new_k=rephase_temporal_keys(source_k,64)[0].permute(1,0,2).float().contiguous()
            old_logits=torch.bmm(q,old_k.transpose(1,2))*record['softmax_scale']
            new_logits=torch.bmm(q,new_k.transpose(1,2))*record['softmax_scale']
            old_p=old_logits.softmax(-1);new_p=new_logits.softmax(-1)
            new_output=torch.bmm(new_p,v);new_z=torch.logsumexp(new_logits,-1)
            control=match_group_mass(base['log_z'],base['role_outputs'],new_z,new_output)
            discrepancy=float((control['mass']-control['matched_mass']).abs().max());assert discrepancy<1e-6
            original=base['fp32_output'];retimed=control['retimed'];mass_only=control['mass_only']
            change=(retimed-original).double();residual=(retimed-mass_only).double()
            change_norm=float(change.norm());residual_norm=float(residual.norm())
            constants={}
            for gain in (2.,4.):
                z=base['log_z'].clone();z[...,1]+=torch.tensor(gain).log();mass=z.softmax(-1)
                out=(mass[...,None]*base['role_outputs']).sum(-2)
                constants[str(gain)]=dict(source_mass=summary(mass[...,1]),error_to_retimed=output_error(retimed,out))
            row=dict(phase=key[0],layer=key[1],FP32_original_replay_gate=True,
                original_source_mass=summary(base['mass'][...,1]),retimed_source_mass=summary(control['mass'][...,1]),
                matched_log_bias=summary(control['bias']),mass_matching_max_error=discrepancy,
                source_conditional_probability_TV=summary((old_p-new_p).abs().sum(-1)*.5),
                retimed_output_change=output_error(original,retimed),matched_mass_only_error_to_retimed=output_error(retimed,mass_only),
                matched_mass_only_residual_over_retime_change=(residual_norm/change_norm if change_norm>1e-12 else None),
                denominator_change_norm=change_norm,ratio_is_not_a_bounded_causal_fraction=True,constant_gain_controls=constants)
            if key==(0,0):
                actual=next(r for r in witness['rows'] if (r['phase'],r['layer'])==key)
                assert all(actual['input_witness'].values())
                assert abs(row['retimed_source_mass']['mean']-actual['retimed_roles']['source']['probability_mass']['mean'])<1e-6
                row['first_L0_matches_prior_actual_retimed_mass']=True
            report['rows'].append(row);print(json.dumps(dict(phase=key[0],layer=key[1],
                mass_only_residual_ratio=row['matched_mass_only_residual_over_retime_change'],
                within_source_TV=row['source_conditional_probability_TV']['mean'])),flush=True)
        report.update(status='pass',capture_sha256=capture_sha,summary_sha256=summary_sha,
            actual_comparison_sha256=hashlib.sha256(args.actual_comparison.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),CPU_only=True,
            offline_teacher_only=True,online_policy_frozen=False,quality_evaluated=False,
            query_sample='32 geometric queries/head, all24heads; not full-Q or semantic masks',
            all_rows_fixed_original_Q_and_V=True,late_rows_not_actual_retimed_trajectories=True,
            clean_commit_is_not_an_extra_denoising_step=True,temporal_delta=64)
    except Exception:report.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'mass_control.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
