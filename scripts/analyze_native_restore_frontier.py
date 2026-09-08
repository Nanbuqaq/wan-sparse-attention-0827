#!/usr/bin/env python3
"""Finite-sample deadline counts and an explicitly unmeasured staging model."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_native_hybrid_restore_figure import rows_from_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();raw=args.report.read_bytes();data=json.loads(raw);rows=rows_from_report(data)
    names=set(data['samples_s']);n=data['repeats']
    if len(data['blocked_randomized_order'])!=n or any(len(v)!=n for v in data['samples_s'].values()) or any(set(order)!=names or len(order)!=len(names) for order in data['blocked_randomized_order']):
        raise ValueError('unmatched repeat blocks')
    deadlines=(.75,1.,1.25,1.5)
    for row in rows:
        samples=data['samples_s'][row['path']]
        row['mean_s']=statistics.mean(samples)
        row['deadline_met_counts']={str(t):sum(x<=t for x in samples) for t in deadlines}
        row['measured_repeats']=n
    # Model assumes serial transfer to CPU, then the measured warm restore.
    # It does NOT measure a network, NVMe, GPUDirect, pipelining or interference.
    raw_row=next(r for r in rows if r['path']=='raw_pageable')
    log_row=next(r for r in rows if r['path']=='clean_log_replay')
    crossover=(raw_row['state_tensor_bytes']-log_row['state_tensor_bytes'])/(log_row['median_s']-raw_row['median_s'])
    bandwidths=(.25,.5,1.,2.,4.,6.,8.,16.,32.,64.)
    modeled=[]
    for bandwidth in bandwidths:
        costs={r['path']:r['median_s']+r['state_tensor_bytes']/(bandwidth*1e9) for r in rows}
        modeled.append(dict(external_bandwidth_GBps=bandwidth,winner=min(costs,key=costs.get),predicted_serial_median_proxy_s=costs))
    result=dict(status='pass',source=str(args.report.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),
        rows=rows,descriptive_deadline_grid_seconds=deadlines,
        observed_deadline_counts_are_not_a_production_SLO=True,one_process_repeats_not_independent_sessions=True,
        grid_is_posthoc_description_not_online_admission_calibration=True,
        unmeasured_serial_CPU_staged_migration_model=dict(formula='state_bytes / external_bandwidth + measured_warm_restore_median',
            endpoint_crossover_GBps=crossover/1e9,points=modeled,
            no_actual_network_NVMe_GPUDirect_or_overlap_experiment=True,
            per_object_memory_caps_and_creation_costs_not_in_this_model=True))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table='\n'.join('| '+r['label']+' | '+' | '.join(f"{r['deadline_met_counts'][str(t)]}/{n}" for t in deadlines)+' |' for r in rows)
    (args.output/'INTERPRETATION.md').write_text('# 恢复延迟：看deadline，但不把30次采样当SLO\n\n'
        '| 表示 | ≤0.75s | ≤1.0s | ≤1.25s | ≤1.5s |\n|---|---:|---:|---:|---:|\n'+table+'\n\n'
        '这是一个进程内30次blocked随机测量的描述，阈值为事后展示网格，不是预注册的在线选择规则或生产SLO。\n\n'
        '## 尚未实测的迁移模型\n\n'
        '仅假定外部存储/网络→CPU和warm restore完全串行，则 T = bytes/B + 本次warm median。'
        f'该模型下raw与日志端点交点约{crossover/1e9:.3f} GB/s。所列带宽点中间检查点并不必然赢得最小延迟；'
        '它们的价值可能来自每对象硬容量约束，而非一条普遍更快的迁移路径。\n\n'
        '没有实测网络/NVMe/GPUDirect、copy-compute重叠、创建和背景干扰。不要用此阈值提交正式方法、宣称跨节点速度，'
        '或把中位数相加当成真实组合延迟分布；这是选择下一项系统实验的分析模型。\n')
    print(json.dumps(dict(status='pass',endpoint_crossover_GBps=crossover/1e9,
                         modeled_winners=sorted(set(p['winner'] for p in modeled)),rows=rows)))


if __name__=='__main__':main()
