#!/usr/bin/env python3
"""Source-derived stage tables, hardware feeds/speeds and readable review material."""
import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import re
from pathlib import Path
import shutil

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


STAGE_LABELS={
    'text.encode_with_dynamic_swap':'文本编码（含动态参数换入）',
    'self_attention.q':'Self-Attn Q投影＋LoRA','self_attention.k':'Self-Attn K投影＋LoRA',
    'self_attention.v':'Self-Attn V投影＋LoRA','self_attention.o':'Self-Attn O投影＋LoRA',
    'transformer.ffn':'FFN＋LoRA','transformer.cross_attn':'文本Cross-Attention＋投影/LoRA',
    'history_and_exact_attention_core':'Self-Attention核心（exact＋history）',
    'history.selected_RoPE':'选中history RoPE','transformer.self_attn':'Self-Attn模块（GPU列仅剩余kernel）',
    'attention.query_group_KV_replication':'GPU query/head分组KV整理',
    'attention.complete_grouped_backend':'Attention wrapper其余GPU工作',
    'archive.route_indexed':'CPU history route','history.Q_summary_and_D2H':'Q摘要＋D2H',
    'history.CPU_pack_archive_runs':'CPU archive pack','archive.materialize_transfer_plan':'物理计划/传输物化',
    'history.materialize_route':'逻辑route物化入口（含子阶段）','archive.index_frame':'archive索引/原型（CPU＋GPU）',
    'archive_offload_stager.launch':'archive D2H发起','archive_offload_stager.complete':'archive D2H完成/提交',
    'history.coarse_frame_retrieval':'历史帧粗检索','history.descriptor_update':'已提交latent检索描述符',
    'vae.decode_complete':'VAE decode（含helper内像素D2H）'}


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join('---' for _ in headers)+'|\n'+''.join('| '+' | '.join(html.escape(str(x).replace('|','/'),quote=False) for x in row)+' |\n' for row in rows)


def fmt(value,digits=3):return '—' if value is None else f'{value:,.{digits}f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();work=args.work.resolve();base=work/'results/metrics/mentor_system_audit_20260908';out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir();sources=[]
    def load(path):
        raw=path.read_bytes();sources.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()));return json.loads(raw)
    hw=load(base/'hardware_calibration_local_v1/summary.json')
    catalog=load(base/'final39_kernel_resources_v1.json')
    streaming=load(work/'results/metrics/sprint24h_20260907/streaming_priority_trace_v2.activity.json')
    profiles={};all_rows=[]
    for name,method,trace_name in (('Dense','rag_dense','perfetto_dense39_v3'),('Final','transfer_vaware_hybrid_history','perfetto_final39_v3')):
        report=load(base/f'profile39_{method}_v2/report.json');trace=load(base/trace_name/'summary.json')
        stats=load(base/f'profile39_{method}_v2/generation_call_stats.json')
        if report['status']!='pass' or not report['instrumented_latent_exact_control']:raise ValueError('invalid profile')
        aggregate=report['trace']['aggregates'];work_stages=report['operator_work']['stages']
        gpu=trace['quantitative']['stages'];rows=[]
        for stage in sorted(set(aggregate)|set(work_stages)|{r['stage'] for r in gpu}):
            if stage=='unattributed':continue
            activity=[r for r in gpu if r['stage']==stage and r['major'] in ('generation.complete','vae.decode_complete')]
            kernel=sum(r['GPU_kernel_service_sum_s'] for r in activity)
            known=work_stages.get(stage);flops=known['FLOPs'] if known else None
            row=dict(case=name,stage=stage,label=STAGE_LABELS.get(stage,stage),CPU_inclusive_wall_s=aggregate.get(stage,{}).get('host_wall_s'),
                CPU_self_uncovered_s=aggregate.get(stage,{}).get('host_self_s'),GPU_kernel_service_s=kernel,
                useful_or_direct_equivalent_FLOPs=flops,LoRA_FLOPs=known['unmerged_lora_FLOPs'] if known else None,
                effective_TFLOP_per_s=flops/kernel/1e12 if flops and kernel else None,
                softmax_score_elements=known['softmax_pairs'] if known else None)
            for direction in ('H2D','D2H','D2D'):
                n=sum(r[direction]['bytes'] for r in activity);t=sum(r[direction]['service_sum_s'] for r in activity)
                row[direction+'_bytes']=n;row[direction+'_service_s']=t;row[direction+'_ops']=sum(r[direction]['operations'] for r in activity)
                row[direction+'_payload_GBps']=n/t/1e9 if t else None
            rows.append(row)
        profiles[name]=dict(report=report,trace=trace,stats=stats,rows=rows);all_rows+=rows
    with (out/'stage_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (out/'stage_metrics.json').write_text(json.dumps(all_rows,indent=2,ensure_ascii=False)+'\n')
    final=profiles['Final'];dense=profiles['Dense'];fr={r['stage']:r for r in final['rows']}
    hrows=hw['bandwidth'];cp=next(r for r in hrows if r['operation']=='CPU_contiguous_copy' and r['affinity']=='local_CPU_affinity')
    pin=next(r for r in hrows if r['operation']=='pinned_H2D' and r['affinity']=='local_CPU_affinity' and r['payload_bytes']==256*1024**2)
    pack=next(r for r in hrows if r['operation']=='CPU_pack_plus_pinned_H2D' and r['affinity']=='local_CPU_affinity' and r['payload_bytes']==256*1024**2)
    page=next(r for r in hrows if r['operation']=='pageable_H2D' and r['affinity']=='local_CPU_affinity' and r['payload_bytes']==256*1024**2)
    d2d=next(r for r in hrows if r['operation']=='GPU_D2D_copy')
    serial_prediction=1/(1/cp['effective_payload_GBps']+1/pin['effective_payload_GBps'])
    geom=[]
    for label,nk in (('Dense',18720),('Final',11700)):
        nq,h,d=4680,12,128;f=4*h*nq*nk*d;byte=4*h*d*(nq+nk);qtiles=math.ceil(nq/128)
        reread=4*h*d*(nq+qtiles*nk)
        geom.append(dict(case=label,Q=nq,K=nk,heads=h,head_dim=d,GFLOPs=f/1e9,
            one_read_write_tensor_MB=byte/1e6,ideal_tensor_AI=f/byte,
            QOut_tile128_KV_reread_model_MB=reread/1e6,QOut_no_L2_reuse_AI=f/reread))
    gen_s=final['report']['trace']['aggregates']['generation.complete']['host_wall_s']
    core_s=fr['history_and_exact_attention_core']['GPU_kernel_service_s']
    generation_activity=final['trace']['quantitative']['major_stages']['generation.complete']['GPU_all_activity_union_s']
    gen_flops={key:sum(v['FLOPs'] for k,v in value['report']['operator_work']['stages'].items() if k not in ('text.encode_with_dynamic_swap','vae.decode_complete')) for key,value in profiles.items()}
    params=final['report']['operator_work']['model_inventory']
    text=['# LongLive系统参数、逐阶段工作量与瓶颈审阅稿\n',
        '本次先完成导师系统分析；**等待用户过目后，才恢复新方法探索**。正文数据来自官方规格、真实模块shape、CUPTI/NVTX时间线及独立校准，四种证据不混用。\n',
        '## 0. 先看结论\n',
        f'- 本机4090的BF16/FP32累加密集Tensor Core峰值为**165.2 TFLOP/s**，不是330；大GEMM标定达到{max(r["effective_TFLOPs"] for r in hw["gemms"]):.1f} TFLOP/s。\n'
        f'- 256MiB pinned H2D约**{pin["effective_payload_GBps"]:.2f} GB/s**；CPU pack一起计入约**{pack["effective_payload_GBps"]:.2f} GB/s**。\n'
        f'- 本次Final39基线诊断generation为**{gen_s:.2f}s**，其中launch归因的Self-Attention核心kernel服务约**{core_s:.2f}s**。不能把其余时间统称FFN或GPU计算。\n'
        '- QOut/KVOut既可描述GPU内tile所有权，也可描述CPU-GPU供给顺序；固定边集合不自动减少QK/PV FLOPs。\n'
        '- FA4值得借鉴的是分资源建模→依赖/资源改造→真实时间线与消融，不是把SM100指令直接移植到4090。\n',
        '## 1. 测量协议与适用范围\n',
        table(['数据组','用途','不能当作'],[
            ['本次Dense/Final 39 latent /153 pixel，480×832，seed20260904','两独立4090进程的完整分阶段诊断；无采集latent对照精确一致','正式质量比较、重复端到端benchmark'],
            ['该诊断系统preset','archive-run＋per-chunk cache，metadata=recompute，grouped_fa2，batch VAE','不是叠加最新metadata复用/streaming VAE的最强执行配置'],
            ['已有Dense477完整Nsys','长轨迹的启动→生成→VAE/输出及last-call观察','不与39的FLOPs分子随意拼接算TFLOP/s'],
            ['已有current streaming153 Nsys','validated_reuse＋resident_grouped_fa2＋生成/VAE流水','不把其GPU忙碌比例搬到基线版本'],
            ['本次硬件校准','5预热、30测量，含同步host wall，显式pack路径完整计费','GPU架构保证峰值、真实视频收益或完整NUMA策略']]),
        '\n本机软件核对：PyTorch2.7.0a0+ecf3bae40a.nv25.02、CUDA12.8、flash-attn2.7.3、Triton3.2.0。采集器v2源码0689f7d；FA2论文发布期源码用于解释算法，不冒充本机安装版本。\n',
        '\nv1采集器访问DynamicSwap的weight属性造成额外参数H2D，且遗漏T5 attention核心计数；v2已修复并通过回归。v1不用于最终计费；原始结果保留。\n',
        '## 2. 硬件：规格、单位与实测\n',
        table(['参数','RTX4090（本机）','H200官方参照','含义'],[
            ['架构/SM','Ada SM8.9；实测128SM','Hopper SM9.0；具体SXM/NVL部署需核对','B200/SM100与RTX Blackwell/SM120不是同一kernel目标'],
            ['BF16输入、FP32累加，密集Tensor Core','165.2 TFLOP/s','SXM约989.5；NVL约835.5 TFLOP/s','H200官网1979/1671带2:4 sparsity脚注，密集需除2'],
            ['FP16输入、FP16累加','330.3 TFLOP/s','不同口径不混用','不是本项目BF16/FP32累加分母'],
            ['FP32 CUDA Core','82.6 TFLOP/s','SXM67；NVL60','norm/普通ALU与Tensor Core不同'],
            ['FP64 CUDA Core','1.29 TFLOP/s','SXM34；NVL30','原生RoPE的FP64/complex路径不能套BF16峰值'],
            ['显存','厂商24GB GDDR6X；CUDA可见字节见hardware JSON','141GB HBM3e','容量不等于带宽；实验cache预算不是硬件容量'],
            ['显存标称带宽','1008 GB/s','4800 GB/s','4090不是HBM'],
            ['PCIe单向编码后链路上限','Gen4×16约31.51 GB/s','Gen5×16约63.02 GB/s','H200宣传128GB/s为双向汇总量级，不能作为单向H2D分母'],
            ['GPU互联','本机GPU间SYS跨CPU互联，无NVLink','NVLink900GB/s标称','不能将GPU互联规格当CPU archive H2D带宽'],
            ['L2','实测75,497,472 bytes＝72MiB','本次未采集H200 cache属性','不能套完整AD102的96MiB'],
            ['每SM registers / shared','64K个32bit regs；100KiB shared','硬件guide与实际kernel另核对','本机单CTA动态shared opt-in99KiB'],
            ['warp/thread上限','48warps/1536threads每SM，32threads/warp','Hopper与Ada不同','这些不是实测occupancy']]),
        '\n官方来源：NVIDIA Ada白皮书Appendix A pp.29–31；H200规格页及sparsity脚注；Ada/Hopper/Blackwell tuning guide。文末source locks记录URL与SHA。CPU为2×AMD EPYC7Y83（各64核），256逻辑CPU、8 NUMA节点，L3合计512MiB；进程设置2个PyTorch CPU线程。本次没有读取到可核实DRAM DIMM速率，不编造CPU整机内存峰值。\n',
        table(['本机256MiB传输/复制校准','有效payload GB/s','包含什么'],[
            ['pinned H2D',fmt(pin['effective_payload_GBps']),'CPU pinned已就绪→GPU，含调用与完成同步'],
            ['pageable H2D',fmt(page['effective_payload_GBps']),'含驱动pageable路径成本'],
            ['CPU pack＋pinned H2D',fmt(pack['effective_payload_GBps']),'包含CPU copy/pack，再H2D'],
            ['1GiB CPU连续copy，2线程',fmt(cp['effective_payload_GBps']),'payload；若按读＋写算是'+fmt(cp['modeled_read_plus_write_GBps'])+'GB/s'],
            ['1GiB GPU D2D copy',fmt(d2d['effective_payload_GBps']),'payload；读＋写等价'+fmt(d2d['modeled_read_plus_write_GBps'])+'GB/s，不是DRAM counter']]),
        f'\n**分资源模型直接核对：**简单串行copy＋H2D的有效带宽=1/(1/{cp["effective_payload_GBps"]:.3f}+1/{pin["effective_payload_GBps"]:.3f})≈{serial_prediction:.3f}GB/s，与实测{pack["effective_payload_GBps"]:.3f}接近。理想流水上界受较慢阶段限制，但还会受到内存争用、buffer和发射开销影响，不能保证达到。CPU affinity切换不是严格NUMA membind；本次远端pageable更快的观察不推广为NUMA规律。\n',
        '![Measured transfer stages](figures/bandwidth.png)\n',
        '## 3. 模型shape、实际参数与计算量公式\n',
        table(['参数','值/规则'],[
            ['生成骨干','Wan2.1-T2V-1.3B，30层，width1536，FFN8960，12heads，head_dim128'],
            ['实际generator参数（含LoRA）',f'{params[0]["parameters"]:,}；tensor bytes={params[0]["parameter_tensor_bytes"]:,}'],
            ['未融合LoRA','rank256，300个目标Linear；349,962,240参数；A/B矩阵乘已计数'],
            ['Text encoder',f'UMT5-XXL encoder24层，width4096/FFN10240/64heads；{params[1]["parameters"]:,}参数'],
            ['VAE',f'{params[2]["parameters"]:,}参数，latent C16，实际Conv形状逐模块记录'],
            ['视频时间单位','39 latent→153 pixel；120→477；240→957；pixel=4×latent−3'],
            ['空间与chunk','latent60×104，patch1×2×2→1560tokens/latent；3latent/chunk→Q4680'],
            ['forward次数','4去噪＋1clean commit；本次13chunks×5×30=1950个transformer block calls'],
            ['算法块/物理页','Block64是(layer,head,frame,within-frame block)；Page256/Frame1560是物理layout'],
            ['cache预算','本profile GPU union cache4GiB，host pinned128MiB；不是GPU硬件上限'],
            ['路由/执行分组','Q摘要64-token分组（74组/head）；legacy union执行实测12组（每head一个），每组Q4680']]),
        '\nFMA=2 FLOPs。Linear：2×rows×Din×Dout；LoRA额外2×rows×r×(Din+Dout)。Conv：2×输出元素数×Cin/groups×kernel体积（direct等价量，不是实际cuDNN指令数）。Attention：2×有效边数×(dk+dv)。exp另记score元素数，不硬折算为Tensor FLOPs。\n',
        table(['稳态单层/单forward','Q','K','head数/d','Attention GFLOP','最小一次读写tensor MB','理想tensor FLOP/byte'],[
            [g['case'],g['Q'],g['K'],'12/128',fmt(g['GFLOPs']),fmt(g['one_read_write_tensor_MB']),fmt(g['ideal_tensor_AI'])] for g in geom]),
        '\n上述一次读写tensor量不是实际HBM/DRAM流量。以Q tile128、完全没有L2复用为另一种模型，QOut反复读取KV时AI约'+', '.join(g['case']+' '+fmt(g['QOut_no_L2_reuse_AI']) for g in geom)+' FLOP/byte；而165.2TFLOP/s÷1008GB/s的标称DRAM ridge约163.9。两种模型会给出不同瓶颈判断，因此不能用逻辑字节数代替L2/DRAM counter。\n',
        f'Dense39的已计DiT矩阵/Attention工作{gen_flops["Dense"]/1e12:.3f} TFLOP，Final39为{gen_flops["Final"]/1e12:.3f} TFLOP，降低{100*(1-gen_flops["Final"]/gen_flops["Dense"]):.2f}%。历史密度25%不等于整个模型只算25%；FFN、投影、cross-attn与VAE仍在。\n',
        '\n### 非GEMM阶段如何计工作量\n',
        table(['阶段','可明确列出的量','不能混淆'],[
            ['Q摘要','每层/冷调用输入7,188,480个Q元素；74×12×128个FP32摘要＝454,656 bytes D2H','归约/转换/拷贝不是一块Tensor Core GEMM'],
            ['CPU Q·K prototype项','6历史帧×每帧25个Block64＝150块/head；12heads×74groups×150blocks×128维×2＝34.0992 MFLOP','仅该dot-product项，完整Final还含归约、V-aware/coverage、索引、排序和哈希；不拿此数除完整route时间冒充CPU算力'],
            ['Block选择/索引','每head150候选块的元数据、选中坐标、run数与copy数，见call统计/CSV','整数/比较/排序以规模和时间计，不强行换成Tensor FLOPs'],
            ['RoPE','标准复数乘每complex-pair约6个real FLOPs（每real元素3）；另有dtype转换和多层次读写','缓存命中/位置策略影响实际处理量；FP64/complex路径与BF16峰值不是同一资源'],
            ['archive/输出/编码','按payload、padding、实际copy次数/耗时、文件大小和CPU范围计费','加载、pack、编码不应伪造GPU TFLOP/s']]),
        '## 4. 从开始到最后：时间分层\n',
        '以下是各自诊断轨迹，不是跨卡配对速度。加载单列，generation包含文本编码、缓存初始化和DiT，VAE随后批解码。初始import/config/device准备在首NVTX前单列；未把未采集区间画成真实GPU时间线。\n']
    major_names=['startup.load_pipeline','generation.complete','vae.decode_complete','output.normalize_VAE_CPU','output.RGB_convert_CPU','output.latent_save','output.MP4_encode_write','output.decode_integrity_check']
    for name,pf in profiles.items():
        agg=pf['report']['trace']['aggregates'];gpu=pf['trace']['quantitative']['major_stages']
        text+=['### '+name+'39：顶层时间\n',table(['阶段','CPU范围wall s','关联GPU全部activity union s','范围含义'],[
            [s,fmt(agg.get(s,{}).get('host_wall_s')),fmt(gpu.get(s,{}).get('GPU_all_activity_union_s')),
             'GPU异步服务与CPU范围不同' if s in ('generation.complete','vae.decode_complete') else 'CPU/IO或准备'] for s in major_names if s in agg])]
    text+=['\n### Final39：阶段算量、GPU服务与主机范围\n',
        'GPU有效TFLOP/s=已计FLOPs÷**GPU kernel服务时间**，不是除CPU函数返回时间。GPU按leaf拆分，CPU列仍是同名函数的inclusive与uncovered self；例如self-attn的CPU48.09s包含子阶段，GPU2.77s则不含已归到QKV/核心Attention的kernel。父子范围不能相加。非线性、整数、路由/索引、未单独计数的RoPE/整理用—，不填成零计算。\n']
    wanted=list(STAGE_LABELS)
    text.append(table(['阶段','已计TFLOP','其中LoRA TFLOP','GPU kernel s','有效TFLOP/s','CPU inclusive s','CPU未覆盖self s'],[
        [STAGE_LABELS[s],fmt(fr[s]['useful_or_direct_equivalent_FLOPs']/1e12 if fr[s]['useful_or_direct_equivalent_FLOPs'] is not None else None),
         fmt(fr[s]['LoRA_FLOPs']/1e12 if fr[s]['LoRA_FLOPs'] is not None else None),fmt(fr[s]['GPU_kernel_service_s']),
         fmt(fr[s]['effective_TFLOP_per_s']),fmt(fr[s]['CPU_inclusive_wall_s']),fmt(fr[s]['CPU_self_uncovered_s'])] for s in wanted if s in fr]))
    text+=['\n![Arithmetic and GPU service](figures/work_and_time.png)\n',
        '![Actual observed timeline overview](figures/final_timeline.png)\n',
        '### 数据搬运计费\n',table(['范围','方向','操作数','payload GB','GPU copy服务 s','payload/服务 GB/s'],[
            [name+'/'+stage,direction,pf['trace']['quantitative']['major_stages'][stage][direction]['operations'],
             fmt(pf['trace']['quantitative']['major_stages'][stage][direction]['bytes']/1e9),
             fmt(pf['trace']['quantitative']['major_stages'][stage][direction]['service_sum_s']),
             fmt(pf['trace']['quantitative']['major_stages'][stage][direction]['bytes']/pf['trace']['quantitative']['major_stages'][stage][direction]['service_sum_s']/1e9
                if pf['trace']['quantitative']['major_stages'][stage][direction]['service_sum_s'] else None)]
            for name,pf in profiles.items() for stage in ('generation.complete','vae.decode_complete') for direction in ('H2D','D2H','D2D')]),
        '\nH2D包含KV、文本模型参数和metadata，不能全部叫history payload；D2D是显式device copy，不包含全部kernel DRAM读写。完整逐阶段bytes/copy-count/service见stage_metrics.csv。\n',
        f'本次Final39里，**文本阶段H2D {fr["text.encode_with_dynamic_swap"]["H2D_bytes"]/1e9:.3f}GB / {fr["text.encode_with_dynamic_swap"]["H2D_service_s"]:.3f}s**；archive物化H2D **{fr["archive.materialize_transfer_plan"]["H2D_bytes"]/1e9:.3f}GB / {fr["archive.materialize_transfer_plan"]["H2D_service_s"]:.3f}s**（约{fr["archive.materialize_transfer_plan"]["H2D_payload_GBps"]:.2f}GB/s）。这里历史KV的实际DMA已接近大copy标定，CPU pack/索引/命中路径/整理更值得关注。文本换入是首prompt开销，不应当成每个稳态chunk都发生的历史KV搬运。\n',
        '## 5. 瓶颈程度：结论及证据强度\n',
        table(['对象','量化/性质','结论'],[
            ['加载',f'Final本次{final["report"]["trace"]["aggregates"]["startup.load_pipeline"]["host_wall_s"]:.2f}s；旧Dense477约296.57s且GPU活动约0.318s','CPU/storage/构建主导；mmap会把触页成本推迟，不能仅用torch.load wall当磁盘带宽'],
            ['生成主机路径',f'Final39 wall{gen_s:.2f}s，关联GPU activity union{generation_activity:.2f}s','存在大量主机发射/准备/同步空隙；不是CPU核心或DRAM饱和的直接证明'],
            ['Self-Attention核心',f'{core_s:.3f}s，占generation wall约{100*core_s/gen_s:.2f}%；有效{fr["history_and_exact_attention_core"]["effective_TFLOP_per_s"]:.1f}TFLOP/s','核心kernel本身效率不低；不是整个self-attention模块时间'],
            ['只优化核心kernel的条件上界',f'其他工作固定、核心无限快：generation至多约{1/(1-core_s/gen_s):.3f}×','条件Amdahl上界，不是承诺，也不能外推到新流水线'],
            ['FFN',f'GPU服务{fr["transformer.ffn"]["GPU_kernel_service_s"]:.3f}s，有效{fr["transformer.ffn"]["effective_TFLOP_per_s"]:.1f}TFLOP/s','显著非Attention GPU工作，未融合LoRA也在此计费'],
            ['VAE',f'CPU范围{final["report"]["trace"]["aggregates"]["vae.decode_complete"]["host_wall_s"]:.3f}s','GPU活动主导；compute vs memory需具体kernel证据，不凭busy比例定性'],
            ['CPU pack/H2D',f'{cp["effective_payload_GBps"]:.2f}与{pin["effective_payload_GBps"]:.2f}GB/s，串行合成仅{pack["effective_payload_GBps"]:.2f}','供给链瓶颈不能用PCIe峰值掩盖；pageable路径不一定输给额外pack'],
            ['GPU DRAM/L2/SMEM硬件事务','Ncu权限缺口保留，未修改权限','不声称实测HBM-bound；现有copy与launch资源可量化但不是counter']]),
        '\n### 静态资源：Nsight也能告诉我们什么\n',
        '本次主FA2 kernel为head128、Q tile128/K tile32、128threads/CTA；launch记录255 registers/thread、48KiB dynamic shared。结合4090每SM64K regs/100KiB shared，乐观上界为2CTAs、8warps，即16.7% warp occupancy。**这不是实测occupancy；低warp occupancy也不等于Tensor Core吞吐低。**\n',
        table(['代表性kernel/launch','calls','GPU service s','threads/CTA','regs/thread','SMEM/CTA KiB','CTA上界','warp占用上界'],[
            [r['kernel'][:80]+'…',r['calls'],fmt(r['GPU_service_sum_s']),r['threads_per_CTA'],r['registers_per_thread'],
             fmt((r['static_shared_bytes']+r['dynamic_shared_bytes'])/1024,1),r['resource_bounds']['CTA_upper_bound'],fmt(100*r['resource_bounds']['warp_occupancy_upper_bound'],1)+'%'] for r in catalog['rows'][:7]]),
        '\nregister分配粒度、subpartition约束和运行时活跃率未实测，上表只能做资源上界；local-memory metadata也不等于实测spill流量。完整448种launch形状在kernel资源JSON。\n',
        '## 6. 当前优化流水线与QOut/KVOut证据\n',
        f'当前streaming153代表性trace：parent{streaming["parent_wall_s"]:.3f}s，generation/control kernel{streaming["GPU_kernel_activity"]["generation_or_control"]["service_sum_s"]:.3f}s、VAE kernel{streaming["GPU_kernel_activity"]["vae"]["service_sum_s"]:.3f}s，两类kernel同时运行{streaming["simultaneous_VAE_and_generation_kernels_s"]:.3f}s。GPU全部activity union为{streaming["GPU_active_union_s"]:.3f}s（约{100*streaming["GPU_active_union_s"]/streaming["parent_wall_s"]:.1f}%时间），无本进程GPU活动约{streaming["GPU_idle_in_parent_s"]:.3f}s。这是不同执行版本的瓶颈观察，**不能把旧batch基线的空隙比例直接当作当前流水线比例**。此前无插桩完整视频重复实验独立验证收益，本次不以两张不同trace直接排名。\n',
        '![Real H2D and kernel overlap](figures/page_overlap.png)\n',
        '页流水图来自真实13.884ms窗口，不是示意图；H2D2.783ms，其中2.652ms与kernel重叠。这是包含CPU pack的组件试验，尚非生产history-onload端到端收益。\n',
        'QOut/KVOut含义、原论文/现成算子、FA4逐资源cycle推导及适用限制，见[FA4/QOut/KVOut详解](FA4_QOut_KVOut.html)。FA4 Figure1如下（原图，注明出处；不是本机trace）。\n',
        '![FA4 original forward pipeline](figures/fa4_forward.png)\n',
        '*来源：Zadouri等，FlashAttention-4，arXiv:2603.05451v1 Figure1；原文CC BY-NC-SA 4.0。*\n',
        '## 7. 待用户审阅的系统动作，不先扩新探索\n',
        '- 优先围绕主机准备/发射、GPU数据整理、生成/VAE供给链继续；所有优化要在完整服务时间中独立归因。\n'
        '- LoRA融合可减少额外matmul，但BF16下改变计算顺序，不保证逐位相同；不能直接当无损系统变换。\n'
        '- 本机kernel寄存器/SMEM上界已清楚；是否缩tile/减少live state需测实效。当前chunk内causal=False、legacy每head同长度union，不是天然LPT长短不均任务。\n'
        '- H200的141GB与实验4GiB cache预算不是一回事。旧120/240 latent CPU archive约31.05/65.56GB，数量级上值得审查更大驻留/全GPU强基线；不能仅凭容量相加声称已验证可行，也不能把受限offload路径代表H200最佳Dense。\n'
        '- 不在本次把FA4数学近似、自动causal memory或新正式质量矩阵启动；先交用户过目。\n',
        '## 8. 交付入口与复核\n',
        '先读[Perfetto阅读说明](Perfetto_阅读说明.html)，在ui.perfetto.dev本地打开trace文件。推荐顺序：Final39 overview→detail→current streaming153→page overlap；Dense477提供长轨迹参考。原Nsight/SQLite不随轻量包压缩搬运，保留在工作区。\n'
        '所有stage数字来自脚本生成，source_audit.json保存输入SHA。Counter为5ms活动时间占比，不是SM利用率；FLOPs为已列数学工作，卷积为direct等价量；绝不以估计字节冒充硬件事务。\n']
    text+=['### 官方参数与原始来源\n',
        '- [NVIDIA Ada GPU白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) Appendix A pp.29–31（165.2、1008、72MiB等）。\n'
        '- [NVIDIA H200规格](https://www.nvidia.com/en-us/data-center/h200/)（SXM/NVL与sparsity脚注）。\n'
        '- [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)、[Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)、[Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)。\n'
        '- [FA4原文v1](https://arxiv.org/html/2603.05451v1)与版本化算子链接见配套详解；来源锁副本在sources/。\n']
    report_text='\n'.join(text)
    (out/'REPORT.md').write_text(report_text)
    shutil.copy2(work/'publish_repo/docs/MENTOR_FA4_QOUT_KVOUT_20260908.md',out/'FA4_QOut_KVOut.md')
    shutil.copy2(work/'publish_repo/docs/MENTOR_PERFETTO_GUIDE_20260908.md',out/'Perfetto_阅读说明.md')
    figstyle={'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False};plt.rcParams.update(figstyle)
    fig,ax=plt.subplots(figsize=(9,4));labels=['CPU copy\npayload','Pinned H2D','Serial model','Pack + H2D\nmeasured','Pageable H2D']
    values=[cp['effective_payload_GBps'],pin['effective_payload_GBps'],serial_prediction,pack['effective_payload_GBps'],page['effective_payload_GBps']]
    ax.bar(labels,values,color=['#64748b','#0284c7','#94a3b8','#e08725','#64748b']);ax.set_ylabel('Effective payload GB/s (higher is better)')
    for i,v in enumerate(values):ax.text(i,v+.35,f'{v:.2f}',ha='center')
    ax.set_ylim(0,max(values)*1.2);ax.set_title('Preparation + transfer is a pipeline, not just PCIe bandwidth');fig.tight_layout();fig.savefig(out/'figures/bandwidth.png',dpi=160);plt.close(fig)
    chosen=[('QKV/O',[f'self_attention.{x}' for x in 'qkvo']),('FFN',['transformer.ffn']),('Self-Attn core',['history_and_exact_attention_core']),('Cross-Attn',['transformer.cross_attn']),('VAE',['vae.decode_complete'])]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,field,div,title in ((axes[0],'useful_or_direct_equivalent_FLOPs',1e12,'Counted work (TFLOP)'),(axes[1],'GPU_kernel_service_s',1,'GPU kernel service (seconds)')):
        values=[sum(fr[s].get(field) or 0 for s in group)/div for _,group in chosen];ax.barh([c[0] for c in chosen],values,color='#0284c7');ax.set_xlabel(title);ax.invert_yaxis()
    fig.suptitle('Final39 diagnostic: arithmetic and GPU time are separate from host wall');fig.tight_layout();fig.savefig(out/'figures/work_and_time.png',dpi=160);plt.close(fig)
    with gzip.open(base/'perfetto_final39_v3/overview.trace.json.gz','rt') as h:events=json.load(h)['traceEvents']
    fig,axes=plt.subplots(2,1,figsize=(11,4),sharex=True,gridspec_kw={'height_ratios':[1,3]})
    major={'startup.load_pipeline':'Load','generation.complete':'Generation','vae.decode_complete':'VAE','output.MP4_encode_write':'Encode'}
    for e in events:
        if e.get('ph')=='X' and e['name'] in major:
            color={'Load':'#dbeafe','Generation':'#bfdbfe','VAE':'#fde68a','Encode':'#e2e8f0'}[major[e['name']]]
            axes[0].broken_barh([(e['ts']/1e6,e['dur']/1e6)],(.1,.8),facecolors=color)
            if e['dur']>3e6:axes[0].text((e['ts']+e['dur']/2)/1e6,.5,major[e['name']],ha='center',va='center',fontsize=9)
    for name,color in (('kernel active %','#0284c7'),('H2D active %','#e08725'),('D2D active %','#64748b')):
        counters=[e for e in events if e.get('ph')=='C' and e['name']==name]
        # Display100ms average of the exact5ms counter data; raw stays in trace.
        n=len(counters)//20;x=[counters[i*20]['ts']/1e6 for i in range(n)];y=[np.mean([e['args']['percent'] for e in counters[i*20:(i+1)*20]]) for i in range(n)]
        axes[1].plot(x,y,label=name,color=color,linewidth=.8)
    axes[0].set_yticks([]);axes[1].set_ylabel('Traced activity %\nNOT SM utilization');axes[1].set_xlabel('Seconds from first captured NVTX range');axes[1].legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(out/'figures/final_timeline.png',dpi=160);plt.close(fig)
    with gzip.open(base/'perfetto_page_overlap_v2/detail.trace.json.gz','rt') as h:page_events=json.load(h)['traceEvents']
    gpu_events=[e for e in page_events if e.get('ph')=='X' and e.get('pid')==20];streams=sorted({e['tid'] for e in gpu_events})
    fig,ax=plt.subplots(figsize=(11,3))
    for i,stream in enumerate(streams):
        for kind,color in (('kernel','#0284c7'),('H2D','#e08725'),('D2D','#64748b'),('memset','#aaa')):
            ranges=[(e['ts']/1000,e['dur']/1000) for e in gpu_events if e['tid']==stream and e['cat']==kind]
            ax.broken_barh(ranges,(i-.3,.6),facecolors=color)
    ax.set_yticks(range(len(streams)),[f'GPU stream track {s}' for s in streams]);ax.set_xlabel('Milliseconds from actual NVTX window start');ax.set_title('Actual component trace: blue kernels / orange H2D (13.884 ms window)');fig.tight_layout();fig.savefig(out/'figures/page_overlap.png',dpi=160);plt.close(fig)
    from PIL import Image
    with Image.open(base/'sources_operators_v1/fa4_original_forward_pipeline.png') as im:
        im.thumbnail((1500,900));im.save(out/'figures/fa4_forward.png')
    (out/'assets').mkdir()
    shutil.copy2(base/'viewer_assets_v1/mathjax_tex_svg_3_2_2.js',out/'assets/mathjax_tex_svg_3_2_2.js')
    shell='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LongLive导师系统审阅稿</title><style>body{max-width:1250px;margin:32px auto;padding:0 22px;font:16px/1.8 system-ui;color:#1e293b}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:8px;border:1px solid #cbd5e1;vertical-align:top}th{background:#f1f5f9}h2{margin-top:40px}img{max-width:100%}a{color:#0369a1}code,pre{background:#f1f5f9}pre{padding:15px;overflow:auto}.math{overflow:auto}</style>'
    def render(text):
        equations=[]
        def protect(match,inline=False):
            i=len(equations);tag='span' if inline else 'div';delimiters=('\\(','\\)') if inline else ('\\[','\\]')
            equations.append(f'<{tag} class="math">'+delimiters[0]+html.escape(match.group(1))+delimiters[1]+f'</{tag}>')
            return 'MENTORMATHTOKEN'+str(i)+'END'
        text=re.sub(r'\\\[(.*?)\\\]',protect,text,flags=re.S)
        text=re.sub(r'\\\((.*?)\\\)',lambda m:protect(m,True),text,flags=re.S)
        body=markdown.markdown(text,extensions=['tables','fenced_code','toc'])
        for i,e in enumerate(equations):
            token='MENTORMATHTOKEN'+str(i)+'END'
            if e.startswith('<div'):body=body.replace('<p>'+token+'</p>',e)
            body=body.replace(token,e)
        return shell+'<p><a href="index.html">主报告</a> · <a href="FA4_QOut_KVOut.html">FA4与数据流</a> · <a href="Perfetto_阅读说明.html">Perfetto阅读说明</a></p>'+body+'<script src="assets/mathjax_tex_svg_3_2_2.js"></script></html>'
    (out/'index.html').write_text(render(report_text))
    for name in ('FA4_QOut_KVOut','Perfetto_阅读说明'):
        (out/(name+'.html')).write_text(render((out/(name+'.md')).read_text()))
    traces={'final39':base/'perfetto_final39_v3','dense39':base/'perfetto_dense39_v3',
            'dense477':base/'perfetto_dense477_reference_v1','current_streaming':base/'perfetto_streaming_current_v2'}
    (out/'traces').mkdir()
    for label,path in traces.items():
        for view in ('overview','detail'):
            shutil.copy2(path/(view+'.trace.json.gz'),out/'traces'/(label+'_'+view+'.trace.json.gz'))
    shutil.copy2(base/'perfetto_page_overlap_v2/detail.trace.json.gz',out/'traces/page_overlap_detail.trace.json.gz')
    (out/'sources').mkdir()
    for folder in ('sources_v1','sources_operators_v1','sources_corrections_v1','viewer_assets_v1'):
        shutil.copy2(base/folder/'source_lock.json',out/'sources'/(folder+'_lock.json'))
    for path in (work/'publish_repo/docs/MENTOR_FA4_QOUT_KVOUT_20260908.md',work/'publish_repo/docs/MENTOR_PERFETTO_GUIDE_20260908.md',
                 base/'sources_v1/source_lock.json',base/'sources_operators_v1/source_lock.json',base/'sources_corrections_v1/source_lock.json',base/'viewer_assets_v1/source_lock.json'):
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    audit=dict(status='pass',sources=sources,hardware_calibration=hw['status'],profile_latent_controls_exact=True,
        source_scope='baseline_work_and_timeline_plus_separate_current_pipeline_evidence',geometry_models=geom,
        serial_bandwidth_model_GBps=serial_prediction,stage_metrics_rows=len(all_rows),not_a_new_algorithm_or_formal_video_stage=True)
    (out/'source_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(status='pass',output=str(out),stage_rows=len(all_rows))))


if __name__=='__main__':main()
