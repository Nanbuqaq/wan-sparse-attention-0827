#!/usr/bin/env python3
"""Source-derived frame lineage, not measured attention probabilities.

For the locked native block8/global8/shot8, absolute-RoPE, clean-recache-off
pipeline, track frame IDs through the same rolling and pinning rule. Validate
pin locations against actual terminal reports before using this explanation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def simulate(arm):
    if arm not in ('none','global','global_one_chunk','shot','window128'):
        raise ValueError('unknown arm')
    capacity=128 if arm=='window128' else 32
    slots=[];pin=-1;pin_len=0;saved=None;rows=[];pins=[]
    for start in range(0,128,8):
        if start==96 and arm in ('global','global_one_chunk','shot'):
            destination=pin if arm=='shot' else 0
            saved=slots[:8].copy()
            slots[destination:destination+8]=list(range(40,48))
        if start==104 and arm=='global_one_chunk':slots[:8]=saved
        effective=8+pin_len if pin==8 and pin_len else 8
        evicted=max(0,len(slots)+8-capacity)
        if evicted:
            slots=slots[:effective]+slots[effective+evicted:]
            if pin>=effective and pin_len:pin-=evicted
        slots+=list(range(start,start+8))
        # All occupied slots fit max_attention_size in this specific workload.
        roles=Counter('initial' if f<24 else 'reveal' if f<48 else 'away' if f<96 else 'return' for f in slots)
        rows.append(dict(query_start_latent=start,visible_frame_ids=slots.copy(),visible_latents=len(slots),
                         role_latent_counts=dict(roles),pin_before_clean_repin=pin))
        if start in (24,48,96):
            pin=len(slots)-8;pin_len=8
            pins.append(dict(completed_latent=start+8,pinned_start=pin*880,pinned_tokens=8*880))
    return dict(arm=arm,capacity=capacity,calls=rows,native_shot_pin_events=pins)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    reference=json.loads(args.reference.read_text())
    if (reference['status']!='pass' or reference['latent_shape']!=[1,128,48,44,80]
            or reference['local_frames']!=32 or reference['episode_memory_mode']!='none'
            or reference['expected_scene_cut_block_indices']!=[3,6,12]):
        raise ValueError('not the frozen native full-length control')
    arms=[simulate(a) for a in ('none','global','global_one_chunk','shot','window128')]
    if arms[0]['native_shot_pin_events']!=reference['native_shot_pin_events']:
        raise ValueError('shadow pin locations disagree with actual native execution')
    result=dict(schema='native_source_derived_frame_accounting_v1',measured_attention_mass=False,
        pin_metadata_validated_against_real_GPU=True,reference=str(args.reference),
        reference_summary_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        scope='source-derived frame IDs; not observed per-layer K/V hashes or softmax mass',arms=arms)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'frame_lineage.json').write_text(json.dumps(result,indent=2)+'\n')
    table=['# 首返回调用可见上下文（源码推导）','',
           '实际GPU的pin位置与推导逐项一致；以下是latent帧数，不是Attention概率质量。',
           '每latent帧880 token。query为96–103，共7040 token/head。',
           '', '| 配置 | 初始空物体 | 源身份/状态 | 离开的花朵场景 | 当前返回 | 总K latent |',
           '|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        call=arm['calls'][12];c=call['role_latent_counts']
        table.append(f'| {arm["arm"]} | {c.get("initial",0)} | {c.get("reveal",0)} | {c.get("away",0)} | {c.get("return",0)} | {call["visible_latents"]} |')
    table+=['','原global召回仍与16帧away同时竞争；shot召回减少为8帧away，但没有清空。',
            '大窗口保留24帧源信息，也保留48帧away；容量充足不等于上下文正确。',
            '首返回结束后，混合内容会成为新的clean KV。后续释放原错误KV不保证能逆转已写入的污染。',
            '最后一句是机制假设；需返回前上下文失效/相同容量错误源控制实验，不能当已验证因果结论。','']
    (args.output/'INTERPRETATION.md').write_text('\n'.join(table))
    print('\n'.join(table))


if __name__=='__main__':main()
