#!/usr/bin/env python3
"""A compact discussion entry linking complete evidence, including negatives."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();work=args.work.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=work/'results/metrics/sprint24h_20260907';sources=[];targets=[]
    def source(path):
        data=path.read_bytes();sources.append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest()));return data
    def url(path):
        if not path.is_file():raise FileNotFoundError(path)
        targets.append(str(path));return quote(os.path.relpath(path,out),safe='/')
    def link(label,path):return '<a href="'+url(path)+'">'+html.escape(label)+'</a>'
    def picture(path,caption):return '<figure><a href="'+url(path)+'"><img src="'+url(path)+'" loading="lazy"></a><figcaption>'+caption+'</figcaption></figure>'
    inventory=json.loads(source(base/'video_inventory_final_20260908.json'))
    boundary=json.loads(source(base/'native_boundary_terminal_audit_20260908.json'))
    semantic=json.loads(source(base/'longlive2_native_episode509_review_v1/semantic_verdicts.json'))
    placement=json.loads(source(base/'longlive2_native_placement_gate64_review_v1/semantic_verdicts.json'))
    if inventory['remaining'] or boundary['status']!='pass' or not semantic['review_complete'] or not placement['review_complete']:
        raise ValueError('discussion index requires terminal evidence and completed reviews')
    report=work/'publish_repo/docs/SPRINT24H_RESEARCH_REPORT_20260908.md';source(report)
    parts=['<h1>流式长视频：信息、组织与生命周期</h1>',
        '<p class="lead">24小时研究讨论入口。已有可重复系统收益和新的记忆/表示证据；尚无全面优于LongLive/LongLive2的算法—系统共同赢家。</p>',
        '<p>'+link('完整研究总结',report)+' · '+link('导师三项要求',work/'publish_repo/docs/MENTOR_SYSTEM_KVOUT_TETHER_UPDATE_20260908.md')+' · '+link('时间序进度',work/'publish_repo/docs/SPRINT24H_PROGRESS_20260907.md')+'</p>',
        '<h2>1. 系统收益是真实的，层级也必须讲清楚</h2>',
        '<p>H200477组合配对中位1.16–1.20×；4090新增957两类中位1.133×/1.155×，对应route/latent/RGB精确一致。首MP4包提前不是客户端显示延迟，也不是16fps实时。</p>',
        picture(base/'system_insight_figures_v1/h200_complete_service_factorial.png','同硬件、同输出的完整服务对照；negative重复保留。'),
        '<p>'+link('所有系统图和来源SHA',base/'system_insight_figures_v1/index.html')+' · '+link('完整设备/存储/阶段解析',work/'publish_repo/docs/SYSTEM_FULL_FLOW_AND_REPRODUCTION_AUDIT_20260907.md')+' · '+link('477逐阶段诊断量化表',work/'results/metrics/full_flow_20260907/final120_v2/REPORT_V2.md')+'</p>',
        '<h2>2. 相关信息可以取回，正确使用它仍是问题</h2>',
        picture(base/'native_memory_figures_v3_recorded/native_memory_failure.png','原生H800分镜开发任务：生成特定信息 → 真正离开 → 返回后身份/状态未保住。这不是方法胜出。'),
        '<p>后续H200四臂16视频：raw/log相关记忆完整latent与RGB逐位一致、归档张量90.10×差异；但四组都未得到稳定整体质量成功。以下列全四组，不以局部颜色/面孔恢复替代质量结论。</p>',
        '<table><thead><tr><th>场景/seed</th><th>观察与结论</th><th>审查板</th></tr></thead><tbody>']
    labels=['部分原面孔回来，但花朵/碎片污染','雏菊遮盖主体，无稳定身份成功',
            '红珠回来，但绿色珠子和碎片破坏状态','仍为花朵/罐混合，无稳定状态成功']
    review=base/'longlive2_native_episode509_review_v1'
    for group,label in zip(semantic['groups'],labels):
        lane=group['lane'];name=('玩具' if lane<2 else '红珠')+'/'+str(group['seed'])
        boards=' · '.join(link(title,review/f'lane{lane}__{mode}__{part}.png') for title,mode,part in
                          [('source','none','source'),('baseline','none','q4'),('相关KV','raw_reveal','q4'),('无关KV','raw_away','q4')])
        parts.append('<tr><td>'+name+'</td><td>'+label+'</td><td>'+boards+'</td></tr>')
    parts+=['</tbody></table><p>'+link('16视频完整解释',review/'INTERPRETATION.md')+' · '+link('最后6条放置门禁：任务有效性失败',base/'longlive2_native_placement_gate64_review_v1/INTERPRETATION.md')+'</p>',
        '<p>Tether：官方CF+AE与neutral控制已审查，未得到稳定整体质量赢家；mask失败与手工oracle分开。没有同条件AdaCluster/SVOO/SCOPE新排名。'+link('Tether效果及可用机制',work/'publish_repo/docs/TETHER_LONG_VIDEO_FINDINGS_20260908.md')+'</p>',
        '<h2>3. 热KV与冷日志之间，存在可测的恢复折中</h2>',
        picture(base/'native_hybrid_restore_figures_v1/tradeoff.png','本次同一物理GPU0：全部原KV哈希通过；每路径5预热、30随机交错重复，红标记为p95。仅warm恢复，不含创建/驱逐/冷存储，不是实际进程RSS或视频加速。'),
        '<p>'+link('数值表与来源SHA',base/'native_hybrid_restore_figures_v1/index.html')+' · '+link('已提交日志与数值执行配方',work/'publish_repo/docs/NATIVE_CLEAN_COMMIT_REPLAY_20260908.md')+'</p>',
        '<h2>下一轮只追三个关键问题</h2><ol><li>如何激活原生成的信息，同时不携带错误场景上下文？</li><li>哪些记忆值得常驻KV，哪些适合检查点/日志重建，如何计入创建、迁移、重访和deadline？</li><li>有限驻留KV-major的取数收益，何时能在真实供给—消费流水中成为完整视频收益？</li></ol>',
        '<h2>审计与边界</h2><p>'+str(inventory['terminal_executions'])+'/'+str(inventory['expected_executions'])+'命名视频执行有终态，missing=0；包含重复、门禁、技术失败，不是独立实验或质量成功数。原正式和Tether各有独立审计。</p>',
        '<p>'+link('视频/latent完整SHA清单',base/'video_inventory_final_20260908.json')+' · '+link('恢复失败/Blackwell未启动/资源拦截审计',base/'native_boundary_terminal_audit_20260908.json')+'</p>']
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LongLive 24h研究讨论</title><style>body{max-width:1100px;margin:40px auto;padding:0 22px;color:#1e293b;font:16px/1.8 system-ui}h1,h2{line-height:1.4}h2{margin-top:44px}a{color:#0369a1}img{max-width:100%;height:auto}figure{margin:20px 0}figcaption{font-size:14px;color:#475569}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #cbd5e1}.lead{font-size:18px}</style>'+''.join(parts)+'</html>'
    (out/'index.html').write_text(page)
    (out/'source_audit.json').write_text(json.dumps(dict(status='pass',sources=sources,local_targets=sorted(set(targets)),
        all_local_targets_exist=True,no_new_experimental_data_created=True),indent=2)+'\n')
    print(json.dumps(dict(status='pass',linked_targets=len(set(targets)),output=str(out))))


if __name__=='__main__':main()
