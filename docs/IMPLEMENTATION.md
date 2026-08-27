# 实现与复现入口

## 核心接口

- `adapters/types.py`：`MethodConfig`、`RoutePlan`与累计审计统计。
- `adapters/routing.py`：Q/K聚类、permutation、Top-p校准、实际Q-K pair预算和padding/load统计。
- `adapters/kernels.py`：`fixed64_bf16`与`varlen_triton`严格执行，无Dense fallback。
- `adapters/wan_sparse.py`：只替换Wan self-attention；cross-attention、非BF16和异常输入直接报错。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/preflight.py
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/run_matrix.py --suite configs/smoke_1step.json
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/run_matrix.py --suite configs/core_screen_4step.json
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/run_matrix.py --suite configs/formal_50step.final.json
```

任务以MP4和同名`stats.json`为完成单元，可断点续跑。`--prompt`、`--matrix`、`--include`与shard参数可裁剪任务。

## 评测与审计

```bash
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/evaluate_matrix.py --suite configs/formal_50step.final.json
/usr/bin/python3 scripts/plot_results.py --metrics-dir results/metrics/formal_50step.final
/usr/bin/python3 scripts/audit_results.py --suite configs/formal_50step.final.json
/usr/bin/python3 scripts/build_manifest.py
```

正式速度同时报告logical pairs、scheduled tile pairs、padding和load imbalance。首次Triton编译与warm p50/p90保存在每个任务的stats中。
