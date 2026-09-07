#!/usr/bin/env python3
"""Render measured full-flow ranges without adding nested/device timers."""
import argparse
import hashlib
import json
from pathlib import Path


def render(report, source):
    if report.get("status") != "pass" or not report.get("instrumented_latent_exact_control"):
        raise ValueError("a completed latent-equivalent diagnostic is required")
    aggregates = report["trace"]["aggregates"]
    top = [r for r in report["trace"]["records"] if r["parent"] is None]
    lines = ["# 全流程实测（诊断口径）", "",
        f"硬件：{report['gpu']}；{report['latent_frames']} latent / {report['pixel_frames']} pixel；{report['method']}。",
        "插桩与同 seed 无插桩 latent 精确一致。数值代表这次诊断，不是重复计时统计或生产服务延迟。", "",
        "## 从启动到交付", "", "| 阶段 | wall 秒 | 设备 |", "|---|---:|---|"]
    lines.append(f"| 初始 import | {report['initial_import_wall_s']:.4f} | CPU/storage |")
    for row in top:
        lines.append(f"| {row['name']} | {row['host_wall_s']:.4f} | {row['device']} |")
    lines += ["", "这些 top-level 范围按顺序执行；控制回归和 report 写入另列，不把嵌套范围再相加。", "",
              "## 子阶段：包含其子调用的 host wall", "",
              "GPU 操作是异步的，host wall 不等于 GPU 服务时间。完整 Attention backend 包含重排与 kernel；self-attention 又包含 backend。", "",
              "| 范围 | 调用数 | inclusive wall 秒 | 未被子范围覆盖的 wall 秒 |", "|---|---:|---:|---:|"]
    for name, row in aggregates.items():
        if name not in {t["name"] for t in top}:
            lines.append(f"| {name} | {row['calls']} | {row['host_wall_s']:.4f} | {row['host_self_s']:.4f} |")
    lines += ["", "## 解释限制", "",
        f"- 随后运行的无插桩生成：{report['control_generation_wall_s']:.4f} s。它在暖缓存/后序执行，差额不能全部归因于插桩。",
        "- CUDA stream spans 含 host launch gaps；实际 GPU kernel/copy 活动、重叠与暴露等待需 Nsight。",
        "- 未归因 host self 包含 Python、dispatch、残差、同步/等待等，不称纯 CPU compute。",
        "- 本报告不提供未测得的 HBM/片上 transactions 或 CPU DRAM 饱和结论。",
        f"- 输入事实源：`{source}`；SHA256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = render(json.loads(args.input.read_text()), args.input.resolve())
    with args.output.open("x") as handle:
        handle.write(content)


if __name__ == "__main__":
    main()
