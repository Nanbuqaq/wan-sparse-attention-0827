#!/usr/bin/env python3
"""Aggregate fixed-route LongLive2 timeline batches without mixing service/wall time."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

INTERVAL_NAMES = {
    "pack_attention_reshape": ("pack", "payload_hash", "attention", "reshape"),
    "pack_attention_scatter": ("pack", "attention", "scatter"),
    "gather_attention": ("gather", "attention"),
    "attention_only": ("attention",),
}


def _sum(rows, key):
    return sum(float(row.get(key) or 0.0) for row in rows)


def summarize_case(path: Path) -> dict:
    data = json.loads(path.read_text())
    wave2 = data.get("wave2") or {}
    timeline = wave2.get("route_timeline_records") or []
    intervals = defaultdict(float)
    intervals_by_state = defaultdict(lambda: defaultdict(float))
    for row in timeline:
        state = row.get("state", "unknown")
        for series, values in row.items():
            if not series.endswith("_interval_device_s") or not isinstance(values, list):
                continue
            base = series[: -len("_interval_device_s")]
            names = INTERVAL_NAMES.get(base, tuple(f"interval_{i}" for i in range(len(values))))
            if base == "gather_attention" and "selected_K" not in row and len(values) == 2:
                # Commit 4430237 recorded native/no-sparse FA2 in the first
                # gather_attention slot and a near-zero sentinel in the second.
                names = ("attention", "sentinel")
            for name, value in zip(names, values):
                if name != "sentinel":
                    key = f"{base}.{name}"
                    intervals[key] += float(value)
                    intervals_by_state[state][key] += float(value)
    rows = wave2.get("rows") or []
    pairs = sum(float(row.get("logical_pairs") or 0) for row in rows)
    full_pairs = sum(float(row.get("full_native_pairs") or 0) for row in rows)
    pipeline = data.get("video_pipeline") or {}
    pixels = data.get("pixels") or {}
    method = wave2.get("method") or data.get("wave2_method") or "native"
    if method == "w2_native":
        method = "native"
    return {
        "case": path.parent.name,
        "status": data.get("status", "missing"),
        "method": method,
        "selector": wave2.get("selector"),
        "preparation": wave2.get("preparation"),
        "generation_s": data.get("native_DiT_s"),
        "delivery_s": pipeline.get("complete_s"),
        "first_packet_s": pixels.get("first_packet_s") or pipeline.get("first_packet_s"),
        "producer_backpressure_s": pipeline.get("producer_backpressure_s"),
        "timeline_records": len(timeline),
        "timeline_states": dict(Counter(row.get("state") for row in timeline)),
        "prepare_host_s": _sum(timeline, "prepare_host_s"),
        "selection_host_s": _sum(timeline, "selection_host_s"),
        "execution_submit_host_s": _sum(timeline, "execution_submit_host_s"),
        "dispatch_host_s": _sum(timeline, "dispatch_host_s"),
        "device_interval_s": dict(sorted(intervals.items())),
        "device_interval_by_state_s": {state: dict(sorted(values.items())) for state, values in sorted(intervals_by_state.items())},
        "packed_QKV_bytes": _sum(timeline, "packed_QKV_bytes"),
        "payload_hash_D2H_bytes": _sum(timeline, "payload_hash_D2H_bytes"),
        "payload_hash_records": sum(1 for row in timeline if row.get("packed_K_sha256")),
        "route_audit_sha256": wave2.get("route_audit_sha256"),
        "route_audit_records": wave2.get("route_audit_records"),
        "selected_mask_hash_records": sum(1 for row in rows if row.get("selected_mask_sha256")),
        "logical_pairs": pairs,
        "full_native_pairs": full_pairs,
        "pair_ratio": pairs / full_pairs if full_pairs else None,
        "GPU_gather_output_bytes": _sum(rows, "GPU_gather_output_bytes"),
        "summary_build_host_s": wave2.get("summary_build_host_s"),
        "deferred_statistics_flush_host_s": wave2.get("deferred_statistics_flush_host_s"),
        "latent_sha256": data.get("latent_sha256"),
        "raw_RGB_sha256": pixels.get("raw_RGB_sha256"),
        "noise_sha256": data.get("noise_sha256"),
        "runner_commit": data.get("runner_commit"),
        "gpu": data.get("gpu"),
    }


def aggregate(cases):
    grouped = defaultdict(list)
    for row in cases:
        grouped[(row["method"], row.get("selector"), row.get("preparation"))].append(row)
    result = []
    for (method, selector, preparation), rows in sorted(grouped.items()):
        passed = [row for row in rows if row["status"] == "pass"]
        def med(key):
            values = [row[key] for row in passed if row.get(key) is not None]
            return median(values) if values else None
        attention = [sum(v for k, v in row["device_interval_s"].items() if k.endswith(".attention")) for row in passed]
        pack = [sum(v for k, v in row["device_interval_s"].items() if k.endswith(".pack")) for row in passed]
        result.append({
            "method": method,
            "selector": selector,
            "preparation": preparation,
            "cases": len(rows),
            "pass": len(passed),
            "generation_median_s": med("generation_s"),
            "delivery_median_s": med("delivery_s"),
            "pair_ratio_median": med("pair_ratio"),
            "attention_device_s_median": median(attention) if attention else None,
            "pack_device_s_median": median(pack) if pack else None,
            "prepare_host_s_median": med("prepare_host_s"),
            "execution_submit_host_s_median": med("execution_submit_host_s"),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = sorted(path for path in args.root.rglob("summary.json") if path.parent.name != "screen")
    cases = [summarize_case(path) for path in summaries]
    result = {
        "root": str(args.root),
        "case_count": len(cases),
        "cases": cases,
        "aggregate": aggregate(cases),
        "timing_scope": "CUDA event service intervals are separate from host spans and wall time; do not add overlapped intervals as wall time.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "pass", "cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
