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
    "gather_attention": ("gather", "payload_hash", "attention"),
    "attention_only": ("attention",),
}


def _sum(rows, key):
    return sum(float(row.get(key) or 0.0) for row in rows)


def _ms_sum(rows, key):
    return sum(float(row.get(key) or 0.0) for row in rows) / 1000.0


def summarize_delivery(pipeline: dict, generation_s) -> dict:
    """Decompose complete-delivery time from committed-chunk records.

    GPU stream spans are service time; host timestamps are wall clock and may
    include synchronization waits (e.g. pixels_D2H host span waits on decode).
    The two families are never added together here.
    """
    records = pipeline.get("records") or []
    groups = [g for r in records for g in (r.get("groups") or [])]
    if not records:
        return {"records": 0}
    decode_service_s = _ms_sum(groups, "decode_GPU_stream_span_ms")
    pixel_d2h_service_s = _ms_sum(groups, "pixel_D2H_GPU_stream_span_ms")
    latent_h2d_service_s = _ms_sum(records, "H2D_GPU_stream_span_ms")
    latent_d2h_service_s = _ms_sum(records, "D2H_GPU_stream_span_ms")
    chunk_queue_waits = [float(r["decode_started_s"]) - float(r["submitted_s"])
                         for r in records if r.get("decode_started_s") is not None and r.get("submitted_s") is not None]
    first_decode_s = min((float(r["decode_started_s"]) for r in records if r.get("decode_started_s") is not None), default=None)
    last_ready_s = max((float(g["CPU_pixels_ready_s"]) for g in groups if g.get("CPU_pixels_ready_s") is not None), default=None)
    decode_window_s = (last_ready_s - first_decode_s) if first_decode_s is not None and last_ready_s is not None else None
    decode_idle_s = (decode_window_s - decode_service_s - pixel_d2h_service_s) if decode_window_s is not None else None
    post_generation_decode_service_s = None
    delivery_tail_s = None
    if generation_s is not None:
        delivery_tail_s = (pipeline.get("complete_s") or 0.0) - float(generation_s)
        post_generation_decode_service_s = sum(
            float(g.get("decode_GPU_stream_span_ms") or 0.0) / 1000.0
            for g in groups
            if g.get("CPU_pixels_ready_s") is not None and float(g["CPU_pixels_ready_s"]) > float(generation_s))
    return {
        "records": len(records),
        "groups": len(groups),
        "decode_service_s": decode_service_s,
        "decode_service_per_group_ms": (decode_service_s / len(groups) * 1000.0) if groups else None,
        "pixel_D2H_service_s": pixel_d2h_service_s,
        "latent_H2D_service_s": latent_h2d_service_s,
        "latent_D2H_service_s": latent_d2h_service_s,
        "chunk_queue_wait_median_s": median(chunk_queue_waits) if chunk_queue_waits else None,
        "chunk_queue_wait_max_s": max(chunk_queue_waits) if chunk_queue_waits else None,
        "decode_window_s": decode_window_s,
        "decode_idle_s": decode_idle_s,
        "delivery_tail_s": delivery_tail_s,
        "post_generation_decode_service_s_approx": post_generation_decode_service_s,
        "producer_backpressure_s": pipeline.get("producer_backpressure_s"),
        "pixel_output_backpressure_s": pipeline.get("pixel_output_backpressure_s"),
    }


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
    first_packet_s = pixels.get("first_packet_muxed_s")
    if first_packet_s is None:
        first_packet_s = pixels.get("first_packet_s") or pipeline.get("first_packet_s")
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
        "first_packet_s": first_packet_s,
        "delivery": summarize_delivery(pipeline, data.get("native_DiT_s")),
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
        def med_delivery(key):
            values = [row["delivery"].get(key) for row in passed if row.get("delivery") and row["delivery"].get(key) is not None]
            return median(values) if values else None
        result.append({
            "method": method,
            "selector": selector,
            "preparation": preparation,
            "cases": len(rows),
            "pass": len(passed),
            "generation_median_s": med("generation_s"),
            "delivery_median_s": med("delivery_s"),
            "first_packet_median_s": med("first_packet_s"),
            "pair_ratio_median": med("pair_ratio"),
            "attention_device_s_median": median(attention) if attention else None,
            "pack_device_s_median": median(pack) if pack else None,
            "prepare_host_s_median": med("prepare_host_s"),
            "execution_submit_host_s_median": med("execution_submit_host_s"),
            "decode_service_s_median": med_delivery("decode_service_s"),
            "decode_window_s_median": med_delivery("decode_window_s"),
            "decode_idle_s_median": med_delivery("decode_idle_s"),
            "delivery_tail_s_median": med_delivery("delivery_tail_s"),
            "post_generation_decode_service_s_median": med_delivery("post_generation_decode_service_s_approx"),
            "chunk_queue_wait_median_s_median": med_delivery("chunk_queue_wait_median_s"),
            "producer_backpressure_s_median": med_delivery("producer_backpressure_s"),
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
