#!/usr/bin/env python3
"""Join technical, paired-quality and manual evidence into case-level metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.case_identity import validate_case_identity


COUNT_FIELDS = [
    "irreversible_state_reset_count",
    "action_loop_count",
    "action_discontinuity_count",
    "freeze_count",
    "flicker_count",
    "camera_cut_count",
]
SCORE_FIELDS = [
    "subject_identity_1to5",
    "background_consistency_1to5",
    "late_quarter_quality_1to5",
]


def baseline_method(method: str) -> str:
    if method in {"native_dense", "native_block"}:
        return "native_dense"
    return "rag_dense"


def pairing_key(case: dict) -> tuple:
    return (
        case.get("commit"),
        case.get("prompt_id"),
        int(case.get("seed")),
        int(case.get("latent_frames")),
    )


def _number(value, *, field: str, case_id: str) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing manual field {field}: {case_id}")
    return float(value)


def negative_reasons(
    case: dict,
    baseline: dict,
    *,
    finalize: bool,
    skip_speed_negative: bool = False,
) -> list[str]:
    if case.get("status") != "pass" or case["method"] == baseline["method"]:
        return []
    reasons = []
    manual = case.get("manual_review")
    baseline_manual = baseline.get("manual_review")
    if finalize:
        if manual is None or baseline_manual is None:
            raise ValueError(f"missing complete-video manual review: {case['id']}")
        for field in COUNT_FIELDS:
            value = _number(manual.get(field), field=field, case_id=case["id"])
            base = _number(
                baseline_manual.get(field), field=field, case_id=baseline["id"]
            )
            if value > base:
                reasons.append(f"new_{field}")
        for field in SCORE_FIELDS:
            value = _number(manual.get(field), field=field, case_id=case["id"])
            base = _number(
                baseline_manual.get(field), field=field, case_id=baseline["id"]
            )
            if value <= base - 2:
                reasons.append(f"severe_drop_{field}")
        degradation = _number(
            manual.get("late_quarter_degradation_0to2"),
            field="late_quarter_degradation_0to2",
            case_id=case["id"],
        )
        base_degradation = _number(
            baseline_manual.get("late_quarter_degradation_0to2"),
            field="late_quarter_degradation_0to2",
            case_id=baseline["id"],
        )
        if degradation >= 2 and degradation > base_degradation:
            reasons.append("severe_late_quarter_degradation")

    if not skip_speed_negative:
        end_to_end = case.get("end_to_end_s")
        base_end_to_end = baseline.get("end_to_end_s")
        attention = case.get("attention_s")
        base_attention = baseline.get("attention_s")
        h2d = case.get("h2d_s")
        base_h2d = baseline.get("h2d_s")
        if all(value is not None for value in (end_to_end, base_end_to_end, attention, base_attention, h2d, base_h2d)):
            slower = float(end_to_end) > 1.05 * float(base_end_to_end)
            attention_gain = float(attention) <= 0.90 * float(base_attention)
            h2d_gain = float(h2d) <= 0.90 * float(base_h2d)
            if slower and not attention_gain and not h2d_gain:
                reasons.append("slower_than_dense_without_attention_or_h2d_gain")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--expected")
    parser.add_argument("--quality", action="append", default=[])
    parser.add_argument("--manual-review")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--skip-speed-negative",
        action="store_true",
        help="skip runtime-based negative classification for mixed-hardware evidence",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-output")
    args = parser.parse_args()

    payload = json.loads(Path(args.states).read_text(encoding="utf-8"))
    cases = [dict(case) for case in payload["cases"]]
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise RuntimeError("duplicate case ids")
    for case in cases:
        errors = validate_case_identity(case)
        if errors:
            raise ValueError(f"invalid case identity {case.get('id')}: {errors}")
    if args.expected:
        expected_cases = json.loads(
            Path(args.expected).read_text(encoding="utf-8")
        )["cases"]
        expected_by_id = {case["id"]: case for case in expected_cases}
        if len(expected_by_id) != len(expected_cases):
            raise RuntimeError("expected manifest contains duplicate case ids")
        if set(expected_by_id) != set(by_id):
            raise RuntimeError("states and expected manifest case sets differ")
        for case in cases:
            expected = expected_by_id[case["id"]]
            if expected.get("case_key_sha256") != case.get("case_key_sha256"):
                raise RuntimeError(f"expected identity mismatch: {case['id']}")
            case["expansion_axes"] = expected.get("expansion_axes", [])

    quality = {}
    for value in args.quality:
        summary = json.loads(Path(value).read_text(encoding="utf-8"))
        for case_id, metrics in summary.get("candidates", {}).items():
            if case_id in quality and quality[case_id] != metrics:
                raise RuntimeError(f"conflicting quality metrics: {case_id}")
            quality[case_id] = metrics

    manual = {}
    if args.manual_review:
        with Path(args.manual_review).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                case_id = row["case_id"]
                if case_id in manual:
                    raise RuntimeError(f"duplicate manual review: {case_id}")
                manual[case_id] = row

    for case in cases:
        case["baseline_method"] = baseline_method(case["method"])
        if case["id"] in quality:
            case["quality_metrics"] = quality[case["id"]]
        if case["id"] in manual:
            case["manual_review"] = manual[case["id"]]

    baseline_index = {
        (case["method"], *pairing_key(case)): case
        for case in cases
        if case["method"] in {"native_dense", "rag_dense"}
    }
    rows = []
    for case in cases:
        baseline_name = case["baseline_method"]
        baseline = baseline_index.get((baseline_name, *pairing_key(case)))
        if baseline is None:
            raise RuntimeError(f"missing same-commit Dense baseline: {case['id']}")
        reasons = negative_reasons(
            case,
            baseline,
            finalize=args.finalize,
            skip_speed_negative=args.skip_speed_negative,
        )
        if reasons:
            case["status"] = "negative"
            case["negative_reasons"] = reasons
        quality_metrics = case.get("quality_metrics", {})
        row = {
            "case_id": case["id"],
            "case_key_sha256": case["case_key_sha256"],
            "commit": case["commit"],
            "execution_commit": case.get("execution_commit", case["commit"]),
            "execution_change_scope": case.get(
                "execution_change_scope", "same_checkout"
            ),
            "method": case["method"],
            "baseline_method": baseline_name,
            "routing_stage": case.get("routing_stage"),
            "prompt_id": case.get("prompt_id"),
            "seed": case.get("seed"),
            "latent_frames": case.get("latent_frames"),
            "pixel_frames": case.get("pixel_frames"),
            "decoded_frames": case.get("decoded_frames"),
            "status": case.get("status"),
            "history_density": case.get("history_density"),
            "history_pair_density": case.get("history_pair_density"),
            "history_transfer_density": case.get("history_transfer_density"),
            "global_executed_density": case.get("global_executed_density"),
            "rope_policy": case.get("rope_policy"),
            "refresh_policy": case.get("refresh_policy"),
            "expansion_axes": json.dumps(
                case.get("expansion_axes", []), sort_keys=True
            ),
            "negative_reasons": json.dumps(reasons, sort_keys=True),
            "psnr": quality_metrics.get("psnr_mean"),
            "ssim": quality_metrics.get("ssim_mean"),
            "lpips": quality_metrics.get("lpips_mean"),
            "late_ssim": quality_metrics.get("late_quarter_ssim_mean"),
            "temporal_delta_l1": quality_metrics.get("temporal_delta_l1_mean"),
            "flow_epe": quality_metrics.get("flow_epe_mean"),
            "end_to_end_s": case.get("end_to_end_s", case.get("elapsed_s")),
            "attention_s": case.get("attention_s"),
            "routing_s": case.get("routing_s"),
            "q_summary_s": case.get("q_summary_s"),
            "d2h_s": case.get("d2h_s"),
            "cpu_gather_s": case.get("cpu_gather_s"),
            "h2d_s": case.get("h2d_s"),
            "h2d_bytes": case.get("transferred_bytes"),
            "peak_allocated_gb": case.get("peak_allocated_gb"),
            "model_load_s_total": case.get("model_load_s_total"),
            "model_load_s_amortized": case.get("model_load_s_amortized"),
            "end_to_end_with_amortized_load_s": case.get(
                "end_to_end_with_amortized_load_s"
            ),
            "archive_bytes": case.get("archive_bytes"),
            "index_bytes": case.get("index_bytes"),
            "index_transfer_bytes": case.get("index_transfer_bytes"),
            "query_summary_bytes": case.get("query_summary_bytes"),
            "candidate_transfer_bytes": case.get("candidate_transfer_bytes"),
            "staging_padding_tokens": case.get("staging_padding_tokens"),
            "candidate_history_tokens": case.get("candidate_history_tokens"),
            "selected_history_tokens": case.get("selected_history_tokens"),
            "dense_qk_pairs": case.get("dense_qk_pairs"),
            "executed_qk_pairs": case.get("executed_qk_pairs"),
        }
        manual_review = case.get("manual_review", {})
        row.update(
            {
                field: manual_review.get(field)
                for field in COUNT_FIELDS
                + SCORE_FIELDS
                + ["late_quarter_degradation_0to2"]
            }
        )
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.state_output:
        Path(args.state_output).write_text(
            json.dumps({**payload, "cases": cases}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"cases": len(cases), "negative": sum(case.get("status") == "negative" for case in cases)}, indent=2))


if __name__ == "__main__":
    main()
