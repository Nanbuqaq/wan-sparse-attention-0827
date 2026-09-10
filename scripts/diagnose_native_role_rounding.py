#!/usr/bin/env python3
"""CPU-only precision audit of saved statistics; preserves original gate."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--payload', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with args.payload.open('rb') as h:
        digest = hashlib.file_digest(h, 'sha256').hexdigest()
    prior = json.loads(args.analysis.read_text())
    assert digest == prior['payload_sha256']
    payload = torch.load(args.payload, map_location='cpu', weights_only=True)
    assert payload['complete'] and payload['full_QKV_not_stored']
    rows = []
    for r in payload['records']:
        z, values, native = r['log_z'], r['role_outputs'], r['native_output']
        assert native.dtype == torch.bfloat16
        ref = (z.softmax(-1)[..., None] * values).sum(-2)
        ref64 = (z.double().softmax(-1)[..., None] * values.double()).sum(-2)
        rounded = ref.to(torch.bfloat16).float()
        error = (ref - native.float()).abs()
        index = torch.unravel_index(error.argmax(), error.shape)
        index = tuple(int(x) for x in index)
        rounding = (ref - rounded).abs()
        rows.append(dict(query_frame=r['query_frame'], phase=r['phase'], layer=r['layer'],
            original_max_abs=float(error.max()),
            unavoidable_BF16_rounding_max_abs=float(rounding.max()),
            rounded_reference_vs_native_max_abs=float((rounded-native.float()).abs().max()),
            BF16_elements_matching_rounded_reference=float((rounded==native.float()).float().mean()),
            FP64_merge_vs_FP32_merge_max_abs=float((ref64-ref.double()).abs().max()),
            worst_index=list(index), reference_at_worst=float(ref[index]),
            native_at_worst=float(native[index]), rounded_at_worst=float(rounded[index]),
            original_max_abs_failed=float(error.max()) > .02,
            rounding_alone_exceeds_original_threshold=float(rounding.max()) > .02))
    report = dict(status='diagnostic_complete_original_gate_unchanged', rows=rows,
                  payload_sha256=digest, records=len(rows),
                  original_failures=sum(r['original_max_abs_failed'] for r in rows),
                  rounding_floor_failures=sum(r['rounding_alone_exceeds_original_threshold'] for r in rows),
                  full_QKV_not_available=True, original_gate_status=prior['status'],
                  second_seed_authorized_by_this_audit=False,
                  note='FP64 merge audits stored FP32 group summaries only; does not recompute QK/PV.')
    (args.output/'rounding_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))
    print(json.dumps([r for r in rows if r['original_max_abs_failed']], indent=2))


if __name__ == '__main__':
    main()
