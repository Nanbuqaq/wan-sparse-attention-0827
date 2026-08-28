#!/usr/bin/env python3
"""Build commit-aware expected states for the 32-case Dense prompt screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.case_identity import build_case_identity


SPECS = {
    "native_dense": {
        "backend": "packed_fa2",
        "history_density": 1.0,
        "refresh_policy": "not_applicable",
        "rope_policy": "not_applicable",
        "routing_stage": "N/A",
    },
    "rag_dense": {
        "backend": "packed_fa2",
        "history_density": 1.0,
        "refresh_policy": "per_chunk",
        "rope_policy": "upstream_zero",
        "routing_stage": "post-transfer",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="configs/prompts/dense_candidates.json")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--runtime", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--latent-frames", type=int, default=120)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / args.candidates).read_text(encoding="utf-8"))
    runtimes = list(SPECS) if args.runtime == "all" else [args.runtime]
    cases = []
    for runtime in runtimes:
        spec = SPECS[runtime]
        for candidate in manifest["candidates"]:
            for seed in manifest["seeds"]:
                identity = build_case_identity(
                    commit=args.commit,
                    method=runtime,
                    prompt_id=candidate["prompt_id"],
                    prompt=candidate["prompt"],
                    seed=int(seed),
                    latent_frames=args.latent_frames,
                    history_density=spec["history_density"],
                    rope_policy=spec["rope_policy"],
                    refresh_policy=spec["refresh_policy"],
                    backend=spec["backend"],
                )
                cases.append(
                    {
                        **identity,
                        "method": runtime,
                        "runtime": runtime,
                        "routing_stage": spec["routing_stage"],
                        "backend": spec["backend"],
                        "prompt_id": candidate["prompt_id"],
                        "seed": int(seed),
                        "latent_frames": args.latent_frames,
                        "history_density": spec["history_density"],
                        "rope_policy": spec["rope_policy"],
                        "refresh_policy": spec["refresh_policy"],
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"commit": args.commit, "cases": cases}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"expected_cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
