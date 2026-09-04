#!/usr/bin/env python3
"""Fail closed unless the system holdout manifest is frozen and auditable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.formal_gate import load_frozen_system_holdouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="configs/formal/system_holdout_prompts.json"
    )
    args = parser.parse_args()
    payload = load_frozen_system_holdouts(args.manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "artifact_id": payload["artifact_id"],
                "prompt_ids": [item["prompt_id"] for item in payload["prompts"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
