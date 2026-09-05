#!/usr/bin/env python3
"""Fail closed unless the four online formal configurations are frozen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.system_formal import validate_system_method_freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="configs/formal/system_method_freeze.json"
    )
    args = parser.parse_args()
    path = Path(args.manifest)
    if not path.is_file():
        raise FileNotFoundError("formal video is blocked until system method freeze exists")
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_system_method_freeze(payload)
    if errors:
        raise ValueError("invalid system method freeze: " + "; ".join(errors))
    print(
        json.dumps(
            {
                "status": "pass",
                "artifact_id": payload.get("artifact_id"),
                "config_ids": [item["config_id"] for item in payload["configs"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
