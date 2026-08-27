#!/usr/bin/env python3
"""Merge sharded case-state files without dropping failures or negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = {}
    sources = {}
    for value in args.input:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            case_id = case.get("id", case.get("case_id"))
            if not case_id:
                raise ValueError(f"case without id in {path}")
            if case_id in merged and merged[case_id] != case:
                raise RuntimeError(
                    f"conflicting duplicate case {case_id}: {sources[case_id]} and {path}"
                )
            merged[case_id] = case
            sources[case_id] = str(path.resolve())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sources": sorted(set(sources.values())),
        "cases": [merged[key] for key in sorted(merged)],
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(merged), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
