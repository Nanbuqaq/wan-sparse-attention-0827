#!/usr/bin/env python3
"""Replace matching case states while preserving canonical case identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--override", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_path = Path(args.base).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    by_id = {case["id"]: dict(case) for case in base["cases"]}
    if len(by_id) != len(base["cases"]):
        raise RuntimeError("duplicate base case identities")
    replacements = []
    sources = []
    for value in args.override:
        path = Path(value).resolve()
        sources.append({"path": str(path), "sha256": sha256(path)})
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            case_id = case["id"]
            previous = by_id.get(case_id)
            if previous is None:
                raise RuntimeError(f"override case is absent from base: {case_id}")
            if previous.get("case_key_sha256") != case.get("case_key_sha256"):
                raise RuntimeError(f"override identity mismatch: {case_id}")
            by_id[case_id] = dict(case)
            replacements.append(
                {
                    "case_id": case_id,
                    "case_key_sha256": case["case_key_sha256"],
                    "old_video": previous.get("video"),
                    "new_video": case.get("video"),
                    "old_end_to_end_s": previous.get("end_to_end_s"),
                    "new_end_to_end_s": case.get("end_to_end_s"),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "cases": [by_id[case["id"]] for case in base["cases"]],
        "replacement_provenance": {
            "base": {"path": str(base_path), "sha256": sha256(base_path)},
            "overrides": sources,
            "replacements": replacements,
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(result["cases"]), "replacements": len(replacements), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
