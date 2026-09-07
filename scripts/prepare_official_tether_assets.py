#!/usr/bin/env python3
"""Link existing public Wan/SAM2 assets and verify the original Tether weights.

No checkpoint is converted, replaced, or trained. Existing destinations must
already point to the exact intended input. New downloaded blobs are SHA locked.
"""
import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "longlive_rag/checkpoints/causal_forcing.pt": (5676282643, "cf75ee5cc6f4e2e336c59c973f5544655d8f0aa481761efe6de1b9cb2eb0cd9d"),
    "longlive_rag/checkpoints/ae_latent_mem.pt": (362866531, "977c14ff89a03969346918a59062d28b0ee72c789698e60433d9b6474625ab7e"),
    "sam2/sam2_hiera_large.pt": (897952466, "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"),
}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def link_exact(destination, source):
    source = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(f"refusing to replace existing model location: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--wan-dir", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--link-only", action="store_true")
    args = parser.parse_args()
    root = args.model_root.resolve()
    link_exact(root / "wan_models/Wan2.1-T2V-1.3B", args.wan_dir)
    link_exact(root / "sam2/sam2_hiera_large.pt", args.sam2_checkpoint)
    if args.link_only:
        print(json.dumps({"status": "linked_existing_Wan_SAM2", "weights_verified": False}))
        return
    records = []
    for relative, (size, expected) in EXPECTED.items():
        path = root / relative
        if path.stat().st_size != size:
            raise ValueError(f"wrong checkpoint size: {relative}")
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"wrong checkpoint SHA: {relative}")
        records.append({"path": relative, "bytes": size, "sha256": actual})
        print(json.dumps(records[-1]), flush=True)
    source = args.source.resolve()
    snapshot = json.loads((source / ".source_snapshot.json").read_text())
    source_files = ["configs/tethermem.yaml", "scripts/inference_tethermem.py", "scripts/run_tethermem_pipeline.py",
                    "tethermem/routing.py", "pipeline/causal_inference.py", "wan/modules/causal_model_latentmem.py"]
    manifest = {"status": "pass", "HF_revision": "aaafe325726e0204ada9d1f019356fde97e08c2e",
        "source": snapshot, "source_files": {name: sha(source / name) for name in source_files}, "weights": records,
        "wan_existing_directory": str(args.wan_dir.resolve()),
        "wan_reused_not_re_downloaded_or_newly_hash_verified_here": True,
        "canonical_generator": "causal_forcing", "retrieval": "ae", "lora": False}
    output = root / "official_assets_verified.json"
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise FileExistsError("existing verification differs; use a new asset root")
    else:
        with output.open("x") as handle:
            handle.write(json.dumps(manifest, indent=2) + "\n")
    print(str(output))


if __name__ == "__main__":
    main()
