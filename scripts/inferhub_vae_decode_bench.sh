#!/usr/bin/env bash
# Single-process VAE decode benchmark on the worker GPU: eager vs compiled
# decoder variants on fixed real latents. Also serves as the single-flight
# Inductor cache warmer (avoids concurrent autotune writes across lanes).
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
w2_libs=$VIRTUAL_ENV/lib
for w2_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$w2_lib" ]] || w2_libs="$w2_libs:$w2_lib"
done
export LD_LIBRARY_PATH="$w2_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/kaimm-distill/zhouhe08/.triton-cache/longlive2}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/kaimm-distill/zhouhe08/.inductor-cache/longlive2}"
cd "$INFER_CODE_DIR"
python scripts/bench_vae_decode_grouping.py \
  --vae "$INFER_WEIGHTS_DIR/wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
  --latents "${BENCH_LATENTS:-/kaimm-distill/zhouhe08/longlive-system/outputs/shared125_reuse_pipeline4_eb17c3e_s20261010_retry1/screen/w2_rotating_wooden_bird__s20261010__shared125_reuse_r1/latents.pt}" \
  --latent-count "${BENCH_LATENT_COUNT:-17}" \
  --group-sizes ${BENCH_GROUP_SIZES:-2} \
  --compile-modes ${BENCH_COMPILE_MODES:-default max-autotune-no-cudagraphs} \
  --output "$INFER_OUTPUT_DIR/vae_decode_bench.json"
