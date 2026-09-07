#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
metadata_length=${SYSTEM_METADATA_LATENT_FRAMES:-120}
metadata_batch_root=$INFER_OUTPUT_DIR
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  metadata_libraries=$VIRTUAL_ENV/lib
  for metadata_library in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
    [[ ! -d "$metadata_library" ]] || metadata_libraries="$metadata_libraries:$metadata_library"
  done
  export LD_LIBRARY_PATH="$metadata_libraries:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
  python -c 'import torch,av,flash_attn; print("CPU_PREP_IMPORT_PASS",torch.__version__)'
  python scripts/build_metadata_video_suite.py --latent-frames "$metadata_length" --output-dir "$metadata_batch_root/control" --validate-only
  touch "$metadata_batch_root/prepared.ok"
  exit 0
fi
source "$INFER_CODE_DIR/scripts/inferhub_runtime_env.sh"
IFS=',' read -r -a metadata_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#metadata_gpus[@]} == 2 ]] || { echo 'requires exactly two assigned GPUs' >&2; exit 2; }
export LONGLIVE_NVTX=0 LONGLIVE_CAPTURE_QKV=0 LONGLIVE_CAPTURE_COMPLETE_ATTENTION=0
python scripts/build_metadata_video_suite.py --latent-frames "$metadata_length" --output-dir "$metadata_batch_root/control"
metadata_pids=()
for metadata_lane in "${!metadata_gpus[@]}"; do
  mkdir -p "$metadata_batch_root/lane$metadata_lane"
  CUDA_VISIBLE_DEVICES=${metadata_gpus[$metadata_lane]} INFER_OUTPUT_DIR=$metadata_batch_root/lane$metadata_lane \
    python scripts/run_loaded_method_suite.py --suite "$metadata_batch_root/control/lane$metadata_lane.json" \
    --shard-index 0 --shard-count 1 >"$metadata_batch_root/lane$metadata_lane/runner.log" 2>&1 &
  metadata_pids+=("$!")
done
metadata_status=0
for metadata_pid in "${metadata_pids[@]}"; do
  wait "$metadata_pid" || metadata_status=1
done
python scripts/audit_metadata_video_suite.py --root "$metadata_batch_root" --expected "$metadata_batch_root/control/expected.json" \
  --output "$metadata_batch_root/metadata_video_audit.json" || metadata_status=1
exit "$metadata_status"
