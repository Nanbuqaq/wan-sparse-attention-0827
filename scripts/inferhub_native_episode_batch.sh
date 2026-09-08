#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
episode_root=$INFER_OUTPUT_DIR
episode_libs=$VIRTUAL_ENV/lib
for episode_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$episode_lib" ]] || episode_libs="$episode_libs:$episode_lib"
done
export LD_LIBRARY_PATH="$episode_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python -c 'from adapters.longlive_sparse.native_episode_memory import NativeEpisodeMemory; print("EPISODE_IMPORT_PASS")'
  touch "$episode_root/prepared.ok";exit 0
fi
IFS=',' read -r -a episode_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#episode_gpus[@]} == 4 ]] || { echo 'requires four assigned GPUs' >&2; exit 2; }
episode_scenarios=(generated_patchwork_toy_cut_revisit generated_patchwork_toy_cut_revisit generated_bead_state_cut_revisit generated_bead_state_cut_revisit)
episode_seeds=(20260913 20260914 20260913 20260914)
episode_pids=()
for episode_lane in "${!episode_gpus[@]}"; do
  (
    mkdir -p "$episode_root/lane$episode_lane"
    episode_order=(none raw_reveal raw_away log_reveal)
    if (( episode_lane % 2 )); then episode_order=(log_reveal raw_away raw_reveal none); fi
    for episode_mode in "${episode_order[@]}"; do
      episode_code=0
      CUDA_VISIBLE_DEVICES=${episode_gpus[$episode_lane]} python scripts/run_longlive2_native_reference.py \
        --assets "$INFER_WEIGHTS_DIR" --output "$episode_root/lane$episode_lane/$episode_mode" \
        --seed "${episode_seeds[$episode_lane]}" --cut-scenario "${episode_scenarios[$episode_lane]}" \
        --episode-memory-mode "$episode_mode" >"$episode_root/lane$episode_lane/$episode_mode.log" 2>&1 || episode_code=$?
      echo "episode_mode=$episode_mode exit=$episode_code"
    done
  ) >"$episode_root/lane$episode_lane.log" 2>&1 &
  episode_pids+=("$!")
done
for episode_pid in "${episode_pids[@]}"; do wait "$episode_pid" || true; done
python scripts/collect_native_episode_batch.py --root "$episode_root"
