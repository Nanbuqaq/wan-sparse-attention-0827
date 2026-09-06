#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}"
: "${INFER_WEIGHTS_DIR:?}"
: "${INFER_OUTPUT_DIR:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
: "${VIRTUAL_ENV:?}"
export LONGLIVE_INPUT_BUNDLE_ROOT="$INFER_WEIGHTS_DIR/input_bundle"
source "$INFER_CODE_DIR/scripts/inferhub_runtime_env.sh"
IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#assigned_gpus[@]} == 3 ]] || { echo 'requires3 assigned GPUs' >&2; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LONGLIVE_CAPTURE_COMPLETE_ATTENTION=0 LONGLIVE_NVTX=0
cd "$INFER_CODE_DIR"
python scripts/build_aligned_seed_replication.py --output-dir "$INFER_OUTPUT_DIR/control"
batch_root=$INFER_OUTPUT_DIR
pids=()
for lane in 0 1 2; do
  mkdir -p "$batch_root/lane$lane"
  CUDA_VISIBLE_DEVICES=${assigned_gpus[$lane]} INFER_OUTPUT_DIR=$batch_root/lane$lane \
    python scripts/run_loaded_method_suite.py --suite "$batch_root/control/suite.json" \
    --shard-axis case --shard-index "$lane" --shard-count 3 >"$batch_root/lane$lane/runner.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || exit "$failed"
python scripts/merge_case_states.py --input "$batch_root/lane0/shard_0_states.json" \
 --input "$batch_root/lane1/shard_1_states.json" --input "$batch_root/lane2/shard_2_states.json" \
 --expected "$batch_root/control/expected.json" --output "$batch_root/states.json" >"$batch_root/merge.log"
python scripts/audit_case_states.py --expected "$batch_root/control/expected.json" \
 --states "$batch_root/states.json" --output "$batch_root/terminal_audit.json" >"$batch_root/terminal_audit.log"
