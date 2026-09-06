#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}" "${CUDA_VISIBLE_DEVICES:?}"
source "$INFER_CODE_DIR/scripts/inferhub_runtime_env.sh"
IFS=',' read -r -a assigned_gpus <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#assigned_gpus[@]} == 2 ]] || exit 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$INFER_CODE_DIR"
python scripts/record_dataflow_hardware.py --output "$INFER_OUTPUT_DIR/hardware.json"
counter_cache=$(mktemp -d /tmp/longlive-counter-triton.XXXXXX)
run_lane() {
  local lane=$1 backend code
  shift
  for backend in "$@"; do
    code=0
    if command -v ncu >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES=${assigned_gpus[$lane]} TRITON_CACHE_DIR=$counter_cache/lane$lane \
        ncu --target-processes all --profile-from-start off --section MemoryWorkloadAnalysis \
        --section LaunchStats --section Occupancy --section SpeedOfLight \
        --export "$INFER_OUTPUT_DIR/$backend" python scripts/profile_dataflow_counters.py \
        --backend "$backend" --output "$INFER_OUTPUT_DIR/$backend.replay.json" \
        >"$INFER_OUTPUT_DIR/$backend.log" 2>&1 || code=$?
    else
      code=127
    fi
    python - "$INFER_OUTPUT_DIR" "$backend" "$code" <<'PY'
import json,sys
from pathlib import Path
root,backend,code=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3])
log=root/f'{backend}.log'
denied=log.exists() and 'ERR_NVGPUCTRPERM' in log.read_text()
report=root/f'{backend}.ncu-rep'
passing=code==0 and not denied and report.is_file() and report.stat().st_size>0
(root/f'{backend}.status.json').write_text(json.dumps({'status':'pass' if passing else 'fail',
 'exit_code':code,'counter_permission_denied':denied,'backend':backend,
 'failure_scope':None if passing else 'counter_environment_not_algorithm_quality'},indent=2)+'\n')
PY
  done
}
run_lane 0 qout fa2 & first_pid=$!
run_lane 1 kvout & second_pid=$!
wait "$first_pid"
wait "$second_pid"
python - "$INFER_OUTPUT_DIR" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[json.loads((root/f'{name}.status.json').read_text()) for name in ('qout','kvout','fa2')]
passing=all(r['status']=='pass' for r in rows)
(root/'terminal.json').write_text(json.dumps({'status':'pass' if passing else 'fail',
 'cases':rows,'missing':0,'counter_values_require_postprocessing':True},indent=2)+'\n')
sys.exit(0 if passing else 1)
PY
