#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${CUDA_VISIBLE_DEVICES:?}" "${VIRTUAL_ENV:?}"
IFS=',' read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES}"; [[ ${#gpus[@]} == 8 ]] || exit 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
library_paths=("${VIRTUAL_ENV}/lib")
for p in "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/*/lib "${VIRTUAL_ENV}"/lib/python*/site-packages/torch/lib "${VIRTUAL_ENV}"/lib/python*/site-packages/triton/backends/nvidia/lib; do [[ -d "$p" ]] && library_paths+=("$p"); done
joined=$(IFS=:; echo "${library_paths[*]}"); export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR:$INFER_CODE_DIR/third_party/LongLive2"
cd "$INFER_CODE_DIR"; mkdir -p "$INFER_OUTPUT_DIR"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total --format=csv > "$INFER_OUTPUT_DIR/gpu_mapping.csv"
nvidia-smi -L > "$INFER_OUTPUT_DIR/hardware.log" 2>&1; nvidia-smi topo -m >> "$INFER_OUTPUT_DIR/hardware.log" 2>&1
pids=()
for lane in 0 1 2 3; do
 pair="${gpus[$((2*lane))]},${gpus[$((2*lane+1))]}"
 env CUDA_VISIBLE_DEVICES="$pair" WAN_SPARSE_PHYSICAL_GPUS="$pair" \
 python scripts/run_longlive2_native_reference.py --assets "$INFER_WEIGHTS_DIR" --output "$INFER_OUTPUT_DIR/lane$lane" \
  --seed "$((20260909 + lane % 2))" --pipeline-mode overlap --resident-output-pairs 4 \
  --resident-barrier-root "$INFER_OUTPUT_DIR/barriers" --resident-lane "$lane" > "$INFER_OUTPUT_DIR/lane$lane.log" 2>&1 &
 pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
python - "$INFER_OUTPUT_DIR/summary_8gpu.json" "$status" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).parent;rows=[]
for i in range(4):
 p=root/f'lane{i}/summary.json'; rows.append(json.loads(p.read_text()) if p.exists() else {'status':'missing','lane':i})
out={'status':'pass' if int(sys.argv[2])==0 and all(r.get('status')=='pass' for r in rows) else 'fail','lanes':rows,'exit_status':int(sys.argv[2]),'resident_pairs':4}
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out)); raise SystemExit(0 if out['status']=='pass' else 1)
PY
