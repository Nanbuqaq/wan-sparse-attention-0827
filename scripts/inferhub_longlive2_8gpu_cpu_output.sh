#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?missing INFER_CODE_DIR}"
: "${INFER_WEIGHTS_DIR:?missing INFER_WEIGHTS_DIR}"
: "${INFER_OUTPUT_DIR:?missing INFER_OUTPUT_DIR}"
: "${CUDA_VISIBLE_DEVICES:?missing CUDA_VISIBLE_DEVICES}"
: "${VIRTUAL_ENV:?missing VIRTUAL_ENV}"
IFS=',' read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#gpus[@]} == 8 ]] || { echo "requires exactly 8 assigned GPUs, got ${#gpus[@]}" >&2; exit 2; }
mode=${LONGLIVE_8GPU_MODE:-sync}
case "$mode" in
  sync) pipeline_mode=serial; encode_mode=inline ;;
  async) pipeline_mode=overlap; encode_mode=thread ;;
  smoke-sync) pipeline_mode=serial; encode_mode=inline ;;
  smoke-async) pipeline_mode=overlap; encode_mode=thread ;;
  *) echo "unknown LONGLIVE_8GPU_MODE=$mode" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
libs=("${VIRTUAL_ENV}/lib")
for p in "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/*/lib "${VIRTUAL_ENV}"/lib/python*/site-packages/torch/lib "${VIRTUAL_ENV}"/lib/python*/site-packages/triton/backends/nvidia/lib; do [[ -d "$p" ]] && libs+=("$p"); done
joined=$(IFS=:; echo "${libs[*]}")
export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${INFER_WEIGHTS_DIR}/python-overlay:${INFER_CODE_DIR}:${INFER_CODE_DIR}/third_party/LongLive2${PYTHONPATH:+:${PYTHONPATH}}"
cd "$INFER_CODE_DIR"
mkdir -p "$INFER_OUTPUT_DIR"
{
  date -u +%FT%TZ
  echo "mode=$mode"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi -L
  nvidia-smi --query-gpu=index,uuid,name,memory.total,pci.bus_id --format=csv
  nvidia-smi topo -m
  python - <<'PY'
import json, os, platform, torch
rows=[]
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    rows.append(dict(logical=i,name=p.name,total_memory=p.total_memory,capability=[p.major,p.minor]))
print(json.dumps(dict(host=platform.node(),visible=os.environ['CUDA_VISIBLE_DEVICES'],devices=rows),indent=2))
PY
} > "$INFER_OUTPUT_DIR/hardware.log" 2>&1
python - "$INFER_OUTPUT_DIR/hardware.json" <<'PY'
import json, os
from pathlib import Path
ids=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
lanes=[dict(lane=i,physical_gpus=ids[2*i:2*i+2],dit_gpu=ids[2*i],vae_gpu=ids[2*i+1]) for i in range(4)]
Path(__import__('sys').argv[1]).write_text(json.dumps(dict(status='hardware_recorded',visible_devices=ids,lanes=lanes,mode=os.environ['LONGLIVE_8GPU_MODE']),indent=2)+'\n')
PY
seeds=(20260909 20260910 20260909 20260910)
pids=()
for lane in 0 1 2 3; do
  pair="${gpus[$((2*lane))]},${gpus[$((2*lane+1))]}"
  extra=()
  [[ "$lane" == 2 ]] && extra+=(--control duck)
  [[ "$lane" == 3 ]] && extra+=(--control empty)
  args=(--assets "$INFER_WEIGHTS_DIR" --output "$INFER_OUTPUT_DIR/lane$lane" --seed "${seeds[$lane]}" --pipeline-mode "$pipeline_mode" --pipeline-encode-mode "$encode_mode" --pipeline-slots 2 --pipeline-pixel-slots 2 --pipeline-pinned-mib 128 --pipeline-profile)
  [[ "$mode" == smoke-* ]] && args+=(--gate)
  env CUDA_VISIBLE_DEVICES="$pair" python scripts/run_longlive2_native_reference.py "${args[@]}" "${extra[@]}" >"$INFER_OUTPUT_DIR/lane$lane.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
python scripts/collect_longlive2_8gpu_cpu_output.py --root "$INFER_OUTPUT_DIR" --mode "$mode" --exit-status "$status"
exit "$status"
