#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "${repo_root}/.." && pwd)
control_root=${workspace_root}/results/manifests/formal_basic_residual_694da9e_local2/control
method_repo=${workspace_root}/.runtime/publish_repo_694da9e
output_root=${workspace_root}/results/videos/local_formal_basic_residual_694da9e_local2
log_root=${workspace_root}/results/logs/local_formal_basic_residual_694da9e_local2
final_root=${workspace_root}/results/manifests/formal_basic_residual_694da9e_local2/final
experiment_commit=694da9e70e8af4202d75734667af847a0ceaf286

(
  cd "${control_root}"
  sha256sum -c CONTROL_SHA256SUMS.txt
)
[[ $(git -C "${method_repo}" rev-parse HEAD) == "${experiment_commit}" ]] || exit 3
mkdir -p "${output_root}" "${log_root}" "${final_root}"

pids=()
for lane in 0 1; do
  /usr/bin/python3 "${repo_root}/scripts/run_on_free_gpu.py" \
    --physical-gpu "${lane}" \
    --max-memory-mib 1024 --max-utilization 20 -- \
    bash "${repo_root}/scripts/local_formal_basic_residual_lane.sh" \
      "${lane}" "${control_root}" "${method_repo}" "${output_root}" \
      "${log_root}" "${experiment_commit}" \
      >"${log_root}/lane${lane}_launcher.log" 2>&1 &
  pids+=("$!")
done
statuses=()
set +e
for pid in "${pids[@]}"; do
  wait "${pid}"
  statuses+=("$?")
done
set -e

state_inputs=(--input "${control_root}/checkpoint_states.json")
for lane in 0 1; do
  suite_count=$(python3 - "${control_root}/lane${lane}_plan.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["suites"]))
PY
  )
  for ((suite_index=0; suite_index<suite_count; suite_index++)); do
    state=${output_root}/lane${lane}/suite${suite_index}/shard_0_states.json
    if [[ ! -f "${state}" ]]; then
      mkdir -p "$(dirname "${state}")"
      python3 - "${state}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"cases": []}) + "\n", encoding="utf-8")
PY
    fi
    state_inputs+=(--input "${state}")
  done
done

python3 "${repo_root}/scripts/merge_case_states.py" \
  "${state_inputs[@]}" \
  --expected "${control_root}/expected_basic_477.json" \
  --fill-missing-reason "local dual-GPU residual runner did not emit a terminal state" \
  --output "${final_root}/merged_case_states.json"
python3 "${repo_root}/scripts/audit_case_states.py" \
  --expected "${control_root}/expected_basic_477.json" \
  --states "${final_root}/merged_case_states.json" \
  --output "${final_root}/terminal_state_audit.json"
python3 - "${final_root}/batch_status.json" "${statuses[@]}" <<'PY'
import json, sys
from pathlib import Path
payload = {
    "lane_statuses": [int(value) for value in sys.argv[2:]],
    "terminal_audit_completed": True,
    "local_dual_gpu": True,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
python3 - "${final_root}/merged_case_states.json" "${final_root}/SHA256SUMS.txt" <<'PY'
import hashlib, json, sys
from pathlib import Path
states = json.load(open(sys.argv[1], encoding="utf-8"))["cases"]
paths = set()
for case in states:
    if case.get("status") not in {"pass", "negative"}:
        continue
    video = Path(case["video"])
    latent = Path(case.get("latent", video.parent / "latents.pt"))
    paths.update((video, latent))
lines = []
for path in sorted(paths, key=str):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    lines.append(f"{digest.hexdigest()}  {path}")
Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
