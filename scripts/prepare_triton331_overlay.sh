#!/usr/bin/env bash
# CPU-only InferHub preparation. Never install into the declared shared env.
set -Eeuo pipefail
: "${INFER_OUTPUT_DIR:?}"
overlay=$INFER_OUTPUT_DIR/private-triton331
wheel_dir=$INFER_OUTPUT_DIR/private-wheels
[[ ! -e $overlay/ready.json ]] || exit 0
mkdir -p "$wheel_dir" "$overlay"
python -m pip download --no-deps --only-binary=:all: --dest "$wheel_dir" triton==3.3.1
python - "$wheel_dir" <<'PY'
import hashlib,sys
from pathlib import Path
files=list(Path(sys.argv[1]).glob('triton-3.3.1-cp312-cp312-*.whl'))
assert len(files)==1, 'exact CPython3.12 wheel required'
assert hashlib.sha256(files[0].read_bytes()).hexdigest()=='9999e83aba21e1a78c1f36f21bce621b77bcaa530277a50484a7cb4a822f6e43'
PY
python -m pip install --no-deps --no-index --target "$overlay" "$wheel_dir"/triton-3.3.1-cp312-cp312-*.whl
PYTHONPATH="$overlay" python - "$overlay" <<'PY'
import json,sys,triton
from pathlib import Path
root=Path(sys.argv[1]).resolve()
assert root in Path(triton.__file__).resolve().parents and triton.__version__=='3.3.1'
(root/'ready.json').write_text(json.dumps({'status':'pass','triton':'3.3.1',
 'wheel_sha256':'9999e83aba21e1a78c1f36f21bce621b77bcaa530277a50484a7cb4a822f6e43',
 'shared_environment_modified':False},indent=2)+'\n')
PY
