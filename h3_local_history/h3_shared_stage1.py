"""Load Agent 5's exact rules and route them through the inference FSDP root."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

CONTRACT = 'h3_per_block_text_clean_noisy_fixed_video_KV_v1'
NATIVE = 'vdn-minimax-h3/diffusers/src/diffusers/models/transformers/transformer_minimax_h3.py'


def shared_modules(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent/'shared_stage1'
    manifest = json.loads((root/'SHARED_STAGE1_MANIFEST.json').read_text())
    if manifest['contract'] != CONTRACT:
        raise ValueError('Unknown shared stage1 contract')
    for name, digest in manifest['hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Shared training/inference source changed: '+name)
    sys.path.insert(0, str(root))
    modules = {}
    for name in ['h3_fixed_kv_stage1', 'h3_fixed_kv_runtime', 'h3_stage1_abc_data', 'h3_stable_modulation', 'h3_attention_backend']:
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != manifest['hashes'][name+'.py']:
            raise ValueError('Another shared implementation is already imported: '+name)
        modules[name] = module
    from diffusers import MiniMaxH3Transformer3DModel
    native = Path(inspect.getfile(MiniMaxH3Transformer3DModel))
    if hashlib.sha256(native.read_bytes()).hexdigest() != manifest['hashes'][NATIVE]:
        raise ValueError('Native H3 differs from the version declared by Agent5')
    return modules, manifest


class TransformerCallProxy:
    """Expose native attributes, but execute through all FSDP root hooks."""
    def __init__(self, model, observer=None):
        self.model = model
        self.observer = observer

    def __getattr__(self, name):
        return getattr(self.model.transformer, name)

    def __call__(self, *, attention_mask, **inputs):
        if self.observer is not None:
            self.observer.before(inputs, attention_mask)
        result = self.model(dict(inputs=inputs, context=attention_mask, raw=True))
        if self.observer is not None:
            self.observer.after(result)
        return result
