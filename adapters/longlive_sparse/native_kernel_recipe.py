"""Capture/restore native numerical launch recipes without teacher data."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch


def capture_native_recipe(frame_tokens,dim=3072):
    import triton
    from utils import adaln_triton
    rows=[]
    for key,cfg in adaln_triton._adaln_modulate_kernel.cache.items():
        if key[:2]!=(dim,frame_tokens):continue
        if any(not isinstance(v,(str,int,float,bool,type(None))) for v in key):raise ValueError('unsupported kernel signature')
        if cfg.pre_hook is not None:raise ValueError('nonserializable native kernel prehook')
        rows.append(dict(key=list(key),kwargs=cfg.kwargs,num_warps=cfg.num_warps,num_stages=cfg.num_stages,
            num_ctas=cfg.num_ctas,maxnreg=cfg.maxnreg))
    if not rows:raise RuntimeError('native adaLN recipe must already be observed')
    return dict(schema='native_numeric_recipe_v1',torch=str(torch.__version__),triton=str(triton.__version__),
        CUDA=torch.version.cuda,compute_capability=list(torch.cuda.get_device_capability()),
        kernel_source_sha256=hashlib.sha256(Path(adaln_triton.__file__).read_bytes()).hexdigest(),adaln=rows,
        grad_enabled=torch.is_grad_enabled(),inference_mode=torch.is_inference_mode_enabled())


def recipe_digest(recipe):
    return hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@contextmanager
def native_recipe_scope(recipe):
    if recipe is None:
        yield;return
    import triton
    from utils import adaln_triton
    if recipe['torch']!=torch.__version__ or recipe['triton']!=triton.__version__ or recipe['CUDA']!=torch.version.cuda:
        raise RuntimeError('numeric replay environment differs from recorded recipe')
    if recipe['compute_capability']!=list(torch.cuda.get_device_capability()):raise RuntimeError('unvalidated replay hardware')
    if recipe['kernel_source_sha256']!=hashlib.sha256(Path(adaln_triton.__file__).read_bytes()).hexdigest():
        raise RuntimeError('native adaLN source differs')
    if recipe['grad_enabled']!=torch.is_grad_enabled() or recipe['inference_mode']!=torch.is_inference_mode_enabled():
        raise RuntimeError('numeric replay mode differs')
    tuner=adaln_triton._adaln_modulate_kernel;saved=dict(tuner.cache)
    try:
        for row in recipe['adaln']:
            tuner.cache[tuple(row['key'])]=triton.Config(row['kwargs'],num_warps=row['num_warps'],
                num_stages=row['num_stages'],num_ctas=row['num_ctas'],maxnreg=row['maxnreg'])
        yield
    finally:
        tuner.cache.clear();tuner.cache.update(saved)


def fix_native_adaln_for_fresh_run(warps,stages):
    """Explicit calibration choice before any generation; not a runtime retune."""
    import triton
    from utils import adaln_triton
    if warps not in (4,8,16) or stages not in (1,2,3):raise ValueError('unsupported native recipe')
    tuner=adaln_triton._adaln_modulate_kernel
    if tuner.cache:raise RuntimeError('fix recipe only before first native generation')
    tuner.configs=[triton.Config({},num_warps=warps,num_stages=stages)]
