#!/usr/bin/env python3
"""Observer-only breakdown of the native runner's constructor-to-checkpoint span.

Same released architecture and complete strict merged checkpoint. No generator
forward, training, video, alternative loading algorithm or source/weight edits.
Host durations include any waits; this is not a CPU FLOPs/HBM/GPU-service counter.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.startup_host_trace import StartupHostTrace


def tensor_metadata(tensor,*args,**kwargs):
    return dict(device=str(tensor.device),dtype=str(tensor.dtype),elements=tensor.numel(),
        tensor_payload_bytes=tensor.numel()*tensor.element_size(),not_measured_DRAM_bytes=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for key in ('source','assets','output'):setattr(args,key,getattr(args,key).resolve())
    if args.output.is_relative_to(args.assets):raise ValueError('assets are read-only')
    args.output.mkdir(parents=True,exist_ok=False);report=dict(status='running');trace=StartupHostTrace()
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('real locked GPU required for native T5 placement')
        source_sha=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
        assert source_sha=='6b36d20ec6f7958d29d11a704dfa64611a9f2572'
        manifest=args.assets/'assets_manifest.json';assert json.loads(manifest.read_text())['status']=='pass'
        sys.path.insert(0,str(args.source));os.chdir(args.assets)
        from omegaconf import OmegaConf
        from pipeline import CausalDiffusionInferencePipeline
        from utils.config import normalize_config
        from utils import wan_5b_wrapper as wrapper
        from utils.inference_utils import load_generator_checkpoint
        raw=OmegaConf.load(args.source/'configs/inference.yaml');del raw.adapter
        raw.checkpoints.lora_ckpt=None;raw.checkpoints.generator_ckpt=str(args.assets/'checkpoints/model_bf16.pt')
        raw.inference.streaming_vae=False;raw.inference.async_vae=False;raw.inference.vae_device=None
        raw.data.image_or_video_shape=[1,128,48,44,80]
        raw.model_kwargs.local_attn_size=32;raw.inference.local_attn_size=32
        config=normalize_config(raw);OmegaConf.save(raw,args.output/'config.yaml')
        def architecture(path,**kwargs):
            with trace.phase('generator.from_config'):
                cfg=json.loads((Path(path)/'config.json').read_text())
                return wrapper.CausalWanModel.from_config(cfg,**kwargs)
        def load_metadata(path,*args,**kwargs):
            try:
                p=Path(path);return dict(path=str(p),file_bytes=p.stat().st_size,OS_page_cache_state_unknown=True)
            except (TypeError,OSError):return dict(file_like_input=True)
        with ExitStack() as stack:
            for name in ('normal_','uniform_','kaiming_uniform_','kaiming_normal_','xavier_uniform_',
                         'xavier_normal_','trunc_normal_','zeros_','ones_','constant_'):
                function=getattr(torch.nn.init,name)
                stack.enter_context(patch.object(torch.nn.init,name,trace.wrap('nn.init.'+name,function,tensor_metadata)))
            stack.enter_context(patch.object(torch,'load',trace.wrap('torch.load',torch.load,load_metadata)))
            stack.enter_context(patch.object(torch.nn.Module,'cuda',trace.wrap('module.cuda_host_span',torch.nn.Module.cuda,
                lambda module,*args,**kwargs:dict(module_type=type(module).__name__,
                    parameter_payload_bytes=sum(p.numel()*p.element_size() for p in module.parameters()),
                    includes_allocation_context_and_copy_waits=True,not_DMA_service_time=True))))
            for cls in (wrapper.WanTextEncoder,wrapper.WanVAEWrapper):
                stack.enter_context(patch.object(cls,'__init__',trace.wrap(cls.__name__+'.constructor',cls.__init__)))
            stack.enter_context(patch.object(wrapper.CausalWanModel,'from_pretrained',side_effect=architecture))
            with trace.phase('native_load_span'):
                with trace.phase('pipeline.constructor'):
                    pipe=CausalDiffusionInferencePipeline(config,device=torch.device('cuda'))
                with trace.phase('native_T5.cast_BF16'):
                    pipe.text_encoder.to(dtype=torch.bfloat16)
                with trace.phase('generator.strict_checkpoint_load'):
                    loaded=load_generator_checkpoint(pipe.generator,str(args.assets/'checkpoints/model_bf16.pt'),strict=True)
                    assert not loaded.missing_keys and not loaded.unexpected_keys
                with trace.phase('final_CUDA_readiness'):
                    torch.cuda.synchronize()
        report.update(status='pass',scope='native_load_span_host_nested_observer_only',aggregate=trace.aggregate(),
            events=trace.events,source_sha=source_sha,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            runner_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),GPU=torch.cuda.get_device_name(),
            physical_GPU=os.environ.get('WAN_SPARSE_PHYSICAL_GPU'),torch=torch.__version__,CPU_threads=torch.get_num_threads(),
            strict_checkpoint_missing_keys=loaded.missing_keys,strict_checkpoint_unexpected_keys=loaded.unexpected_keys,
            generator_state_dict_entries=len(pipe.generator.state_dict()),no_training_or_generator_forward=True,
            no_weight_or_initialization_operations_skipped=True,import_and_asset_checks_outside_window=True,
            observer_overhead_not_calibrated=True,OS_page_cache_state_unknown=True,
            host_duration_is_not_CPU_compute_or_GPU_service=True,not_an_optimized_loader_or_speedup_test=True)
    except Exception:report.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'startup_profile.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output/'startup_host.perfetto.json').write_text(json.dumps(trace.chrome())+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2))


if __name__=='__main__':main()
