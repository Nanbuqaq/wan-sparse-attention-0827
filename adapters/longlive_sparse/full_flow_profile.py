"""Opt-in, non-mutating full-flow attribution for diagnostic runs.

CPU ranges are nested wall times. CUDA event spans include host launch gaps;
they are NOT GPU service time and must not be summed into a critical path.
Actual activity/overlap comes from Nsight, never from these event spans.
"""
from __future__ import annotations

import contextlib
import functools
import ast
import inspect
import sys
import textwrap
import time
from collections import defaultdict

import torch


class FullFlowTrace:
    def __init__(self, *, cuda=False, nvtx=False):
        self.cuda = bool(cuda)
        self.nvtx = bool(nvtx)
        self.origin = time.perf_counter()
        self.records = []
        self.stack = []
        self.pending = []
        self.restorations = []

    @contextlib.contextmanager
    def span(self, name, *, device="mixed", metadata=None, cuda=None):
        use_cuda = self.cuda if cuda is None else self.cuda and cuda
        row = {"id": len(self.records), "parent": self.stack[-1] if self.stack else None,
               "name": name, "device": device, "metadata": metadata or {},
               "start_s": time.perf_counter() - self.origin, "status": "running"}
        self.records.append(row)
        self.stack.append(row["id"])
        begin = end = None
        if use_cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        if self.nvtx:
            torch.cuda.nvtx.range_push("fullflow/" + name)
        try:
            yield row
            row["status"] = "pass"
        except BaseException as error:
            row["status"] = "fail"
            row["error_type"] = type(error).__name__
            raise
        finally:
            if end is not None:
                end.record()
                self.pending.append((row, begin, end))
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            row["end_s"] = time.perf_counter() - self.origin
            row["host_wall_s"] = row["end_s"] - row["start_s"]
            assert self.stack.pop() == row["id"]

    def wrap(self, owner, attribute, name, *, device="mixed", metadata=None, cuda=None):
        """Patch only this process; retain descriptor ownership when restoring."""
        original = getattr(owner, attribute)
        owned = attribute in vars(owner)
        stored = vars(owner).get(attribute)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            info = metadata(args, kwargs) if callable(metadata) else metadata
            with self.span(name, device=device, metadata=info, cuda=cuda):
                return original(*args, **kwargs)

        setattr(owner, attribute, wrapped)
        self.restorations.append((owner, attribute, owned, stored))

    def restore(self):
        for owner, attribute, owned, stored in reversed(self.restorations):
            if owned:
                setattr(owner, attribute, stored)
            else:
                delattr(owner, attribute)
        self.restorations.clear()

    def result(self):
        if self.stack:
            raise RuntimeError("close every profiling scope before collecting results")
        if self.pending:
            torch.cuda.synchronize()
            for row, begin, end in self.pending:
                row["cuda_stream_span_s"] = begin.elapsed_time(end) / 1000
            self.pending.clear()
        children = defaultdict(float)
        for row in self.records:
            if row["parent"] is not None:
                children[row["parent"]] += row["host_wall_s"]
        aggregates = {}
        for row in self.records:
            row["host_self_s"] = max(0., row["host_wall_s"] - children[row["id"]])
            item = aggregates.setdefault(row["name"], {"calls": 0, "host_wall_s": 0., "host_self_s": 0.})
            item["calls"] += 1
            item["host_wall_s"] += row["host_wall_s"]
            item["host_self_s"] += row["host_self_s"]
        return {"records": self.records, "aggregates": aggregates,
                "host_self_is_unattributed_python_and_dispatch_not_CPU_compute": True,
                "cuda_stream_spans_are_not_service_or_exposed_wait": True,
                "nested_host_ranges_not_additive": True,
                "actual_GPU_activity_requires_Nsight": True}


def instrument_pipeline(trace, pipeline):
    """Cover transformer submodules, cache init, scheduler and archive helpers."""
    counts = defaultdict(int)

    def generator_metadata(args, kwargs):
        start = int(kwargs.get("current_start", -1))
        phase = counts[start]
        counts[start] += 1
        return {"current_start_token": start, "call_in_chunk": phase,
                "phase": "clean_commit" if phase == 4 else "denoising"}

    trace.wrap(pipeline.text_encoder, "forward", "text.encode_with_dynamic_swap", device="CPU+GPU")
    trace.wrap(pipeline.generator, "forward", "generator.forward", metadata=generator_metadata)
    for name in ("_initialize_kv_cache", "_initialize_crossattn_cache"):
        trace.wrap(pipeline, name, "cache." + name.removeprefix("_"))
    trace.wrap(pipeline.scheduler, "add_noise", "scheduler.add_noise", device="GPU")
    model = pipeline.generator.model
    for layer, block in enumerate(model.blocks):
        trace.wrap(block, "forward", "transformer.block", metadata={"layer": layer})
        for attribute in ("self_attn", "cross_attn", "ffn", "norm1", "norm2", "norm3"):
            module = getattr(block, attribute, None)
            if isinstance(module, torch.nn.Module):
                trace.wrap(module, "forward", "transformer." + attribute, metadata={"layer": layer})
        for attribute in ("q", "k", "v", "o", "norm_q", "norm_k"):
            module = getattr(block.self_attn, attribute, None)
            if isinstance(module, torch.nn.Module):
                trace.wrap(module, "forward", "self_attention." + attribute, device="GPU", metadata={"layer": layer})
    for attribute in ("patch_embedding", "text_embedding", "time_embedding", "time_projection", "head"):
        module = getattr(model, attribute, None)
        if isinstance(module, torch.nn.Module):
            trace.wrap(module, "forward", "generator." + attribute, device="GPU")
    archive = getattr(pipeline, "sparse_history_archive", None)
    if archive is not None:
        for attribute in ("materialize", "materialize_transfer_plan", "dense_history_tensors", "add_frame"):
            if hasattr(archive, attribute):
                trace.wrap(archive, attribute, "archive." + attribute)


@contextlib.contextmanager
def pipeline_regions(trace, function):
    """Trace just the pinned pipeline's inline retrieval/commit regions.

No source rewrite, tensor replacement, or synchronization is performed. The
diagnostic overhead is explicit and needs the uninstrumented latent control.
"""
    if sys.gettrace() is not None:
        raise RuntimeError("do not replace an existing debugger/trace hook")
    source, first_line = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(source)))
    markers = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "memory_indices" in names and isinstance(node.value, ast.Constant) and node.value.value is None:
                markers[first_line + node.lineno - 1] = "history.coarse_frame_retrieval"
            elif "context_timestep" in names:
                markers[first_line + node.lineno - 1] = None
            elif any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "output"
                     for t in node.targets):
                markers[first_line + node.lineno - 1] = "latent.commit_to_CPU_output"
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Tuple) and any(isinstance(t, ast.Name) and t.id == "current_timestep" for t in node.target.elts):
                markers[first_line + node.lineno - 1] = None
            elif isinstance(node.target, ast.Name) and node.target.id == "f_idx":
                markers[first_line + node.lineno - 1] = "history.descriptor_update"
    required = {"history.coarse_frame_retrieval", "latent.commit_to_CPU_output", "history.descriptor_update"}
    if not required.issubset(set(markers.values())):
        raise RuntimeError("pinned pipeline region structure changed; audit instrumentation before running")
    active = []

    def close():
        if active:
            active.pop().__exit__(None, None, None)

    def local(frame, event, arg):
        if event == "line" and frame.f_lineno in markers:
            name = markers[frame.f_lineno]
            if active and getattr(local, "name", None) == name:
                return local
            close()
            local.name = name
            if name is not None:
                context = trace.span(name, metadata={"current_frame": int(frame.f_locals.get("current_start_frame", -1))})
                context.__enter__()
                active.append(context)
        elif event == "return":
            close()
            local.name = None
        return local

    def dispatch(frame, event, arg):
        return local if frame.f_code is function.__code__ else None

    sys.settrace(dispatch)
    try:
        yield
    finally:
        sys.settrace(None)
        close()
