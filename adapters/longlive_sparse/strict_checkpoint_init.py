"""Experimental skip of Parameter initializers before complete strict loading.

Only direct nn.Parameter initializers are skipped. Ordinary tensors/views and
derived buffers retain their original initialization. RNG consumption changes;
callers must strictly load every model and independently seed inference noise.
This context does not itself prove a model is fully loaded or output-equivalent.
"""
from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch

import torch


class StrictCheckpointParameterInit:
    FUNCTIONS=('normal_','uniform_','kaiming_uniform_','kaiming_normal_','xavier_uniform_',
               'xavier_normal_','trunc_normal_','zeros_','ones_','constant_')

    def __init__(self,*,enabled=False):
        self.enabled=enabled;self.rows=[];self.stack=None

    def _wrap(self,name,function):
        @wraps(function)
        def call(tensor,*args,**kwargs):
            if isinstance(tensor,torch.nn.Parameter):
                self.rows.append(dict(function=name,device=str(tensor.device),dtype=str(tensor.dtype),
                    elements=tensor.numel(),logical_payload_bytes=tensor.numel()*tensor.element_size()))
                return tensor
            return function(tensor,*args,**kwargs)
        return call

    def __enter__(self):
        if self.stack is not None:raise RuntimeError('initialization guard is single-use')
        self.stack=ExitStack()
        try:
            if self.enabled:
                for name in self.FUNCTIONS:
                    self.stack.enter_context(patch.object(torch.nn.init,name,self._wrap(name,getattr(torch.nn.init,name))))
        except BaseException:self.stack.close();raise
        return self

    def __exit__(self,*exc):return self.stack.__exit__(*exc)

    def record(self):
        return dict(enabled=self.enabled,skipped_parameter_initializer_calls=len(self.rows),
            skipped_logical_payload_bytes=sum(r['logical_payload_bytes'] for r in self.rows),
            details=self.rows,tensor_and_view_initializers_preserved=True,
            RNG_consumption_changed=self.enabled,complete_strict_checkpoint_loading_required=True,
            derived_buffer_or_full_output_equivalence_not_proven_by_this_record=True)
