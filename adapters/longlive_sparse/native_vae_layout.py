"""Experimental decoder memory-format context; values and upstream stay unchanged.

Changing a layout may change cuDNN/reduction numerical execution. This helper
does not assert pixel equivalence or speedup; both require separate real gates.
"""
import torch


class NativeVAEMemoryFormat:
    MODES=('baseline','weights_only','entry_layout','all_conv_inputs')

    def __init__(self,model,mode):
        if mode not in self.MODES:raise ValueError('unknown native VAE layout mode')
        self.model,self.mode=model,mode;self.saved=[];self.handles=[];self.entered=False
        self.input_calls=0;self.input_reformats=0;self.input_reformat_payload_bytes=0

    @staticmethod
    def layout(module):
        return torch.channels_last_3d if isinstance(module,torch.nn.Conv3d) else torch.channels_last

    def _hook(self,module,args,kwargs):
        self.input_calls+=1;ndim=5 if isinstance(module,torch.nn.Conv3d) else 4
        def convert(x):
            if torch.is_tensor(x) and x.ndim==ndim and not x.is_contiguous(memory_format=self.layout(module)):
                self.input_reformats+=1;self.input_reformat_payload_bytes+=x.numel()*x.element_size()
                return x.contiguous(memory_format=self.layout(module))
            return x
        return tuple(convert(x) for x in args),{k:convert(v) for k,v in kwargs.items()}

    def __enter__(self):
        if self.entered:raise RuntimeError('layout context cannot be entered twice')
        self.entered=True
        if self.mode=='baseline':return self
        modules=[];seen=set()
        for root in (self.model.conv2,self.model.decoder):
            for module in root.modules():
                if id(module) in seen or not isinstance(module,(torch.nn.Conv2d,torch.nn.Conv3d)):continue
                seen.add(id(module));modules.append(module)
        for module in modules:
            target=self.layout(module)
            original=torch.contiguous_format if module.weight.is_contiguous() else target
            if not module.weight.is_contiguous(memory_format=original):
                raise ValueError('unqualified strided convolution weight')
            self.saved.append((module,original))
        try:
            for module,_ in self.saved:
                module.to(memory_format=self.layout(module))
                if self.mode=='all_conv_inputs' or (self.mode=='entry_layout' and module is self.model.conv2):
                    self.handles.append(module.register_forward_pre_hook(self._hook,with_kwargs=True))
        except BaseException:
            self.__exit__(None,None,None);raise
        return self

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles=[]
        for module,original in self.saved:module.to(memory_format=original)
        self.saved=[]
        return False

    def record(self):
        return dict(mode=self.mode,input_hook_calls=self.input_calls,input_reformats=self.input_reformats,
                    estimated_input_reformat_payload_bytes=self.input_reformat_payload_bytes,
                    estimate_is_not_a_DRAM_counter=True,does_not_modify_upstream_source=True)
