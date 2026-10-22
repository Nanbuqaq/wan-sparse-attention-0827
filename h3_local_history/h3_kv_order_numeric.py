"""Diagnostic-only K/V concatenation-order intervention; visibility unchanged.

Do not install in production. This preserves Context.visible_splits and all
tensor values, while putting already selected cached keys before local keys.
It tests whether a packed-TF/cache BF16 gap is due to reduction/key order.
"""


def install(core):
    original_torch=core.torch
    original_forward=core.forward

    class TorchView:
        def __init__(self):
            self.cached_ids=set()
            self.reordered_calls=0
        def __getattr__(self,name):
            return getattr(original_torch,name)
        def cat(self,tensors,dim=0,*args,**kwargs):
            if dim==2 and isinstance(tensors,(list,tuple)):
                past=[x for x in tensors if id(x) in self.cached_ids]
                local=[x for x in tensors if id(x) not in self.cached_ids]
                if past and local:
                    tensors=[*past,*local]
                    self.reordered_calls+=1
            return original_torch.cat(tensors,dim,*args,**kwargs)

    view=TorchView()
    def forward(transformer,inputs,context):
        previous=view.cached_ids
        view.cached_ids={id(t) for entries in (context.cached or {}).values() for entry in entries for t in (entry.key,entry.value)}
        try:
            return original_forward(transformer,inputs,context)
        finally:
            view.cached_ids=previous
    core.torch=view
    core.forward=forward
    def restore():
        core.torch=original_torch
        core.forward=original_forward
    return view,restore
