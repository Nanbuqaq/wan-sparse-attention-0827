"""Inference-only native cache data update with original metadata commit order.

Common baseline optimization candidate. Avoid full-cache temporary clones and
the later duplicate data write; retain overlap-safe rolling clones. No layout,
positions, selection, arithmetic, or weights change. Not a new memory algorithm.
"""
import ast
import hashlib
import inspect
import textwrap
from types import MethodType
import torch


def transform_forward(source):
    tree=ast.parse(textwrap.dedent(source));clones=tags=0
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal clones
            value=node.func.value if isinstance(node.func,ast.Attribute) else None
            cache_value=(isinstance(value,ast.Subscript) and isinstance(value.value,ast.Name)
                and value.value.id=='kv_cache' and isinstance(value.slice,ast.Constant)
                and value.slice.value in ('k','v'))
            cache_alias=isinstance(value,ast.Name) and value.id in ('cache_k','cache_v')
            if (isinstance(node.func,ast.Attribute) and node.func.attr=='clone'
                and not node.args and not node.keywords and (cache_value or cache_alias)):
                clones+=1
                return ast.copy_location(ast.IfExp(
                    test=ast.parse("getattr(self, '_research_inplace_cache', False)",mode='eval').body,
                    body=node.func.value,orelse=node),node)
            return self.generic_visit(node)
        def visit_Assign(self,node):
            nonlocal tags
            node=self.generic_visit(node)
            if any(isinstance(t,ast.Name) and t.id=='window_start' for t in node.targets):
                tags+=1
                tag=ast.parse("cache_update_info['research_inplace_data_applied'] = getattr(self, '_research_inplace_cache', False)").body[0]
                return [ast.copy_location(tag,node),node]
            return node
    tree=Rewrite().visit(tree)
    if clones!=4 or tags!=1:raise ValueError(f'upstream cache structure changed: clones={clones}, tags={tags}')
    guard=ast.parse("if getattr(self, '_research_inplace_cache', False) and torch.is_grad_enabled():\n    raise RuntimeError('in-place native cache requires no-grad inference')").body[0]
    body=tree.body[0].body
    body.insert(1 if isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) else 0,guard)
    ast.fix_missing_locations(tree)
    return tree


def metadata_only_updates(caches,updates):
    result=[];stats=dict(calls=0,full_cache_clone_logical_bytes_avoided=0,duplicate_new_KV_write_bytes_avoided=0)
    for layer,(end,local,info) in updates:
        if info is not None and info.get('research_inplace_data_applied'):
            cache=caches[layer]
            if info['action'] not in ('direct_insert','roll_and_insert') or cache.get('quantized',False):
                raise ValueError('unregistered already-applied cache update')
            shift=info.get('pinned_shift',0)
            if shift:
                cache['pinned_start'].sub_(shift)
            stats['calls']+=1
            stats['full_cache_clone_logical_bytes_avoided']+=sum(cache[k].numel()*cache[k].element_size() for k in ('k','v'))
            stats['duplicate_new_KV_write_bytes_avoided']+=sum(info[k].numel()*info[k].element_size() for k in ('new_k','new_v'))
            result.append((layer,(end,local,None)))
        else:result.append((layer,(end,local,info)))
    return result,stats


class NativeInplaceCache:
    def __init__(self,pipe):
        if pipe.use_relative_rope or pipe.quantize_kv or pipe.guidance_scale!=1:
            raise ValueError('qualified absolute BF16 CFG1 only')
        if pipe.generator._compiled_model_call is not None:
            raise ValueError('compiled/autograd cache ownership needs a separate gate')
        self.pipe=pipe;self.model=pipe._dit_model;self.original_forward=None
        self.counters=dict(calls=0,full_cache_clone_logical_bytes_avoided=0,duplicate_new_KV_write_bytes_avoided=0)
    def attach(self):
        import wan_5b.modules.causal_model as native
        if native._CGRAPH_OUTPLACE_KV_ENABLED:
            raise ValueError('outplace CUDA graph cache mode is not this experiment')
        self.native=native;cls=native.CausalWanSelfAttention
        self.original_forward=cls.forward
        source=getattr(cls.forward,'_research_source',None) or inspect.getsource(cls.forward)
        tree=transform_forward(source);scope={}
        exec(compile(tree,'<native_inplace_cache_forward>','exec'),native.__dict__,scope)
        derived=scope['forward'];self.derived_source=ast.unparse(tree);derived._research_source=self.derived_source
        cls.forward=derived
        for block in self.model.blocks:block.self_attn._research_inplace_cache=True
        self.original_apply=self.model._apply_cache_updates
        def apply(owner,caches,updates):
            rewritten,stats=metadata_only_updates(caches,updates)
            for key,value in stats.items():self.counters[key]+=value
            return self.original_apply(caches,rewritten)
        self.model._apply_cache_updates=MethodType(apply,self.model)
        self.original_sha256=hashlib.sha256(textwrap.dedent(source).encode()).hexdigest()
        self.derived_sha256=hashlib.sha256(self.derived_source.encode()).hexdigest()
    def detach(self):
        if self.original_forward is not None:
            self.native.CausalWanSelfAttention.forward=self.original_forward;self.original_forward=None
            self.model._apply_cache_updates=self.original_apply
            for block in self.model.blocks:
                if hasattr(block.self_attn,'_research_inplace_cache'):del block.self_attn._research_inplace_cache
    def audit(self):
        return dict(schema='native_inplace_cache_v1',**self.counters,
            original_forward_sha256=self.original_sha256,derived_forward_sha256=self.derived_sha256,
            actual_HBM_transactions_measured=False,metadata_commit_order_preserved=True,
            rolling_overlap_clone_preserved=True,common_baseline_optimization=True)
