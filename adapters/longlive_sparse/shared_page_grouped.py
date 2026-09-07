"""FA2 varlen paged KV: share identical physical pages across query groups.

Logical per-group KV order is unchanged. Exact prefix pages naturally dedupe;
mixed exact/history boundary pages remain private when their contents differ.
This changes GPU packing/storage, not original history H2D admission.
"""
from dataclasses import dataclass, replace
import time

import numpy as np
import torch

from .grouped_staging import build_grouped_packing_recipe
from .resident_grouped import ResidentGroupedExecutor
from .profiling import synchronize_cuda


@dataclass(frozen=True)
class SharedPageRecipe:
    packing: object
    block_table: torch.Tensor
    page_tokens: int
    logical_packed_tokens: int
    unique_valid_coordinates: int

    @property
    def physical_tokens(self):
        return self.packing.key_indices.numel()

    @property
    def metadata_bytes(self):
        return self.packing.metadata_bytes+self.block_table.numel()*self.block_table.element_size()


def build_shared_page_recipe(plan, *, exact_tokens, union_tokens, page_tokens=256):
    if page_tokens < 256 or page_tokens % 256:
        raise ValueError('FA2 pages must be multiples of256 tokens')
    packed = build_grouped_packing_recipe(plan, exact_tokens=exact_tokens, union_tokens=union_tokens)
    values = packed.key_indices.numpy()
    pages, table_rows, lookup = [], [], {}
    cursor = 0
    for length in packed.key_lengths:
        sequence = values[cursor:cursor+length]
        cursor += length
        row = []
        for begin in range(0, length, page_tokens):
            page = np.full(page_tokens, -1, dtype=np.int64)
            count = min(page_tokens, length-begin)
            page[:count] = sequence[begin:begin+count]
            signature = page.tobytes()
            if signature not in lookup:
                lookup[signature] = len(pages)
                pages.append(page)
            row.append(lookup[signature])
        table_rows.append(row)
    table = torch.zeros(len(table_rows), max(map(len, table_rows)), dtype=torch.int32)
    for i, row in enumerate(table_rows):
        table[i, :len(row)] = torch.tensor(row, dtype=torch.int32)
    physical = torch.from_numpy(np.stack(pages)).reshape(-1)
    return SharedPageRecipe(replace(packed, key_indices=physical), table, page_tokens,
                            sum(packed.key_lengths), int(np.unique(values).size))


class SharedPageGroupedExecutor:
    def __init__(self):
        self.identity = self.recipe = None
        self.packer = ResidentGroupedExecutor()

    def prepare(self, plan, query, exact_key, history_key):
        identity = (plan.digest(), query.shape, exact_key.shape, history_key.shape, query.dtype, query.device)
        if identity == self.identity:
            return self.recipe, 0
        recipe = build_shared_page_recipe(plan, exact_tokens=exact_key.shape[1], union_tokens=history_key.shape[1])
        packing = replace(recipe.packing, **{name: getattr(recipe.packing, name).to(query.device) for name in
            ('query_indices', 'key_indices', 'cu_query', 'cu_key')})
        self.recipe = replace(recipe, packing=packing, block_table=recipe.block_table.to(query.device))
        self.identity = identity
        return self.recipe, recipe.metadata_bytes

    def execute(self, query, exact_key, exact_value, history_key, history_value, plan):
        from flash_attn import flash_attn_varlen_func
        from .backends import BackendResult
        started = time.perf_counter()
        recipe, metadata_bytes = self.prepare(plan, query, exact_key, history_key)
        packed = recipe.packing
        q, k, v = self.packer.pack(packed, query, exact_key, exact_value, history_key, history_value)
        dim = query.shape[-1]
        k = k.view(-1, recipe.page_tokens, 1, dim)
        v = v.view(-1, recipe.page_tokens, 1, dim)
        output = flash_attn_varlen_func(q, k, v, packed.cu_query, packed.cu_key,
            max(packed.query_lengths), max(packed.key_lengths), dropout_p=0., causal=False,
            block_table=recipe.block_table)
        restored = torch.empty(query.numel()//dim, dim, device=query.device, dtype=query.dtype)
        restored.index_copy_(0, packed.query_indices, output[:, 0])
        synchronize_cuda(query.device)
        pairs = sum(qn*kn for qn, kn in zip(packed.query_lengths, packed.key_lengths))
        result = BackendResult(restored.reshape(query.shape), 'shared_page_grouped_fa2_reference',
            (time.perf_counter()-started)*1000, pairs, pairs, 0, packed.route_sha,
            metadata_H2D_bytes=metadata_bytes, resident_metadata_bytes=recipe.metadata_bytes)
        result.storage_accounting = dict(logical_packed_KV_bytes=recipe.logical_packed_tokens*dim*query.element_size()*2,
            physical_paged_KV_bytes=recipe.physical_tokens*dim*query.element_size()*2,
            unique_original_KV_bytes=recipe.unique_valid_coordinates*dim*query.element_size()*2,
            physical_pages=recipe.physical_tokens//recipe.page_tokens,
            original_H2D_admission_unchanged=True, HBM_transactions_not_measured=True)
        return result
