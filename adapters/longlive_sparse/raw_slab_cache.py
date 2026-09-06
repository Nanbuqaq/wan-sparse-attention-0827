"""Bounded raw KV slab with per-token validity, not speculative full-block load."""
from collections import OrderedDict
from dataclasses import dataclass
from typing import NamedTuple

import torch


class SlabKey(NamedTuple):
    batch_id: int
    layer_id: int
    head_id: int
    archive_epoch: int
    frame_id: int
    frame_storage_version: int
    token_start: int
    token_end: int
    dtype: str
    device: str
    storage_kind: str = 'raw_unrotated_kv'


@dataclass
class SlabEntry:
    slot: int
    valid_mask: int = 0


class RawTokenSlabCache:
    def __init__(self, budget_bytes, *, head_dim, dtype, device, block_tokens=64):
        if budget_bytes < 0 or head_dim < 1 or not 1 <= block_tokens <= 64:
            raise ValueError('invalid raw slab geometry/budget')
        self.budget_bytes, self.head_dim = int(budget_bytes), int(head_dim)
        self.block_tokens, self.dtype = int(block_tokens), dtype
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.bytes_per_token = 2*head_dim*torch.empty((), dtype=dtype).element_size()
        self.capacity = budget_bytes//(self.bytes_per_token*block_tokens)
        self.key = self.value = None
        self._entries = OrderedDict()
        self._free_slots = list(reversed(range(self.capacity)))
        self.hit_bytes = self.miss_bytes = self.evictions = 0

    def allocate(self):
        if self.key is None:
            if self.capacity < 1:
                raise MemoryError('raw slab cannot hold one block')
            self.key = torch.empty((self.capacity*self.block_tokens, self.head_dim), dtype=self.dtype, device=self.device)
            self.value = torch.empty_like(self.key)

    def reserve(self, keys, masks):
        if len(keys) != len(masks) or len(set(keys)) != len(keys):
            raise ValueError('unique physical block keys and masks required')
        if len(keys) > self.capacity:
            raise MemoryError('current raw working set exceeds slab capacity; no silent fallback')
        for key, mask in zip(keys, masks):
            if key.storage_kind != 'raw_unrotated_kv' or min(key.batch_id, key.layer_id, key.head_id,
                    key.archive_epoch, key.frame_id, key.frame_storage_version, key.token_start) < 0:
                raise ValueError('invalid raw slab storage identity')
            if not 0 < mask < (1 << (key.token_end-key.token_start)) or key.token_end-key.token_start > self.block_tokens:
                raise ValueError('invalid requested raw token mask')
            if key.dtype != str(self.dtype) or key.device != str(self.device):
                raise ValueError('raw slab dtype/device key mismatch')
        self.allocate()
        protected = set(keys)
        for key in keys:
            if key in self._entries:
                self._entries.move_to_end(key)
        slots, missing = [], []
        for key, required in zip(keys, masks):
            entry = self._entries.get(key)
            if entry is None:
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    victim = next((k for k in self._entries if k not in protected), None)
                    if victim is None:
                        raise MemoryError('all slab slots protected')
                    slot = self._entries.pop(victim).slot
                    self.evictions += 1
                entry = SlabEntry(slot)
                self._entries[key] = entry
            absent = required & ~entry.valid_mask
            slots.append(entry.slot)
            missing.append(absent)
            self.hit_bytes += (required & entry.valid_mask).bit_count()*self.bytes_per_token
            self.miss_bytes += absent.bit_count()*self.bytes_per_token
        return slots, missing

    def commit(self, keys, masks):
        """Call only after the corresponding GPU index-copy has completed."""
        for key, mask in zip(keys, masks):
            self._entries[key].valid_mask |= mask

    def reset(self):
        self._entries.clear()
        self._free_slots = list(reversed(range(self.capacity)))
        self.hit_bytes = self.miss_bytes = self.evictions = 0

    def as_dict(self):
        backing = 0 if self.key is None else self.key.untyped_storage().nbytes()+self.value.untyped_storage().nbytes()
        valid_tokens = sum(e.valid_mask.bit_count() for e in self._entries.values())
        return {'cache_kind': 'raw_token_valid_slab', 'budget_bytes': self.budget_bytes,
            'allocated_backing_bytes': backing, 'current_bytes': backing,
            'capacity_blocks': self.capacity, 'entries': len(self._entries),
            'valid_kv_bytes': valid_tokens*self.bytes_per_token,
            'hit_bytes': self.hit_bytes, 'miss_bytes': self.miss_bytes, 'evictions': self.evictions,
            'all_backing_allocation_counted': True}
