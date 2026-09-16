"""Bounded two-state preservation for ONE object with two confirmed states.

This is a diagnostic, not an online algorithm. The caller (a human diagnostician)
explicitly names two 8-frame windows of the SAME object in two distinguishable
states (e.g. closed latent 8-15 and open latent 24-31 of one red toolbox). Each
state's trailing four frames are saved to CPU **while that state is still the
live window** (closed saved before the open update, open saved after), because a
rolling window evicts the earlier state. We never auto-promote the first two
archives into object versions, and we never relax the frozen similarity rule to
chase success.

Capacity is charged at an eight-frame-equivalent budget; the per-state staging
copies are billed to the same budget so the diagnostic never hides copies.
"""
import torch


class TwoStateBank:
    """CPU bank holding old4+new4 (eight frames) of ONE object across ONE update."""

    def __init__(self, frame_tokens, capacity_bytes):
        self.frame_tokens = frame_tokens
        self.capacity_bytes = capacity_bytes
        self.halves = {}          # 'old'/'new' -> list per layer of (k[4ft], v[4ft])
        self.half_records = {}    # 'old'/'new' -> 4 records
        self.raw_bytes = 0
        self.D2H_bytes = 0
        self.records = None       # 8 records: old4 (version1) then new4 (version2)
        self.frozen = False

    def _alloc_half(self, caches):
        ft = self.frame_tokens
        return [(torch.empty_like(c['k'][:, :4 * ft], device='cpu'),
                 torch.empty_like(c['v'][:, :4 * ft], device='cpu')) for c in caches]

    def save_state(self, caches, records, *, state):
        """Save the trailing four frames of one just-committed state to CPU (D2H).

        Must be called while that state still occupies the live window. records is
        the 8-frame window; we keep the trailing four (old4/new4).
        """
        if self.frozen:
            raise RuntimeError('two-state bank already frozen')
        if state in self.halves:
            raise RuntimeError(f'state {state!r} already saved')
        if state not in ('old', 'new'):
            raise ValueError('state must be old or new')
        if len(records) != 8:
            raise ValueError('exactly eight frames per state window required')
        ft = self.frame_tokens
        keep = records[-4:]
        half = self._alloc_half(caches)
        for slot, rec in enumerate(keep):
            for cache, (key, value) in zip(caches, half):
                end = int(cache['local_end_index'])
                src = end - 8 * ft + (rec['frame'] % 8) * ft
                for source, target in ((cache['k'], key), (cache['v'], value)):
                    target[:, slot * ft:(slot + 1) * ft].copy_(source[:, src:src + ft])
                    self.D2H_bytes += ft * target.shape[0] * target.shape[2] * target.shape[3] * target.element_size()
        self.halves[state] = half
        self.half_records[state] = keep
        # Charge the staged half against the shared eight-frame budget.
        staged = sum(t.numel() * t.element_size() for pair in half for t in pair)
        if self.raw_bytes + staged > self.capacity_bytes:
            raise RuntimeError('two-state staging exceeds the eight-frame budget')
        self.raw_bytes += staged

    def freeze(self, *, same_object_cosine, min_cosine=0.8):
        """Freeze old4+new4 as one object's two versions after BOTH states saved.

        same_object_cosine: precomputed cosine between the two states' text
        descriptors; must clear the SAME frozen floor (no tuning). Raises rather
        than substituting windows.
        """
        if self.frozen:
            raise RuntimeError('two-state bank already frozen')
        if set(self.halves) != {'old', 'new'}:
            raise RuntimeError('both old and new states must be saved before freeze')
        old4 = self.half_records['old']
        new4 = self.half_records['new']
        if max(r['frame'] for r in old4) >= min(r['frame'] for r in new4):
            raise ValueError('old state must be strictly earlier than new state')
        if same_object_cosine < min_cosine:
            raise RuntimeError(
                f'the two states do not describe the same object (cosine {same_object_cosine:.3f} < {min_cosine}); refusing to freeze as versions')
        self.records = [dict(r, version=1) for r in old4] + [dict(r, version=2) for r in new4]
        self.frozen = True

    def kv(self):
        """Return per-layer (k[8ft], v[8ft]) with old4 in slots 0-3, new4 in 4-7."""
        if not self.frozen:
            raise RuntimeError('two-state bank not frozen')
        out = []
        for (ok, ov), (nk, nv) in zip(self.halves['old'], self.halves['new']):
            out.append((torch.cat([ok, nk], dim=1), torch.cat([ov, nv], dim=1)))
        return out

    def audit(self):
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                    for half in self.halves.values() for pair in half for t in pair}
        owned = sum(storages.values())
        if owned != self.raw_bytes:
            raise RuntimeError('two-state bank has hidden/noncompact storage')
        return dict(capacity_bytes=self.capacity_bytes, unique_owned_raw_bytes=owned,
                    D2H_bytes=self.D2H_bytes, records=self.records, frozen=self.frozen,
                    scope='ONE object, two human-confirmed states saved pre/post update; not auto archive promotion')
