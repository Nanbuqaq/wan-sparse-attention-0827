from __future__ import annotations

import hashlib

import torch

from adapters.longlive_sparse.route_timeline import (
    finalize_records,
    stable_metadata_sha256,
    tensor_identity,
)
from adapters.longlive_sparse.wave2_temporal_budget import Wave2TemporalBudget


class _FakeEvent:
    def __init__(self, timestamp=0.0):
        self.timestamp = timestamp
        self.synced = False

    def synchronize(self):
        self.synced = True

    def elapsed_time(self, other):
        return (other.timestamp - self.timestamp) * 1000.0


def test_tensor_identity_records_storage_without_payload_read():
    base = torch.arange(16, dtype=torch.float32)
    view = base[4:12].view(2, 4)
    identity = tensor_identity(view)
    assert identity["shape"] == [2, 4]
    assert identity["stride"] == [4, 1]
    assert identity["data_ptr"] == view.data_ptr()
    assert identity["storage_data_ptr"] == base.untyped_storage().data_ptr()
    assert identity["storage_nbytes"] == base.untyped_storage().nbytes()
    assert identity["nbytes"] == view.numel() * view.element_size()
    assert identity["is_view"] is True


def test_stable_route_metadata_digest_ignores_tensor_payload_values():
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([99.0, -1.0])
    left = stable_metadata_sha256({"frames": (1, 2), "tensor": first})
    right = stable_metadata_sha256({"frames": (1, 2), "tensor": second})
    assert left == right
    changed = stable_metadata_sha256({"frames": (1, 3), "tensor": first})
    assert changed != left


def test_finalize_records_exports_interval_service_times():
    events = [_FakeEvent(0.0), _FakeEvent(0.25), _FakeEvent(0.75)]
    record = {"_event_series": {"pack_attention": events}}
    (row,) = finalize_records([record])
    assert row["pack_attention_device_s"] == 0.75
    assert row["pack_attention_interval_device_s"] == [0.25, 0.5]
    assert all(event.synced for event in events)
    assert "_event_series" not in row


def test_route_audit_adds_per_row_selected_mask_hash():
    controller = Wave2TemporalBudget.__new__(Wave2TemporalBudget)
    controller.defer_stats = False
    controller.route_audit = True
    controller.age_observer = False
    controller.age_D2H_bytes = 0
    controller.route_records = 0
    controller.route_hasher = hashlib.sha256()
    controller.stats_D2H_bytes = 0
    controller.stats_queue_bytes = 0
    controller.stats_flush_host_s = 0.0
    row = {"protected_union_tokens": 8, "route_reused": False, "layer": 0}
    stats = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False]], dtype=torch.bool)
    binding = (1, 0, 4, [(0, ("native", 1, 0, 0.0))], [1])
    controller.stats_queue = [(row, stats, mask, binding, 1, 8, 16)]
    controller.flush_statistics()
    assert row["selected_mask_sha256"] == hashlib.sha256(mask.numpy().tobytes()).hexdigest()
    assert row["selected_mask_shape"] == [1, 2]
    assert controller.route_records == 1
    assert controller.stats_queue == []
