import pytest
import torch

from adapters.longlive_sparse.dataflow_reference import DataflowInputs
from scripts.gate_dataflow_references import allowed_keys


def test_reuse_labels_define_identical_logical_mask_for_all_backends():
    tags = torch.tensor([0, 1, 2, 0, 1, 2])
    for reuse in (1, 2, 3):
        visible = torch.stack([allowed_keys(tags, group, reuse) for group in range(3)])
        assert torch.equal(visible.sum(0), torch.full((6,), reuse))


def test_gpu_reference_cannot_be_mislabeled_as_cpu_execution():
    q, k = torch.zeros(2, 7, 64), torch.zeros(2, 9, 64)
    with pytest.raises(ValueError, match='CUDA'):
        DataflowInputs(q, k, k, torch.zeros(9, dtype=torch.int32), 1).validate()
