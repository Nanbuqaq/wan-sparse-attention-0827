import pytest
import torch
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.precision_runtime import PrecisionRecipe
from adapters.longlive_sparse.archive import HistoryArchive


def test_precision_method_has_explicit_identity_and_no_unused_legacy_index():
    cfg=SparseHistoryConfig(method='whole_block_precision_history',backend='resident_grouped_fa2',
        refresh_policy='per_chunk',history_density=.14,method_params={'precision_admission':'random'})
    assert cfg.routing_stage=='pre-transfer' and PrecisionRecipe.from_config(cfg).admission=='random'
    archive=HistoryArchive(cfg,spatial_height=8,spatial_width=16)
    frame=archive.index_frame(0,1,torch.zeros(1,128,2,64),torch.zeros(1,128,2,64))
    assert frame.index_bytes==0 and frame.cluster_labels.numel()==0


@pytest.mark.parametrize('change',[{'refresh_policy':'per_step'},{'rope_policy':'recency_rank'},
    {'method_params':{'precision_query_samples':0}}])
def test_invalid_precision_contract_rejected(change):
    kwargs=dict(method='whole_block_precision_history',backend='resident_grouped_fa2',refresh_policy='per_chunk',history_density=.14)
    kwargs.update(change)
    with pytest.raises(ValueError):SparseHistoryConfig(**kwargs)
