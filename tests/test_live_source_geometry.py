import time
import torch
import pytest
from adapters.longlive_sparse.live_source_geometry import SourceGeometryWorker
from adapters.longlive_sparse.live_source_geometry import LiveSourceGeometryMemory


class Model:
    def extract(self,window,**kwargs):
        if window['archive_version']==1:raise ValueError('no component')
        return dict(status='mask_ready',indices=torch.tensor([1,2]),archive_version=window['archive_version'])


def window(version):
    return dict(archive_version=version,source_end=version*8,source_phase=0.,pixel_start=0,pixel_end=29,ready_s=0.)


def test_unselected_error_does_not_block_later_source_and_results_are_versioned(tmp_path):
    worker=SourceGeometryWorker(Model(),started=time.perf_counter())
    worker.submit(window(1));worker.submit(window(2))
    assert worker.get(2,timeout=2)['archive_version']==2
    with pytest.raises(RuntimeError,match='selected source geometry failed'):worker.get(1,timeout=2)
    with pytest.raises(ValueError):worker.submit(window(2))
    worker.finish();worker.export(tmp_path/'results.pt')
    assert len(worker.audit()['rows'])==2 and worker.audit()['no_dense_or_manual_fallback']


def test_unready_source_has_explicit_timeout_and_no_fallback():
    worker=SourceGeometryWorker(Model(),started=time.perf_counter())
    with pytest.raises(RuntimeError,match='readiness timeout'):worker.get(99,timeout=.01)
    worker.abort()


def test_live_memory_checks_selected_source_and_reuses_only_its_plan():
    from types import SimpleNamespace
    memory=object.__new__(LiveSourceGeometryMemory)
    memory.mask_fill='uniform_midpoint'
    memory.active_bank={'descriptor':SimpleNamespace(archive_version=2,source_end=16,source_phase=8.)}
    memory.frame_tokens=4;memory.grid=(2,2);memory.geometry_plans={};memory.geometry_sources=[]
    memory.geometry_used_layers=set();memory.ledger={};calls=[]
    mask=torch.zeros(8,2,2,dtype=torch.bool);mask.flatten()[:3]=True
    result=dict(source_start=8,source_end=16,source_phase=8.,all_masks_nonempty=True,
        pixel_input_kind='raw_stream_before_codec',token_masks=mask,indices=torch.arange(3),
        source_latent_sha256='a'*64,source_raw_pixel_sha256='b'*64)
    def get(version):calls.append(version);return result
    memory.geometry_worker=SimpleNamespace(get=get)
    assert memory.mask_source_indices(0,2,8).shape==(2,8)
    memory.mask_source_indices(1,2,8);assert calls==[2]
    memory.geometry_plans.clear();result['source_end']=24
    with pytest.raises(RuntimeError,match='ownership'):memory.mask_source_indices(0,2,8)
