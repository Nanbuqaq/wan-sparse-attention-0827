import threading
import pytest
from adapters.longlive_sparse.staged_scene_archive import SnapshotJob


def test_completed_job_propagates_worker_error_instead_of_reading_partial_archive():
    job=object.__new__(SnapshotJob);job.done=threading.Event();job.done.set();job.error='copy failed'
    with pytest.raises(RuntimeError,match='copy failed'):job.wait()
    job.error=None;job.wait()


def test_oversized_staging_is_rejected_before_any_pinned_or_cuda_allocation():
    import torch
    from adapters.longlive_sparse.staged_scene_archive import StagedSceneArchive
    archive=object.__new__(StagedSceneArchive);archive.worker=None
    archive.stats={'pinned_limit_bytes':256*1024**2}
    example=torch.empty((1,8*880,48,128),dtype=torch.bfloat16,device='meta')
    with pytest.raises(RuntimeError,match='exceed256MiB'):archive._initialize(example)
