from pathlib import Path


def test_legacy_d2h_is_fenced_before_cpu_prototype_indexing():
    source=(Path(__file__).resolve().parents[1]/'adapters/longlive_sparse/runtime_attention.py').read_text()
    copied=source.index('evicted_value_frames = [item.to("cpu", non_blocking=True)')
    fence=source.index('torch.cuda.current_stream(query.device).synchronize()',copied)
    indexed=source.index('self.history_archive.index_frame(',copied)
    assert copied<fence<indexed
