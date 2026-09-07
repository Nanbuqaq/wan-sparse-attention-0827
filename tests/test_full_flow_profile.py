import pytest

from adapters.longlive_sparse.full_flow_profile import FullFlowTrace, pipeline_regions


def test_nested_wall_is_not_summed_and_leaf_self_matches():
    trace = FullFlowTrace()
    with trace.span("root"):
        with trace.span("child"):
            pass
    result = trace.result()
    root, child = result["records"]
    assert child["parent"] == root["id"]
    assert root["host_self_s"] + child["host_self_s"] == pytest.approx(root["host_wall_s"])
    assert "cuda_stream_span_s" not in root


def test_exception_is_preserved_and_scope_closed():
    trace = FullFlowTrace()
    with pytest.raises(ValueError, match="original"):
        with trace.span("failure"):
            raise ValueError("original")
    assert trace.result()["records"][0]["status"] == "fail"


def test_wrap_restores_inherited_method_and_preserves_result():
    class Sample:
        def run(self, x):
            return x + 2
    item = Sample()
    trace = FullFlowTrace()
    trace.wrap(item, "run", "sample", metadata=lambda args, kwargs: {"x": args[0]})
    assert item.run(3) == 5
    trace.restore()
    assert "run" not in vars(item)
    assert item.run(4) == 6
    assert trace.result()["records"][0]["metadata"] == {"x": 3}


def test_open_scope_cannot_be_flushed():
    trace = FullFlowTrace()
    with trace.span("open"):
        with pytest.raises(RuntimeError, match="close every"):
            trace.result()


def test_inline_regions_do_not_change_result_and_close_each_chunk():
    def small_pipeline():
        output = [0, 0]
        for current_start_frame in range(2):
            memory_indices = None
            memory_indices = current_start_frame
            for index, current_timestep in enumerate((1, 2)):
                memory_indices += current_timestep
            output[current_start_frame] = memory_indices
            for f_idx in range(3):
                pass
            context_timestep = 0
        return output
    expected = small_pipeline()
    trace = FullFlowTrace()
    with pipeline_regions(trace, small_pipeline):
        assert small_pipeline() == expected
    aggregates = trace.result()["aggregates"]
    assert aggregates["history.coarse_frame_retrieval"]["calls"] == 2
    assert aggregates["history.descriptor_update"]["calls"] == 2
    assert aggregates["latent.commit_to_CPU_output"]["calls"] == 2
