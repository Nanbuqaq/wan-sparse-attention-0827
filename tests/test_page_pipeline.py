import pytest

from adapters.longlive_sparse.page_pipeline import FramePage, frame_pages


def test_pages_never_cross_frame_boundaries_and_count_padding():
    pages = frame_pages([1560, 1560], 256)
    assert len(pages) == 14
    assert sum(p.count for p in pages) == 3120
    assert pages[6] == FramePage(0, 1536, 24)
    assert pages[7] == FramePage(1, 0, 256)
    assert len(pages)*256 - sum(p.count for p in pages) == 464


@pytest.mark.parametrize('lengths,size', [([], 64), ([0], 64), ([5], 0)])
def test_invalid_geometry(lengths, size):
    with pytest.raises(ValueError):
        frame_pages(lengths, size)
