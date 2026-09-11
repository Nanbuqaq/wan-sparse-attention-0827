import hashlib
import torch
import pytest
from adapters.longlive_sparse.native_raw_source_input import load_raw_source_window


def test_raw_window_is_case_bound_and_payload_verified(tmp_path):
    pixels=torch.arange(96,dtype=torch.uint8).reshape(32,1,1,3)
    row=dict(archive_version=2,source_start=40,source_end=48,source_phase=8.,pixel_start=157,pixel_end=189,
        source_latent_sha256='a'*64,raw_pixel_bytes_sha256=hashlib.sha256(memoryview(pixels.numpy())).hexdigest(),pixels=pixels)
    summary={'source_pixel_witness':{'records':[{k:v for k,v in row.items() if k!='pixels'}]}}
    path=tmp_path/'raw.pt';torch.save(dict(complete=True,pixels_before_lossy_codec=True,records=[row]),path)
    assert torch.equal(load_raw_source_window(path,2,summary)['pixels'],pixels)
    summary['source_pixel_witness']['records'][0]['source_end']=56
    with pytest.raises(ValueError,match='case witness'):load_raw_source_window(path,2,summary)
    summary['source_pixel_witness']['records'][0]['source_end']=48
    pixels.zero_();torch.save(dict(complete=True,pixels_before_lossy_codec=True,records=[row]),path)
    with pytest.raises(ValueError,match='SHA mismatch'):load_raw_source_window(path,2,summary)
