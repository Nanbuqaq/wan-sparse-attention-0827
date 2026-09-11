"""Read a completed, case-bound raw source window without decoding a video."""
import hashlib
import torch


def load_raw_source_window(path,version,summary):
    data=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
    if not data.get('complete') or not data.get('pixels_before_lossy_codec'):
        raise ValueError('complete pre-codec source witness required')
    rows=[r for r in data['records'] if r['archive_version']==version]
    declared=[r for r in summary['source_pixel_witness']['records'] if r['archive_version']==version]
    if len(rows)!=1 or len(declared)!=1:raise ValueError('unique case-bound source archive required')
    row=rows[0]
    for field in ('source_start','source_end','source_phase','pixel_start','pixel_end','source_latent_sha256','raw_pixel_bytes_sha256'):
        if row[field]!=declared[0][field]:raise ValueError('raw source differs from case witness: '+field)
    pixels=row['pixels']
    if (row['source_end']-row['source_start']!=8 or row['pixel_start']!=max(0,4*row['source_start']-3)
        or row['pixel_end']!=4*row['source_end']-3):raise ValueError('raw source causal window bounds differ')
    if pixels.dtype!=torch.uint8 or pixels.ndim!=4 or pixels.shape[-1]!=3 or pixels.shape[0]!=row['pixel_end']-row['pixel_start']:
        raise ValueError('raw source pixel geometry differs')
    if hashlib.sha256(memoryview(pixels.numpy())).hexdigest()!=row['raw_pixel_bytes_sha256']:
        raise ValueError('raw source pixel payload SHA mismatch')
    return row
