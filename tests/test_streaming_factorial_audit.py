import copy
import pytest
from scripts.collect_streaming_factorial import summarize_lane


def fixture():
    arms=['batch_current_stream__rope_upstream','batch_current_stream__rope_direct_output',
          'async_priority_current_stream__rope_upstream','async_priority_current_stream__rope_direct_output']
    rows=[dict(variant=arm,repetition=r,status='pass',exact_reference=True,diagnostic_timing=False,
        identity={'noise':'a','latent':'b','raw_RGB':'c','ordered_routes':'d'},
        generation_decode_encode_s=100-i*10,sink={'first_packet_muxed_s':5},peak_GPU_bytes=123)
          for r in range(3) for i,arm in enumerate(arms)]
    return dict(variants=rows,method='rag_dense',prompt={'prompt_id':'dev'},gpu='test',
                seed=1,latent_frames=120,source_commit='a'*40)


def test_all_arms_and_paired_contrasts():
    result=summarize_lane(fixture(),3)
    assert result['cases']==12 and result['contrasts']['combined']['median']==100/70


@pytest.mark.parametrize('defect',['duplicate','identity','profile'])
def test_bad_complete_matrix_rejected(defect):
    sample=fixture()
    if defect=='duplicate':sample['variants'][-1]=copy.deepcopy(sample['variants'][0])
    elif defect=='identity':sample['variants'][0]['identity']['latent']='different'
    else:sample['variants'][0]['diagnostic_timing']=True
    with pytest.raises(ValueError):summarize_lane(sample,3)
