import sqlite3

import pytest

from scripts.audit_native_pipeline_timeline import analyze,trace_events


def fixture_db(second=True):
    db=sqlite3.connect(':memory:')
    db.executescript('''
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,streamId INTEGER,demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,streamId INTEGER,bytes INTEGER,copyKind INTEGER);
    ''')
    db.execute("INSERT INTO StringIds VALUES(1,'test kernel')")
    db.execute("INSERT INTO NVTX_EVENTS VALUES(9000000000,21000000000,'native_pipeline/full_run',NULL,7)")
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)',[
        (10*10**9,14*10**9,0,7,1,1),(16*10**9,18*10**9,0,7,1,1)])
    if second:db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)',(12*10**9,17*10**9,1,7,1,1))
    return db


def summary():
    return dict(status='pass',observer_noise_latent_RGB_equivalence=True,physical_GPU_mapping='0,1',
                video_pipeline=dict(latent_D2H_bytes=0,latent_H2D_bytes=0,pixel_D2H_bytes=0))


def test_overlap_is_cross_device_interval_intersection_not_sum():
    result,raw=analyze(fixture_db(),summary())
    assert result['cross_device_kernel_overlap_s']==3
    assert result['parent_wall_s']==12
    assert result['any_device_activity_union_s']==8
    assert result['per_device'][0]['GPU_kernel_active_union_s']==6
    assert result['per_device'][1]['GPU_kernel_active_union_s']==5
    events=trace_events(raw)
    assert {e['pid'] for e in events if e.get('cat')=='kernel'}=={100,101}
    raw['scopes']=[(a,b,n,2**50+t) for a,b,n,t in raw['scopes']]
    assert all(e['tid']<2**32 for e in trace_events(raw))


def test_missing_second_device_or_payload_is_rejected():
    with pytest.raises(ValueError,match='both device'):analyze(fixture_db(False),summary())
    invalid=summary();invalid['video_pipeline']['pixel_D2H_bytes']=1
    with pytest.raises(ValueError,match='traffic incomplete'):analyze(fixture_db(),invalid)


def test_numerical_failure_and_aggregate_graph_trace_are_not_overlap_evidence():
    invalid=summary();invalid['observer_noise_latent_RGB_equivalence']=False
    with pytest.raises(ValueError,match='equivalence'):analyze(fixture_db(),invalid)
    db=fixture_db();db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(value INTEGER)')
    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(1)')
    with pytest.raises(ValueError,match='graph-level'):analyze(db,summary())


def test_CPU_encode_overlap_is_measured_against_actual_device_kernels():
    db=fixture_db()
    db.execute("INSERT INTO NVTX_EVENTS VALUES(13000000000,15000000000,'native_pipeline/encode',NULL,8)")
    db.execute("INSERT INTO NVTX_EVENTS VALUES(12000000000,16000000000,'native_pipeline/VAE_decode_group',NULL,9)")
    result,_=analyze(db,summary());output=result['CPU_output']
    assert output['NVTX_ranges']==1 and output['encode_host_union_s']==2
    assert output['overlap_GPU0_kernel_s']==1 and output['overlap_GPU1_kernel_s']==2
    assert output['encode_and_decode_CPU_threads_distinct'] and output['encode_fraction_overlapping_GPU1_kernel']==1
