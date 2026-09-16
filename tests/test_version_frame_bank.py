import torch
from adapters.longlive_sparse.version_frame_bank import EightFrameBank
from adapters.longlive_sparse.version_scene_memory import frame_binding


def test_three_policies_share_eight_frame_storage_and_preserve_actual_payload():
    first=[dict(frame=i,phase=0.,version=1) for i in range(16,24)]
    second=[dict(frame=i,phase=8.,version=2) for i in range(40,48)]
    expected={'latest8':list(range(40,48)),'old4_new4':[20,21,22,23,44,45,46,47],
              'uniform8':[16,18,20,22,41,43,45,47]}
    for policy,frames in expected.items():
        cache=dict(k=torch.arange(16,24,dtype=torch.bfloat16).reshape(1,8,1,1),
                   v=torch.arange(116,124,dtype=torch.bfloat16).reshape(1,8,1,1),local_end_index=8)
        bank=EightFrameBank(1,32);bank.update([cache],first,policy)
        pointer=bank.kv[0][0].data_ptr()
        cache['k']=torch.arange(40,48,dtype=torch.bfloat16).reshape(1,8,1,1)
        cache['v']=cache['k']+100
        bank.update([cache],second,policy)
        assert bank.kv[0][0].data_ptr()==pointer
        slots=bank.ordered_slots()
        assert [bank.records[i]['frame'] for i in slots]==frames
        assert bank.kv[0][0][0,slots,0,0].tolist()==frames
        assert bank.kv[0][1][0,slots,0,0].tolist()==[x+100 for x in frames]
        assert bank.audit()['unique_owned_raw_bytes']==32


def test_role_halves_store_only_the_role_tensor_of_trailing_four_frames():
    first=[dict(frame=i,phase=0.,version=1) for i in range(16,24)]
    second=[dict(frame=i,phase=8.,version=2) for i in range(40,48)]
    expected={'oldk_newv':([20,21,22,23],[144,145,146,147]),'newk_oldv':([44,45,46,47],[120,121,122,123])}
    for role,(k_values,v_values) in expected.items():
        cache=dict(k=torch.arange(16,24,dtype=torch.bfloat16).reshape(1,8,1,1),
                   v=torch.arange(116,124,dtype=torch.bfloat16).reshape(1,8,1,1),local_end_index=8)
        bank=EightFrameBank(1,32,role_halves=role);bank.update([cache],first,'old4_new4')
        cache['k']=torch.arange(40,48,dtype=torch.bfloat16).reshape(1,8,1,1);cache['v']=cache['k']+100
        bank.update([cache],second,'old4_new4')
        assert bank.kv[0][0].shape[1]==4 and bank.kv[0][1].shape[1]==4
        assert bank.kv[0][0][0,:,0,0].tolist()==k_values
        assert bank.kv[0][1][0,:,0,0].tolist()==v_values
        assert [r['frame'] for r in bank.records]==[20,21,22,23,44,45,46,47]
        assert [r['version'] for r in bank.records]==[1,1,1,1,2,2,2,2]
        assert bank.raw_bytes==16 and bank.audit()['unique_owned_raw_bytes']==16
        assert bank.D2H_bytes==16
    try:
        EightFrameBank(1,32,role_halves='oldk_newv').update([dict(k=torch.zeros(1,8,1,1,dtype=torch.bfloat16),v=torch.zeros(1,8,1,1,dtype=torch.bfloat16),local_end_index=8)],first,'uniform8')
        raise SystemExit('role halves must fail fast off the frozen old4_new4 policy')
    except ValueError:pass


def test_mixed_versions_use_actual_per_frame_phase_not_latest_scalar():
    records=[dict(frame=f,phase=0.) for f in (20,21,22,23)]+[dict(frame=f,phase=8.) for f in (44,45,46,47)]
    deltas=frame_binding(records,96,24.)
    assert deltas==[92.]*4+[64.]*4
    assert [r['frame']+r['phase']+delta for r,delta in zip(records,deltas)]==list(range(112,120))
