from types import SimpleNamespace
from pathlib import Path

from adapters.longlive_sparse.payload_aware_scene import PayloadAwareScene
from adapters.longlive_sparse.write_origin_scene import WriteOriginScene
from adapters.longlive_sparse.write_origin_protocol import write_origin_schedule


def fixture(policy):
    x=WriteOriginScene.__new__(WriteOriginScene)
    x.write_policy=policy;x.write_events=[];x.root_replacements=0
    x.last_commit={'phase':24.};x.installations=[{'current_phase':24.,'installation':{'admission_plan':{'archive_version':2}}}]
    x.identity_roots={1:1,2:2,3:3};x.banks=[{'descriptor':SimpleNamespace(archive_version=i),'owned_bytes':100} for i in (1,2,3)]
    x.archives=[]
    return x


def test_skip_derived_makes_no_payload_or_new_descriptor(monkeypatch):
    x=fixture('skip_derived')
    def forbidden(*args):raise AssertionError('raw producer should not be called')
    monkeypatch.setattr(PayloadAwareScene,'_archive_last_scene',forbidden)
    x._archive_last_scene(88)
    assert [b['descriptor'].archive_version for b in x.banks]==[1,2,3]
    assert x.write_events[0]['raw_KV_bytes_written']==0 and x.archives==[]


def test_root_latest_releases_old_payload_before_copy_and_preserves_root_FIFO(monkeypatch):
    x=fixture('root_latest_fifo')
    def producer(self,frame):
        assert [b['descriptor'].archive_version for b in self.banks]==[1,3]
        self.banks.append({'descriptor':SimpleNamespace(archive_version=4),'owned_bytes':100})
        self.identity_roots[4]=2;self.archives.append({'KV_bytes':96})
    monkeypatch.setattr(PayloadAwareScene,'_archive_last_scene',producer)
    x._archive_last_scene(88)
    assert [b['descriptor'].archive_version for b in x.banks]==[1,4,3]
    assert x.write_events[-1]['removed_payload_version']==2
    assert x.root_replacements==1


def test_final_request_changes_only_after_the_shared_prefix():
    root=Path(__file__).resolve().parents[1]
    b,pb=write_origin_schedule(root,'w2_write_return_b');a,pa=write_origin_schedule(root,'w2_write_return_a')
    assert b[:-1]==a[:-1] and pb[0][:15]==pa[0][:15]
    assert a[-1]['prompt']==a[3]['prompt'] and len(pa[0])==16


def test_full_resolution_cohort_never_enables_gate_budget_scaling(tmp_path):
    import json
    from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'write_origin',tmp_path,tmp_path,tmp_path,20260913)
    assert len(rows)==8 and list(map(len,case_lane_indices(rows,2)))==[4,4]
    assert all('--scene-archive-scale-for-gate' not in x['cmd'] for x in rows)
    assert {x['method'] for x in rows}=={'native','append','root_latest_fifo','skip_derived'}
