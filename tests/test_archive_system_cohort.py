import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices


def test_archive_factors_have_matched_source_controls_and_reversed_order(tmp_path):
    root=Path(__file__).resolve().parents[1];spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'archive_system',tmp_path,tmp_path,tmp_path,20261023)
    lanes=case_lane_indices(rows,2);assert list(map(len,lanes))==[12,12]
    for lane,ids in enumerate(lanes):
        items=[rows[i] for i in ids]
        assert [r['method'] for r in items]==['native_start','sync_r0','dedup_r0','serial_r0','async_device_r0','async_stream_r0','async_stream_r1','async_device_r1','serial_r1','dedup_r1','sync_r1','native_end']
        for row in items:
            assert '--archive-digest' not in row['cmd'] and '--source-clean-cache-witness' not in row['cmd']
            assert ('--state-quarter-turn' in row['cmd'])==(lane==0)
            if row['method'].startswith('native'):assert '--archive-write-backend' not in row['cmd']
            else:assert row['source_readiness_scope']==('generation' if row['method'].startswith('async_stream') else 'device')
        assert items[-1]['reference_case_index']==ids[0]
        assert all(r['reference_case_index']==ids[1] for r in items[2:-1])
