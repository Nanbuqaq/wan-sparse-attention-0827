import json
import pytest
from scripts.summarize_resident_output import summarize

def fixture(tmp_path):
 for pair in range(4):
  for mode in ('inline','thread'):
   for lane in range(4):
    p=tmp_path/f'lane{lane}'/f'pair{pair}_{mode}';p.mkdir(parents=True)
    d=dict(status='pass',measured=True,case=dict(id=f'pair{pair}_{mode}',mode=mode,measured=True,pair=pair),
      pixels=dict(frames=509,raw_RGB_sha256='rgb',first_packet_muxed_s=1),
      video_pipeline=dict(encoded_pixels=509,encode_mode=mode,scheduling='overlap_two_gpu',complete_s=10,records=[]),
      full_latent_equal_to_first_warmup=True,runner_commit='sha',assets_manifest_sha256='weights',seed=lane,
      latent_sha256='latent',physical_GPU_mapping=f'{2*lane},{2*lane+1}',gpu='test',
      monotonic_start_ns=lane*1_000_000_000,monotonic_delivery_ns=(lane+10)*1_000_000_000)
    (p/'summary.json').write_text(json.dumps(d))

def test_machine_throughput_uses_common_epoch_not_longest_lane(tmp_path):
 fixture(tmp_path);r,_=summarize(tmp_path)
 assert r['batches'][0]['batch_delivery_s']==13
 assert r['batches'][0]['aggregate_delivered_fps']==2036/13
 assert r['batches'][0]['start_skew_s']==3
 assert r['batches'][0]['common_active_window_s']==7

def test_missing_lane_or_changed_identity_rejected(tmp_path):
 fixture(tmp_path)
 p=tmp_path/'lane2/pair3_thread/summary.json';d=json.loads(p.read_text());d['seed']=999;p.write_text(json.dumps(d))
 with pytest.raises(AssertionError,match='identity'):summarize(tmp_path)
 p.unlink()
 with pytest.raises(FileNotFoundError):summarize(tmp_path)
