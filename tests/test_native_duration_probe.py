import torch
import pytest
from adapters.longlive_sparse.native_duration_probe import duration_geometry,stretch_away_schedule,duration_noise


def test_time_uses_real_native24fps_and_preserves_block_geometry():
    assert duration_geometry(728,128)['media_duration_s']==2909/24
    assert duration_geometry(3608,128)['media_duration_s']>600
    with pytest.raises(ValueError):duration_geometry(729,128)


def test_stretch_changes_only_away_hold_and_return_time():
    segments=[dict(start_latent=t,role=r) for t,r in zip((0,8,16,48),('initial','source','away','return_without_restatement'))]
    prompts=[['initial','The scene transitions. source','The scene transitions. away','away','away','away','The scene transitions. return','return']]
    extended,expanded=stretch_away_schedule(segments,prompts,96,64)
    assert extended[:3]==segments[:3] and extended[3]['start_latent']==80
    assert expanded[0][:6]==prompts[0][:6] and expanded[0][10:]==prompts[0][6:]
    assert expanded[0][6:10]==['away']*4
    assert sum(p.startswith('The scene transitions. ') for p in expanded[0])==3
    assert segments[3]['start_latent']==48


def test_base_noise_and_native_global_rng_are_preserved_and_tails_are_prefix_stable():
    torch.manual_seed(61);reference=torch.randn(1,64,2,2,2,dtype=torch.bfloat16);expected_state=torch.get_rng_state()
    torch.manual_seed(61);short=duration_noise((1,128,2,2,2),base_length=64,seed=61,device='cpu')
    assert torch.equal(torch.get_rng_state(),expected_state)
    assert torch.equal(short[:,:64],reference)
    torch.manual_seed(61);long=duration_noise((1,192,2,2,2),base_length=64,seed=61,device='cpu')
    assert torch.equal(short,long[:,:128]) and torch.equal(torch.get_rng_state(),expected_state)
    torch.manual_seed(61);base=duration_noise((1,64,2,2,2),base_length=64,seed=61,device='cpu')
    assert torch.equal(base,reference)
