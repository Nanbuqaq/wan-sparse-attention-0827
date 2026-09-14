"""Frozen current-motion counterexample; source requests are unchanged."""
from copy import deepcopy

TURN=('The entire open toolbox slowly turns through a quarter turn on a rotating display base, '
      'continuing the turn smoothly across the shot while the camera remains fixed. '
      'Its lid stays fully raised and the handle and paint scratches stay attached to the same surfaces.')


def append_quarter_turn(scenario,segments,prompts):
    if scenario!='w2_state_red_toolbox_keep':raise ValueError('quarter-turn feasibility freezes the red open toolbox')
    changed=deepcopy(segments);output=deepcopy(prompts)
    last=changed[-1];first=last['start_latent']//8
    if last['role']!='return_without_restatement':raise ValueError('unexpected motion return boundary')
    last['prompt']+=' '+TURN
    for i in range(first,len(output[0])):output[0][i]+=' '+TURN
    return changed,output,dict(kind='fixed_quarter_turn_counterexample',changed_blocks=list(range(first,len(output[0]))),
        source_requests_unchanged=True,not_a_memory_method=True)
