"""Frozen intervention on discarded-away content; never a retrieval label."""
from copy import deepcopy
import hashlib

PENDULUM = ('A close-up of a single brushed steel pendulum swinging slowly from side to side '
            'against a plain gray background. The metal sphere and thin suspension wire '
            'fill the frame in soft realistic studio lighting.')


def replace_away_content(segments, prompts, control):
    if control != 'pendulum':
        raise ValueError('unknown frozen away content control')
    changed = deepcopy(segments)
    output = deepcopy(prompts)
    away = [i for i, segment in enumerate(changed) if segment['role'] == 'away']
    if len(away) != 1 or len(output) != 1:
        raise ValueError('requires one away phase and one sample')
    index = away[0]
    if not 0 < index < len(changed)-1:
        raise ValueError('away must be between source and return')
    first = changed[index]['start_latent']//8
    end = changed[index+1]['start_latent']//8
    old = changed[index]['prompt']
    changed[index]['prompt'] = PENDULUM
    for block in range(first, end):
        prefix = 'The scene transitions. ' if output[0][block].startswith('The scene transitions. ') else ''
        if output[0][block].removeprefix(prefix) != old:
            raise ValueError('away prompts differ from declared phase')
        output[0][block] = prefix + PENDULUM
    audit = dict(control=control,changed_blocks=list(range(first,end)),
                 old_prompt_sha256=hashlib.sha256(old.encode()).hexdigest(),
                 new_prompt_sha256=hashlib.sha256(PENDULUM.encode()).hexdigest(),
                 source_and_return_prompts_unchanged=True,
                 changes_only_away_request=True,not_a_memory_method=True)
    return changed, output, audit
