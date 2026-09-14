"""Cheap, privileged past-text control; no automatic identity extraction claim."""
import hashlib
import json


def append_past_appearance(root,scenario,segments,prompts):
    task,operation=scenario.removeprefix('w2_state_').rsplit('_',1)
    if task not in ('silver_case','red_toolbox') or operation not in ('keep','close'):
        raise ValueError('past-appearance control is restricted to registered state requests')
    spec=json.loads((root/'configs/system/state_update_protocols.json').read_text())['protocols'][task]
    clause=spec['past_appearance_clause']
    if clause not in spec['source_prompt'] or len(clause.encode())>512:
        raise ValueError('appearance must be an exact bounded span of the past source request')
    sources=[s for s in segments if s['prompt']==spec['source_prompt']]
    if len(sources)!=1 or sources[0]['start_latent']>=segments[-1]['start_latent']:
        raise ValueError('appearance clause is not from a strictly past source request')
    result=[dict(s) for s in segments];texts=[list(prompts[0])]
    result[-1]['prompt']+=' '+clause
    result[-1]['role']='return_with_past_appearance_text'
    for i in range(result[-1]['start_latent']//8,len(texts[0])):texts[0][i]+=' '+clause
    audit=dict(source_start_latent=sources[0]['start_latent'],return_start_latent=result[-1]['start_latent'],
        clause=clause,clause_UTF8_bytes=len(clause.encode()),clause_sha256=hashlib.sha256(clause.encode()).hexdigest(),
        exact_past_prompt_span=True,manually_selected_identity_clause=True,autonomous_extraction_claim=False,
        observed_identity_or_state_certificate=False,current_state_request_unchanged=True,
        changes_text_conditioning=True,raw_history_KV_required=False)
    return result,texts,audit
