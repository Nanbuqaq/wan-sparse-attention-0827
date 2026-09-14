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


def append_past_state_request(root,scenario,segments,prompts):
    """Privileged text-memory control on an independently valid updated prefix."""
    if scenario!='w2_state_red_toolbox_last_open':
        raise ValueError('past-state control is restricted to the qualified opening history')
    spec=json.loads((root/'configs/system/state_update_protocols.json').read_text())['protocols']['red_toolbox']
    appearance=spec['past_appearance_clause']
    update=[s for s in segments if s['role']=='open_requested_same_scene']
    if len(update)!=1 or not segments[1]['start_latent']<update[0]['start_latent']<segments[-1]['start_latent']:
        raise ValueError('one strictly past state request required')
    if appearance not in segments[1]['prompt']:raise ValueError('appearance is not in the source request')
    clause=appearance+' '+update[0]['prompt']
    if len(clause.encode())>1024:raise ValueError('bounded text memory exceeded')
    result=[dict(s) for s in segments];texts=[list(prompts[0])]
    result[-1]['prompt']+=' '+clause;result[-1]['role']='return_with_past_appearance_text'
    for i in range(result[-1]['start_latent']//8,len(texts[0])):texts[0][i]+=' '+clause
    audit=dict(clause=clause,clause_UTF8_bytes=len(clause.encode()),clause_sha256=hashlib.sha256(clause.encode()).hexdigest(),
        source_start_latent=segments[1]['start_latent'],past_request_start_latent=update[0]['start_latent'],
        return_start_latent=segments[-1]['start_latent'],past_requested_not_observed_state=True,
        generated_state_not_supplied_to_conditioner=True,privileged_known_object_and_request=True,
        autonomous_extraction_claim=False,changes_text_conditioning=True,raw_history_KV_required=False)
    return result,texts,audit


def append_past_pattern_text(root,scenario,segments,prompts):
    if scenario!='w2_state_pattern_tile_keep':raise ValueError('pattern text control is registered only for the tile')
    source=json.loads((root/'configs/system/state_update_protocols.json').read_text())['protocols']['pattern_tile']['source_prompt']
    candidates=[s for s in segments if s['prompt']==source]
    if len(candidates)!=1 or candidates[0]['start_latent']>=segments[-1]['start_latent']:
        raise ValueError('one strictly past source prompt required')
    if len(source.encode())>1024:raise ValueError('bounded past pattern text exceeded')
    result=[dict(s) for s in segments];texts=[list(prompts[0])]
    result[-1]['prompt']+=' '+source;result[-1]['role']='return_with_past_appearance_text'
    for i in range(result[-1]['start_latent']//8,len(texts[0])):texts[0][i]+=' '+source
    return result,texts,dict(clause=source,clause_UTF8_bytes=len(source.encode()),
        exact_past_prompt=True,generated_fill_or_edge_details_not_supplied=True,
        privileged_known_object_and_request=True,autonomous_extraction_claim=False,
        changes_text_conditioning=True)
