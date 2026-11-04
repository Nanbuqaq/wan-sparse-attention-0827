"""Validate dispatch fields for the shared A/B/C inference contract on CPU."""


def validate_shared_request(request):
    strategy=request.get('strategy')
    spec=request.get('weights_spec',{})
    if strategy not in ('A','B','C') or spec.get('strategy')!=strategy:
        raise ValueError('Sampler strategy and checkpoint strategy differ')
    if request.get('final_training_strategy',strategy)!=strategy:
        raise ValueError('Training strategy label and actual sampler strategy differ')
    if spec.get('attention_contract')!='h3_per_block_text_clean_noisy_fixed_video_KV_v1':
        raise ValueError('Shared fixed-KV worker requires its explicit attention contract')
    if request.get('NFE')!=50 or request.get('FPS')!=24 or request.get('history_recomputation_allowed') is not False:
        raise ValueError('Shared A/B/C sampling requires NFE50, 24FPS, immutable history')
    blocks=request.get('blocks',[])
    if (not blocks or request.get('actual_frames')!=17*len(blocks)
            or request.get('generation_video_latents_per_chunk')!=5):
        raise ValueError('Frame count does not match the trained fixed5 contract')
    prompts=request.get('prompts',[])
    if not prompts or len(request.get('prompt_paths',[]))!=len(prompts):
        raise ValueError('Prompt texts and embedding paths differ')
    for i,row in enumerate(blocks):
        if (row.get('block')!=i or row.get('end_frame')-row.get('start_frame')!=17
                or row.get('start_frame')!=i*17):
            raise ValueError('Invalid shared fixed5 block geometry')
        if not 0<=row['prompt_index']<len(prompts):
            raise ValueError('Prompt index is out of range')
    return dict(strategy=strategy,effective_text_reads_history=strategy=='B',NFE=50,
                FPS=24,chunks=len(blocks),frames=17*len(blocks),
                attention_contract=spec['attention_contract'])
