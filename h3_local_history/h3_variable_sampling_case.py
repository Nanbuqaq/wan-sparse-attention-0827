"""Merge contiguous retained blocks of an annotation for operational comparison."""
import copy


def annotation_variable_cases(cases):
    result=copy.deepcopy(cases)
    for case in result:
        merged=[]
        for row in case['chunks']:
            if merged and merged[-1]['parent_window_id']==row['parent_window_id'] and merged[-1]['video_latent_end']==row['video_latent_start']:
                previous=merged[-1]
                assert previous['prompt_path']==row['prompt_path']
                assert previous['audio_latent_end']==row['audio_latent_start']
                previous['video_latent_end']=row['video_latent_end']
                previous['audio_latent_end']=row['audio_latent_end']
                previous['source_target_end_sec']=row['source_target_end_sec']
                previous['raw_video_end_frame_index']=row['raw_video_end_frame_index']
                previous['audio_valid_latent_count']+=row['audio_valid_latent_count']
                previous['member_window_ids'].append(row['window_id'])
            else:
                merged.append(copy.deepcopy(row))
                merged[-1]['member_window_ids']=[row['window_id']]
            merged[-1]['window_id']=merged[-1]['parent_window_id']+':variable_common_retained_span'
            merged[-1]['chunk_kind']='annotation_variable_common_retained_span'
        case['chunks']=merged
        case['sampling_protocol']='annotation_variable_common_retained_span'
        case['original_annotation_boundary_expansion_used']=False
    return result


def decoder_chunks(chunk,start):
    count=chunk.shape[2];offset=0
    if start==0:
        if count<7 or (count-2)%5:raise ValueError('Native first span requires2+5*n latents.')
        yield chunk[:,:,:7];offset=7
    elif count%5:
        raise ValueError('Native later span must contain complete5-latent groups.')
    while offset<count:
        yield chunk[:,:,offset:offset+5];offset+=5
