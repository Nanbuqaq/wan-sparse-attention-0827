"""CPU planning after source-time24FPS resampling; no media or model changes."""
import argparse
import json
import math


def resolve_codec_mode(sampling_contract,training_metadata=None,requested=None):
    modes={'continuous_sequence':'continuous','native_shot':'independent_shots','native_stock':'independent_shots'}
    if sampling_contract not in modes:raise ValueError('Explicit sampler codec contract required')
    inferred=modes[sampling_contract]
    if requested is not None and requested!=inferred:raise ValueError('Sampler and declared codec boundary differ')
    metadata=training_metadata or {};args=metadata.get('args') or {}
    trained=metadata.get('video_codec_contract') or args.get('video_codec_contract')
    if trained is None:
        policy=args.get('target_policy')
        if policy=='teacher_native_shot_startup7_fixed5':trained='independent_shots'
        elif policy in ['strict_fixed5_startup7','fixed5_within_chunk_cut_startup7']:trained='continuous'
    if trained is not None and trained!=inferred:
        raise ValueError('Checkpoint codec contract differs: choose the matching sampling entry')
    return inferred


def independent_shot_plan(valid_frames,fps=24):
    if type(valid_frames) is not int or valid_frames<1 or fps!=24:
        raise ValueError('Need a positive sampled-frame count at the declared24FPS clock')
    steps=max(0,math.ceil((valid_frames-22)/17))
    encoded_frames=22+17*steps;latents=7+5*steps
    return dict(schema='h3_independent_codec_shot_v1',fps=24,valid_source_frames=valid_frames,
        encode_input_frames=encoded_frames,encoded_video_latents=latents,
        right_padding_frames=encoded_frames-valid_frames,padding_policy='repeat_last_frame_of_this_shot_only',
        source_frame_valid_mask=[True]*valid_frames+[False]*(encoded_frames-valid_frames),
        decoder_boundary='finish_previous_codec_shot_then_start_new_empty_buffer',
        model_video_history_reset=False,model_generation_chunk_size='declared_by_checkpoint_not_fixed_by_codec',
        current_native_7_then5_partition=[7]+[5]*steps,
        latent_loss_mask='derive_from_actual_encoder_support_and_valid_source_mask_not_integer_rounding',
        padding_has_real_source_PTS=False,formal_causal_admission=False,
        video_encoder_future_support='requires_explicit_audit',audio_policy='preserve_source_window_and_declared_lookahead')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=int,required=True)
    print(json.dumps(independent_shot_plan(p.parse_args().frames),indent=2))
