"""Native H3 first-frame input contract; no model imports or fixed aspect ratio."""
import hashlib,json,math

VERSION='native_H3_first_frame_I2VA_v1'
INSTRUCTION=('For the target video, at 0.00 seconds into the target video, '
             '<Picture 1> (from [Shot 1]) is fully referenced.')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def conditioning_digest(conditioning):
    # Worker-visible aliases can change between runs. Content, raster geometry,
    # native preprocessing, prompt and model assets determine the encoding.
    value=dict(conditioning,image={k:v for k,v in conditioning['image'].items() if k!='path'})
    return digest(value)


def canvas_size(width,height):
    if type(width) is not int or type(height) is not int or min(width,height)<=0:
        raise ValueError('Positive original raster dimensions are required')
    ratio=width/height
    if not .25<=ratio<=4:raise ValueError('Native H3 supports aspect ratios from1:4 to4:1')
    w,h=(768*ratio,768.) if ratio>=1 else (768.,768/ratio)
    if w*h>768*1344:
        scale=math.sqrt((768*1344)/(w*h));w,h=w*scale,h*scale
    return max(32,round(h/32)*32),max(32,round(w/32)*32)


def task(case_id,prompt,image,num_frames,seed,model_contract_sha256):
    if not isinstance(case_id,str) or not case_id or not isinstance(prompt,str) or not prompt.strip():
        raise ValueError('A case ID and verbatim benchmark prompt are required')
    if type(num_frames) is not int or num_frames<22 or num_frames%17!=5:
        raise ValueError('Choose an explicit native frame count17*n+5, at least22; no silent extension')
    if type(seed) is not int or seed<0:raise ValueError('Nonnegative explicit seed required')
    if set(['path','sha256','width','height'])-set(image):raise ValueError('Image path/hash/raster geometry missing')
    if len(image['sha256'])!=64 or any(c not in '0123456789abcdef' for c in image['sha256']):
        raise ValueError('A verified image SHA256 is required')
    if (not isinstance(model_contract_sha256,str) or len(model_contract_sha256)!=64 or
        any(c not in '0123456789abcdef' for c in model_contract_sha256)):
        raise ValueError('A frozen encoder/VAE/base asset contract SHA256 is required')
    height,width=canvas_size(image['width'],image['height'])
    conditioning=dict(protocol=VERSION,original_prompt=prompt,encoder_prompt=INSTRUCTION+'\n'+prompt,
                      prompt_protocol='native_I2VA_instruction_prefix_plus_verbatim_benchmark_prompt_v1',
                      image=image,canvas=dict(height=height,width=width,policy='native_first_image_short_edge768_area_cap32_multiple',
                                              operation='RGB_LANCZOS_resize_no_crop_no_EXIF_transpose',
                                              scale_x=width/image['width'],scale_y=height/image['height']),
                      anchors=['first'],model_contract_sha256=model_contract_sha256,conditioner_hidden_state_layer=50,keyframe_posterior_seed=42,
                      keyframe_posterior_rounding='float16_then_float32_then_native_normalization',keyframe_noise_aug=.999)
    value=dict(schema='h3_native_I2AV_task_v1',case_id=case_id,seed=seed,conditioning=conditioning,
               conditioning_sha256=conditioning_digest(conditioning),num_frames=num_frames,fps=24.,
               video_latent_frames=(num_frames-5)//17*5+2,audio_latent_frames=round(num_frames/24*40),
               duration_sec=num_frames/24,NFE=50,CFG=1.,video_shift=12.,audio_shift=3.,
               input_modality='I2AV',history_mode='native_single_clip_no_generated_history',
               decoder_policy='complete_clip_native_offline_AV_decode',source_time_admission=False,
               streaming_quality_claim=False,formal_memory_quality_claim=False)
    value['task_id']='i2av_'+digest(value)[:24]
    return value


def validate(value):
    if value.get('schema')!='h3_native_I2AV_task_v1':raise ValueError('Unknown I2AV task schema')
    cond=value['conditioning']
    original=task(value['case_id'],cond['original_prompt'],cond['image'],value['num_frames'],value['seed'],cond['model_contract_sha256'])
    for key,want in original.items():
        if value.get(key)!=want:raise ValueError('I2AV task/condition/clock changed: '+key)
    if any(key in value for key in ['source_video','source_audio','source_latents','reference_video']):
        raise ValueError('This adapter accepts exactly the declared image and text')
    return value


def validate_cache(value,cache):
    """Actual CPU tensors only; never construct or call an encoder/model."""
    import torch
    validate(value);cond=value['conditioning'];canvas=cond['canvas']
    if cache.get('conditioning_sha256')!=value['conditioning_sha256'] or cache.get('prompt')!=cond['encoder_prompt']:
        raise ValueError('Joint conditioning belongs to another image/prompt')
    if (cache.get('keyframe_anchors')!=['first'] or len(cache.get('keyframe_files',[]))!=1 or
        cache.get('source_image_sha256')!=cond['image']['sha256']):
        raise ValueError('I2AV keyframe source/anchor changed')
    if (cache.get('height'),cache.get('width'))!=(canvas['height'],canvas['width']):raise ValueError('Native canvas changed')
    embeds=cache['prompt_embeds'];tags=cache['text_token_tags'];conditions=cache['condition_latents']
    if (embeds.ndim!=2 or embeds.shape[1]!=5120 or embeds.dtype!=torch.bfloat16 or
        tags.shape!=(embeds.shape[0],) or tags.dtype!=torch.long or not torch.isfinite(embeds).all()):
        raise ValueError('Invalid joint Qwen hidden-state50 embeddings/tags')
    if set(tags.tolist())!={0,1}:raise ValueError('The joint presentation must retain both image and text tags')
    if cache.get('image_token_count')!=int((tags==0).sum())-2 or cache['image_token_count']<=0:
        raise ValueError('Image-pad token count differs from visual block tags')
    if len(conditions)!=1 or conditions[0].shape!=(1,24,1,canvas['height']//16,canvas['width']//16):
        raise ValueError('Invalid first-frame VAE condition geometry')
    if conditions[0].dtype!=torch.float32 or not torch.isfinite(conditions[0]).all():
        raise ValueError('Native VAE condition must be finite normalizedFP32')
    if cache.get('hidden_state_layer')!=50 or cache.get('keyframe_posterior_seed')!=42:
        raise ValueError('Native conditioning layer/posterior policy changed')
    return cache
