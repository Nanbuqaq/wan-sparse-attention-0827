"""Official-worker image/text+VAE encoding, keeping the native H3 presentation."""
import argparse,gc,hashlib,inspect,json,os,time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def verify_assets(root,expected):
    file=root/'MODEL_ASSET_CONTRACT.json'
    if sha(file)!=expected:raise ValueError('Model asset contract changed')
    value=json.loads(file.read_text())
    for item in value['files']:
        path=root/item['relative'];stat=path.stat()
        # Full SHA verification is done by CPU preparation before submission;
        # read-only worker inputs must retain those exact file identities.
        if stat.st_size!=item['bytes'] or stat.st_mtime_ns!=item['mtime_ns']:
            raise ValueError('Prepared model asset changed: '+item['relative'])
    return value


def encode_joint(model,processor,ids,vision,device):
    import torch
    ids_tensor=torch.tensor([ids],dtype=torch.long,device=device)
    kwargs={k:v.to(device=device,dtype=model.dtype) if k.startswith('pixel_') else v.to(device) for k,v in vision.items()}
    if 'mm_token_type_ids' in inspect.signature(model.model.forward).parameters:
        method=getattr(processor,'create_mm_token_type_ids',None)
        if method is None:raise ValueError('Qwen processor does not match its explicit multimodal API')
        kwargs['mm_token_type_ids']=torch.tensor(method([ids]),dtype=torch.long,device=device)
    with torch.inference_mode():
        result=model.model(input_ids=ids_tensor,attention_mask=torch.ones_like(ids_tensor),use_cache=False,
                           output_hidden_states=True,**kwargs)
    if len(result.hidden_states)<=50:raise ValueError('H3 requires Qwen hidden state50')
    out=result.hidden_states[50][0].to(torch.bfloat16).cpu()
    if out.shape!=(len(ids),5120) or not torch.isfinite(out).all():raise ValueError('Invalid image/text hidden states')
    return out


def require_complete_weights(info):
    problems={k:info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'] if info.get(k)}
    if problems:raise ValueError('Native conditioner checkpoint is not completely loaded: '+str(problems))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);a=parser.parse_args()
    if not os.environ.get('INFER_CODE_DIR') or not os.environ.get('INFER_WEIGHTS_DIR'):
        raise RuntimeError('Image encoder must run through an official InferHub job')
    import torch
    from train_h3_v2_local_history_ddp import bootstrap
    bootstrap()
    from h3_i2av_contract import validate,validate_cache
    from src.inference import encode_keyframes as native
    from transformers import Qwen3VLForConditionalGeneration,Qwen3VLProcessor
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from PIL import Image
    import numpy as np
    if not torch.cuda.is_available():raise RuntimeError('Actual conditioning forward requires assigned H GPU')
    torch.set_num_threads(2);torch.set_grad_enabled(False);device=torch.device('cuda',0);torch.cuda.set_device(device)
    manifest=json.loads(Path(a.manifest).read_text());root=Path(os.environ['INFER_WEIGHTS_DIR']);out=Path(os.environ['INFER_OUTPUT_DIR'])
    verify_assets(root,manifest['model_contract_sha256']);tasks=manifest['tasks']
    if len({t['conditioning_sha256'] for t in tasks})!=len(tasks):raise ValueError('Encode each unique image/text condition only once')
    pending=[]
    for task in tasks:
        validate(task);cond=task['conditioning'];image=cond['image'];folder=out/task['conditioning_sha256'];folder.mkdir(exist_ok=True)
        if sha(image['path'])!=image['sha256']:raise ValueError('Original I2AV image changed')
        if task['conditioning']['model_contract_sha256']!=manifest['model_contract_sha256']:raise ValueError('Encoder model source changed')
        done=folder/'ENCODE_RESULT.json'
        if done.exists():
            r=json.loads(done.read_text());cache=folder/'CONDITIONING.pt'
            if r['conditioning_sha256']!=task['conditioning_sha256'] or sha(cache)!=r['cache_sha256']:
                raise ValueError('Completed conditioning receipt differs')
            validate_cache(task,torch.load(cache,map_location='cpu',weights_only=True));continue
        pending.append((task,folder))
    processor=None;model=None
    for task,folder in pending:
        text_file=folder/'JOINT_TEXT.pt';cond=task['conditioning']
        if text_file.exists():
            old=torch.load(text_file,map_location='cpu',weights_only=True)
            if old['conditioning_sha256']!=task['conditioning_sha256']:raise ValueError('Partial image/text condition changed')
            continue
        if model is None:
            processor=Qwen3VLProcessor.from_pretrained(root/'native_text_source',subfolder='processor',local_files_only=True)
            model,info=Qwen3VLForConditionalGeneration.from_pretrained(root/'native_text_source',subfolder='text_encoder',
                    dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
            require_complete_weights(info);model=model.to(device).eval().requires_grad_(False)
        if sha(cond['image']['path'])!=cond['image']['sha256']:raise ValueError('Image changed before joint encoding')
        with Image.open(cond['image']['path']) as source:keyframes,height,width=native.put_on_canvas([source])
        if (height,width)!=(cond['canvas']['height'],cond['canvas']['width']):raise ValueError('Native canvas and frozen case disagree')
        ids,tags,vision=native.build_presentation(processor,cond['encoder_prompt'],keyframes)
        image_count=int(vision['image_grid_thw'].prod(-1).sum())//processor.image_processor.merge_size**2
        if sum(v==native.VIDEO_TAG for v in tags)!=image_count+2:raise ValueError('Native image presentation token counts differ')
        start=time.monotonic();embeds=encode_joint(model,processor,ids,vision,device);torch.cuda.synchronize()
        value=dict(conditioning_sha256=task['conditioning_sha256'],prompt=cond['encoder_prompt'],prompt_embeds=embeds,
                   text_token_tags=torch.tensor(tags,dtype=torch.long),image_token_count=image_count,
                   joint_encoder_seconds=time.monotonic()-start,hidden_state_layer=50)
        temporary=text_file.with_suffix('.partial');torch.save(value,temporary);temporary.replace(text_file)
        keyframes[0].save(folder/'PREPARED_IMAGE.png')
        print(json.dumps(dict(event='I2AV_joint_text_complete',case=task['case_id'],tokens=len(ids),image_tokens=image_count)),flush=True)
    del model,processor;gc.collect();torch.cuda.empty_cache();vae=None
    for task,folder in pending:
        if vae is None:
            vae,info=AutoencoderKLMiniMaxH3.from_pretrained(root/'h3-base',subfolder='vae',local_files_only=True,output_loading_info=True)
            require_complete_weights(info);vae=vae.to(device).eval()
        cond=task['conditioning']
        if sha(cond['image']['path'])!=cond['image']['sha256']:raise ValueError('Image changed before native VAE encoding')
        with Image.open(cond['image']['path']) as source:keyframes,height,width=native.put_on_canvas([source])
        if not (folder/'PREPARED_IMAGE.png').exists():keyframes[0].save(folder/'PREPARED_IMAGE.png')
        with torch.inference_mode():
            pixels=torch.from_numpy(np.array(keyframes[0])).to(device).permute(2,0,1)[None,:,None]
            condition=encode_vae_condition(vae,pixels,native.PIXEL_MEAN,native.PIXEL_STD,42).cpu()
        joint=torch.load(folder/'JOINT_TEXT.pt',map_location='cpu',weights_only=True)
        joint.update(keyframe_anchors=['first'],keyframe_files=[cond['image']['path']],condition_latents=[condition],
                     source_image_sha256=cond['image']['sha256'],height=height,width=width,keyframe_posterior_seed=42,conditioning='joint_image_and_text_native_I2VA')
        validate_cache(task,joint);path=folder/'CONDITIONING.pt';temporary=path.with_suffix('.partial')
        torch.save(joint,temporary);temporary.replace(path)
        atomic_json(folder/'ENCODE_RESULT.json',dict(status='native_I2AV_condition_encoded',conditioning_sha256=task['conditioning_sha256'],
                    cache=str(path),cache_sha256=sha(path),image_sha256=cond['image']['sha256'],model_contract_sha256=manifest['model_contract_sha256'],
                    encoder_worker_source_sha256=sha(__file__),prepared_image_sha256=sha(folder/'PREPARED_IMAGE.png'),
                    native_encoder_source_sha256=sha(inspect.getfile(native)),GPU=torch.cuda.get_device_name(device),torch_version=torch.__version__,
                    transformers_version=__import__('transformers').__version__,complete_conditioner_weights_loaded=True,
                    formal_memory_quality_claim=False))
        print(json.dumps(dict(event='I2AV_condition_complete',case=task['case_id'])),flush=True)
    atomic_json(out/('ENCODER_COMPLETE_'+str(manifest['shard'])+'.json'),dict(expected=len(tasks),complete=len(tasks),model_contract_sha256=manifest['model_contract_sha256']))

if __name__=='__main__':main()
