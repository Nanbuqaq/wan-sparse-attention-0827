"""RetinaFace/ArcFace primitives on InferHub; no automatic identity pass."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import av
import cv2
import numpy as np
import torch

from h3_face_return_contract import GROUPS, cosine, digest, requested_samples, validate_task


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda:source.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def sample_frames(task):
    samples=sorted(requested_samples(task),key=lambda x:x[2]);found={}
    with av.open(task['video']) as container:
        container.streams.video[0].codec_context.thread_count=1
        for frame in container.decode(video=0):
            pts=float(frame.pts*frame.time_base)
            while samples and pts+1e-8>=samples[0][2]:
                name,index,requested,end=samples.pop(0)
                found[name,index]=(dict(group=name,sample_index=index,requested_sec=requested,actual_pts_sec=pts,
                    frame_in_requested_window=pts<end-1e-8,width=frame.width,height=frame.height),frame.to_ndarray(format='rgb24'))
            if not samples:break
    for name,index,requested,end in samples:
        found[name,index]=(dict(group=name,sample_index=index,requested_sec=requested,actual_pts_sec=None,
                              frame_in_requested_window=False,status='missing_decoded_frame'),None)
    return found


def initialize(assets):
    from retinaface.predict_single import Model
    detector=Model(max_size=2048,device='cuda')
    detector.load_state_dict(torch.load(assets/'retinaface_resnet50_state.pt',map_location='cpu',weights_only=True));detector.eval()
    source=Path(__file__).parent/'metric_arcface_resnet.py'
    spec=importlib.util.spec_from_file_location('h3_metric_arcface',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    arc=module.resnet_face18(use_se=False)
    state=torch.load(assets/'arcface_resnet18.pth',map_location='cpu',weights_only=True)
    arc.load_state_dict({k.replace('module.',''):v for k,v in state.items()},strict=True)
    return detector,arc.cuda().eval()


def score(task,directory,detector,arc,weight_hashes,driver_hashes):
    validate_task(task)
    started=time.perf_counter()
    if sha(task['video'])!=task['video_sha256']:raise ValueError('Frozen face diagnostic media changed')
    frames=sample_frames(task);entries=[];images=[];valid_entries=[]
    for key,(entry,rgb) in frames.items():
        if rgb is None or not entry['frame_in_requested_window']:
            entry.setdefault('status','decoded_after_requested_window');entries.append(entry);continue
        detections=detector.predict_jsons(rgb,confidence_threshold=.7,nms_threshold=.4)
        faces=[d for d in detections if len(d['bbox'])==4]
        eligible=[d for d in faces if min(d['bbox'][2]-d['bbox'][0],d['bbox'][3]-d['bbox'][1])>=32]
        entry.update(detections=faces,detected_faces=len(faces),eligible_faces=len(eligible),feature_available=False)
        if len(eligible)!=1:
            entry['status']='multiple_eligible_faces' if len(eligible)>1 else ('face_too_small' if faces else 'no_detected_face')
            entries.append(entry);continue
        box=eligible[0]['bbox'];x1,y1,x2,y2=[int(v) for v in box]
        x1,x2=max(0,x1),min(rgb.shape[1],x2);y1,y2=max(0,y1),min(rgb.shape[0],y2)
        crop=rgb[y1:y2,x1:x2]
        if not crop.size:
            entry['status']='empty_clamped_crop';entries.append(entry);continue
        gray=cv2.cvtColor(cv2.resize(crop,(128,128)),cv2.COLOR_RGB2GRAY)
        images.extend([gray,np.fliplr(gray)]);valid_entries.append(entry)
        name=entry['group']+'_'+str(entry['sample_index'])+'_face.png'
        cv2.imwrite(str(directory/name),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
        entry.update(status='pending_feature',crop=name,crop_sha256=sha(directory/name),
                     crop_bbox_pixels=[x1,y1,x2,y2],crop_width=x2-x1,crop_height=y2-y1)
        entries.append(entry)
    features={};feature_rows=[];feature_names=[]
    if images:
        batch=torch.from_numpy((np.stack(images)[:,None].astype(np.float32)-127.5)/127.5).cuda()
        with torch.inference_mode():output=arc(batch).float().cpu().numpy()
        if output.shape[0]!=len(images) or not np.isfinite(output).all():raise ValueError('Invalid ArcFace feature batch')
        for i,entry in enumerate(valid_entries):
            feature=np.concatenate([output[2*i],output[2*i+1]])
            norm=float(np.linalg.norm(feature))
            if not np.isfinite(norm) or norm<=1e-12:raise ValueError('Invalid face feature norm')
            key=entry['group']+'_'+str(entry['sample_index'])
            features[key]=feature.tolist();feature_rows.append(feature);feature_names.append(key)
            entry.update(status='feature_available',feature_available=True,feature_norm=norm,feature_key=key)
    comparisons={}
    ref=[v for k,v in features.items() if k.startswith('reference_anchor_')]
    for name in GROUPS:
        values=[v for k,v in features.items() if k.startswith(name+'_')]
        pairs=([cosine(a,b) for i,a in enumerate(values) for b in values[i+1:]] if name=='reference_anchor'
               else [cosine(a,b) for a in ref for b in values])
        enough=(len(values)>=2 if name=='reference_anchor' else len(ref)>=2 and len(values)>=2)
        comparisons[name]=dict(valid_features=len(values),requested_samples=3 if task['windows'][name] is not None else 0,
                              pair_cosines=pairs,median_cosine=statistics.median(pairs) if enough else None,
                              status='available_uncalibrated' if enough else 'insufficient_valid_faces')
    target=directory/'FACE_FEATURES.npz'
    with target.with_suffix('.npz.tmp').open('wb') as handle:
        np.savez_compressed(handle,keys=np.asarray(feature_names,dtype='U64'),
                            features=np.asarray(feature_rows,dtype=np.float32).reshape(len(feature_rows),-1) if feature_rows else np.zeros((0,1024),np.float32))
    target.with_suffix('.npz.tmp').replace(target)
    value=dict(schema='face_return_diagnostic_result_v2',status='complete',task_id=task['task_id'],task_sha256=digest(task),
               video=task['video'],video_sha256=task['video_sha256'],windows=task['windows'],associations=task['associations'],
               comparisons=comparisons,frame_detections=entries,feature_archive=str(target),feature_archive_sha256=sha(target),
               preprocessing='clamped bbox; RGB to gray;128x128;original and horizontal flip;[-1,1];concatenate embeddings;no landmark alignment',
               minimum_face_pixels=32,retina_confidence_threshold=.7,retina_NMS_threshold=.4,
               weight_hashes=weight_hashes,driver_hashes=driver_hashes,identity_threshold_adopted=False,identity_pass=None,
               event_eligibility='not_inferred_from_face_detection_or_cosine',formal_memory_admitted=False,
               elapsed_seconds=time.perf_counter()-started,hardware=torch.cuda.get_device_name(),
               hardware_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
    atomic(directory/'FACE_RESULT.json',value)


def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    stage=Path(os.environ['INFER_WEIGHTS_DIR']);out=Path(os.environ['INFER_OUTPUT_DIR'])
    manifest=json.loads((stage/'TASKS.json').read_text())
    if digest(manifest['tasks'])!=manifest['tasks_sha256']:raise ValueError('Face manifest changed')
    lane=int(os.environ['FACE_LANE']);lanes=int(os.environ['FACE_LANES'])
    if not 4<=lanes<=8 or not 0<=lane<lanes:raise ValueError('Use4..8 independent metric lanes')
    hashes={name:sha(stage/'assets'/name) for name in manifest['weight_hashes']}
    if hashes!=manifest['weight_hashes']:raise ValueError('Face model asset changed')
    driver_hashes={name:sha(Path(__file__).parent/name) for name in manifest['driver_hashes']}
    if driver_hashes!=manifest['driver_hashes']:raise ValueError('Published face metric code differs')
    assigned=manifest['tasks'][lane::lanes]
    if not assigned:raise ValueError('No padding-only GPU lanes')
    loaded=time.perf_counter();detector,arc=initialize(stage/'assets');load_seconds=time.perf_counter()-loaded
    status=[]
    for task in assigned:
        directory=out/task['task_id'];directory.mkdir(exist_ok=True)
        try:
            score(task,directory,detector,arc,hashes,driver_hashes)
            status.append(dict(task_id=task['task_id'],status='complete'));print('FACE_COMPLETE',task['task_id'],flush=True)
        except Exception as error:
            value=dict(task_id=task['task_id'],status='error',error=str(error),traceback=traceback.format_exc())
            atomic(directory/'FACE_ERROR.json',value);status.append(value)
    atomic(out/f'FACE_LANE{lane}.json',dict(lane=lane,lanes=lanes,load_seconds=load_seconds,results=status))
    if any(v['status']!='complete' for v in status):raise RuntimeError('Some face diagnostic tasks failed; inspect per-task errors')


if __name__=='__main__':main()
