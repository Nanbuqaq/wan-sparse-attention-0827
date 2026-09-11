"""Reusable source-only SAM2 extraction primitive; no online routing or queue yet."""
import hashlib
import time

import numpy as np
import torch

from scripts.derive_native_source_component_box import component_box
from scripts.probe_native_source_foreground import source_token_masks


class CachedSourceGeometry:
    def __init__(self,checkpoint,*,device='cuda:0',compact_return=False,mask_stride=1):
        if mask_stride not in (1,32):raise ValueError('registered mask schedules are all frames or causal hold-first')
        self.mask_stride=mask_stride
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from scripts.build_sam2_oracle_masks import EXPECTED_CHECKPOINT_SHA,sha256
        verify_started=time.perf_counter();self.checkpoint_sha=sha256(checkpoint)
        self.checkpoint_verify_s=time.perf_counter()-verify_started
        if self.checkpoint_sha!=EXPECTED_CHECKPOINT_SHA:raise ValueError('SAM2 source lock differs')
        self.device=torch.device(device);self.stream=torch.cuda.Stream(device=self.device)
        began=time.perf_counter()
        # Construction is outside generation; preserve every visible generator state.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.cuda.device(self.device),torch.cuda.stream(self.stream):
            self.model=build_sam2('configs/sam2/sam2_hiera_l.yaml',str(checkpoint),device=str(self.device),apply_postprocessing=False)
            self.model.eval().requires_grad_(False)
            self.automatic=SAM2AutomaticMaskGenerator(self.model,points_per_side=16,points_per_batch=32,
                pred_iou_thresh=.8,stability_score_thresh=.95,crop_n_layers=0,min_mask_region_area=0,output_mode='binary_mask')
            self.predictor=SAM2ImagePredictor(self.model)
        self.compact_return=compact_return
        if compact_return:
            from .compact_source_mask import verify_predictor_source
            verify_predictor_source(self.predictor)
        self.stream.synchronize();self.load_s=time.perf_counter()-began
        self.model_tensor_bytes=sum(t.numel()*t.element_size() for t in list(self.model.parameters())+list(self.model.buffers()))
        self.calls=0

    @torch.inference_mode()
    def extract(self,window,*,component_policy='mutual_geometry',keep_pixel_masks=True):
        pixels=window['pixels']
        if pixels.device.type!='cpu' or pixels.dtype!=torch.uint8 or pixels.ndim!=4 or pixels.shape[-1]!=3:
            raise ValueError('owned canonical CPU RGB source window required')
        began=time.perf_counter();before=hashlib.sha256(memoryview(pixels.numpy())).hexdigest()
        if before!=window['raw_pixel_bytes_sha256']:raise ValueError('source input ownership hash differs')
        height,width=pixels.shape[1:3];records=[];output=np.empty((len(pixels),height,width),dtype=np.bool_)
        try:
            with torch.cuda.device(self.device),torch.cuda.stream(self.stream),torch.autocast('cuda',dtype=torch.bfloat16):
                started=time.perf_counter();proposals=self.automatic.generate(pixels[0].numpy());self.stream.synchronize()
                proposal_s=time.perf_counter()-started;meta=[]
                for i,p in enumerate(proposals):
                    mask=np.asarray(p['segmentation'],dtype=np.bool_)
                    token=mask.reshape(height//32,32,width//32,32).any(axis=(1,3))
                    meta.append(dict(id=i,bbox=p['bbox'],pixel_area=int(mask.sum()),spatial_tokens=int(token.sum())))
                del proposals
                started=time.perf_counter();box,selected,components=component_box(meta,width=width,height=height,policy=component_policy)
                box_s=time.perf_counter()-started;started=time.perf_counter();returned_bytes=0
                for i,frame in enumerate(pixels):
                    if i%self.mask_stride:
                        output[i]=output[i-1]
                        records.append(dict(records[-1],pixel_frame=window['pixel_start']+i))
                        continue
                    self.predictor.set_image(frame.numpy())
                    if self.compact_return:
                        from .compact_source_mask import predict_selected_mask
                        selected_mask,scores,chosen=predict_selected_mask(self.predictor,box)
                        output[i]=selected_mask
                        returned_bytes+=selected_mask.nbytes+scores.nbytes
                    else:
                        masks,scores,logits=self.predictor.predict(box=np.asarray(box,dtype=np.float32),multimask_output=True)
                        chosen=int(np.argmax(scores));output[i]=np.asarray(masks[chosen],dtype=np.bool_)
                        returned_bytes+=masks.nbytes+scores.nbytes+logits.nbytes
                    records.append(dict(pixel_frame=window['pixel_start']+i,segmented_pixel_frame=window['pixel_start']+i,
                        chosen=chosen,scores=np.asarray(scores).tolist()))
                self.stream.synchronize();mask_s=time.perf_counter()-started
        finally:
            self.automatic.predictor.reset_predictor();self.predictor.reset_predictor()
        if hashlib.sha256(memoryview(pixels.numpy())).hexdigest()!=before:raise RuntimeError('geometry model mutated source pixels')
        tokens=source_token_masks(output,window['source_start']);indices=np.flatnonzero(tokens.reshape(-1)).astype(np.int64)
        self.calls+=1
        result=dict(status='mask_ready',archive_version=window['archive_version'],source_start=window['source_start'],
            source_end=window['source_end'],source_latent_sha256=window['source_latent_sha256'],
            source_raw_pixel_sha256=before,pixel_input_kind='raw_stream_before_codec',bbox=box,
            component_policy=component_policy,selected_component_ids=selected,components=components,
            indices=torch.from_numpy(indices),token_masks=torch.from_numpy(tokens),records=records,
            all_masks_nonempty=bool(output.reshape(len(pixels),-1).any(1).all()),
            proposal_s=proposal_s,component_CPU_s=box_s,mask_s=mask_s,total_extract_wall_s=time.perf_counter()-began,
            predictor_returned_CPU_array_bytes=returned_bytes,source_pixels_unchanged=True,
            compact_selected_boolean_return=self.compact_return,
            mask_stride=self.mask_stride,segmented_frames=len({r['segmented_pixel_frame'] for r in records}),
            complete_online_H2D_or_queue_cost_not_measured=True,model_reused=True)
        if keep_pixel_masks:result['pixel_masks']=torch.from_numpy(output)
        return result
