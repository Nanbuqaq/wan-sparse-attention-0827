"""Real GPU C2 timing on the original ClusterVLM model path."""
import json,time
from pathlib import Path
import torch,numpy as np
from decord import VideoReader,cpu
from transformers import LlavaOnevisionForConditionalGeneration,LlavaOnevisionProcessor
MODEL='/kaimm-distill/zhouhe08/clustervlm/models/llava-onevision-qwen2-7b-ov-hf'; VIDEO='/kaimm-distill/zhouhe08/clustervlm/datasets/mlvu/MLVU/video/2_needle/needle_113.mp4'; POINTS=[100,500,1000,2000,4000]; WINDOW=64; BATCH=8; K=64; OUT=Path('/kaimm-distill/zhouhe08/clustervlm/outputs/c2_real/c2_real_raw.json')
def sync():
 if torch.cuda.is_available(): torch.cuda.synchronize()
def main():
 processor=LlavaOnevisionProcessor.from_pretrained(MODEL); model=LlavaOnevisionForConditionalGeneration.from_pretrained(MODEL,device_map='auto',torch_dtype=torch.float16,low_cpu_mem_usage=True).eval(); vr=VideoReader(VIDEO,ctx=cpu(0)); idx=np.linspace(0,len(vr)-1,max(POINTS),dtype=np.int64); frames=vr.get_batch(idx).asnumpy(); fs=[]
 with torch.inference_mode():
  for st in range(0,len(frames),BATCH):
   batch=list(frames[st:st+BATCH]); inp=processor.video_processor(batch,return_tensors='pt'); px=inp.pixel_values_videos.to(model.device,model.dtype); f=model.get_video_features(px,vision_feature_layer=model.config.vision_feature_layer,vision_feature_select_strategy=model.config.vision_feature_select_strategy).reshape(len(batch),-1,model.config.vision_config.hidden_size).mean(1); fs.append(f.float().cpu())
 x=torch.cat(fs).to(model.device); rows=[]
 for n in POINTS:
  z=x[:n]; c=z[torch.linspace(0,n-1,min(K,n),device=z.device).long()].clone(); sync(); t=time.perf_counter()
  for _ in range(10):
   labels=torch.cdist(z,c).argmin(1); sums=torch.zeros_like(c); counts=torch.zeros(c.shape[0],device=z.device); sums.index_add_(0,labels,z); counts.index_add_(0,labels,torch.ones(n,device=z.device)); c=sums/counts.clamp_min(1).unsqueeze(1)
  sync(); km=(time.perf_counter()-t)*1000; q=z[-1:]; w=z[-WINDOW:]; sync(); t=time.perf_counter(); scores=q@w.T; probs=scores.softmax(-1); _=probs@w; sync(); att=(time.perf_counter()-t)*1000; rows.append({'frames':n,'global_kmeans_ms':km,'window_attention_ms':att}); print(rows[-1],flush=True)
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps({'model':MODEL,'video':VIDEO,'feature':'projected vision feature mean pool','k':K,'window_frames':WINDOW,'points':rows},indent=2)); print('saved',OUT)
if __name__=='__main__': main()
