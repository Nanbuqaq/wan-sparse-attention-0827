#!/usr/bin/env python3
"""Equal-slot spatial/random/feature group representation probe on complete captures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.feature_prototypes import build_feature_tail
from adapters.longlive_sparse.prototype_tail import build_prototype_tail
from adapters.longlive_sparse.offline_eval import dense_history_attention, output_error_metrics
from scripts.probe_prototype_tail import weighted_attention, selected_raw_indices
from scripts.evaluate_complete_attention_capture import construct_routes


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture-manifest',type=Path,required=True)
    p.add_argument('--kind',choices=('motion','state'),required=True)
    p.add_argument('--layers',default='0,9,19,29')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    manifest=json.loads(args.capture_manifest.read_text())
    paths={r['layer']:r for r in manifest['captures'] if r['kind']==args.kind}
    report=dict(status='running',gpu=torch.cuda.get_device_name(),scope='offline_group_representation_not_video_or_speed',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/probe_feature_prototypes.py','adapters/longlive_sparse/feature_prototypes.py')},cases=[])
    try:
        for layer in map(int,args.layers.split(',')):
            record=paths[layer];path=(args.capture_manifest.parent/record['file']).resolve()
            if not path.is_relative_to(args.capture_manifest.parent.resolve()):raise ValueError('capture escapes bundle')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('capture SHA mismatch')
            capture=torch.load(path,map_location='cpu',weights_only=True)
            route=construct_routes(capture,device='cuda',value_candidates=())['legacy_cap25']
            chosen=selected_raw_indices(route,capture['frame_ids'],capture['token_ids']).cuda()
            q,k,v,ek,ev=[capture[name].cuda() for name in ('query','key','value','exact_key','exact_value')]
            frame_tokens=capture['spatial_height']*capture['spatial_width']
            variants={}
            for raw_name,indices in (('no_raw',chosen[:,:,:0]),('legacy25',chosen)):
                for grouping in ('spatial_block64','spatial_groups','random_groups','key_kmeans'):
                    begin=time.perf_counter()
                    if grouping=='spatial_block64':
                        tail=build_prototype_tail(k,v,indices,frame_tokens=frame_tokens)
                        info=dict(grouping=grouping,groups_per_block=1,representation_bytes=tail.bytes)
                    else:
                        tail,info=build_feature_tail(k,v,indices,frame_tokens=frame_tokens,grouping=grouping)
                    torch.cuda.synchronize()
                    info['GPU_index_reconstruction_wall_s']=time.perf_counter()-begin
                    variants[raw_name+'_'+grouping]=(indices,tail,info)
            # All raw choices and groups exist before Dense output is read.
            reference=dense_history_attention(q,torch.cat((ek,k),1),torch.cat((ev,v),1))
            results={}
            for name,(indices,tail,info) in variants.items():
                output=torch.empty_like(q,dtype=torch.float32)
                for b in range(q.shape[0]):
                    for h in range(q.shape[2]):
                        fullk=torch.cat((ek[b,:,h],k[b,indices[b,h],h],tail.key[b,:,h]))
                        fullv=torch.cat((ev[b,:,h],v[b,indices[b,h],h],tail.value[b,:,h]))
                        counts=torch.cat((torch.ones(ek.shape[1]+indices.shape[-1],device='cuda'),tail.counts[b,h]))
                        output[b,:,h]=weighted_attention(q[b,:,h],fullk,fullv,counts)
                results[name]=dict(error=output_error_metrics(reference,output),index=info,
                    prototype_slots=tail.key.shape[1],raw_tokens_per_head=indices.shape[-1],
                    actual_raw_plus_prototype_value_bytes=2*(indices.numel()+tail.key.shape[0]*tail.key.shape[1]*tail.key.shape[2])*q.shape[-1]*q.element_size())
            row=dict(layer=layer,kind=args.kind,capture_sha256=record['sha256'],results=results,
                     four_group_variants_have_equal_physical_slots=True,groups_built_before_teacher=True)
            report['cases'].append(row)
            (args.output/f'layer{layer:02d}.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps({'layer':layer,'kind':args.kind,'relative_l2':{n:r['error']['relative_l2'] for n,r in results.items()}}),flush=True)
            del capture,q,k,v,ek,ev,reference,output,variants
        report['status']='pass'
    except BaseException:
        report.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
