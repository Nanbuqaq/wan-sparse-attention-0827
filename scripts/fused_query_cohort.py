"""Same-graph delivery costs, paired orders, and strong native controls."""


def build_fused_query(spec,assets,source,output,seed,base_builder):
    if seed!=20261010:raise ValueError('same-input backend study reuses the frozen development inputs')
    base=base_builder(spec,'query_groups',assets,source,output,seed)
    rows=[]
    schedule=(('native','torch',0),('shared','torch',0),('shared','fused',0),
              ('split_specific','torch',0),('split_specific','fused',0),
              ('split_specific','fused',1),('split_specific','torch',1),('native','torch',1))
    for lane,task in enumerate(('w2_rotating_wooden_bird','w2_tracking_delivery_cart')):
        references={}
        for method,backend,repeat in schedule:
            template=next(x for x in base if x['scenario']==task and x['method']==method)
            row=dict(template,cmd=list(template['cmd']))
            row['id']=f'{task}__s{seed}__pack_{method}_{backend}_r{repeat}'
            row['cmd'][row['cmd'].index('--output')+1]=str(output/row['id'])
            if method!='native':row['cmd']+=['--wave2-query-pack-backend',backend]
            row.update(method=f'{method}_{backend}',pack_backend=backend,repeat=repeat,
                not_independent_quality_sample=True,repeat_reason='matched_exact_graph_backend_and_same_pair_complete_cost')
            if method in references:
                ref_index,ref_id=references[method]
                row['reference_case_index']=ref_index
                row['cmd']+=['--equivalence-reference',str(output/ref_id/'summary.json')]
            else:references[method]=(len(rows),row['id'])
            rows.append(row)
    return rows
