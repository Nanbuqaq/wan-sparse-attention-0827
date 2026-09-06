"""Post-source-layer diagnostic snapshots, before the target layer executes.

Only Q summaries and already-indexed target-layer CPU prototypes are exposed.
No target Q, target selected coordinates, full candidate KV or teacher outputs
enter a prediction. This observer never returns a route to the model.
"""
from dataclasses import fields
import torch
from .selectors import summarize_query_for_pretransfer


def context_state(context):
    return {f.name: (getattr(context, f.name).detach().cpu().clone()
                    if isinstance(getattr(context, f.name), torch.Tensor) else getattr(context, f.name))
            for f in fields(context)}


class NextPrototypeObserver:
    def __init__(self, *, starts=(46800, 51480, 56160), layers=(0, 1, 2, 3)):
        self.starts, self.layers = set(starts), set(layers)
        self.records = []

    def __call__(self, *, module, query, route_plan, candidate_frame_ids,
                 current_start, denoising_pass, route_was_reused):
        if current_start not in self.starts or module.layer_id not in self.layers or denoising_pass != 0:
            return
        if route_was_reused:
            raise ValueError('first-call probe unexpectedly saw an enforced cache reuse')
        archive, layer = module.history_archive, module.layer_id
        ids = [int(v) for v in candidate_frame_ids.detach().cpu().reshape(-1)]
        summary = summarize_query_for_pretransfer(query.detach(), 64)
        own = context_state(archive.online_routing_context(layer, summary, ids))
        target = layer+1
        available = [f for f in ids if f in archive._layers.get(target, {})] if target in self.layers else []
        future = context_state(archive.online_routing_context(target, summary, available)) if available else None
        self.records.append({'source_layer': layer, 'target_layer': target,
            'current_start': current_start, 'denoising_pass': denoising_pass,
            'source_candidate_frame_ids': ids, 'target_frames_already_indexed': available,
            'target_missing_at_prediction': [f for f in ids if f not in available],
            'source_context': own, 'next_context': future, 'source_route': route_plan.state_dict(),
            'source_route_sha': route_plan.digest(), 'source_output_completed': True,
            'target_layer_not_yet_executed': True, 'target_Q_or_route_used': False,
            'raw_candidate_KV_exposed_to_predictor': False})
