"""Source-aligned DMD algebra and native-H3 few-step sampling contracts.

These helpers do not start DMD training or admit a checkpoint. They separate
true Dense score-model roles from a chunk-causal student and expose the
stochastic clean-prediction/re-noising transition used upstream.
"""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class ScoreRoles:
    generator_attention:str='chunk_causal_generated_history'
    real_score_attention:str='native_dense_noncausal'
    fake_score_attention:str='native_dense_noncausal'
    real_guidance:float=3.
    fake_guidance:float=0.


def guided_clean_prediction(conditional,unconditional,guidance):
    if guidance==0:return conditional
    if unconditional is None:raise ValueError('unconditional score required by nonzero guidance')
    return conditional+(conditional-unconditional)*guidance


def normalized_kl_gradient(generated,fake_clean,real_clean):
    if generated.shape!=fake_clean.shape or generated.shape!=real_clean.shape or generated.ndim<2:
        raise ValueError('matching batch-first clean predictions required')
    # Match model/dmd.py exactly: per-sample mean over non-batch dimensions,
    # then nan_to_num. No normalization across unrelated batch members.
    axes=tuple(range(1,generated.ndim))
    scale=(generated-real_clean).abs().mean(dim=axes,keepdim=True)
    return torch.nan_to_num((fake_clean-real_clean)/scale)


def distribution_matching_surrogate(generated,gradient,mask=None):
    if generated.shape!=gradient.shape:raise ValueError('gradient shape mismatch')
    # Double precision and detached pseudo-target match upstream semantics.
    value=generated.double();target=(value-gradient.double()).detach()
    if mask is not None:
        if mask.shape!=generated.shape or mask.dtype!=torch.bool:raise ValueError('explicit full-shaped gradient mask required')
        if not bool(mask.any()):raise ValueError('empty gradient mask')
        value,target=value[mask],target[mask]
    return .5*torch.nn.functional.mse_loss(value,target,reduction='mean')


def flow_to_clean(noisy,velocity,clean_weight):
    """H3 uses x=t*x0+(1-t)*noise; model predicts dx/dt=x0-noise."""
    if not 0<=clean_weight<=1:raise ValueError('native H3 clean weight outside [0,1]')
    return noisy+(1-clean_weight)*velocity


def renoise_clean(clean,noise,next_clean_weight):
    if clean.shape!=noise.shape or not 0<=next_clean_weight<=1:raise ValueError('invalid re-noising contract')
    return next_clean_weight*clean+(1-next_clean_weight)*noise


def stochastic_backward_simulation(initial,clean_weights,exit_index,predict,noise_like):
    """One modality/chunk; retained graph only at the synchronized exit step.

    predict(noisy, t) returns native flow velocity. The caller supplies the
    chunk-causal model/history and broadcasts exit_index. Clean writeback is
    a separate detached forward. This is not a repeated Euler integrator.
    """
    weights=tuple(float(t) for t in clean_weights)
    if not weights or any(not 0<=t<1 for t in weights) or any(b<=a for a,b in zip(weights,weights[1:])):
        raise ValueError('strictly increasing native H3 denoise clean weights required')
    if not 0<=exit_index<len(weights):raise ValueError('invalid synchronized exit index')
    noisy=initial
    for i,t in enumerate(weights):
        with (torch.enable_grad() if i==exit_index else torch.no_grad()):
            clean=flow_to_clean(noisy,predict(noisy,t),t)
        if i==exit_index:return clean
        noisy=renoise_clean(clean,noise_like(clean),weights[i+1])
    raise AssertionError('unreachable')
