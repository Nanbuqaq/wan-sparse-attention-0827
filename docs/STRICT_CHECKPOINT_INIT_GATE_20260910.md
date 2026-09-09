# Conditional elimination of overwritten Parameter initialization

The actual startup observer measured145.258s total, including~97.33s exclusive
initializer scope time and40.089s in torch.load. This is our source-equivalent
from_config/merged-checkpoint bootstrap, not an inherent LongLive2 limitation.

Implement a disabled-by-default constructor mode that skips only direct
nn.Parameter initializer functions, retaining ordinary Tensor/view and derived
buffer initialization. Keep original device/dtype placement and complete strict
checkpoint loads for generator/T5/VAE. Missing weights must abort; independently
seed inference noise exactly as before. Initialization RNG consumption changes,
so complete weights alone are not proof that unregistered buffers or outputs match.

First qualify the real existing64-latent causal-original gate against saved
full actual latent/decoded RGB. Then one full native509 original-prompt gate,
not a new quality sample. Do not enable it in frozen chest/text protocols yet.
Only after full equality can it be offered as a common loading optimization to
all baselines. No steady-state/Attention speedup or new algorithm claim follows
from shorter construction; startup timing needs separate repeated controls.
