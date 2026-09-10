from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.native_shared_conditioning import SharedNativeConditioning


def encode_prompt_blocks(*args):
    raise AssertionError('original encoder should only be restored, not consumed in this fixture')


class FakePipeline:
    guidance_scale = 1
    num_frame_per_block = 8
    independent_first_frame = False

    def inference(self, noise, text_prompts, initial_latent=None, return_latents=False, start_frame_index=0):
        return encode_prompt_blocks(self.text_encoder, text_prompts, noise.shape[0])


def test_repeated_aliases_share_storage_with_exact_block_values():
    encoder = SimpleNamespace(values={'a': torch.randn(1, 3, 5), 'b': torch.randn(1, 3, 5)},
        aliases={'alias': 'a'}, target_device='cpu')
    shared = SharedNativeConditioning(encoder)
    aggregate, blocks = shared.encode(encoder, [['a', 'b', 'alias', 'a']], 1)
    assert aggregate == {}
    assert torch.equal(torch.cat([x['prompt_embeds'] for x in blocks]),
        torch.cat([encoder.values[p] for p in ['a', 'b', 'a', 'a']]))
    assert blocks[0]['prompt_embeds'].data_ptr() == blocks[2]['prompt_embeds'].data_ptr()
    assert shared.unique_gpu.numel() == 30
    shared.verify_unchanged()
    blocks[0]['prompt_embeds'].add_(1)
    with pytest.raises(RuntimeError, match='changed'):
        shared.verify_unchanged()


def test_invalid_encoder_and_batches_fail_without_fallback():
    encoder = SimpleNamespace(values={'a': torch.ones(1, 3, 5)}, aliases={}, target_device='cpu')
    shared = SharedNativeConditioning(encoder)
    for prompts, batch in [([], 1), ([[]], 1), ([['a'], ['a']], 2)]:
        with pytest.raises(ValueError):
            shared.encode(encoder, prompts, batch)
    with pytest.raises(ValueError):
        shared.encode(object(), [['a']], 1)


def test_consumption_audit_rejects_wrong_block_and_detached_copy():
    encoder = SimpleNamespace(values={'a': torch.ones(1, 3, 5), 'b': torch.zeros(1, 3, 5)},
        aliases={}, target_device='cpu')
    shared = SharedNativeConditioning(encoder)
    _, blocks = shared.encode(encoder, [['a', 'b']], 1)
    shared.verify_consumption(880, None, (), dict(current_start=8*880, conditional_dict=blocks[1]))
    for wrong in [blocks[0], {'prompt_embeds': blocks[1]['prompt_embeds'].clone()}]:
        with pytest.raises(RuntimeError):
            shared.verify_consumption(880, None, (), dict(current_start=8*880, conditional_dict=wrong))


def test_scope_guard_and_module_restoration_after_error():
    pipe = FakePipeline()
    pipe.text_encoder = SimpleNamespace(values={'a': torch.ones(1, 3, 5)}, aliases={}, target_device='cpu')
    original = pipe.inference
    shared = SharedNativeConditioning(pipe.text_encoder)
    with shared.activate(pipe):
        aggregate, blocks = pipe.inference(torch.zeros(1, 8), [['a']])
        assert aggregate == {} and len(blocks) == 1
        with pytest.raises(ValueError, match='prefill'):
            pipe.inference(torch.zeros(1, 8), [['a']], initial_latent=torch.zeros(1))
        with pytest.raises(ValueError, match='continuation'):
            pipe.inference(torch.zeros(1, 8), [['a']], start_frame_index=8)
        with pytest.raises(KeyError):
            pipe.inference(torch.zeros(1, 8), [['missing']])
        with pytest.raises(AssertionError):
            encode_prompt_blocks()
    assert pipe.inference == original
