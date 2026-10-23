"""Explicit legacy-weight migration into the unchanged shared fixed-KV rules.

Legacy history projection is applied only when the newly completed clean block
is committed. Already committed blocks are never forwarded or reprojected.
"""
from contextlib import contextmanager

TRACK = 'legacy_weights_new_fixed_kv_interface'
PROJECTION_POLICY = 'trained_projection_on_new_clean_commit_only'


def require_migration_spec(spec):
    if spec.get('comparison_track') != TRACK or spec.get('legacy_history_projection_policy') != PROJECTION_POLICY:
        raise ValueError('Legacy migration requires an explicit track and clean-commit projection policy')


@contextmanager
def commit_projection(transformer, projection, inputs, context):
    if context.write_bank is None:
        yield
        return
    clean = [s for s in context.splits if s.role == 'clean']
    if len(clean) != 1 or any(s.role not in ('text', 'clean') for s in context.splits):
        raise ValueError('Legacy projection may commit exactly one new clean video block')
    expected = clean[0].end - clean[0].start
    if inputs['hidden_states'].shape[1] != expected:
        raise ValueError('Clean commit contains video rows outside the new block')
    calls = []

    def replace(module, args, native_output):
        if args[0].shape[1] != expected:
            raise ValueError('Unexpected clean video projection shape')
        result = projection(args[0])
        if result.shape != native_output.shape:
            raise ValueError('Legacy projection is incompatible with native H3 width')
        calls.append(expected)
        return result

    handle = transformer.proj_in.register_forward_hook(replace)
    try:
        yield
        if calls != [expected]:
            raise ValueError('Clean history projection must execute exactly once for the new block')
    finally:
        handle.remove()
