from adapters.longlive_sparse.native_kernel_recipe import recipe_digest,native_recipe_scope


def test_recipe_is_semantic_state_and_none_scope_is_explicit_legacy():
    assert recipe_digest({'num_warps':4})!=recipe_digest({'num_warps':16})
    with native_recipe_scope(None):
        assert recipe_digest({'x':1,'y':2})==recipe_digest({'y':2,'x':1})
