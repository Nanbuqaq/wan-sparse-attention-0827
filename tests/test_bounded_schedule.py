from adapters.longlive_sparse.bounded_schedule import actions, route_digest, simulate


def test_finite_capacity_creates_natural_reloads_but_kv_major_does_not():
    groups = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    assert simulate(groups, "query_major", 1)["misses"] == 9
    assert simulate(groups, "kv_major", 1)["misses"] == 3
    assert simulate(groups, "query_major", 3)["misses"] == 3
    assert simulate(groups, "eager_union", 1)["status"] == "infeasible"


def test_all_schedules_preserve_exact_logical_edges_not_shared_union_edges():
    groups = [[0, 1], [0, 2], [2, 3]]
    wanted = {(i, b) for i, group in enumerate(groups) for b in group}
    for policy in ("query_major", "kv_major", "windowed_union", "eager_union"):
        emitted = [(group, block) for block, neighbors in actions(groups, policy) for group in neighbors]
        assert set(emitted) == wanted
        assert len(emitted) == len(wanted)
        assert simulate(groups, policy, 4)["route_sha256"] == route_digest(groups)


def test_disjoint_groups_do_not_invent_redundant_transfer():
    groups = [[0, 1], [2, 3], [4, 5]]
    assert simulate(groups, "query_major", 1)["misses"] == 6
    assert simulate(groups, "kv_major", 1)["misses"] == 6
