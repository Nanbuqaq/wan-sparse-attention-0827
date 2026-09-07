"""Fixed logical edges, variable physical schedules under explicit KV capacity.

This model does not cap the global union or make every query consume the union.
IDs are abstract immutable physical tiles; GPU replay supplies actual tensors.
"""
from collections import OrderedDict
import hashlib
import json


def route_digest(groups):
    return hashlib.sha256(json.dumps([sorted(set(g)) for g in groups], separators=(",", ":")).encode()).hexdigest()


def actions(groups, policy, *, window=2):
    groups = [set(g) for g in groups]
    if not groups or any(not g for g in groups):
        raise ValueError("this fixture requires nonempty groups")
    if policy in ("query_major", "eager_union"):
        return [(block, (group,)) for group, selected in enumerate(groups) for block in sorted(selected)]
    if policy == "kv_major":
        return [(block, tuple(i for i, g in enumerate(groups) if block in g)) for block in sorted(set.union(*groups))]
    if policy == "windowed_union":
        if window <= 0:
            raise ValueError("positive query-group window required")
        result = []
        for begin in range(0, len(groups), window):
            stop = min(begin + window, len(groups))
            result += [(block, tuple(i for i in range(begin, stop) if block in groups[i]))
                       for block in sorted(set.union(*groups[begin:stop]))]
        return result
    raise ValueError("unknown schedule")


def simulate(groups, policy, capacity, *, window=2):
    if capacity <= 0:
        raise ValueError("positive capacity required")
    union = sorted(set().union(*map(set, groups)))
    if policy == "eager_union" and capacity < len(union):
        return {"status": "infeasible", "reason": "union_exceeds_declared_GPU_KV_capacity"}
    resident = OrderedDict()
    misses = hits = 0
    def visit(block):
        nonlocal misses, hits
        if block in resident:
            hits += 1
            resident.move_to_end(block)
        else:
            misses += 1
            if len(resident) == capacity:
                resident.popitem(last=False)
            resident[block] = True
    if policy == "eager_union":
        for block in union:
            visit(block)
    for block, _ in actions(groups, policy, window=window):
        visit(block)
    return {"status": "pass", "misses": misses, "hits": hits, "unique_tiles": len(union),
            "logical_edges": sum(len(set(g)) for g in groups), "route_sha256": route_digest(groups)}
