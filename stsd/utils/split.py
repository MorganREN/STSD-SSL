"""
Spatially aware data splits for STSD.

STSD segments are consecutive ~10 m slices of three continuous tunnels, so random
segment-level splits put almost every test segment next to a training segment.
This module rebuilds each segment's chainage (position along the tunnel), cuts
contiguous folds with a buffer, carves a contiguous validation block and samples
nested, type-stratified label budgets. Pure numpy, no Pointcept dependency.
"""

import numpy as np

from .meta import TUNNEL_TYPES

TYPES = tuple(TUNNEL_TYPES.values())


def chainage(scenes):
    """scenes: {name: meta} -> {name: dict(type, order, ch0, ch1)} along each tunnel."""
    out = {}
    for ttype in TYPES:
        names = sorted(
            (n for n, m in scenes.items() if m["type"] == ttype),
            key=lambda n: (scenes[n]["seq"], scenes[n]["part"]),
        )
        centre = np.array([scenes[n]["centroid"] for n in names])
        position = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
        for order, (name, pos) in enumerate(zip(names, position)):
            length = scenes[name]["s_range"][1] - scenes[name]["s_range"][0]
            out[name] = dict(type=ttype, order=order, ch0=pos - length / 2, ch1=pos + length / 2)
    return out


def edge_distance(a, b):
    """Gap in metres between two segments along the tunnel (0 if touching/overlapping)."""
    return max(0.0, max(a["ch0"], b["ch0"]) - min(a["ch1"], b["ch1"]))


def nearest_distance(chain, names, others):
    """For each name, distance to the closest segment of the same tunnel in `others`."""
    dist = {}
    for n in names:
        same = [o for o in others if chain[o]["type"] == chain[n]["type"]]
        dist[n] = min((edge_distance(chain[n], chain[o]) for o in same), default=np.inf)
    return dist


def contiguous_folds(chain, num_folds=5):
    """Cut every tunnel into `num_folds` contiguous blocks; fold k = block k of every tunnel."""
    folds = [[] for _ in range(num_folds)]
    for ttype in TYPES:
        names = sorted((n for n in chain if chain[n]["type"] == ttype), key=lambda n: chain[n]["order"])
        for k, block in enumerate(np.array_split(np.array(names), num_folds)):
            folds[k] += block.tolist()
    return folds


def drop_buffer(chain, train, test, buffer=1.0):
    """Remove training segments closer than `buffer` metres to any test segment."""
    dist = nearest_distance(chain, train, test)
    return [n for n in train if dist[n] >= buffer]


def carve_val(chain, train, ratio=0.1):
    """Per tunnel, take a contiguous block (middle of the longest training run) as validation."""
    val = []
    for ttype in TYPES:
        names = sorted((n for n in train if chain[n]["type"] == ttype), key=lambda n: chain[n]["order"])
        if not names:
            continue
        runs, run = [], [names[0]]
        for prev, cur in zip(names[:-1], names[1:]):
            if chain[cur]["order"] == chain[prev]["order"] + 1:
                run.append(cur)
            else:
                runs.append(run)
                run = [cur]
        runs.append(run)
        longest = max(runs, key=len)
        size = max(1, int(round(ratio * len(names))))
        start = (len(longest) - size) // 2
        val += longest[start : start + size]
    return [n for n in train if n not in set(val)], val


def make_fold(chain, train, test, buffer=1.0, val_ratio=0.1):
    """Assemble one fold dict with buffer and validation carved from the training side."""
    kept = drop_buffer(chain, train, test, buffer) if buffer > 0 else list(train)
    train_, val = carve_val(chain, kept, val_ratio)
    excluded = sorted(set(chain) - set(train_) - set(val) - set(test))
    return dict(train=sorted(train_), val=sorted(val), test=sorted(test), excluded=excluded)


def budget_allocation(type_sizes, budget):
    """
    Split `budget` over tunnel types roughly in proportion to their size, >= 1 per type.

    Greedy: start from one scene per type and repeatedly add a scene to the type furthest
    below its proportional share. Allocations are monotone in `budget` by construction,
    which keeps sampled subsets nested (largest-remainder rounding is not monotone).
    """
    types = [t for t in TYPES if type_sizes.get(t, 0) > 0]
    total = sum(type_sizes[t] for t in types)
    budget = min(budget, total)
    assert budget >= len(types), f"budget {budget} smaller than number of tunnel types"
    alloc = {t: 1 for t in types}
    for b in range(len(types) + 1, budget + 1):
        open_types = [t for t in types if alloc[t] < type_sizes[t]]
        t = max(open_types, key=lambda t: (b * type_sizes[t] / total - alloc[t], type_sizes[t]))
        alloc[t] += 1
    return alloc


def sample_budget(scene_types, pool, budget, seed):
    """
    Nested, type-stratified subset of `pool` with `budget` segments.

    A fixed per-type permutation (seeded) is truncated by the allocation, so for the same
    seed a smaller budget is always a subset of a larger one. budget=None/'all' -> pool.
    """
    if budget is None or budget == "all" or budget >= len(pool):
        return sorted(pool)
    rng = np.random.default_rng(seed)
    by_type = {t: sorted(n for n in pool if scene_types[n] == t) for t in TYPES}
    perm = {t: list(rng.permutation(names)) for t, names in by_type.items()}
    alloc = budget_allocation({t: len(v) for t, v in by_type.items()}, budget)
    return sorted(n for t, k in alloc.items() for n in perm[t][:k])
