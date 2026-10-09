import numpy as np
import pytest

from stsd.utils.split import (
    budget_allocation,
    chainage,
    contiguous_folds,
    make_fold,
    nearest_distance,
    sample_budget,
)

TYPES = {"horseshoe": "horse", "quasi_rectangular": "rec", "circular": "round"}


def toy_scenes(sizes=dict(horseshoe=12, quasi_rectangular=40, circular=30), length=10.0):
    """Contiguous 10 m segments along a straight line per tunnel type."""
    scenes = {}
    for ttype, n in sizes.items():
        for i in range(n):
            name = f"{TYPES[ttype]}-1-{i // 5}_{i % 5 + 1}"
            scenes[name] = dict(type=ttype, seq=i // 5, part=i % 5 + 1, centroid=[i * length, 0.0, 0.0], s_range=[0.0, length])
    return scenes


@pytest.fixture
def chain():
    return chainage(toy_scenes())


def test_folds_partition_every_scene_once(chain):
    folds = contiguous_folds(chain)
    flat = [n for f in folds for n in f]
    assert sorted(flat) == sorted(chain) and len(flat) == len(set(flat))


def test_buffer_removes_touching_training_scenes(chain):
    for block in contiguous_folds(chain):
        fold = make_fold(chain, [n for n in chain if n not in block], block, buffer=1.0)
        dist = nearest_distance(chain, fold["test"], fold["train"] + fold["val"])
        assert min(dist.values()) >= 1.0
        assert not set(fold["train"]) & set(fold["val"]) and not set(fold["train"]) & set(fold["test"])


def test_budget_is_nested_and_stratified(chain):
    pool = sorted(chain)
    types = {n: chain[n]["type"] for n in pool}
    previous = set()
    for budget in (3, 12, 48):
        subset = set(sample_budget(types, pool, budget, seed=1))
        assert len(subset) == budget
        assert {types[n] for n in subset} == set(TYPES)
        assert previous <= subset
        previous = subset
    assert sample_budget(types, pool, None, seed=1) == pool


def test_budget_allocation_monotone():
    sizes = dict(horseshoe=23, quasi_rectangular=104, circular=84)
    allocs = [budget_allocation(sizes, b) for b in range(3, 212)]
    for a, b in zip(allocs[:-1], allocs[1:]):
        assert all(b[t] >= a[t] for t in sizes)
    assert allocs[0] == dict(horseshoe=1, quasi_rectangular=1, circular=1)
