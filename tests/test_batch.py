import pytest

from stsd.engines.batch import split_batch


@pytest.mark.parametrize(
    "world, cap, expected",
    [(1, 2, (2, 12)), (1, 1, (1, 24)), (3, 8, (8, 1)), (3, 6, (4, 2)), (4, 6, (6, 1)), (4, 8, (6, 1))],
)
def test_global_batch_is_preserved(world, cap, expected):
    per_gpu, accum = split_batch(24, world, cap)
    assert (per_gpu, accum) == expected
    assert per_gpu * accum * world == 24 and per_gpu <= cap
