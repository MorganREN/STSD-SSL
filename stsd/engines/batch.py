"""
Hardware-independent batch configuration.

Configs specify `global_batch_size` (the optimisation batch) and `max_batch_size_per_gpu`
(a memory limit). At launch the per-step batch and gradient accumulation are derived from
the world size, so 1x4090, 3xH100 and 4xH100 runs optimise the same objective.
"""

from pointcept.utils import comm


def split_batch(global_bs, world, max_per_gpu):
    """(per-GPU batch, accumulation steps): largest per-GPU divisor not above the memory cap."""
    assert global_bs % world == 0, f"global batch {global_bs} not divisible by {world} GPUs"
    local = global_bs // world
    per_gpu = max(d for d in range(1, min(local, max_per_gpu) + 1) if local % d == 0)
    return per_gpu, local // per_gpu


def resolve_batch(cfg):
    if cfg.get("global_batch_size") is None:
        return cfg
    world = comm.get_world_size()
    global_bs, max_per_gpu = cfg.global_batch_size, cfg.max_batch_size_per_gpu
    per_gpu, accum = split_batch(global_bs, world, max_per_gpu)
    cfg.batch_size = per_gpu * world
    cfg.gradient_accumulation_steps = accum
    if cfg.get("num_worker_per_gpu_target") is not None:
        cfg.num_worker = cfg.num_worker_per_gpu_target * world
    if cfg.get("sync_bn") == "auto":
        cfg.sync_bn = world > 1 and per_gpu <= 4
    return cfg
