"""
STSD-SSL training entry. Identical to Pointcept's tools/train.py, plus the global-batch
resolver (stsd.engines.resolve_batch) applied once the world size is known.

    python tools/train.py --config-file configs/stsd/<cfg>.py --num-gpus 3 \
        --options save_path=exp/stsd/<name>
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.launch import launch
from pointcept.engines.train import TRAINERS
from pointcept.utils import comm

import stsd.datasets  # noqa: F401  (registers STSD datasets and transforms)
from stsd.engines import resolve_batch


def main_worker(cfg):
    cfg = default_setup(resolve_batch(cfg))
    print(
        f"DDP rank={comm.get_rank()} local_rank={comm.get_local_rank()} "
        f"world_size={comm.get_world_size()} batch_per_gpu={cfg.batch_size_per_gpu} "
        f"accumulation={cfg.gradient_accumulation_steps}",
        flush=True,
    )
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    if cfg.get("smoke_test", False):
        cfg.hooks = [h for h in cfg.hooks if h["type"] != "PreciseEvaluator"]
        cfg.dump(f"{cfg.save_path}/config.py")
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
