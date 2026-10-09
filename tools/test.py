"""
STSD-SSL testing entry (Pointcept's tools/test.py + STSD component registration).

    python tools/test.py --config-file configs/stsd/<cfg>.py --num-gpus 1 \
        --options save_path=exp/stsd/<name> weight=exp/stsd/<name>/model/model_best.pth
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.launch import launch
from pointcept.engines.test import TESTERS

import stsd.datasets  # noqa: F401  (registers STSD datasets and transforms)


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(cfg=cfg, **cfg.test))
    tester.test()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
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
