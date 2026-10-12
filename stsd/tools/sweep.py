"""
Expand experiment matrices into a run table consumed by Slurm job arrays or local runs.

    python -m stsd.tools.sweep --matrix a_budget --output exp/sweeps/a_budget.csv
    python -m stsd.tools.sweep --matrix a_budget --run 17      # run row 17 locally

Each row: name, config, num_gpus, save_path, options (Pointcept --options string).
Every run writes <save_path>/run_meta.json so evaluation can group results without
parsing paths.
"""

import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

SONATA = "weights/sonata/pretrain-sonata-v1m1-0-base.pth"
DAPT = "exp/stsd/pretrain/dapt-f{fold}/model/model_last.pth"
SPLITS = dict(
    block="configs/splits/stsd_contiguous5_buffer.json",
    random="configs/splits/stsd_random_segments5.json",
    brunklaus="configs/splits/stsd_brunklaus2026.json",
    loto="configs/splits/stsd_leave_one_type_out.json",
)
NUM_FOLDS = dict(block=5, random=5, brunklaus=1, loto=3)
# provisional step budgets (epochs of 500 optimiser steps), see semseg-ptv3-stsd-ft.py
EPOCHS = {3: 4, 12: 6, 48: 8, "all": 12}


def weight_of(method, fold):
    return dict(B0=None, B1=SONATA, B3=DAPT.format(fold=fold))[method]


def finetune(protocol, method, split, fold, budget="all", subset=0, seed=0):
    cfg = dict(ft="configs/stsd/semseg-ptv3-stsd-ft.py", lin="configs/stsd/semseg-ptv3-stsd-lin.py")[protocol]
    name = f"{protocol}-{method}-{split}-f{fold}-b{budget}-s{subset}-r{seed}"
    epoch = EPOCHS[budget] if protocol == "ft" else 10
    options = dict(
        seed=seed,
        epoch=epoch,
        eval_epoch=epoch,
        **{f"data.{s}.split_file": SPLITS[split] for s in ("train", "val", "test")},
        **{f"data.{s}.fold": fold for s in ("train", "val", "test")},
        **{"data.train.budget_seed": subset},
    )
    if budget != "all":  # Pointcept parses "None" as a string, so omit instead
        options["data.train.label_budget"] = budget
    if weight_of(method, fold) is not None:
        options["weight"] = weight_of(method, fold)
    meta = dict(protocol=protocol, method=method, split=split, fold=fold, budget=budget, subset=subset, seed=seed)
    return dict(name=name, config=cfg, num_gpus=1, save_path=f"exp/stsd/{protocol}/{name}", options=options, meta=meta)


def pretrain_dapt(fold, split="block"):
    name = f"dapt-f{fold}"
    options = {"weight": SONATA, "data.train.fold": fold, "data.train.split_file": SPLITS[split]}
    meta = dict(protocol="pretrain", method="B3", split=split, fold=fold)
    return dict(
        name=name,
        config="configs/stsd/pretrain-sonata-stsd-dapt.py",
        num_gpus=4,
        save_path=f"exp/stsd/pretrain/{name}",
        options=options,
        meta=meta,
    )


def matrix(name):
    """Paper A experiment matrices (see docs/RESEARCH_PLAN_v3.md, section 2.3)."""
    if name == "a_dapt":
        return [pretrain_dapt(f) for f in range(5)] + [pretrain_dapt(f, "loto") for f in range(3)]
    if name == "a_leakage":  # A-T1
        runs = [finetune("ft", m, "brunklaus", 0) for m in ("B0", "B1")]
        runs += [finetune("ft", m, s, f) for m in ("B0", "B1") for s in ("random", "block") for f in range(5)]
        return runs
    if name == "a_budget":  # A-T2 (budget=all comes from a_leakage / B3 below)
        runs = []
        for method, fold in itertools.product(("B0", "B1", "B3"), range(5)):
            runs += [finetune("ft", method, "block", fold, b, s) for b in (3, 12) for s in (0, 1)]
            runs += [finetune("ft", method, "block", fold, 48, 0)]
            if method == "B3":
                runs += [finetune("ft", method, "block", fold, "all", 0)]
        return runs
    if name == "a_linear":  # A-T3
        return [finetune("lin", m, "block", f, b) for m in ("B0", "B1", "B3") for f in range(5) for b in (12, "all")]
    if name == "a_loto":  # A-T4 (LOTO part)
        return [finetune("ft", m, "loto", f) for m in ("B1", "B3") for f in range(3)]
    raise KeyError(name)


def option_string(options):
    return " ".join(f"{k}={v}" for k, v in options.items())


def command(run, num_gpus=None):
    gpus = num_gpus or run["num_gpus"]
    opts = option_string(dict(save_path=run["save_path"], **run["options"]))
    return f"python tools/train.py --config-file {run['config']} --num-gpus {gpus} --options {opts}"


def write_table(runs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "name", "config", "num_gpus", "save_path", "options", "meta"])
        for i, r in enumerate(runs):
            w.writerow([i, r["name"], r["config"], r["num_gpus"], r["save_path"], option_string(r["options"]), json.dumps(r["meta"])])


def effective_save_path(run, extra=""):
    overrides = dict(token.split("=", 1) for token in shlex.split(extra) if "=" in token)
    return overrides.get("save_path", run["save_path"])


def launch(run, num_gpus=None, extra=""):
    save_path = effective_save_path(run, extra)
    os.makedirs(save_path, exist_ok=True)
    with open(Path(save_path) / "run_meta.json", "w") as f:
        json.dump(dict(name=run["name"], **run["meta"]), f)
    cmd = f"{command(run, num_gpus)} {extra}".strip()
    print(cmd, flush=True)
    return subprocess.call(shlex.split(cmd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", default=None, help="write the run table (CSV)")
    parser.add_argument("--run", type=int, default=None, help="launch one row")
    parser.add_argument("--num-gpus", type=int, default=None, help="override GPUs for --run")
    parser.add_argument("--extra", default="", help="extra --options appended (e.g. hardware profile)")
    parser.add_argument("--save-path", type=int, default=None, help="print the save path of one row")
    args = parser.parse_args()

    runs = matrix(args.matrix)
    if args.save_path is not None:
        print(effective_save_path(runs[args.save_path], args.extra))
        return
    if args.output:
        write_table(runs, args.output)
        print(f"{len(runs)} runs -> {args.output}")
    if args.run is not None:
        sys.exit(launch(runs[args.run], args.num_gpus, args.extra))
    if not args.output:
        for i, r in enumerate(runs):
            print(i, r["name"])


if __name__ == "__main__":
    main()
