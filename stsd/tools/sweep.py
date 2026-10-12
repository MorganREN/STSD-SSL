"""
Expand experiment matrices into a run table consumed by Slurm job arrays or local runs.

    python -m stsd.tools.sweep --matrix a_budget --output exp/sweeps/a_budget.csv
    python -m stsd.tools.sweep --matrix a_budget --run 17      # run row 17 locally

Each row: name, config, num_gpus, save_path, options (Pointcept --options string).
Every run writes <save_path>/run_meta.json so evaluation can group results without
parsing paths, and <save_path>/DONE after a successful exit; rows that are already DONE
are skipped, so matrices may share runs (e.g. a_gate rows reappear in a_budget*).
"""

import argparse
import csv
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

SONATA = "weights/sonata/pretrain-sonata-v1m1-0-base.pth"
DAPT = "exp/stsd/pretrain/dapt-f{fold}/model/{ckpt}.pth"
DAPT_LOTO = "exp/stsd/pretrain/dapt-f{fold}-loto/model/model_last.pth"
SPLITS = dict(
    block="configs/splits/stsd_contiguous5_buffer.json",
    random="configs/splits/stsd_random_segments5.json",
    brunklaus="configs/splits/stsd_brunklaus2026.json",
    brunklaus_orig="configs/splits/stsd_brunklaus2026_original.json",
    loto="configs/splits/stsd_leave_one_type_out.json",
)
CONFIGS = dict(
    ft="configs/stsd/semseg-ptv3-stsd-ft.py",
    lin="configs/stsd/semseg-ptv3-stsd-lin.py",
    ftbr="configs/stsd/semseg-ptv3-stsd-ft-brunklaus.py",
)
# step budgets (epochs of 500 optimiser steps), see semseg-ptv3-stsd-ft.py; None = config default
EPOCHS = dict(ft={3: 4, 12: 6, 48: 8, "all": 12}, lin={b: 10 for b in (3, 12, 48, "all")}, ftbr={"all": None})
DAPT_LENGTHS = (4, 12, 20)  # DAPT saves epoch_N.pth every 4 epochs (N x 500 steps: 2k, 6k, 10k); 20k = model_last


def weight_of(method, split, fold):
    """B0 scratch | B1 Sonata | B3 Sonata-DAPT (model_last) | B3e<N> DAPT checkpoint at epoch N."""
    if method == "B0":
        return None
    if method == "B1":
        return SONATA
    if split == "loto":
        return DAPT_LOTO.format(fold=fold)
    ckpt = "model_last" if method == "B3" else f"epoch_{method[3:]}"
    return DAPT.format(fold=fold, ckpt=ckpt)


def finetune(protocol, method, split, fold, budget="all", subset=0, seed=0):
    name = f"{protocol}-{method}-{split}-f{fold}-b{budget}-s{subset}-r{seed}"
    epoch = EPOCHS[protocol][budget]
    options = dict(
        seed=seed,
        **({} if epoch is None else dict(epoch=epoch, eval_epoch=epoch)),
        **{f"data.{s}.split_file": SPLITS[split] for s in ("train", "val", "test")},
        **{f"data.{s}.fold": fold for s in ("train", "val", "test")},
        **{"data.train.budget_seed": subset},
    )
    if budget != "all":  # Pointcept parses "None" as a string, so omit instead
        options["data.train.label_budget"] = budget
    weight = weight_of(method, split, fold)
    if weight is not None:
        options["weight"] = weight
    meta = dict(protocol=protocol, method=method, split=split, fold=fold, budget=budget, subset=subset, seed=seed)
    return dict(name=name, config=CONFIGS[protocol], num_gpus=1, save_path=f"exp/stsd/{protocol}/{name}", options=options, meta=meta)


def pretrain_dapt(fold, split="block"):
    name = f"dapt-f{fold}" + ("-loto" if split == "loto" else "")
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


def budget_runs(methods, folds=range(5), budgets=(3, 12, 48, "all")):
    """Label-budget fine-tunes: 2 labelled subsets for b<=12, one otherwise."""
    runs = []
    for method, fold, budget in itertools.product(methods, folds, budgets):
        subsets = (0, 1) if budget in (3, 12) else (0,)
        runs += [finetune("ft", method, "block", fold, budget, s) for s in subsets]
    return runs


MATRICES = dict(
    # pretraining (4-GPU DDP); row order is fixed: row 0 (fold 0) was submitted first
    a_dapt=lambda: [pretrain_dapt(f) for f in range(5)] + [pretrain_dapt(f, "loto") for f in range(3)],
    # A-T1 leakage, full labels
    a_leakage=lambda: [finetune("ft", m, "brunklaus", 0) for m in ("B0", "B1")]
    + [finetune("ft", m, s, f) for m in ("B0", "B1") for s in ("random", "block") for f in range(5)],
    # Brunklaus & Reiterer (2026) configuration reproduced verbatim (4 cm, 100 ep, val = test)
    a_brunklaus_repro=lambda: [finetune("ftbr", m, "brunklaus_orig", 0) for m in ("B0", "B1")],
    # decision gate: fold 0, B1 vs B3 at b in {3, 12}, 2 subsets (needs dapt-f0)
    a_gate=lambda: budget_runs(("B1", "B3"), folds=(0,), budgets=(3, 12)),
    # DAPT length curve: linear probe on fold 0 with intermediate DAPT checkpoints (needs dapt-f0)
    a_dapt_len=lambda: [finetune("lin", f"B3e{n}", "block", 0) for n in DAPT_LENGTHS],
    # A-T2 label efficiency; b=all of B0/B1 comes from a_leakage (shared names, skipped when DONE)
    a_budget=lambda: budget_runs(("B0", "B1"), budgets=(3, 12, 48)),
    a_budget_b3=lambda: budget_runs(("B3",)),
    # A-T3 linear probing
    a_linear=lambda: [finetune("lin", m, "block", f, b) for m in ("B0", "B1") for f in range(5) for b in (12, "all")],
    a_linear_b3=lambda: [finetune("lin", "B3", "block", f, b) for f in range(5) for b in (12, "all")],
    # A-T4 leave-one-type-out
    a_loto=lambda: [finetune("ft", m, "loto", f) for m in ("B0", "B1") for f in range(3)],
    a_loto_b3=lambda: [finetune("ft", "B3", "loto", f) for f in range(3)],
)


def matrix(name):
    """Paper A experiment matrices (see docs/RESEARCH_PLAN_v3.md, section 2.3)."""
    return MATRICES[name]()


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
    save_path = Path(effective_save_path(run, extra))
    if (save_path / "DONE").exists():
        print(f"skip (DONE): {save_path}", flush=True)
        return 0
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "run_meta.json", "w") as f:
        json.dump(dict(name=run["name"], **run["meta"]), f)
    cmd = f"{command(run, num_gpus)} {extra}".strip()
    print(cmd, flush=True)
    code = subprocess.call(shlex.split(cmd))
    if code == 0:
        (save_path / "DONE").touch()
    return code


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
