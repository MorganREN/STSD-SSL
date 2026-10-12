"""
Score finished runs from their saved original-resolution predictions.

    python -m stsd.tools.evaluate --runs "exp/stsd/ft/*" --output exp/results/ft.csv \
        --scene-output exp/results/ft_scenes.csv --wandb-project stsd-ssl

For every run directory with run_meta.json and result/<scene>_pred.npy, writes
metrics.json (mIoU, mIoU_rare, mAcc, OA, per-class IoU, tolerance F1 for thin classes)
and appends one row per run to the output CSV. With --wandb-project, the same metrics are
added to the summary of the run's wandb run (test/miou, test/miou_rare, test/bf_joint, ...).
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stsd.engines.tracking import log_test_summary
from stsd.evaluation import boundary_f1, confusion, scores, voxel_majority
from stsd.utils import CLASS_NAMES

THIN_CLASSES = ("joint", "small_bolt_hole", "big_bolt_hole")


SCENE_KEYS = ("miou", "miou_no_person", "oa")


def evaluate_run(run_dir, data_root, tol=0.02):
    """Fold-level scores plus per-scene rows (for segment-paired comparisons)."""
    preds = sorted(glob.glob(os.path.join(run_dir, "result", "*_pred.npy")))
    if not preds:
        return None, []
    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), np.int64)
    thin = {c: [] for c in THIN_CLASSES}
    scene_rows = []
    for path in preds:
        scene = Path(path).name[: -len("_pred.npy")]
        scene_dir = Path(data_root) / scene
        pred = np.load(path).astype(np.int64)
        gt = np.load(scene_dir / "origin" / "segment.npy").astype(np.int64)
        scene_conf = confusion(pred, gt)
        conf += scene_conf
        scene_scores = scores(scene_conf)
        scene_rows.append(dict(scene=scene, **{k: scene_scores[k] for k in SCENE_KEYS}))
        # tolerance F1 on 1 cm points: voxel prediction = majority of its original points
        inverse = np.load(scene_dir / "origin" / "inverse.npy")
        coord = np.load(scene_dir / "coord.npy")
        voxel_pred = voxel_majority(pred, inverse, len(coord))
        voxel_gt = np.load(scene_dir / "segment.npy").astype(np.int64)
        for c in THIN_CLASSES:
            thin[c].append(boundary_f1(coord, voxel_pred, voxel_gt, CLASS_NAMES.index(c), tol)[2])
    result = scores(conf)
    result["bf"] = {c: float(np.nanmean(v)) if np.any(~np.isnan(v)) else np.nan for c, v in thin.items()}
    result["num_scenes"] = len(preds)
    return result, scene_rows


def flatten(meta, result):
    row = dict(meta)
    keys = ("miou", "miou_no_person", "miou_rare", "miou_rare_no_person", "macc", "oa", "num_scenes")
    row.update({k: result[k] for k in keys})
    row.update({f"iou_{c}": v for c, v in result["iou"].items()})
    row.update({f"bf_{c}": v for c, v in result["bf"].items()})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="glob of run directories")
    parser.add_argument("--data_root", default="data/processed/stsd_1cm")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--wandb-project", default=None, help="also write test metrics to each run's wandb summary")
    parser.add_argument("--scene-output", default=None, help="also write per-scene metrics (CSV)")
    args = parser.parse_args()

    rows, scenes = [], []
    for run_dir in sorted(glob.glob(args.runs)):
        meta_path = Path(run_dir) / "run_meta.json"
        result, scene_rows = evaluate_run(run_dir, args.data_root, args.tol)
        if result is None:
            print(f"skip (no predictions): {run_dir}")
            continue
        meta = json.load(open(meta_path)) if meta_path.exists() else dict(name=Path(run_dir).name)
        with open(Path(run_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=1, default=float)
        row = flatten(meta, result)
        rows.append(row)
        scenes += [dict(meta, **r) for r in scene_rows]
        if args.wandb_project:
            metrics = {k: v for k, v in row.items() if k not in meta}
            log_test_summary(run_dir, args.wandb_project, metrics)
        print(f"{meta.get('name')}: mIoU {result['miou']:.4f}  mIoU-11 {result['miou_no_person']:.4f}  rare {result['miou_rare']:.4f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"{len(rows)} runs -> {args.output}")
    if args.scene_output:
        Path(args.scene_output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scenes).to_csv(args.scene_output, index=False)
        print(f"{len(scenes)} scene rows -> {args.scene_output}")


if __name__ == "__main__":
    main()
