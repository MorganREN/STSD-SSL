"""
Generate all STSD split files from the preprocessed meta.json.

    python -m stsd.tools.make_splits --meta data/processed/stsd_1cm/meta.json \
        --output configs/splits [--brunklaus data/splits/brunklaus_stsd_dataset_info.json]

Each file: {name, note, folds: [{train, val, test, excluded}, ...]}.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from stsd.utils.split import TYPES, chainage, contiguous_folds, make_fold, nearest_distance


def random_segment_folds(names, num_repeats=5, test_ratio=0.2):
    folds = []
    for seed in range(num_repeats):
        perm = np.random.default_rng(seed).permutation(sorted(names)).tolist()
        k = int(round(len(perm) * test_ratio))
        folds.append((perm[k:], perm[:k]))
    return folds


def build_splits(scenes, brunklaus=None, buffer=1.0, val_ratio=0.1):
    chain = chainage(scenes)
    names = sorted(chain)
    blocks = contiguous_folds(chain)
    rest = lambda test: [n for n in names if n not in set(test)]
    splits = dict(
        stsd_contiguous5_buffer=(
            "Main protocol: 5 contiguous blocks per tunnel, <1 m buffer removed from training.",
            [make_fold(chain, rest(b), b, buffer, val_ratio) for b in blocks],
        ),
        stsd_contiguous5=(
            "Contiguous blocks without buffer (ablation).",
            [make_fold(chain, rest(b), b, 0.0, val_ratio) for b in blocks],
        ),
        stsd_random_segments5=(
            "Segment-level random 4:1 splits, 5 seeds (leakage quantification).",
            [make_fold(chain, tr, te, 0.0, val_ratio) for tr, te in random_segment_folds(names)],
        ),
        stsd_leave_one_type_out=(
            "Leave one tunnel type out (horseshoe, quasi_rectangular, circular).",
            [
                make_fold(chain, [n for n in names if chain[n]["type"] != t], [n for n in names if chain[n]["type"] == t], 0.0, val_ratio)
                for t in TYPES
            ],
        ),
    )
    if brunklaus is not None:
        splits["stsd_brunklaus2026"] = (
            "Brunklaus & Reiterer 2026 published split (random.seed(123), 211/53); val carved from train.",
            [make_fold(chain, brunklaus["train"], brunklaus["test"], 0.0, val_ratio)],
        )
        splits["stsd_brunklaus2026_original"] = (
            "Brunklaus & Reiterer 2026 protocol as published: train on all 211, model selection on test (val = test). "
            "Only for reproducing their numbers.",
            [dict(train=sorted(brunklaus["train"]), val=sorted(brunklaus["test"]), test=sorted(brunklaus["test"]), excluded=[])],
        )
    return chain, splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", default="configs/splits")
    parser.add_argument("--brunklaus", default=None)
    args = parser.parse_args()

    scenes = json.load(open(args.meta))["scenes"]
    brunklaus = json.load(open(args.brunklaus))["splits"] if args.brunklaus else None
    chain, splits = build_splits(scenes, brunklaus)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for name, (note, folds) in splits.items():
        with open(Path(args.output) / f"{name}.json", "w") as f:
            json.dump(dict(name=name, note=note, folds=folds), f, indent=1)
        touch = [
            np.mean([d < 1.0 for d in nearest_distance(chain, fo["test"], fo["train"]).values()])
            for fo in folds
        ]
        sizes = [(len(fo["train"]), len(fo["val"]), len(fo["test"])) for fo in folds]
        print(f"{name:28s} train/val/test {sizes}  test touching train: {np.round(touch, 2)}")


if __name__ == "__main__":
    main()
