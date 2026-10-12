"""
STSD dataset for Pointcept.

Scenes are read from the folder layout written by `stsd.tools.preprocess`; which scenes
belong to train/val/test comes from a split file (configs/splits/*.json) and a fold index,
so the same preprocessed data serves every protocol.
"""

import json
import os

import numpy as np

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset

from stsd.utils import CLASS_NAMES, parse_scene_name
from stsd.utils.split import sample_budget


@DATASETS.register_module()
class STSDDataset(DefaultDataset):
    VALID_ASSETS = ["coord", "unroll", "strength", "normal", "segment"]
    INDEX_KEYS = ["coord", "unroll", "strength", "normal", "segment", "color"]

    def __init__(
        self,
        split_file,
        fold=0,
        split="train",
        label_budget=None,
        budget_seed=0,
        samples_per_epoch=None,
        windows_per_scene=1,
        mmap=True,
        **kwargs,
    ):
        """
        split: "train" | "val" | "test" | "unlabeled" (train scenes, labels unused).
        label_budget: number of labelled training scenes (None = all), nested per budget_seed.
        samples_per_epoch: fixed epoch length in samples (random crops); None = one per scene.
        windows_per_scene: >1 enumerates each scene as evenly spaced windows (validation),
            consumed by RingWindowCrop(mode="fraction") through `window_fraction`.
        mmap: memory-map assets read-only (cheap when a crop selects points first); set False
            for pipelines that transform whole scenes in place.
        """
        self.split_file = split_file
        self.fold = fold
        self.label_budget = label_budget
        self.budget_seed = budget_seed
        self.samples_per_epoch = samples_per_epoch
        self.windows_per_scene = windows_per_scene
        self.mmap = mmap
        super().__init__(split=split, **kwargs)

    def get_scene_names(self):
        with open(self.split_file) as f:
            fold = json.load(f)["folds"][self.fold]
        if self.split == "unlabeled":
            return fold["train"]
        names = fold[self.split]
        if self.split == "train" and self.label_budget is not None:
            types = {n: parse_scene_name(n)[0] for n in names}
            names = sample_budget(types, names, self.label_budget, self.budget_seed)
        return names

    def get_data_list(self):
        paths = [os.path.join(self.data_root, name) for name in self.get_scene_names()]
        return [p for p in paths for _ in range(self.windows_per_scene)]

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data_dict = {
            asset: np.load(os.path.join(data_path, f"{asset}.npy"), mmap_mode="r" if self.mmap else None)
            for asset in self.VALID_ASSETS
        }
        data_dict["segment"] = np.asarray(data_dict["segment"], dtype=np.int32)
        if self.test_mode:
            # Pointcept's tester maps predictions back with `inverse` and scores on `origin_segment`.
            origin = os.path.join(data_path, "origin")
            data_dict["origin_segment"] = np.load(os.path.join(origin, "segment.npy")).astype(np.int32)
            data_dict["inverse"] = np.load(os.path.join(origin, "inverse.npy"))
        if self.windows_per_scene > 1:
            k = (idx % len(self.data_list)) % self.windows_per_scene
            data_dict["window_fraction"] = k / (self.windows_per_scene - 1)
        data_dict["name"] = self.get_data_name(idx)
        data_dict["split"] = self.split
        data_dict["index_valid_keys"] = list(self.INDEX_KEYS)
        return data_dict

    def get_split_name(self, idx):
        return self.split

    def __len__(self):
        if self.samples_per_epoch is not None and not self.test_mode:
            return self.samples_per_epoch
        return len(self.data_list) * self.loop

    @property
    def class_names(self):
        return CLASS_NAMES
