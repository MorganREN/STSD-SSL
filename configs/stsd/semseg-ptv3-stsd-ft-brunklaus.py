"""
Reproduction of Brunklaus & Reiterer (2026), Automation in Construction 192, 107248,
STSD experiment 12 (configs/stsd/exp11-12-semseg-sonata-v1m1-0c-stsd-ft-100ep.py in
mvg-inatech/synthetic-lidar-training-data-generation-structural-repetition):
whole scenes, 4 cm grid, 80k-point sphere crop, intensity / 4096 for every tunnel,
batch 16, lr 1e-3 (block x0.1), 100 epochs of 211 scenes (~1.3k steps), model selection
on the test split (configs/splits/stsd_brunklaus2026_original.json). Only for explaining
the gap to our protocol; not used for any of our own comparisons.
  B0: weight=None | B1: weight=weights/sonata/pretrain-sonata-v1m1-0-base.pth
"""

_base_ = ["_base_/runtime.py", "_base_/model_ptv3.py"]

seed = 123
global_batch_size = 16
mix_prob = 0.8

epoch = 100
eval_epoch = 100
base_lr = 0.001
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr, base_lr * 0.1],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=base_lr * 0.1)]

dataset_type = "STSDDataset"
data_root = "data/processed/stsd_1cm"
split_file = "configs/splits/stsd_brunklaus2026_original.json"
grid_size = 0.04
intensity = dict(type="LegacyIntensityToColor", scales=dict(horseshoe=692.0, quasi_rectangular=1186.0, circular=183.0))
feat_keys = ("coord", "color", "normal")

data = dict(
    num_classes=12,
    ignore_index=-1,
    names=[
        "tunnel_wall", "rail_track", "power_track", "walkway", "pipe", "cable",
        "attachment", "person", "big_bolt_hole", "small_bolt_hole", "signal_line", "joint",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        split_file=split_file,
        mmap=False,  # whole-scene pipeline transforms in place
        transform=[
            intensity,
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.4, dropout_application_ratio=1.0),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomShift", shift=((0, 0), (0, 0), (-1, 1))),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.01, clip=0.05),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=80000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=feat_keys),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        split_file=split_file,
        mmap=False,  # whole-scene pipeline transforms in place
        transform=[
            intensity,
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        split_file=split_file,
        mmap=False,  # whole-scene pipeline transforms in place
        transform=[intensity, dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=feat_keys),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.student.backbone", replacement="module.backbone"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
