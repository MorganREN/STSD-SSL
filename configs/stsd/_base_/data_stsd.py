# STSD semantic segmentation data. Fold / budget are overridden per run, e.g.
#   --options data.train.fold=2 data.val.fold=2 data.test.fold=2 data.train.label_budget=12
dataset_type = "STSDDataset"
data_root = "data/processed/stsd_1cm"
split_file = "configs/splits/stsd_contiguous5_buffer.json"
grid_size = 0.02
window = 2.5  # metres along the tunnel; a full ring at 2 cm is ~100k points
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
        fold=0,
        label_budget=None,
        budget_seed=0,
        samples_per_epoch=12000,  # 500 optimiser steps at global batch 24
        transform=[
            dict(type="RingWindowCrop", length=window, mode="random"),
            dict(type="StrengthToColor"),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=1.0),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=120000, mode="random"),
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
        fold=0,
        windows_per_scene=4,
        transform=[
            dict(type="RingWindowCrop", length=window, mode="fraction"),
            dict(type="StrengthToColor"),
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
        fold=0,
        transform=[dict(type="StrengthToColor"), dict(type="NormalizeColor")],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="test", return_grid_coord=True),
            crop=dict(type="RingWindowCrop", length=window, mode="all", stride=window / 2),
            post_transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=feat_keys),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
