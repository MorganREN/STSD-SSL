"""
B3 Sonata-DAPT: continue Sonata self-distillation on unlabeled STSD training scenes.

Model, loss and schedule follow Pointcept's pretrain-sonata-v1m1-0-base, with:
  * data: training-fold scenes only (split="unlabeled"), 4 m ring windows (~160k pts @2 cm)
  * batch: global 24 (official 96 on 32 GPUs) -> lr scaled linearly to 1e-3
  * colour: intensity replicated to 3 channels; per-channel ChromaticTranslation dropped
  * init: official Sonata checkpoint (student + teacher), passed via `weight=`
One epoch = 500 optimiser steps. Run per fold:
  --options data.train.fold=k save_path=exp/stsd/pretrain-sonata-stsd-dapt-f{k}
"""

_base_ = ["_base_/runtime.py"]

mix_prob = 0
evaluate = False
find_unused_parameters = False
max_batch_size_per_gpu = 6

model = dict(
    type="Sonata-v1m1",
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    teacher_custom=dict(attn_drop=0.0, proj_drop=0.0, drop_path=0.0),
    head_in_channels=1088,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.4,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
)

epoch = 40  # 20k steps (provisional; length curve 2k/5k/10k/20k checked on fold 0)
eval_epoch = 40
base_lr = 0.004 * 24 / 96
lr_decay = 0.9
base_wd = 0.04
final_wd = 0.2

_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(keyword=f"enc{e}.block{b}.", lr=base_lr * lr_decay ** (sum(_depths) - sum(_depths[:e]) - b - 1))
    for e in range(len(_depths))
    for b in range(_depths[e])
]
del _depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

_geometric = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
]
_intensity = [
    dict(type="RandomColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.02, p=0.8),
    dict(type="NormalizeColor"),
]

data = dict(
    train=dict(
        type="STSDDataset",
        split="unlabeled",
        data_root="data/processed/stsd_1cm",
        split_file="configs/splits/stsd_contiguous5_buffer.json",
        fold=0,
        samples_per_epoch=12000,
        transform=[
            dict(type="RingWindowCrop", length=4.0, mode="random"),
            dict(type="StrengthToColor"),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train"),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(
                type="MultiViewGenerator",
                view_keys=("coord", "origin_coord", "color", "normal"),
                global_view_num=2,
                global_view_scale=(0.4, 1.0),
                local_view_num=4,
                local_view_scale=(0.1, 0.4),
                global_shared_transform=_intensity,
                global_transform=_geometric,
                local_transform=_geometric + _intensity,
                max_size=65536,
            ),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": 0.02}),
            dict(
                type="Collect",
                keys=(
                    "global_origin_coord", "global_coord", "global_color", "global_offset",
                    "local_origin_coord", "local_coord", "local_color", "local_offset",
                    "grid_size", "name",
                ),
                offset_keys_dict=dict(),
                global_feat_keys=("global_coord", "global_color", "global_normal"),
                local_feat_keys=("local_coord", "local_color", "local_normal"),
            ),
        ],
        test_mode=False,
    )
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=4),
]
