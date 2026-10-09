"""
Full fine-tuning of PTv3 on STSD (paper A arms B0 / B1 / B3 differ only in `weight`):
  B0 scratch:     weight=None
  B1 Sonata:      weight=weights/sonata/pretrain-sonata-v1m1-0-base.pth
  B3 Sonata-DAPT: weight=exp/stsd/pretrain-sonata-stsd-dapt-f{k}/model/model_last.pth
Recipe follows Pointcept's semseg-sonata-v1m1-0c-scannet-ft (lr 2e-3, block lr x0.1, wd 0.02).
One epoch = 500 optimiser steps (samples_per_epoch = 500 x global batch 24).
Step budget by label budget (provisional, to be confirmed by P0 convergence curves):
  b=3: epoch=4 (2k) | b=12: epoch=6 (3k) | b=48: epoch=8 (4k) | all: epoch=12 (6k)
"""

_base_ = ["_base_/runtime.py", "_base_/model_ptv3.py", "_base_/data_stsd.py"]

epoch = 12
eval_epoch = 12
base_lr = 0.002
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

hooks = [
    dict(type="CheckpointLoader", keywords="module.student.backbone", replacement="module.backbone"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
