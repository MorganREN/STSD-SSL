"""
Linear probing on STSD: frozen PTv3 encoder, multi-level up-cast features (1232-d) + linear
head, following Pointcept's semseg-sonata-v1m1-0a-scannet-lin. Same `weight` switch as
semseg-ptv3-stsd-ft.py (None = frozen random encoder). 5k optimiser steps.
"""

_base_ = ["semseg-ptv3-stsd-ft.py"]

model = dict(
    backbone_out_channels=1232,
    backbone=dict(enc_mode=True),
    freeze_backbone=True,
)

epoch = 10
eval_epoch = 10
max_batch_size_per_gpu = 12
