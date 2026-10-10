# STSD-SSL

Self-supervised learning for subway tunnel point clouds on
[STSD](https://doi.org/10.1016/j.tust.2024.105829), built as a plugin on
[Pointcept](https://github.com/Pointcept/Pointcept) (PTv3 / Sonata).

## Layout

```
stsd/                 plugin package (registered into Pointcept's registries)
  datasets/           STSDDataset, RingWindowCrop, StrengthToColor
  engines/            hardware-independent batch resolution (global batch -> per-GPU + accumulation)
  evaluation/         mIoU / rare-class mIoU / tolerance F1, paired hierarchical bootstrap
  utils/              LAS reader, metadata, spatial splits and label budgets
  tools/              preprocess, make_splits, sweep, evaluate, stats
configs/stsd/         Pointcept configs (_base_: runtime / model / data)
configs/splits/       split files (contiguous 5-fold + buffer, random, Brunklaus, LOTO)
tools/                train.py / test.py (Pointcept entries + STSD registration)
scripts/slurm/        Bunya job array + DDP submission with checkpoint-requeue
third_party/Pointcept pinned submodule (unmodified)
tests/                unit tests (pytest)
```

## Setup

```bash
git submodule update --init
conda env create -f environment.yml            # PyTorch 2.5.0 + CUDA 12.4 (Pointcept's stack)
conda activate stsd-ssl
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
cd third_party/Pointcept/libs/pointops && \
  CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0" pip install --no-build-isolation . && cd -
export PYTHONPATH=$PWD/third_party/Pointcept:$PWD
python -m pytest
```

Sonata weights: `huggingface-cli download facebook/sonata pretrain-sonata-v1m1-0-base.pth --local-dir weights/sonata`.

**WSL + RTX 4090:** when GPU memory runs out, the Windows driver silently spills into system
RAM and training slows down by up to 80x instead of raising OOM. Keep `max_batch_size_per_gpu`
at 2 (fine-tuning) / 1 (pretraining) locally, or set *CUDA - Sysmem Fallback Policy* to
*Prefer No Sysmem Fallback* in the NVIDIA Control Panel.

## Data

```bash
# STSD v1.1 LAS -> 1 cm voxel scenes (~36 GB) with inverse indices for original-point scoring
python -m stsd.tools.preprocess --las_root data/STSD_v1.1/las --output_root data/processed/stsd_1cm
# spatial splits (already committed in configs/splits)
python -m stsd.tools.make_splits --meta data/processed/stsd_1cm/meta.json --output configs/splits \
    --brunklaus data/splits/brunklaus_stsd_dataset_info.json
```

Main protocol: `configs/splits/stsd_contiguous5_buffer.json` — each tunnel cut into 5 contiguous
blocks along its chainage, training segments within 1 m of the test block removed, and a
contiguous validation block carved from the training side. Label budgets are nested,
type-stratified subsets of training segments (`data.train.label_budget`, `budget_seed`).

## Training

Batching is specified by `global_batch_size` (optimisation batch) and `max_batch_size_per_gpu`
(memory cap); per-GPU batch, gradient accumulation and SyncBN follow from the number of GPUs,
so a run optimises the same objective on 1x4090, 3xH100 or 4xH100.

```bash
# single run
python tools/train.py --config-file configs/stsd/semseg-ptv3-stsd-ft.py --num-gpus 1 \
  --options save_path=exp/stsd/ft/demo weight=weights/sonata/pretrain-sonata-v1m1-0-base.pth \
  data.train.label_budget=12 max_batch_size_per_gpu=2

# experiment matrices (paper A): a_dapt, a_leakage, a_budget, a_linear, a_loto
python -m stsd.tools.sweep --matrix a_budget                 # list rows
python -m stsd.tools.sweep --matrix a_budget --run 0 --extra "max_batch_size_per_gpu=2"
```

### Weights & Biases

Enabled by default (`configs/stsd/_base_/runtime.py`, project `stsd-ssl`). Run `wandb login`
once per machine (or export `WANDB_API_KEY`). Each experiment directory owns one wandb run:
its id is stored in `<save_path>/wandb_id`, so Slurm requeues resume the same run.
Runs are grouped as `<protocol>-<method>-<split>` and tagged `method:` / `fold:` / `budget:`
from `run_meta.json`. Logged: lr and losses per step, epoch averages, validation mIoU /
per-class IoU, plus GPU utilisation/memory (wandb system metrics). Final original-point test
metrics are pushed to each run's summary (`test/miou`, `test/miou_rare`, `test/bf_joint`, ...)
by `stsd.tools.evaluate --wandb-project stsd-ssl`. Disable with `--options enable_wandb=False`
or log offline with `WANDB_MODE=offline` (sync later with `wandb sync`).

### Bunya

```bash
export STSD_ACCOUNT=a_xxx STSD_ROOT=/scratch/user/$USER/STSD-SSL
export STSD_ENV_SETUP="source ~/miniforge3/bin/activate stsd-ssl"   # plus `wandb login` once
scripts/slurm/submit.sh a_leakage smoke 0     # 3-GPU DDP sanity check (debug QoS)
scripts/slurm/submit.sh a_dapt pretrain       # 4x H100 DDP per fold, data staged to $TMPDIR
scripts/slurm/submit.sh a_budget finetune     # 1-GPU array tasks, 4 concurrent
```

Jobs checkpoint every 500 steps; ten minutes before the time limit they requeue themselves and
resume from `model_last.pth`. Short fine-tunes run as independent single-GPU array tasks (better
throughput than DDP); pretraining uses single-node DDP.

## Evaluation

Every fine-tune ends with a test on all original points (predictions on 2 cm voxels propagated
through the stored inverse index). Aggregate and compare:

```bash
python -m stsd.tools.evaluate --runs "exp/stsd/ft/*" --output exp/results/ft.csv --wandb-project stsd-ssl
```

```python
import pandas as pd
from stsd.evaluation import compare
df = pd.read_csv("exp/results/ft.csv")
compare(df, "B3", "B1", metric="miou", by=("budget",))   # paired mean diff, 95% CI, win rate
```
