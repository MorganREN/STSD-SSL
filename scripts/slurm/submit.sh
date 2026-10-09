#!/bin/bash
# Submit a sweep table to Bunya as a job array.
#   scripts/slurm/submit.sh <matrix> <kind> [array-range]
# kind:
#   pretrain  4x H100 SXM (gpu_sxm), DDP, data staged to $TMPDIR, 48 h chunks, one at a time
#   finetune  1x H100 (gpu_cuda), up to 4 concurrent array tasks (per-user limit), 12 h
# Requires: STSD_ACCOUNT, STSD_ROOT, STSD_ENV_SETUP (see README).
set -euo pipefail
MATRIX=$1
KIND=$2
NUM_ROWS=$(python -m stsd.tools.sweep --matrix "$MATRIX" | wc -l)
RANGE=${3:-0-$((NUM_ROWS - 1))}
mkdir -p exp/slurm

case "$KIND" in
pretrain)
    STSD_STAGE=1 sbatch --account="${STSD_ACCOUNT:?}" --export=ALL,STSD_STAGE=1 \
        --partition=gpu_sxm --gres=gpu:h100:4 --cpus-per-task=64 --mem=400G \
        --time=48:00:00 --array="$RANGE%1" --job-name="stsd-$MATRIX" \
        scripts/slurm/run.sbatch "$MATRIX" "max_batch_size_per_gpu=6 num_worker_per_gpu_target=14"
    ;;
finetune)
    sbatch --account="${STSD_ACCOUNT:?}" --export=ALL \
        --partition=gpu_cuda --gres=gpu:h100:1 --cpus-per-task=16 --mem=120G \
        --time=12:00:00 --array="$RANGE%4" --job-name="stsd-$MATRIX" \
        scripts/slurm/run.sbatch "$MATRIX" "max_batch_size_per_gpu=8 num_worker_per_gpu_target=14"
    ;;
smoke)
    # 3-GPU DDP sanity check (debug QoS): a few hundred steps of one row, no requeue needed
    sbatch --account="${STSD_ACCOUNT:?}" --export=ALL --qos=debug \
        --partition=gpu_cuda --gres=gpu:h100:3 --cpus-per-task=48 --mem=300G \
        --time=00:45:00 --array="${3:-0}" --job-name="stsd-smoke-$MATRIX" \
        scripts/slurm/run.sbatch "$MATRIX" \
        "max_batch_size_per_gpu=8 num_worker_per_gpu_target=14 epoch=1 eval_epoch=1 save_path=exp/stsd/smoke/$MATRIX"
    ;;
*)
    echo "unknown kind: $KIND" >&2
    exit 1
    ;;
esac
