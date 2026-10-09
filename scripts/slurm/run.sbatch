#!/bin/bash
# Run one row of a sweep table (one Slurm array task = one run), with checkpoint-resume.
# Submit through scripts/slurm/submit.sh, which sets partition / GPUs / time per job kind.
#SBATCH --job-name=stsd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=gpu
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=exp/slurm/%x-%A_%a.out

set -euo pipefail
MATRIX=$1
PROFILE=${2:-}                     # extra Pointcept options, e.g. hardware batch caps
ROW=${SLURM_ARRAY_TASK_ID:?run as an array job}
NUM_GPUS=$(nvidia-smi -L | wc -l)

cd "${STSD_ROOT:?set STSD_ROOT to the repository on /scratch}"
eval "${STSD_ENV_SETUP:?set STSD_ENV_SETUP, e.g. 'source ~/miniforge3/bin/activate stsd-ssl'}"
export PYTHONPATH=$PWD/third_party/Pointcept:$PWD
export OMP_NUM_THREADS=4

# optional: stage preprocessed data to node-local disk (recommended for pretraining)
DATA_OPTS=""
if [[ "${STSD_STAGE:-0}" == "1" ]]; then
    rsync -a --info=progress2 data/processed/stsd_1cm/ "$TMPDIR/stsd_1cm/"
    DATA_OPTS="data.train.data_root=$TMPDIR/stsd_1cm data.val.data_root=$TMPDIR/stsd_1cm data.test.data_root=$TMPDIR/stsd_1cm"
fi

# resume if a previous attempt of this row left a checkpoint
SAVE_PATH=$(python -m stsd.tools.sweep --matrix "$MATRIX" --save-path "$ROW")
RESUME_OPTS=""
if [[ -f "$SAVE_PATH/model/model_last.pth" ]]; then
    RESUME_OPTS="resume=True weight=$SAVE_PATH/model/model_last.pth"
fi

requeue() {
    echo "[$(date)] time limit approaching: requeue ${SLURM_ARRAY_JOB_ID}_${ROW}"
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${ROW}"
    kill -TERM "$PID" 2>/dev/null || true
}
trap requeue USR1

python -m stsd.tools.sweep --matrix "$MATRIX" --run "$ROW" --num-gpus "$NUM_GPUS" \
    --extra "$PROFILE $DATA_OPTS $RESUME_OPTS" &
PID=$!
wait "$PID"
