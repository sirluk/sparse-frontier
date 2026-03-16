#!/bin/bash
#######################################################################
# Sparse Frontier lm-eval (Quest) - single-node sbatch job, 8 GPUs
#######################################################################
#SBATCH --job-name=quest-cp-tp8
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpu-bind=map_gpu:0,1,2,3,4,5,6,7
#SBATCH --cpus-per-gpu=28
#SBATCH --exclusive=user
#SBATCH --time=14-00:00:00
#SBATCH --chdir=/nfs-gpu/xlstm-distillation/work_lukas/sparse-frontier
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
#SBATCH --wait-all-nodes=1
#######################################################################

set -euo pipefail

WORKDIR=/nfs-gpu/xlstm-distillation/work_lukas/sparse-frontier
cd "$WORKDIR"

echo "Starting job on $(date)"
echo "JobID              : ${SLURM_JOB_ID:-unknown}"
echo "Node list          : ${SLURM_JOB_NODELIST:-unknown}"
echo "GPUs visible       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_GPUS_ON_NODE : ${SLURM_GPUS_ON_NODE:-unset}"
echo "Work dir           : $PWD"

# CPU thread controls (override at submit time with --export=ALL,OMP_NUM_THREADS=...,TORCH_NUM_THREADS=...)
THREADS_DEFAULT="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_DEFAULT}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$THREADS_DEFAULT}"
echo "OMP_NUM_THREADS    : ${OMP_NUM_THREADS}"
echo "TORCH_NUM_THREADS  : ${TORCH_NUM_THREADS}"

# Run script defaults to TP=number of visible GPUs. Keep explicit TP=8 unless overridden.
export TP="${TP:-8}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# Optional: activate your shell env if needed for libraries/tools.
if [[ -f /home/lukash/.bashrc ]]; then
  set +u
  source /home/lukash/.bashrc || true
  set -u
fi

RUN_SCRIPT="${RUN_SCRIPT:-run_scripts/interactive/run_quest_cp.sh}"
if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Run script not found: $RUN_SCRIPT" >&2
  exit 1
fi
echo "Run script         : ${RUN_SCRIPT}"

bash "$RUN_SCRIPT"

echo "Finished at $(date)"
