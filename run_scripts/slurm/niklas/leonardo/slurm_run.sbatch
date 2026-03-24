#!/bin/bash
#######################################################################
# Sparse Frontier lm-eval (Leonardo Booster) - single-node sbatch job
#######################################################################
#SBATCH --job-name=eval-1n4g
#SBATCH --account=aifac_5l0_356
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=14-00:00:00
#SBATCH --chdir=/leonardo_work/EUHPC_D31_137/sparse-frontier
#SBATCH --output=/leonardo_work/EUHPC_D31_137/sparse-frontier/slurm_logs_eval/slurm_%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D31_137/sparse-frontier/slurm_logs_eval/slurm_%x_%j.err
#SBATCH --wait-all-nodes=1
#SBATCH --hint=nomultithread
#######################################################################

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22

parse_first_int() {
  local value="$1"
  local parsed=""
  parsed="$(grep -oE '[0-9]+' <<< "$value" | head -n1 || true)"
  echo "$parsed"
}

WORKDIR="${WORKDIR:-/leonardo_work/EUHPC_D31_137/sparse-frontier}"
LEONARDO_BASHRC="${LEONARDO_BASHRC:-/leonardo/home/userexternal/nschmidi/.bashrc}"
CONDA_ENV="${CONDA_ENV:-sparse_frontier}"
LM_EVAL_HARNESS_PATH="${LM_EVAL_HARNESS_PATH:-/leonardo_work/EUHPC_D31_137/lm-evaluation-harness}"
RUN_SCRIPT="${RUN_SCRIPT:-run_scripts/interactive/run_quest_cp.sh}"

export WORKDIR LEONARDO_BASHRC CONDA_ENV LM_EVAL_HARNESS_PATH
cd "$WORKDIR"

if [[ -f "$LEONARDO_BASHRC" ]]; then
  echo "Activating conda env from ${LEONARDO_BASHRC}"
  set +u
  source "$LEONARDO_BASHRC"
  set -u
  conda activate "$CONDA_ENV"
else
  echo "WARNING: Leonardo bashrc not found: ${LEONARDO_BASHRC}" >&2
fi

GPUS_ON_NODE="$(parse_first_int "${SLURM_GPUS_ON_NODE:-4}")"
if [[ -z "$GPUS_ON_NODE" ]]; then
  GPUS_ON_NODE=4
fi
export SLURM_GPUS_ON_NODE="$GPUS_ON_NODE"

echo "=== Node info ========================================"
module list
nvidia-smi || true
free -m || true
df -h . || true
echo "Starting job on $(date)"
echo "JobID              : ${SLURM_JOB_ID:-unknown}"
echo "Node list          : ${SLURM_JOB_NODELIST:-unknown}"
echo "GPUs visible       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_GPUS_ON_NODE : ${SLURM_GPUS_ON_NODE:-unset}"
echo "CPUs/task          : ${SLURM_CPUS_PER_TASK:-unset}"
echo "Work dir           : $PWD"
echo "Conda env          : ${CONDA_DEFAULT_ENV:-unset}"
echo "LM eval path       : ${LM_EVAL_HARNESS_PATH}"

# CPU thread controls (override at submit time with --export=ALL,OMP_NUM_THREADS=...,TORCH_NUM_THREADS=...)
THREADS_DEFAULT="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_DEFAULT}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$THREADS_DEFAULT}"
echo "OMP_NUM_THREADS    : ${OMP_NUM_THREADS}"
echo "TORCH_NUM_THREADS  : ${TORCH_NUM_THREADS}"

# Run script defaults to TP=number of visible GPUs. Keep explicit TP=4 unless overridden.
export TP="${TP:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Run script not found: $RUN_SCRIPT" >&2
  exit 1
fi
echo "Run script         : ${RUN_SCRIPT}"

srun --ntasks=1 --gpus-per-task="$GPUS_ON_NODE" --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}"   bash "$RUN_SCRIPT"

echo "Finished at $(date)"
