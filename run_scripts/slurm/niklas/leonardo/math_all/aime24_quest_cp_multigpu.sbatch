#!/bin/bash
###############################################################################
# Sparse Frontier Quest AIME24 CP - Leonardo Booster: 1 Node, 4 GPUs total
###############################################################################
#SBATCH --job-name=eval-1n4g
#SBATCH --account=euhpc_d31_137
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/leonardo_work/EUHPC_D31_137/sparse-frontier
#SBATCH --output=/leonardo_work/EUHPC_D31_137/sparse-frontier/slurm_logs_eval/slurm_%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D31_137/sparse-frontier/slurm_logs_eval/slurm_%x_%j.err
#SBATCH --wait-all-nodes=1
#SBATCH --hint=nomultithread
###############################################################################

set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22

module list

echo "=== Node info ========================================"
nvidia-smi || true
free -m || true
df -h . || true
echo "Starting job on $(date)"
echo "JobID        ${SLURM_JOBID:-unknown}"
echo "Node list    ${SLURM_JOB_NODELIST:-unknown}"
echo "GPUs total   4"
echo "CPUs/task    ${SLURM_CPUS_PER_TASK:-unset}"
echo "Work dir     ${SLURM_SUBMIT_DIR:-$PWD}"
echo "======================================================"

export WORKDIR="${WORKDIR:-/leonardo_work/EUHPC_D31_137/sparse-frontier}"
export LEONARDO_BASHRC="${LEONARDO_BASHRC:-/leonardo/home/userexternal/nschmidi/.bashrc}"
export CONDA_ENV="${CONDA_ENV:-sparse_frontier}"
export LM_EVAL_HARNESS_PATH="${LM_EVAL_HARNESS_PATH:-/leonardo_work/EUHPC_D31_137/lm-evaluation-harness}"
export RUN_SCRIPT="${RUN_SCRIPT:-${WORKDIR}/run_scripts/interactive/run_quest_cp.sh}"
export SLURM_GPUS_ON_NODE=4
unset CUDA_VISIBLE_DEVICES

THREADS_PER_GPU=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS_ON_NODE ))
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_GPU}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$THREADS_PER_GPU}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS_PER_GPU}"

export TP="${TP:-4}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-2048}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-15360}"
export TASKS="${TASKS:-custom_aime2024_agg8_verify_instruct}"
export NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
export GEN_KWARGS="${GEN_KWARGS:-max_gen_toks=15360,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,do_sample=True}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MODEL_TAG="${MODEL_TAG:-quest_cp}"
if [[ -n "${experiment_name:-}" && -z "${RUN_TAG:-}" ]]; then
  export RUN_TAG="${experiment_name}"
fi

if [[ -z "${PRETRAINED:-}" ]]; then
  echo "ERROR: PRETRAINED must be set to a local checkpoint/snapshot path for the cp launcher." >&2
  echo "Example: sbatch --export=ALL,experiment_name=my_exp,PRETRAINED=/path/to/model $0" >&2
  exit 2
fi

if [[ -f "$LEONARDO_BASHRC" ]]; then
  echo "Activating conda env from ${LEONARDO_BASHRC}"
  set +u
  source "$LEONARDO_BASHRC"
  set -u
  conda activate "$CONDA_ENV"
else
  echo "WARNING: Leonardo bashrc not found: ${LEONARDO_BASHRC}" >&2
fi

echo "OMP_NUM_THREADS : ${OMP_NUM_THREADS}"
echo "TORCH_NUM_THREADS: ${TORCH_NUM_THREADS}"
echo "Run script       : ${RUN_SCRIPT}"
echo "Pretrained       : ${PRETRAINED}"
echo "Gen kwargs       : ${GEN_KWARGS}"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Run script not found: $RUN_SCRIPT" >&2
  exit 1
fi

srun --ntasks=1 --gpus-per-task="$SLURM_GPUS_ON_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK"   bash "$RUN_SCRIPT"

echo "Finished on $(date)"
