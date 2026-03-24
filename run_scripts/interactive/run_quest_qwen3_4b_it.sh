#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CONDA_CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  CONDA_CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  if [[ -x "$CONDA_CC" && -x "$CONDA_CXX" ]]; then
    export CC="$CONDA_CC"
    export CXX="$CONDA_CXX"
    echo "Using CC: $CC"
    echo "Using CXX: $CXX"
  fi
fi

# vLLM v1 enables FlashInfer sampling by default if flashinfer is installed, which
# can trigger JIT compilation at runtime. Disable by default for robustness.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

export WORKDIR="${WORKDIR:-$REPO_ROOT}"
export LM_EVAL_HARNESS_PATH="${LM_EVAL_HARNESS_PATH:-/leonardo_work/EUHPC_D31_137/lm-evaluation-harness}"
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:$PYTHONPATH}"

cd "$WORKDIR"

if [[ ! -d "$LM_EVAL_HARNESS_PATH" ]]; then
  echo "lm-evaluation-harness path not found: $LM_EVAL_HARNESS_PATH" >&2
  exit 1
fi

parse_first_int() {
  local value="$1"
  local parsed=""
  parsed="$(echo "$value" | grep -oE '[0-9]+' | head -n1 || true)"
  echo "$parsed"
}

VISIBLE_GPU_COUNT=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  VISIBLE_GPU_COUNT="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  VISIBLE_GPU_COUNT="$(parse_first_int "$SLURM_GPUS_ON_NODE")"
fi

TP_DEFAULT=1
if [[ -n "$VISIBLE_GPU_COUNT" ]]; then
  TP_DEFAULT="$VISIBLE_GPU_COUNT"
fi
TP="${TP:-$TP_DEFAULT}"
if ! [[ "$TP" =~ ^[0-9]+$ ]] || [[ "$TP" -lt 1 ]]; then
  echo "TP must be a positive integer (got: $TP)" >&2
  exit 1
fi
if [[ -n "$VISIBLE_GPU_COUNT" && "$TP" -gt "$VISIBLE_GPU_COUNT" ]]; then
  echo "TP=$TP exceeds visible GPUs=$VISIBLE_GPU_COUNT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2
  exit 1
fi

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# model and eval config
DEFAULT_PRETRAINED="/leonardo_work/EUHPC_D31_132/models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
export PRETRAINED="${PRETRAINED:-$DEFAULT_PRETRAINED}"
export ATTENTION="${ATTENTION:-quest}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-2048}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-16384}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-15360}"
TASKS="${TASKS:-custom_aime2024_agg8_verify_instruct}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
GEN_KWARGS="${GEN_KWARGS:-max_gen_toks=${MAX_OUTPUT_TOKENS},temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,do_sample=True}"
if [[ ! -e "$PRETRAINED" ]]; then
  echo "Pretrained model path not found: $PRETRAINED" >&2
  exit 1
fi
MODEL_TAG="${MODEL_TAG:-$(basename "$PRETRAINED")}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:+job${SLURM_JOB_ID}}}"
if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_PATH="${OUTPUT_PATH:-$WORKDIR/results/lm_eval_${ATTENTION}_${MODEL_TAG}_tp${TP}_${RUN_TAG}.json}"
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Tensor parallel size (tp): $TP"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Output path: $OUTPUT_PATH"
echo "Pretrained model: $PRETRAINED"
echo "Attention mechanism: $ATTENTION with token budget: $TOKEN_BUDGET"
echo "Task(s): $TASKS"
echo "Max input tokens: $MAX_INPUT_TOKENS"
echo "Max output tokens: $MAX_OUTPUT_TOKENS"
echo "Generation kwargs: $GEN_KWARGS"
echo "LM eval path: $LM_EVAL_HARNESS_PATH"

cmd=(
  "$PYTHON_BIN"
  -m
  sparse_frontier.lm_eval_harness
  --lm-eval-path
  "$LM_EVAL_HARNESS_PATH"
  --model-args-format
  json
  --pretrained
  "$PRETRAINED"
  --attention
  "$ATTENTION"
  --attention-arg
  "token_budget=$TOKEN_BUDGET"
  --max-input-tokens
  "$MAX_INPUT_TOKENS"
  --max-output-tokens
  "$MAX_OUTPUT_TOKENS"
  --tp
  "$TP"
  --gpu-memory-utilization
  "$GPU_MEMORY_UTILIZATION"
  --force-torch-attention
  --
  --tasks
  "$TASKS"
  --num_fewshot
  "$NUM_FEWSHOT"
  --apply_chat_template
  --gen_kwargs
  "$GEN_KWARGS"
  --output_path
  "$OUTPUT_PATH"
)

"${cmd[@]}"
