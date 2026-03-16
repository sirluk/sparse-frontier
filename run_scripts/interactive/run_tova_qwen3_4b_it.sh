#!/bin/bash
set -euo pipefail

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

PYTHON_BIN="${PYTHON_BIN:-/nfs-gpu/xlstm-distillation/miniconda3/envs/sparse-frontier/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

export WORKDIR=/nfs-gpu/xlstm-distillation/work_lukas/sparse-frontier
export PYTHONPATH=$WORKDIR:${PYTHONPATH:-}

cd "$WORKDIR"

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

# model config
export PRETRAINED=Qwen/Qwen3-4B-Instruct-2507
export ATTENTION=tova
export TOKEN_BUDGET=2048
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

"$PYTHON_BIN" -m sparse_frontier.lm_eval_harness \
  --lm-eval-path /nfs-gpu/xlstm-distillation/work_lukas/lm-evaluation-harness \
  --model-args-format json \
  --pretrained "$PRETRAINED" \
  --attention "$ATTENTION" \
  --attention-arg "token_budget=$TOKEN_BUDGET" \
  --max-input-tokens 8192 \
  --max-output-tokens 7168 \
  --tp "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --force-torch-attention \
  -- \
  --tasks custom_aime2024_agg8_verify_instruct \
  --num_fewshot 0 \
  --apply_chat_template \
  --gen_kwargs "max_gen_toks=7168,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,do_sample=True" \
  --output_path "$OUTPUT_PATH"
