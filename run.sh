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

export PYTHONPATH=/nfs-gpu/xlstm-distillation/work_lukas/sparse-frontier:${PYTHONPATH:-}

"$PYTHON_BIN" -m sparse_frontier.lm_eval_harness \
  --lm-eval-path /nfs-gpu/xlstm-distillation/work_lukas/lm-evaluation-harness \
  --model-args-format json \
  --pretrained Qwen/Qwen2.5-7B-Instruct \
  --attention quest \
  --attention-arg token_budget=2048 \
  --max-input-tokens 8192 \
  --max-output-tokens 256 \
  --tp 1 \
  --force-torch-attention \
  -- \
  --tasks hellaswag,arc_easy,piqa \
  --num_fewshot 0 \
  --output_path lm_eval.json
