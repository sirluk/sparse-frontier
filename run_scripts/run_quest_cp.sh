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

export PRETRAINED=/nfs-gpu/xlstm-distillation/lolcat_models/Qwen/CustomQwen3_it_4b_mathv2medium_8192_5000
export ATTENTION=quest
export TOKEN_BUDGET=2048
echo "Pretrained model: $PRETRAINED"
echo "Attention mechanism: $ATTENTION with token budget: $TOKEN_BUDGET"

"$PYTHON_BIN" -m sparse_frontier.lm_eval_harness \
  --lm-eval-path /nfs-gpu/xlstm-distillation/work_lukas/lm-evaluation-harness \
  --model-args-format json \
  --pretrained "$PRETRAINED" \
  --attention $ATTENTION \
  --attention-arg token_budget=$TOKEN_BUDGET \
  --max-input-tokens 8192 \
  --max-output-tokens 7168 \
  --tp 1 \
  --force-torch-attention \
  -- \
  --tasks custom_aime2024_agg8_verify_instruct \
  --num_fewshot 0 \
  --apply_chat_template \
  --gen_kwargs "max_gen_toks=7168,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,do_sample=True" \
  --output_path /nfs-gpu/xlstm-distillation/work_lukas/sparse-frontier/results/lm_eval.json
