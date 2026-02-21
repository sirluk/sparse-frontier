#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/nfs-gpu/xlstm-distillation/miniconda3/envs/sparse_frontier/bin/python}"
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
  -- \
  --tasks hellaswag,arc_easy,piqa \
  --num_fewshot 0 \
  --output_path lm_eval.json
