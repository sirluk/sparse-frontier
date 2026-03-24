#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"

module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.6

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sparse_frontier}"
WORK_ROOT="${WORK_ROOT:-/leonardo_work/EUHPC_D31_137}"
LM_EVAL_HARNESS_PATH="${LM_EVAL_HARNESS_PATH:-$WORK_ROOT/lm-evaluation-harness}"
PRETRAINED="${PRETRAINED:-/leonardo_work/EUHPC_D31_132/models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
GEN_KWARGS="${GEN_KWARGS:-max_gen_toks=15360,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,do_sample=True}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda activation script not found: $CONDA_SH" >&2
  exit 1
fi
if [[ ! -d "$LM_EVAL_HARNESS_PATH" ]]; then
  echo "lm-evaluation-harness checkout not found: $LM_EVAL_HARNESS_PATH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python - <<'INNERPY'
import importlib
import importlib.metadata as md
import torch

mods = [
    'torch',
    'transformers',
    'vllm',
    'flashinfer',
    'lm_eval',
    'sparse_frontier',
    'sparse_frontier.lm_eval_harness',
]
for name in mods:
    mod = importlib.import_module(name)
    version = getattr(mod, '__version__', None)
    if version is None:
        try:
            version = md.version(name.split('.')[0])
        except Exception:
            version = 'unknown'
    print(f'OK\t{name}\t{version}')

try:
    import sparse_frontier.modelling.attention.minference.minference as _minference_ext
except Exception as exc:
    print(f'WARN\tminference\t{type(exc).__name__}: {exc}')
else:
    print(f'OK\tminference\t{_minference_ext.__file__}')

eps = md.entry_points()
try:
    group = eps.select(group='vllm.general_plugins')
except Exception:
    group = eps.get('vllm.general_plugins', [])
assert any(ep.name == 'swap_vllm_attention' for ep in group), 'swap_vllm_attention entrypoint missing'
print('OK\tvllm.general_plugins\tswap_vllm_attention')
print('torch.cuda.is_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu0', torch.cuda.get_device_name(0))
INNERPY

python -m sparse_frontier.lm_eval_harness           --dry-run           --lm-eval-path "$LM_EVAL_HARNESS_PATH"           --model-args-format json           --pretrained "$PRETRAINED"           --attention quest           --attention-arg token_budget=2048           --max-input-tokens 16384           --max-output-tokens 15360           --tp 1           --gpu-memory-utilization 0.85           --force-torch-attention           --           --tasks custom_aime2024_agg8_verify_instruct           --num_fewshot 0           --apply_chat_template           --gen_kwargs "$GEN_KWARGS"           --output_path /tmp/sparse_frontier_smoke.json
