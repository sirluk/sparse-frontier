#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sparse_frontier}"
WORK_ROOT="${WORK_ROOT:-/leonardo_work/EUHPC_D31_137}"
DEFAULT_WHEELHOUSE_PRIMARY="${WORK_ROOT}/wheelhouse/sparse_frontier"
DEFAULT_WHEELHOUSE_LEGACY="${WORK_ROOT}/wheelhouse/sparse_frontier_torch29"
REQ_FILE="${REQ_FILE:-${SCRIPT_DIR}/requirements_sparse_frontier_torch29.txt}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/cu128}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.5.2}"
FLASHINFER_JIT_CACHE_SPEC="${FLASHINFER_JIT_CACHE_SPEC:-flashinfer-jit-cache==${FLASHINFER_VERSION}+cu128}"
DOWNLOAD_FLASHINFER_JIT_CACHE="${DOWNLOAD_FLASHINFER_JIT_CACHE:-1}"

if [[ -n "${WHEELHOUSE:-}" ]]; then
  WHEELHOUSE="${WHEELHOUSE}"
elif [[ -d "$DEFAULT_WHEELHOUSE_PRIMARY" || ! -d "$DEFAULT_WHEELHOUSE_LEGACY" ]]; then
  WHEELHOUSE="$DEFAULT_WHEELHOUSE_PRIMARY"
else
  WHEELHOUSE="$DEFAULT_WHEELHOUSE_LEGACY"
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda activation script not found: $CONDA_SH" >&2
  exit 1
fi
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Requirements file not found: $REQ_FILE" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$WHEELHOUSE"

echo "Using repo root : $REPO_ROOT"
echo "Using wheelhouse: $WHEELHOUSE"
echo "Using env       : $CONDA_ENV"

echo "Downloading build helpers"
python -m pip download --dest "$WHEELHOUSE" --prefer-binary           pip setuptools wheel packaging ninja psutil

echo "Downloading Sparse Frontier / vLLM / lm-eval runtime wheels"
python -m pip download --dest "$WHEELHOUSE" --prefer-binary           --extra-index-url "$PYTORCH_INDEX_URL"           -r "$REQ_FILE"

if [[ "$DOWNLOAD_FLASHINFER_JIT_CACHE" == "1" ]]; then
  echo "Attempting to download optional flashinfer-jit-cache wheel: $FLASHINFER_JIT_CACHE_SPEC"
  if ! python -m pip download --dest "$WHEELHOUSE" --prefer-binary             --index-url "$FLASHINFER_INDEX_URL"             "$FLASHINFER_JIT_CACHE_SPEC"; then
    echo "WARNING: flashinfer-jit-cache download failed. Install can still proceed," >&2
    echo "but FlashInfer may need local JIT work on first use." >&2
  fi
fi

echo "Downloaded artifacts:"
find "$WHEELHOUSE" -maxdepth 1 -type f | sort

printf "\nCritical wheel check:\n"
for pattern in 'vllm-*.whl' 'flashinfer_python-*.whl' 'flashinfer_cubin-*.whl'; do
  if compgen -G "$WHEELHOUSE/$pattern" > /dev/null; then
    echo "  OK  $pattern"
  else
    echo "  WARN missing wheel matching $pattern" >&2
  fi
done
