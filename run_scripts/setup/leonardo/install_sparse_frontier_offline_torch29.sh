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
DEFAULT_WHEELHOUSE_PRIMARY="${WORK_ROOT}/wheelhouse/sparse_frontier"
DEFAULT_WHEELHOUSE_LEGACY="${WORK_ROOT}/wheelhouse/sparse_frontier_torch29"
REQ_FILE="${REQ_FILE:-${SCRIPT_DIR}/requirements_sparse_frontier_torch29.txt}"
LM_EVAL_HARNESS_PATH="${LM_EVAL_HARNESS_PATH:-$WORK_ROOT/lm-evaluation-harness}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.5.2}"
FLASHINFER_JIT_CACHE_SPEC="${FLASHINFER_JIT_CACHE_SPEC:-flashinfer-jit-cache==${FLASHINFER_VERSION}+cu128}"
BUILD_MINFERENCE="${BUILD_MINFERENCE:-1}"
MINFERENCE_BUILD_LIB="${MINFERENCE_BUILD_LIB:-$REPO_ROOT/sparse_frontier/modelling/attention/minference}"

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
if [[ ! -d "$WHEELHOUSE" ]]; then
  echo "Wheelhouse not found: $WHEELHOUSE" >&2
  exit 1
fi
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Requirements file not found: $REQ_FILE" >&2
  exit 1
fi
if [[ ! -d "$LM_EVAL_HARNESS_PATH" ]]; then
  echo "lm-evaluation-harness checkout not found: $LM_EVAL_HARNESS_PATH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$WHEELHOUSE/tmp" "$WHEELHOUSE/cache"
export TMPDIR="$WHEELHOUSE/tmp"
export PIP_CACHE_DIR="$WHEELHOUSE/cache"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export CC="$(which gcc)"
export CXX="$(which g++)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "Using repo root : $REPO_ROOT"
echo "Using wheelhouse: $WHEELHOUSE"
echo "Using env       : $CONDA_ENV"

echo "Installing build helpers from wheelhouse"
python -m pip install --no-index --find-links="$WHEELHOUSE" --upgrade           pip setuptools wheel packaging ninja psutil

echo "Installing Sparse Frontier runtime deps from wheelhouse"
python -m pip install --no-index --find-links="$WHEELHOUSE" --upgrade           -r "$REQ_FILE"

echo "Removing any pre-existing flashinfer-jit-cache to avoid version mismatches"
python -m pip uninstall -y flashinfer-jit-cache || true

if compgen -G "$WHEELHOUSE/flashinfer_jit_cache-${FLASHINFER_VERSION}+cu128-*.whl" > /dev/null; then
  echo "Installing optional flashinfer-jit-cache from wheelhouse: $FLASHINFER_JIT_CACHE_SPEC"
  python -m pip install --no-index --find-links="$WHEELHOUSE" --upgrade             "$FLASHINFER_JIT_CACHE_SPEC"
elif compgen -G "$WHEELHOUSE/flashinfer_jit_cache-*.whl" > /dev/null; then
  echo "Found flashinfer-jit-cache wheel(s) in $WHEELHOUSE, but none match $FLASHINFER_JIT_CACHE_SPEC; skipping to avoid a version mismatch." >&2
else
  echo "No flashinfer-jit-cache wheel found in $WHEELHOUSE; continuing without it." >&2
fi

echo "Installing local lm-evaluation-harness checkout"
python -m pip install --no-deps --no-build-isolation -e "$LM_EVAL_HARNESS_PATH"

echo "Installing local Sparse Frontier checkout"
python -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT"

if [[ "$BUILD_MINFERENCE" == "1" ]]; then
  if ! command -v nvcc > /dev/null; then
    echo "nvcc not found; cannot build the optional minference CUDA extension." >&2
    exit 1
  fi
  echo "Building local minference CUDA extension"
  python "$REPO_ROOT/compile.py" build_ext --inplace --build-lib "$MINFERENCE_BUILD_LIB"
else
  echo "Skipping minference CUDA extension build because BUILD_MINFERENCE=$BUILD_MINFERENCE"
fi

echo "Running pip check"
python -m pip check
