# Setup on Leonardo

This document describes how to install Sparse Frontier on Leonardo when:

- the login node has internet access but no GPUs
- the compute node has GPUs but no internet access

The repository already includes helper scripts for this workflow under `run_scripts/setup/leonardo/`.

## Environment

Sparse Frontier requires Python `>=3.10`.

The setup below assumes a conda environment named `sparse_frontier`.

You can install this via `conda create -n sparse_frontier python=3.12`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sparse_frontier
python -V
```

## Overview

The setup is split into two phases:

1. On the login node, download all required wheels into a local wheelhouse.
2. On the GPU node, install from that wheelhouse without internet access.

There is also an optional smoke test script to verify the environment afterwards.

## 1. Login Node: Download Wheels

Set `WORK_ROOT` env variable to your project e.g. `/leonardo_work/EUHPC_D31_137`.

Run:

```bash
CONDA_ENV=sparse_frontier \
bash run_scripts/setup/leonardo/download_sparse_frontier_wheels.sh
```

By default, the script will use:

- wheelhouse: `${WORK_ROOT}/wheelhouse/sparse_frontier`
- legacy fallback wheelhouse: `${WORK_ROOT}/wheelhouse/sparse_frontier_torch29`

The script downloads the runtime stack for Sparse Frontier, vLLM, FlashInfer, and the local `lm-evaluation-harness` integration dependencies.

## 2. GPU Node: Offline Install

Run:

```bash
CONDA_ENV=sparse_frontier \
bash run_scripts/setup/leonardo/install_sparse_frontier_offline.sh
```

This script:

- loads the Leonardo compiler and CUDA modules
- installs dependencies from the local wheelhouse with `--no-index`
- installs the local `lm-evaluation-harness` checkout in editable mode
- installs this Sparse Frontier checkout in editable mode
- builds the local `minference` CUDA extension used by `vertical_and_slash` and `block_sparse`

If you do not need that CUDA extension, you can skip it with:

```bash
CONDA_ENV=sparse_frontier BUILD_MINFERENCE=0 \
bash run_scripts/setup/leonardo/install_sparse_frontier_offline.sh
```

## 3. Verification

Run:

```bash
CONDA_ENV=sparse_frontier \
bash run_scripts/setup/leonardo/smoke_test_sparse_frontier.sh
```

The smoke test checks:

- core imports like `torch`, `transformers`, `vllm`, `flashinfer`, `lm_eval`
- the Sparse Frontier vLLM plugin entry point
- the compiled `minference` extension
- a dry-run `lm_eval` command through `sparse_frontier.lm_eval_harness`

On the login node, `torch.cuda.is_available()` being `False` is expected.

## Notes

- `flash-attn` is not required for the default Sparse Frontier + vLLM + FlashInfer setup in this repository.
- The installer avoids mismatched `flashinfer-jit-cache` wheels. If no matching cache wheel is available, installation continues without it.
- On offline compute nodes, use local model checkpoint paths for `--pretrained` instead of relying on first-run Hugging Face downloads.
- The default local `lm-evaluation-harness` checkout path is `${WORK_ROOT}/lm-evaluation-harness`.

## Scripts

- `run_scripts/setup/leonardo/download_sparse_frontier_wheels.sh`
- `run_scripts/setup/leonardo/install_sparse_frontier_offline.sh`
- `run_scripts/setup/leonardo/smoke_test_sparse_frontier.sh`
