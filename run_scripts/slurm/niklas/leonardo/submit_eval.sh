#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
  cat <<'EOF'
Usage:
  submit_eval.sh <sbatch target> <experiment_name> [pretrained_path] [sbatch args...]

Examples:
  submit_eval.sh math_all/aime24_quest_qwen3_4b_it_multigpu.sbatch my_qwen_aime24
  submit_eval.sh math_all/aime24_quest_cp_multigpu.sbatch my_cp_aime24 /path/to/custom_checkpoint
  submit_eval.sh math_all/aime24_quest_qwen3_4b_it_multigpu.sbatch my_qwen_aime24 --time=02:00:00
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

resolve_sbatch_target() {
  local target="$1"
  local resolved=""

  if [[ -f "$target" ]]; then
    resolved="$target"
  elif [[ -f "$SCRIPT_DIR/$target" ]]; then
    resolved="$SCRIPT_DIR/$target"
  elif [[ -f "$SCRIPT_DIR/$target.sbatch" ]]; then
    resolved="$SCRIPT_DIR/$target.sbatch"
  else
    die "Could not find sbatch target: $target"
  fi

  local resolved_dir
  resolved_dir="$(cd -- "$(dirname -- "$resolved")" && pwd -P)"
  echo "$resolved_dir/$(basename -- "$resolved")"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -lt 2 ]]; then
  usage >&2
  exit 1
fi

sbatch_target="$1"
experiment_name="$2"
shift 2

CONDA_ENV="${CONDA_ENV:-sparse_frontier}"

pretrained_path=""
if [[ "$#" -gt 0 && "${1}" != -* ]]; then
  pretrained_path="$1"
  shift
fi

[[ -n "$experiment_name" ]] || die "experiment_name must be non-empty"

sbatch_file="$(resolve_sbatch_target "$sbatch_target")"

has_export=0
for arg in "$@"; do
  case "$arg" in
    --export|--export=*) has_export=1 ;;
  esac
done

sbatch_opts=()
if [[ "$has_export" -eq 0 ]]; then
  sbatch_opts+=(--export=ALL,experiment_name,PRETRAINED,CONDA_ENV)
fi

echo "Submitting sbatch: $sbatch_file"
echo "  experiment_name: $experiment_name"
if [[ -n "$pretrained_path" ]]; then
  echo "  pretrained:      $pretrained_path"
else
  echo "  pretrained:      <launcher default>"
fi
echo "  conda_env:       $CONDA_ENV"

experiment_name="$experiment_name" PRETRAINED="$pretrained_path" CONDA_ENV="$CONDA_ENV"   sbatch "${sbatch_opts[@]}" "$@" "$sbatch_file"
