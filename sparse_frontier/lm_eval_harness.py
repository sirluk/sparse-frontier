import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_kv_list(kv_list: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kv_list:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Empty key in {item!r}")
        try:
            value = json.loads(raw_value)
        except Exception:
            value = raw_value
        out[key] = value
    return out


def _infer_model_dims(
    pretrained: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Tuple[int, int, int]:
    try:
        from transformers import AutoConfig
    except Exception as e:
        raise RuntimeError(
            "transformers is required to infer model dimensions. Install it or pass "
            "--model-num-layers/--model-num-q-heads/--model-num-kv-heads."
        ) from e

    cfg = AutoConfig.from_pretrained(
        pretrained,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    def _first_attr(obj: Any, names: List[str]) -> Optional[int]:
        for name in names:
            value = getattr(obj, name, None)
            if value is None:
                continue
            return int(value)
        return None

    num_layers = _first_attr(
        cfg,
        ["num_hidden_layers", "n_layer", "num_layers", "n_layers"],
    )
    num_q_heads = _first_attr(
        cfg,
        ["num_attention_heads", "n_head", "num_heads"],
    )
    num_kv_heads = _first_attr(
        cfg,
        ["num_key_value_heads", "n_head_kv", "num_kv_heads"],
    )
    if num_kv_heads is None:
        num_kv_heads = num_q_heads

    missing = [
        name
        for name, value in (
            ("num_layers", num_layers),
            ("num_attention_heads", num_q_heads),
            ("num_key_value_heads", num_kv_heads),
        )
        if value is None
    ]
    if missing:
        raise RuntimeError(
            f"Could not infer {', '.join(missing)} from transformers config for {pretrained!r}. "
            "Pass explicit --model-num-* overrides."
        )

    assert num_layers is not None
    assert num_q_heads is not None
    assert num_kv_heads is not None
    return num_layers, num_q_heads, num_kv_heads


def _format_model_args_kv(model_args: Dict[str, Any]) -> str:
    def _fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, (int, float)):
            return str(v)
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return repr(v)
        return str(v)

    parts = []
    for key, value in model_args.items():
        if value is None:
            continue
        parts.append(f"{key}={_fmt(value)}")
    return ",".join(parts)


def _format_model_args_json(model_args: Dict[str, Any]) -> str:
    cleaned = {k: v for k, v in model_args.items() if v is not None}
    return json.dumps(cleaned, sort_keys=True)


def _detect_model_args_format(lm_eval_path: Optional[str]) -> str:
    """
    Determine whether lm_eval expects --model_args in JSON form (new CLI)
    or key=value,key=value form (legacy CLI).
    """
    def _looks_new_cli(text: str) -> bool:
        return "try_parse_json" in text and "MergeDictAction" in text

    if lm_eval_path:
        utils_py = Path(lm_eval_path) / "lm_eval" / "_cli" / "utils.py"
        try:
            text = utils_py.read_text(encoding="utf-8")
            return "json" if _looks_new_cli(text) else "kv"
        except Exception:
            pass

    try:
        import importlib.util

        spec = importlib.util.find_spec("lm_eval._cli.utils")
        if spec and spec.origin:
            text = Path(spec.origin).read_text(encoding="utf-8")
            return "json" if _looks_new_cli(text) else "kv"
    except Exception:
        pass

    return "kv"


def _get_flag_value(argv: List[str], flag: str) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == flag:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for {flag}")
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _contains_flag(argv: List[str], flag: str) -> bool:
    return _get_flag_value(argv, flag) is not None


def _check_sparse_frontier_vllm_entrypoint(enabled: bool) -> None:
    if not enabled:
        return

    try:
        from importlib.metadata import entry_points
    except Exception:
        entry_points = None

    if entry_points is None:
        return

    eps = entry_points()
    try:
        group = eps.select(group="vllm.general_plugins")
    except Exception:
        group = eps.get("vllm.general_plugins", [])

    if not any(ep.name == "swap_vllm_attention" for ep in group):
        raise RuntimeError(
            "Sparse Frontier vLLM plugin entrypoint not found. Install this repo "
            "into the same Python env as vLLM (e.g. `pip install -e .`)."
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sparse_frontier.lm_eval_harness",
        description=(
            "Run EleutherAI lm-evaluation-harness with vLLM while enabling "
            "Sparse Frontier's attention patch (SF_* env vars). "
            "Pass lm_eval CLI args after `--`."
        ),
    )

    parser.add_argument(
        "--lm-eval-path",
        type=str,
        default=None,
        help="Path to a local lm-eval-harness checkout.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run lm_eval with.",
    )
    parser.add_argument(
        "--lm-eval-model",
        type=str,
        default="vllm",
        help="lm_eval model backend name (default: vllm).",
    )
    parser.add_argument(
        "--model-args-format",
        choices=["auto", "json", "kv"],
        default="auto",
        help=(
            "Format used for lm_eval --model_args. "
            "`auto` selects JSON for newer harness CLI and key=value for legacy."
        ),
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="HF model name or local path for vLLM.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (passed to vLLM).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="vLLM dtype (e.g. bfloat16, float16, auto).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM max_model_len.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Sparse Frontier max input tokens (SF_MAX_INPUT_TOKENS).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Sparse Frontier max output tokens (SF_MAX_OUTPUT_TOKENS).",
    )
    parser.add_argument(
        "--kv-cache-block-size",
        type=int,
        default=16,
        help="Sparse Frontier KV cache block size (SF_KV_CACHE_BLOCK_SIZE).",
    )

    parser.add_argument(
        "--attention",
        type=str,
        required=True,
        help="Sparse Frontier attention name (e.g. dense, quest).",
    )
    parser.add_argument(
        "--attention-args-json",
        type=str,
        default="{}",
        help="JSON dict for SF_ATTENTION_ARGS_JSON (overridden by any --attention-arg).",
    )
    parser.add_argument(
        "--attention-arg",
        action="append",
        default=[],
        help=(
            "Override attention arg as KEY=VALUE (VALUE parsed as JSON when possible). "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--vllm-arg",
        action="append",
        default=[],
        help=(
            "Extra vLLM constructor arg as KEY=VALUE (VALUE parsed as JSON when possible). "
            "Repeatable."
        ),
    )

    parser.add_argument(
        "--model-num-layers",
        type=int,
        default=None,
        help="Override inferred number of layers.",
    )
    parser.add_argument(
        "--model-num-q-heads",
        type=int,
        default=None,
        help="Override inferred number of query heads.",
    )
    parser.add_argument(
        "--model-num-kv-heads",
        type=int,
        default=None,
        help="Override inferred number of KV heads.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use trust_remote_code when reading transformers config to infer dims.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not try to download configs from HF when inferring dims.",
    )

    parser.add_argument(
        "--disable-patch",
        action="store_true",
        help="Disable Sparse Frontier vLLM attention patch (sets SF_USE_ATTENTION_PATCH=0).",
    )
    parser.add_argument(
        "--force-torch-attention",
        action="store_true",
        help="Force Sparse Frontier to use torch attention instead of vLLM FA2 kernels.",
    )
    parser.add_argument(
        "--allow-batching",
        action="store_true",
        help=(
            "Allow lm_eval --batch_size != 1 and vLLM max_num_seqs != 1. "
            "WARNING: Sparse Frontier attention patch currently assumes B=1."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the lm_eval command and exit.",
    )

    args, passthrough = parser.parse_known_args(argv)

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if _contains_flag(passthrough, "--model") or _contains_flag(passthrough, "--model_args"):
        raise SystemExit("Do not pass --model/--model_args to lm_eval; the wrapper sets them.")

    if not args.allow_batching:
        batch_size = _get_flag_value(passthrough, "--batch_size")
        if batch_size is None:
            passthrough = ["--batch_size", "1", *passthrough]
        elif str(batch_size) != "1":
            raise SystemExit(
                f"lm_eval --batch_size must be 1 with Sparse Frontier patch "
                f"(got {batch_size!r}). Pass --allow-batching to override."
            )

    max_output_tokens = int(args.max_output_tokens)
    if max_output_tokens <= 0:
        raise SystemExit("--max-output-tokens must be > 0")

    if args.max_model_len is None and args.max_input_tokens is None:
        raise SystemExit("Provide at least one of --max-model-len or --max-input-tokens.")

    if args.max_model_len is None:
        max_input_tokens = int(args.max_input_tokens)
        max_model_len = max_input_tokens + max_output_tokens
    else:
        max_model_len = int(args.max_model_len)
        max_input_tokens = (
            int(args.max_input_tokens)
            if args.max_input_tokens is not None
            else max_model_len - max_output_tokens
        )

    if max_input_tokens <= 0:
        raise SystemExit(
            f"Derived --max-input-tokens is {max_input_tokens}, but must be > 0. "
            "Adjust --max-model-len/--max-output-tokens."
        )

    if (
        args.model_num_layers is not None
        and args.model_num_q_heads is not None
        and args.model_num_kv_heads is not None
    ):
        num_layers = int(args.model_num_layers)
        num_q_heads = int(args.model_num_q_heads)
        num_kv_heads = int(args.model_num_kv_heads)
    else:
        num_layers, num_q_heads, num_kv_heads = _infer_model_dims(
            pretrained=args.pretrained,
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        )
        if args.model_num_layers is not None:
            num_layers = int(args.model_num_layers)
        if args.model_num_q_heads is not None:
            num_q_heads = int(args.model_num_q_heads)
        if args.model_num_kv_heads is not None:
            num_kv_heads = int(args.model_num_kv_heads)

    tp = int(args.tp)
    if num_q_heads % tp != 0:
        raise SystemExit(f"num_q_heads ({num_q_heads}) must be divisible by tp ({args.tp}).")
    if num_kv_heads % tp != 0:
        raise SystemExit(f"num_kv_heads ({num_kv_heads}) must be divisible by tp ({args.tp}).")
    if num_q_heads % num_kv_heads != 0:
        raise SystemExit(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
        )

    try:
        attention_args = json.loads(args.attention_args_json) if args.attention_args_json else {}
    except Exception as e:
        raise SystemExit(f"Failed to parse --attention-args-json: {e}")
    attention_args.update(_parse_kv_list(list(args.attention_arg)))

    # Provide sensible defaults for Quest based on the repo's config conventions.
    if args.attention == "quest":
        attention_args.setdefault("page_size", int(args.kv_cache_block_size))
        attention_args.setdefault("share_pages", True)
        token_budget = attention_args.get("token_budget")
        page_size = attention_args.get("page_size")
        if token_budget is not None and page_size is not None:
            try:
                token_budget_int = int(token_budget)
                page_size_int = int(page_size)
            except Exception:
                raise SystemExit(
                    "Quest attention requires integer token_budget and page_size. "
                    "Pass e.g. --attention-arg token_budget=2048 --attention-arg page_size=16."
                )
            if page_size_int <= 0:
                raise SystemExit("Quest attention page_size must be > 0")
            if token_budget_int <= 0:
                raise SystemExit("Quest attention token_budget must be > 0")
            if token_budget_int % page_size_int != 0:
                raise SystemExit(
                    f"Quest attention requires token_budget divisible by page_size "
                    f"(got token_budget={token_budget_int}, page_size={page_size_int})."
                )
    attention_args_json = json.dumps(attention_args, sort_keys=True)

    extra_vllm_args = _parse_kv_list(list(args.vllm_arg))

    if not args.allow_batching:
        max_num_seqs = extra_vllm_args.get("max_num_seqs", 1)
        if int(max_num_seqs) != 1:
            raise SystemExit(
                f"vLLM max_num_seqs must be 1 with Sparse Frontier patch "
                f"(got {max_num_seqs!r}). Pass --allow-batching to override."
            )

    enabled_patch = not bool(args.disable_patch)
    _check_sparse_frontier_vllm_entrypoint(enabled=enabled_patch)

    env = os.environ.copy()

    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    env.setdefault("VLLM_FLASH_ATTN_VERSION", "2")
    # vLLM v1 enables FlashInfer sampling by default when flashinfer is installed,
    # which can trigger runtime JIT compilation. Disable by default for robustness
    # (can be re-enabled by exporting VLLM_USE_FLASHINFER_SAMPLER=1).
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    env["SF_USE_ATTENTION_PATCH"] = "0" if args.disable_patch else "1"
    if args.force_torch_attention:
        env["SF_FORCE_TORCH_ATTN"] = "1"

    if enabled_patch:
        env["SF_ATTENTION_NAME"] = str(args.attention)
        env["SF_ATTENTION_ARGS_JSON"] = attention_args_json
        env["SF_TP_SIZE"] = str(tp)
        env["SF_MODEL_NUM_Q_HEADS"] = str(int(num_q_heads))
        env["SF_MODEL_NUM_KV_HEADS"] = str(int(num_kv_heads))
        env["SF_MODEL_NUM_LAYERS"] = str(int(num_layers))
        env["SF_MAX_INPUT_TOKENS"] = str(int(max_input_tokens))
        env["SF_MAX_OUTPUT_TOKENS"] = str(int(max_output_tokens))
        env["SF_KV_CACHE_BLOCK_SIZE"] = str(int(args.kv_cache_block_size))

        plugin_name = "swap_vllm_attention"
        existing_plugins = env.get("VLLM_PLUGINS")
        if existing_plugins:
            names = [name.strip() for name in existing_plugins.split(",") if name.strip()]
            if plugin_name not in names:
                names.append(plugin_name)
            env["VLLM_PLUGINS"] = ",".join(names)
        else:
            env["VLLM_PLUGINS"] = plugin_name

    if args.lm_eval_path is not None:
        lm_eval_path = os.path.abspath(args.lm_eval_path)
        env["PYTHONPATH"] = lm_eval_path + os.pathsep + env.get("PYTHONPATH", "")
        cwd = lm_eval_path
    else:
        cwd = None

    model_args = {
        "pretrained": args.pretrained,
        "tensor_parallel_size": int(args.tp),
        "dtype": args.dtype,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(max_model_len),
        "max_num_seqs": 1 if not args.allow_batching else None,
        "max_num_batched_tokens": int(max_model_len),
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "disable_hybrid_kv_cache_manager": True,
        "disable_sliding_window": True,
        "limit_mm_per_prompt": {"image": 0},
    }
    model_args.update(extra_vllm_args)
    if args.model_args_format == "auto":
        detected = _detect_model_args_format(args.lm_eval_path)
    else:
        detected = args.model_args_format
    model_args_str = (
        _format_model_args_json(model_args)
        if detected == "json"
        else _format_model_args_kv(model_args)
    )

    cmd = [
        args.python,
        "-m",
        "lm_eval",
        "--model",
        args.lm_eval_model,
        "--model_args",
        model_args_str,
        *passthrough,
    ]

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    return subprocess.call(cmd, env=env, cwd=cwd)


if __name__ == "__main__":
    raise SystemExit(main())
