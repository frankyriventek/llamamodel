"""Parse model card / README for recommended LLAMA_ARG_* parameters."""

import re
from typing import Any

# Map common names in model cards to LLAMA_ARG_* keys
PARAM_PATTERNS = [
    (r"(?:context\s*size|ctx[- ]?size|n_ctx|n-ctx)[\s:=]+(\d+)", "LLAMA_ARG_N_CTX"),
    (r"(?:context\s*length)[\s:=]+(\d+)", "LLAMA_ARG_N_CTX"),
    (r"(?:n_ctx)[\s:=]+(\d+)", "LLAMA_ARG_N_CTX"),
    (r"(?:n[-_]?gpu[-_]?layers|ngl)[\s:=]+(-?\d+)", "LLAMA_ARG_N_GPU_LAYERS"),
    (r"(?:gpu[- ]?layers)[\s:=]+(-?\d+)", "LLAMA_ARG_N_GPU_LAYERS"),
    (r"(?:batch[- ]?size|batch_size)[\s:=]+(\d+)", "LLAMA_ARG_BATCH"),
    (r"(?:threads)[\s:=]+(-?\d+)", "LLAMA_ARG_THREADS"),
    (r"(?:n[-_]?predict|n_predict)[\s:=]+(-?\d+)", "LLAMA_ARG_N_PREDICT"),
]

# Safe defaults when nothing is found
DEFAULT_PARAMS: dict[str, str] = {
    "LLAMA_ARG_N_CTX": "4096",
    "LLAMA_ARG_N_GPU_LAYERS": "-1",
}


def parse_recommended_params(model_card_content: str) -> dict[str, str]:
    """
    Extract recommended parameters from model card text.
    Returns dict of LLAMA_ARG_* -> value. Does not include defaults; caller can merge.
    """
    content = model_card_content or ""
    out: dict[str, str] = {}
    for pattern, arg_name in PARAM_PATTERNS:
        if arg_name in out:
            continue
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            out[arg_name] = m.group(1).strip()
    return out


def recommended_params_with_defaults(model_card_content: str) -> dict[str, str]:
    """Parse model card and merge with safe defaults (defaults only for missing keys)."""
    parsed = parse_recommended_params(model_card_content)
    result = dict(DEFAULT_PARAMS)
    result.update(parsed)
    return result
