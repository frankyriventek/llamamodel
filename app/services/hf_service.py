"""Hugging Face Hub integration: search GGUF models, list files, model card, download."""

import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, ModelCard

# Cache for list_repo_files (quantizations) to avoid repeated API calls
_repo_files_cache: dict[str, tuple[float, list[str]]] = {}
_CACHE_TTL_SEC = 300  # 5 minutes


def _get_api() -> HfApi:
    return HfApi()


def _infer_capabilities_from_text(text: str, model_id: str) -> dict[str, bool]:
    """Infer vision, tools, thinking from model id and/or card text."""
    s = (text or "").lower() + " " + (model_id or "").lower()
    return {
        "vision": "vision" in s or "visual" in s or "vlm" in s or "multimodal" in s or "image-text" in s,
        "tools": (
            "tool" in s or "function call" in s or "function_call" in s or "function-call" in s
            or "tool-calling" in s or "tool_calling" in s or "tool-use" in s or "tool_use" in s
            or "tool use" in s or "agent" in s
        ),
        "thinking": "thinking" in s or "reasoning" in s or "deepseek" in s or "r1" in s,
    }


def _infer_capabilities_from_tags(tags: list[str] | None) -> dict[str, bool]:
    """Infer vision, tools, thinking from Hugging Face model tags."""
    if not tags:
        return {"vision": False, "tools": False, "thinking": False}
    t = " ".join(t.lower() for t in tags)
    return {
        "vision": "vision" in t or "multimodal" in t or "image-text" in t or "vlm" in t,
        "tools": (
            "tool" in t or "function-call" in t or "function-calling" in t
            or "tool-calling" in t or "tool_calling" in t or "tool-use" in t or "tool_use" in t
            or "tool use" in t or "agent" in t
        ),
        "thinking": "thinking" in t or "reasoning" in t or "reasoner" in t,
    }


def _repo_name(model_id: str) -> str:
    """Return only the repository name (last segment of model_id)."""
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _parse_size_from_repo(repo_name: str) -> str:
    """Parse model size from repo name (e.g. 7B, 0.5B, 70B, 1.5B, 8GB). Returns empty string if not found."""
    import re
    m = re.search(r"(\d+\.?\d*)\s*([BGM])(?:\s|$|[_-])", repo_name, re.IGNORECASE)
    if m:
        return m.group(1) + m.group(2).upper()
    return ""


def _extract_quantization(filename: str) -> str:
    """Extract quantization tag from GGUF filename. Formats: Q(n)_, FP(n), F(n), IQ(n)_, BF(n)."""
    import re
    base = filename.removesuffix(".gguf").removesuffix(".GGUF")
    parts = base.split("-")
    for p in reversed(parts):
        p = p.strip()
        if not p:
            continue
        if re.match(r"^Q\d", p, re.IGNORECASE):
            return p.upper()
        if re.match(r"^FP\d+$", p, re.IGNORECASE):
            return p.upper()
        if re.match(r"^F\d+$", p, re.IGNORECASE):
            return p.upper()
        if re.match(r"^IQ\d", p, re.IGNORECASE):
            return p.upper()
        if re.match(r"^BF\d+$", p, re.IGNORECASE):
            return p.upper()
    return ""


def group_gguf_by_quantization(gguf_files: list[str]) -> list[dict[str, Any]]:
    """
    Group GGUF files by quantization. Multifile models have -00001-of-00003 etc.
    Returns list of {quant, primary_file, all_files}. Primary is the single file or the first shard.
    """
    by_quant: dict[str, list[str]] = {}
    for f in gguf_files:
        q = _extract_quantization(f)
        if not q:
            q = f.removesuffix(".gguf").removesuffix(".GGUF").split("-")[-1].upper() or f
        by_quant.setdefault(q, []).append(f)
    result = []
    for quant, files in by_quant.items():
        files = sorted(files)
        primary = None
        for f in files:
            if "-00001-of-" in f or "-0001-of-" in f:
                primary = f
                break
        if primary is None:
            primary = files[0]
        result.append({"quant": quant, "primary_file": primary, "all_files": files})
    return result


def search_models(
    query: str | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "downloads",
) -> list[dict[str, Any]]:
    """Search for GGUF models. Returns list of {id, repo_name, author, downloads, capabilities, ...}."""
    api = _get_api()
    full_limit = offset + limit
    try:
        it = api.list_models(
            filter="gguf",
            search=query or "",
            sort=sort,
            limit=min(full_limit, 100),
            full=True,
        )
    except TypeError:
        it = api.list_models(
            filter="gguf",
            search=query or "",
            sort=sort,
            limit=min(full_limit, 100),
        )
    items = []
    for i, m in enumerate(it):
        if i < offset:
            continue
        if len(items) >= limit:
            break
        tags = getattr(m, "tags", None) or []
        if isinstance(tags, (list, tuple)):
            caps_from_tags = _infer_capabilities_from_tags(list(tags))
        else:
            caps_from_tags = {"vision": False, "tools": False, "thinking": False}
        caps_from_id = _infer_capabilities_from_text("", m.id)
        caps = {
            "vision": caps_from_tags["vision"] or caps_from_id["vision"],
            "tools": caps_from_tags["tools"] or caps_from_id["tools"],
            "thinking": caps_from_tags["thinking"] or caps_from_id["thinking"],
        }
        repo_name = _repo_name(m.id)
        items.append({
            "id": m.id,
            "repo_name": repo_name,
            "size_display": _parse_size_from_repo(repo_name),
            "author": getattr(m, "author", "") or (m.id.split("/")[0] if "/" in m.id else ""),
            "downloads": getattr(m, "downloads", None) or 0,
            "likes": getattr(m, "likes", None) or 0,
            "vision": caps["vision"],
            "tools": caps["tools"],
            "thinking": caps["thinking"],
        })
    return items


def list_gguf_files(repo_id: str, use_cache: bool = True) -> list[str]:
    """List .gguf filenames in the repo. Cached for CACHE_TTL_SEC."""
    import time
    now = time.time()
    if use_cache and repo_id in _repo_files_cache:
        ts, files = _repo_files_cache[repo_id]
        if now - ts < _CACHE_TTL_SEC:
            return files
    api = _get_api()
    all_files = api.list_repo_files(repo_id)
    gguf_files = [f for f in all_files if f.endswith(".gguf")]
    _repo_files_cache[repo_id] = (now, gguf_files)
    return gguf_files


def get_model_card_content(repo_id: str) -> str:
    """Fetch model card markdown content. Returns empty string on failure."""
    try:
        card = ModelCard.load(repo_id)
        return card.content or ""
    except Exception:
        return ""


def get_model_capabilities(repo_id: str) -> dict[str, bool]:
    """Infer vision, tools, thinking from model card content and repo id."""
    content = get_model_card_content(repo_id)
    return _infer_capabilities_from_text(content, repo_id)


def download_model(
    repo_id: str,
    filename: str,
    models_dir: Path | str,
    local_dir_override: Path | str | None = None,
) -> Path:
    """
    Download a GGUF file into the models directory.
    If local_dir_override is set, download there; else use HF_HOME so file lands under models_dir/hub/.
    Returns path to the downloaded file.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    if local_dir_override is not None:
        local_dir = Path(local_dir_override)
        local_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        return Path(path)
    # Use HF_HOME so cache goes under models_dir
    env = os.environ.copy()
    env["HF_HOME"] = str(models_dir)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=None,
        token=None,
    )
    # hf_hub_download with HF_HOME uses cache under HF_HOME/hub/; path returned is the resolved path
    return Path(path)
