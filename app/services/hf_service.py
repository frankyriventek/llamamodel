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


def search_models(
    query: str | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "downloads",
) -> list[dict[str, Any]]:
    """Search for GGUF models. Returns list of {id, author, downloads, ...}."""
    api = _get_api()
    full_limit = offset + limit
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
        items.append({
            "id": m.id,
            "author": getattr(m, "author", "") or (m.id.split("/")[0] if "/" in m.id else ""),
            "downloads": getattr(m, "downloads", None) or 0,
            "likes": getattr(m, "likes", None) or 0,
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
