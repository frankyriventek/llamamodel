"""Hugging Face Hub integration: search GGUF models, list files, model card, download."""

import time as _time
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, ModelCard

logger = logging.getLogger(__name__)

# Cache for list_repo_files (quantizations) to avoid repeated API calls
_repo_files_cache: dict[str, tuple[float, list[str]]] = {}
# Cache for repo file info (with sizes): repo_id -> (timestamp, {filename: size_bytes})
_repo_file_info_cache: dict[str, tuple[float, dict[str, int]]] = {}
_CACHE_TTL_SEC = 300  # 5 minutes

# Retry / backoff settings for HF API calls
_MAX_RETRIES = 4
_BACKOFF_BASE_SEC = 2.0  # seconds; delay = base * 2^attempt


def _get_api() -> HfApi:
    return HfApi()


def _with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on rate-limit / transient errors.
    Retries up to _MAX_RETRIES times on HTTP 429 or 5xx-style exceptions.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            # Retry on rate-limit (429) or server errors (5xx)
            is_retryable = (
                "429" in msg
                or "rate limit" in msg
                or "too many requests" in msg
                or "503" in msg
                or "502" in msg
                or "500" in msg
                or "connection" in msg
            )
            if not is_retryable or attempt == _MAX_RETRIES:
                raise
            delay = _BACKOFF_BASE_SEC * (2 ** attempt)
            logger.warning(
                "HF API call failed (attempt %d/%d): %s. Retrying in %.1fs…",
                attempt + 1, _MAX_RETRIES, exc, delay,
            )
            last_exc = exc
            _time.sleep(delay)
    raise last_exc  # unreachable but satisfies type checkers


# Tags that indicate a real text-generation / LLM model (not a dataset, embedding, etc.)
_LLM_PIPELINE_TAGS = {
    "text-generation",
    "text2text-generation",
    "conversational",
    "question-answering",
    "summarization",
    "translation",
    "fill-mask",
}

# Tags that indicate the repo is NOT a standalone LLM (skip these)
_SKIP_TAGS = {
    "dataset",
    "datasets",
    "embedding",
    "sentence-similarity",
    "feature-extraction",
    "image-classification",
    "object-detection",
    "audio",
    "speech",
    "automatic-speech-recognition",
    "text-to-speech",
    "image-to-text",
    "text-to-image",
    "image-to-image",
    "video",
    "reinforcement-learning",
    "robotics",
}


def _is_real_llm_model(m: Any) -> bool:
    """
    Return True if the model object looks like a real LLM / chat model.
    Filters out datasets, embedding models, ASR, image models, etc.
    """
    tags_raw = getattr(m, "tags", None) or []
    tags = {t.lower() for t in tags_raw} if isinstance(tags_raw, (list, tuple)) else set()

    # If any skip tag is present, exclude
    if tags & _SKIP_TAGS:
        return False

    # pipeline_tag is the most reliable signal
    pipeline_tag = (getattr(m, "pipeline_tag", None) or "").lower()
    if pipeline_tag and pipeline_tag not in _LLM_PIPELINE_TAGS:
        # Allow empty pipeline_tag (many GGUF repos don't set it)
        # but reject known non-LLM pipelines
        return False

    # Must have at least one GGUF file indicator: either the "gguf" tag or the filter already
    # ensures it, but double-check the repo isn't a pure dataset repo
    model_id_lower = m.id.lower()
    if model_id_lower.startswith("datasets/"):
        return False

    return True


def _infer_capabilities_from_text(text: str, model_id: str) -> dict[str, bool]:
    """Infer vision, tools, thinking from model id and/or card text."""
    s = (text or "").lower() + " " + (model_id or "").lower()
    return {
        "vision": (
            "vision" in s or "visual" in s or "vlm" in s or "multimodal" in s
            or "image-text" in s or "image understanding" in s or "image input" in s
            or "-vl" in s or "_vl" in s or " vl " in s or s.endswith(" vl")
            or "vl-" in s or "vl_" in s
        ),
        "tools": (
            "tool call" in s or "tool use" in s or "tool_use" in s
            or "function call" in s or "function_call" in s or "function-call" in s
            or "tool-calling" in s or "tool_calling" in s or "tool-use" in s
            or "function calling" in s or "function-calling" in s
        ),
        "thinking": (
            "thinking" in s or "<think>" in s or "chain-of-thought" in s
            or "chain of thought" in s or "step-by-step reasoning" in s
        ),
    }


def _infer_capabilities_from_tags(tags: list[str] | None) -> dict[str, bool]:
    """
    Infer vision, tools, thinking from Hugging Face model tags.
    Uses exact tag matching for high-confidence signals.
    """
    if not tags:
        return {"vision": False, "tools": False, "thinking": False}
    tag_set = {t.lower() for t in tags}
    t = " ".join(tag_set)

    vision = (
        "vision" in tag_set
        or "multimodal" in tag_set
        or "image-text-to-text" in tag_set
        or "visual-question-answering" in tag_set
        or "vlm" in tag_set
        or "image-text" in t
    )
    tools = (
        "function-calling" in tag_set
        or "tool-calling" in tag_set
        or "tool-use" in tag_set
        or "function_calling" in tag_set
        or "tool_calling" in tag_set
        or "tool_use" in tag_set
    )
    thinking = (
        "thinking" in tag_set
        or "reasoning" in tag_set
        or "chain-of-thought" in tag_set
        or "cot" in tag_set
    )
    return {"vision": vision, "tools": tools, "thinking": thinking}


def _repo_name(model_id: str) -> str:
    """Return only the repository name (last segment of model_id)."""
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _parse_size_from_text(text: str) -> str:
    """
    Parse parameter count / model size from a string (repo name, tag, or description).
    Recognises patterns like: 7B, 0.5B, 70B, 1.5B, 7b, 7-B, 7_B, 7.0B, 7.0-B
    Also handles 'x' multiplier patterns like 8x7B (MoE).
    Returns a normalised string like '7B', '0.5B', '8x7B', or '' if not found.
    """
    import re
    # MoE pattern: 8x7B, 4x22B etc.
    m = re.search(r"(\d+)[xX](\d+\.?\d*)\s*[Bb]", text)
    if m:
        return f"{m.group(1)}x{m.group(2)}B"
    # Standard: 7B, 0.5B, 70B, 1.5B – must be followed by word boundary / separator
    m = re.search(r"(?<![.\d])(\d+\.?\d*)\s*[Bb](?:[_\-\s]|$)", text)
    if m:
        val = m.group(1)
        # Normalise: drop trailing .0
        if val.endswith(".0"):
            val = val[:-2]
        return val + "B"
    return ""


def _parse_size_from_tags(tags: list[str] | None) -> str:
    """
    Extract parameter count from Hugging Face model tags.
    Tags like '7b', '13b', '70b', '0.5b', '8x7b' are common.
    Returns normalised string like '7B' or '' if not found.
    """
    if not tags:
        return ""
    for tag in tags:
        # Tags are often just the size, e.g. "7b", "13b", "0.5b"
        result = _parse_size_from_text(tag)
        if result:
            return result
    return ""


def _parse_size_from_repo(repo_name: str) -> str:
    """Parse model size from repo name (e.g. 7B, 0.5B, 70B, 1.5B). Returns empty string if not found."""
    return _parse_size_from_text(repo_name)


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


def _format_param_count(n: int) -> str:
    """Format a raw parameter count integer into a human-readable string like '7B', '405M'."""
    if n >= 1_000_000_000:
        val = n / 1_000_000_000
        s = f"{val:.1f}".rstrip("0").rstrip(".")
        return s + "B"
    if n >= 1_000_000:
        val = n / 1_000_000
        s = f"{val:.0f}"
        return s + "M"
    return str(n)


def _format_file_size(n: int) -> str:
    """Format bytes into human-readable string: GB, MB, KB."""
    if n >= 1_073_741_824:  # 1 GiB
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:  # 1 MiB
        return f"{n / 1_048_576:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def get_repo_file_sizes(repo_id: str, use_cache: bool = True) -> dict[str, int]:
    """
    Return a dict of {filename: size_bytes} for all files in the repo.
    Cached for CACHE_TTL_SEC. Returns empty dict on failure.
    """
    now = _time.time()
    if use_cache and repo_id in _repo_file_info_cache:
        ts, info = _repo_file_info_cache[repo_id]
        if now - ts < _CACHE_TTL_SEC:
            return info
    api = _get_api()
    try:
        file_infos = list(_with_retry(api.list_repo_tree, repo_id, recursive=True))
        sizes: dict[str, int] = {}
        for fi in file_infos:
            # RepoFile objects have .path and .size; RepoFolder objects have no .size
            path = getattr(fi, "path", None)
            size = getattr(fi, "size", None)
            if path and size is not None:
                sizes[path] = int(size)
        _repo_file_info_cache[repo_id] = (now, sizes)
        return sizes
    except Exception as exc:
        logger.debug("get_repo_file_sizes(%s) failed: %s", repo_id, exc)
        return {}


def group_gguf_by_quantization(
    gguf_files: list[str],
    file_sizes: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Group GGUF files by quantization. Multifile models have -00001-of-00003 etc.
    Returns list of {quant, primary_file, all_files, total_size_bytes, total_size_display}.
    Primary is the single file or the first shard.
    file_sizes: optional dict of {filename: bytes} from get_repo_file_sizes().
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
        total_bytes: int | None = None
        if file_sizes:
            total = sum(file_sizes.get(f, 0) for f in files)
            if total > 0:
                total_bytes = total
        result.append({
            "quant": quant,
            "primary_file": primary,
            "all_files": files,
            "total_size_bytes": total_bytes,
            "total_size_display": _format_file_size(total_bytes) if total_bytes else "",
        })
    return result


import re as _re

# Tags that are always hidden (internal HF infrastructure tags)
_ALWAYS_HIDDEN_TAGS = {
    "gguf", "transformers", "pytorch", "safetensors",
    "autotrain_compatible", "endpoints_compatible", "has_space",
    "text-generation-inference",
    "region:us", "region:eu", "region:ap",
}

# Mandatory tags: define the primary task/capability of the model
_MANDATORY_TAG_PATTERNS = {
    "text-generation", "text2text-generation", "conversational",
    "question-answering", "summarization", "translation", "fill-mask",
    "image-text-to-text", "visual-question-answering",
    "function-calling", "tool-calling", "tool-use",
    "function_calling", "tool_calling", "tool_use",
    "thinking", "reasoning", "chain-of-thought",
    "vision", "multimodal", "vlm",
}

# Core tags: important model characteristics
_CORE_TAG_PATTERNS = {
    "chat", "instruct", "instruction-tuned", "instruction-following",
    "code", "coding", "math", "science", "medical", "legal",
    "multilingual", "english", "chinese", "japanese", "german", "french",
    "spanish", "arabic", "korean", "russian",
    "base", "pretrained", "finetuned", "fine-tuned",
    "rlhf", "dpo", "sft", "ppo",
    "long-context", "long context", "extended-context",
    "quantized", "4bit", "8bit", "awq", "gptq",
}


def _classify_tags(tags: list[str]) -> dict[str, list[str]]:
    """
    Classify tags into mandatory, core, and optional categories.
    Returns dict with keys: mandatory, core, optional.
    Filters out internal/noisy HF tags.
    """
    mandatory, core, optional = [], [], []
    for t in tags:
        tl = t.lower()
        # Always skip internal tags
        if tl in _ALWAYS_HIDDEN_TAGS:
            continue
        if tl.startswith("arxiv:"):
            continue
        if tl.startswith("base_model:"):
            continue
        if tl.startswith("model-index"):
            continue
        # Skip pure size tags (shown as param count badge)
        if _re.match(r"^\d+\.?\d*[bBmMkK]$", tl):
            continue
        # Classify
        if tl in _MANDATORY_TAG_PATTERNS:
            mandatory.append(t)
        elif tl in _CORE_TAG_PATTERNS or any(c in tl for c in _CORE_TAG_PATTERNS):
            core.append(t)
        else:
            optional.append(t)
    return {"mandatory": mandatory[:6], "core": core[:6], "optional": optional[:6]}


def _extract_license(tags: list[str]) -> str:
    """Extract license string from tags (e.g. 'license:apache-2.0' -> 'apache-2.0')."""
    for t in tags:
        if t.lower().startswith("license:"):
            return t[8:]
    return ""


def search_models(
    query: str | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "downloads",
    tag_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search for GGUF models on Hugging Face.
    Returns list of {id, repo_name, author, downloads, likes, size_display, vision, tools, thinking, ...}.

    Filtering: only real LLM/chat model repos are returned (datasets, embedding models,
    ASR, image models etc. are excluded via _is_real_llm_model()).
    tag_filter: if set, only return models that have this tag.
    Sort: always fetches by downloads from HF API; client-side sorts applied in api.py.
    """
    api = _get_api()
    # Fetch more than needed to compensate for filtered-out non-LLM repos
    fetch_limit = min((offset + limit) * 3, 200)

    # Build HF API filter: combine "gguf" with tag filter if provided
    hf_filter = "gguf"
    if tag_filter:
        hf_filter = ["gguf", tag_filter]

    def _do_list_models(full: bool):
        return list(api.list_models(
            filter=hf_filter,
            search=query or "",
            sort="downloads",          # always fetch by downloads descending from HF
            direction=-1,
            limit=fetch_limit,
            **({"full": True} if full else {}),
        ))

    try:
        raw = _with_retry(_do_list_models, True)
    except TypeError:
        raw = _with_retry(_do_list_models, False)

    # Filter to real LLM repos and apply offset/limit
    items = []
    skipped = 0
    for m in raw:
        if not _is_real_llm_model(m):
            logger.debug("Skipping non-LLM repo: %s (pipeline_tag=%s)", m.id, getattr(m, "pipeline_tag", None))
            continue
        if skipped < offset:
            skipped += 1
            continue
        if len(items) >= limit:
            break

        tags_raw = getattr(m, "tags", None) or []
        tags_list = list(tags_raw) if isinstance(tags_raw, (list, tuple)) else []
        pipeline_tag = (getattr(m, "pipeline_tag", None) or "").lower()

        # Capabilities: use tags (exact matching) + pipeline_tag
        caps_from_tags = _infer_capabilities_from_tags(tags_list)
        vision_from_pipeline = pipeline_tag in {
            "image-text-to-text", "visual-question-answering", "image-to-text",
        }
        # Also check model id for well-known vision model name patterns
        id_lower = m.id.lower()
        vision_from_id = (
            "vision" in id_lower or "vlm" in id_lower or "visual" in id_lower
            or "llava" in id_lower or "qwen-vl" in id_lower or "internvl" in id_lower
            or "minicpm-v" in id_lower or "phi-3-vision" in id_lower
            or "cogvlm" in id_lower or "moondream" in id_lower
            or "-vl" in id_lower or "_vl" in id_lower or "/vl-" in id_lower
            or id_lower.endswith("-vl") or id_lower.endswith("_vl")
        )
        tools_from_id = (
            "tool" in id_lower or "function" in id_lower
            or "hermes" in id_lower  # Hermes models are known for tool use
            or "functionary" in id_lower  # Functionary models
            or "gorilla" in id_lower  # Gorilla models for function calling
        )
        thinking_from_id = (
            "thinking" in id_lower or "qwq" in id_lower or "deepseek-r1" in id_lower
            or "o1" in id_lower or "skywork-o1" in id_lower
        )

        # Need `content` (model card) to reliably infer text-based capabilities inside search pane if basic checks skip it!
        # Search API provides no `content`. So for missing capabilities in search listing, we fetch basic ModelCard parsing minimally via a shared path.
        # But wait - we do NOT want to download the whole raw content loop here (too slow for lists).
        # We assume tag-based + id-based (above) is comprehensive for search mode, while card-info expands it in detail mode. 
        # BUT: It is critical we align how we set tools_from_id, vision_from_id, thinking_from_id here exactly with `get_model_card_info()`.
        
        repo_name = _repo_name(m.id)

        # Parameter count priority:
        # 1. safetensors.total (most accurate, integer param count)
        # 2. HF tags (e.g. "7b", "13b")
        # 3. repo name text
        # 4. full model id text
        size_display = ""
        try:
            st = getattr(m, "safetensors", None)
            if st:
                total = getattr(st, "total", None)
                if total and int(total) > 0:
                    size_display = _format_param_count(int(total))
        except Exception:
            pass
        if not size_display:
            size_display = _parse_size_from_tags(tags_list)
        if not size_display:
            size_display = _parse_size_from_repo(repo_name)
        if not size_display:
            size_display = _parse_size_from_text(m.id)

        # Short description from card metadata (available without extra API call)
        short_desc = ""
        try:
            card_data = getattr(m, "cardData", None)
            if card_data:
                short_desc = (
                    getattr(card_data, "description", None)
                    or getattr(card_data, "summary", None)
                    or ""
                )
                if short_desc:
                    short_desc = str(short_desc).strip()
                    if len(short_desc) > 200:
                        short_desc = short_desc[:197] + "…"
        except Exception:
            pass

        # Update text capabilities based on available description text
        caps_from_text = _infer_capabilities_from_text(short_desc, m.id)
        caps = {
            "vision": caps_from_tags["vision"] or caps_from_text["vision"] or vision_from_pipeline or vision_from_id,
            "tools": caps_from_tags["tools"] or caps_from_text["tools"] or tools_from_id,
            "thinking": caps_from_tags["thinking"] or caps_from_text["thinking"] or thinking_from_id,
        }

        # Classify tags into mandatory/core/optional
        classified_tags = _classify_tags(tags_list)
        # License from tags
        license_str = _extract_license(tags_list)
        author = getattr(m, "author", "") or (m.id.split("/")[0] if "/" in m.id else "")

        items.append({
            "id": m.id,
            "repo_name": repo_name,
            "size_display": size_display,
            "author": author,
            "downloads": getattr(m, "downloads", None) or 0,
            "likes": getattr(m, "likes", None) or 0,
            "vision": caps["vision"],
            "tools": caps["tools"],
            "thinking": caps["thinking"],
            "short_desc": short_desc,
            # Classified tags for display
            "tags_mandatory": classified_tags["mandatory"],
            "tags_core": classified_tags["core"],
            "tags_optional": classified_tags["optional"],
            # Convenience flat list for search pane (mandatory + core only)
            "tags": classified_tags["mandatory"] + classified_tags["core"],
            "license": license_str,
        })

    logger.debug(
        "search_models(q=%r, limit=%d, offset=%d): returned %d items from %d raw",
        query, limit, offset, len(items), len(raw),
    )
    return items


def list_gguf_files(repo_id: str, use_cache: bool = True) -> list[str]:
    """List .gguf filenames in the repo. Cached for CACHE_TTL_SEC."""
    now = _time.time()
    if use_cache and repo_id in _repo_files_cache:
        ts, files = _repo_files_cache[repo_id]
        if now - ts < _CACHE_TTL_SEC:
            return files
    api = _get_api()
    all_files = list(_with_retry(api.list_repo_files, repo_id))
    gguf_files = [f for f in all_files if f.endswith(".gguf")]
    _repo_files_cache[repo_id] = (now, gguf_files)
    return gguf_files


def get_model_card_content(repo_id: str) -> str:
    """Fetch model card markdown content. Returns empty string on failure."""
    try:
        card = _with_retry(ModelCard.load, repo_id)
        return card.content or ""
    except Exception:
        return ""


def get_model_card_info(repo_id: str) -> dict[str, Any]:
    """
    Fetch model card and extract structured info.
    Returns dict with keys:
      content, title, param_count, license, author,
      tags_mandatory, tags_core, tags_optional, short_desc,
      vision, tools, thinking.
    """
    import re
    content = ""
    title = ""
    param_count = ""
    license_str = ""
    author = repo_id.split("/")[0] if "/" in repo_id else ""
    card_tags: list[str] = []
    short_desc = ""
    try:
        card = _with_retry(ModelCard.load, repo_id)
        content = card.content or ""
        try:
            data = card.data
            if data:
                # Title
                title = (
                    getattr(data, "model_name", None)
                    or getattr(data, "model-name", None)
                    or getattr(data, "title", None)
                    or ""
                )
                if title:
                    title = str(title).strip()
                # License
                lic = getattr(data, "license", None)
                if lic:
                    license_str = str(lic).strip()
                # Tags from card metadata
                raw_tags = getattr(data, "tags", None) or []
                if isinstance(raw_tags, (list, tuple)):
                    card_tags = [str(t) for t in raw_tags]
                # Short description from card metadata
                desc = getattr(data, "description", None) or getattr(data, "summary", None) or ""
                if desc:
                    short_desc = str(desc).strip()[:200]
        except Exception:
            pass
        # Fall back to first H1 heading
        if not title and content:
            m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if m:
                title = m.group(1).strip()
                title = re.sub(r"\s*[-–]?\s*GGUF\b", "", title, flags=re.IGNORECASE).strip()
        # Extract parameter count from card text
        m2 = re.search(
            r"(?:parameters?|params?|#\s*params?)\s*[:\-]?\s*(\d+\.?\d*)\s*([BbMmKk])",
            content,
        )
        if m2:
            val = m2.group(1)
            unit = m2.group(2).upper()
            param_count = val + unit
        if not param_count:
            m3 = re.search(r"(\d+\.?\d*)\s*[Bb]\s+(?:parameter|param)", content)
            if m3:
                param_count = m3.group(1) + "B"
    except Exception:
        pass

    # Attempt to use HfApi to fetch complete tags
    try:
        api = _get_api()
        info = _with_retry(api.model_info, repo_id)
        if hasattr(info, "tags") and info.tags:
            # Add missing tags not gathered in ModelCard
            for t in info.tags:
                if t not in card_tags:
                    card_tags.append(str(t))
    except Exception as e:
        logger.warning("Could not fetch full tags via model_info for %s: %s", repo_id, e)

    # Classify tags
    classified = _classify_tags(card_tags)
    # If no license from card metadata, try tags
    if not license_str:
        license_str = _extract_license(card_tags)

    # Capabilities from card content + repo id
    caps_text = _infer_capabilities_from_text(content, repo_id)
    caps_tags = _infer_capabilities_from_tags(card_tags)
    id_lower = repo_id.lower()
    
    # Must explicitly match the search heuristic exactly to avoid desync
    vision_from_id = (
        "vision" in id_lower or "vlm" in id_lower or "visual" in id_lower
        or "llava" in id_lower or "qwen-vl" in id_lower or "internvl" in id_lower
        or "minicpm-v" in id_lower or "phi-3-vision" in id_lower
        or "cogvlm" in id_lower or "moondream" in id_lower
        or "-vl" in id_lower or "_vl" in id_lower or "/vl-" in id_lower
        or id_lower.endswith("-vl") or id_lower.endswith("_vl")
    )
    tools_from_id = (
        "tool" in id_lower or "function" in id_lower
        or "hermes" in id_lower or "functionary" in id_lower or "gorilla" in id_lower
    )
    thinking_from_id = (
        "thinking" in id_lower or "qwq" in id_lower or "deepseek-r1" in id_lower
        or "o1" in id_lower or "skywork-o1" in id_lower
    )

    # Some models have `pipeline_tag` but are fetched here without it being immediately available in ModelCard explicitly
    # We must lean back on the `caps_text` heavily and card tags for the detailed mapping.

    return {
        "content": content,
        "title": title,
        "param_count": param_count,
        "license": license_str,
        "author": author,
        "short_desc": short_desc,
        "tags_mandatory": classified["mandatory"],
        "tags_core": classified["core"],
        "tags_optional": classified["optional"],
        "vision": caps_tags["vision"] or caps_text["vision"] or vision_from_id,
        "tools": caps_tags["tools"] or caps_text["tools"] or tools_from_id,
        "thinking": caps_tags["thinking"] or caps_text["thinking"] or thinking_from_id,
    }


def get_model_capabilities(repo_id: str) -> dict[str, bool]:
    """Infer vision, tools, thinking from model card content and repo id."""
    info = get_model_card_info(repo_id)
    return {"vision": info["vision"], "tools": info["tools"], "thinking": info["thinking"]}


def download_model(
    repo_id: str,
    filename: str,
    models_dir: Path | str,
    local_dir_override: Path | str | None = None,
    job_id: str | None = None,
    download_jobs_dict: dict | None = None
) -> Path:
    """
    Download a GGUF file chunk-by-chunk using requests into a structured layout:
    models_dir / author_name / model_name / filename
    """
    import os
    import time
    import requests
    from huggingface_hub import hf_hub_url
    
    models_dir = Path(models_dir)
    # Parse author and model from repo_id
    parts = repo_id.split('/')
    if len(parts) == 2:
        author, model_name = parts[0], parts[1]
    else:
        author, model_name = "unknown", repo_id
        
    target_dir = models_dir / author / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    
    # Simple check if fully downloaded (could compare sizes if we had total_size apriori)
    if target_path.exists() and target_path.stat().st_size > 0:
        pass # we will overwrite if requested, or the caller skips it.

    tmp_path = target_path.with_suffix(target_path.suffix + ".download")
    
    url = hf_hub_url(repo_id=repo_id, filename=filename)
    
    # Download with progress
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        if download_jobs_dict is not None and job_id is not None:
            if job_id in download_jobs_dict:
                download_jobs_dict[job_id]["total_bytes"] = total_size
                download_jobs_dict[job_id]["downloaded_bytes"] = 0
                download_jobs_dict[job_id]["progress"] = 0
                download_jobs_dict[job_id]["start_time"] = time.time()
                download_jobs_dict[job_id]["speed"] = 0
                download_jobs_dict[job_id]["eta"] = 0
                
        downloaded = 0
        start_time = time.time()
        
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192*4):
                if download_jobs_dict is not None and job_id is not None:
                    # Check for cancellation
                    if download_jobs_dict.get(job_id, {}).get("status") == "cancelled":
                        tmp_path.unlink(missing_ok=True)
                        raise Exception("Download cancelled by user")
                        
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if download_jobs_dict is not None and job_id is not None and job_id in download_jobs_dict:
                        now = time.time()
                        elapsed = now - start_time
                        # throttle updates to avoid locking dict
                        if elapsed > 0:
                            speed = downloaded / elapsed
                            eta = (total_size - downloaded) / speed if speed > 0 else 0
                            
                            job = download_jobs_dict[job_id]
                            job["downloaded_bytes"] = downloaded
                            job["speed"] = speed
                            job["eta"] = eta
                            job["progress"] = (downloaded / total_size * 100) if total_size > 0 else 0

    tmp_path.rename(target_path)
    return target_path
