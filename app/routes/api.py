"""REST API: search, model detail, download, download status."""

import logging
import os
import re
import time
from pathlib import Path

import markdown
from fastapi import APIRouter, HTTPException, BackgroundTasks

logger = logging.getLogger(__name__)

from app.main import get_config
from app.config import get_models_ini_path
from app.services import hf_service, ini_manager, params_parser

router = APIRouter()

# In-memory download status: job_id -> {status, path?, error?}
_download_jobs: dict[str, dict] = {}

# Maximum file size: 100GB (reasonable for large models)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024 * 1024


def _sanitize_section_name(repo_id: str, filename: str) -> str:
    """Derive a valid [section] name from repo and filename."""
    base = filename.removesuffix(".gguf").strip()
    if not base:
        base = repo_id.replace("/", "-")
    return (repo_id.replace("/", "-") + "-" + base).replace(" ", "_")[:80]


def _validate_repo_id(repo_id: str) -> None:
    """Validate repo_id format to prevent path traversal attacks."""
    if not repo_id or len(repo_id) > 200:
        raise HTTPException(status_code=400, detail="Invalid repo_id")
    # HuggingFace repo IDs must match pattern: namespace/model-name
    # Allow alphanumeric, hyphen, underscore, and forward slash
    pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?/[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$"
    if not re.match(pattern, repo_id):
        raise HTTPException(status_code=400, detail="Invalid repo_id format")


def _validate_filename(filename: str) -> None:
    """Validate filename to prevent path traversal attacks."""
    if not filename or len(filename) > 255:
        raise HTTPException(status_code=400, detail="Invalid filename")
    # Only allow safe filenames - no path separators, no parent directory references
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename: path traversal not allowed")
    # Must be a GGUF file
    if not filename.lower().endswith('.gguf'):
        raise HTTPException(status_code=400, detail="Only .gguf files are allowed")


@router.get("/search")
async def api_search(
    q: str | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "downloads",
    vision: bool | None = None,
    tools: bool | None = None,
    thinking: bool | None = None,
    tag: str | None = None,
):
    """
    Search GGUF models on Hugging Face.
    - q: text search query; use [tag_name] syntax to filter by tag
    - sort: downloads (default), likes, name, author, size
    - vision/tools/thinking: filter by capability (true/false)
    - tag: filter by exact tag name
    """
    logger.debug(
        "GET /api/search q=%r limit=%d offset=%d sort=%s vision=%s tools=%s thinking=%s tag=%s",
        q, limit, offset, sort, vision, tools, thinking, tag,
    )
    # Parse [tag_name] syntax from query
    tag_filter: str | None = tag
    text_query: str | None = q
    if q and q.strip().startswith("[") and q.strip().endswith("]"):
        potential_tag = q.strip()[1:-1]
        if potential_tag:
            tag_filter = potential_tag
            text_query = None

    # capabilities sort is client-side; use downloads for HF API
    api_sort = sort if sort not in ("capabilities",) else "downloads"

    items = hf_service.search_models(
        query=text_query,
        limit=limit + offset,  # fetch enough to slice
        offset=0,
        sort=api_sort,
        tag_filter=tag_filter,
    )

    # Filter by capabilities if requested
    if vision is not None:
        items = [m for m in items if bool(m.get("vision")) == vision]
    if tools is not None:
        items = [m for m in items if bool(m.get("tools")) == tools]
    if thinking is not None:
        items = [m for m in items if bool(m.get("thinking")) == thinking]

    # Handle offset/limit manually since we might filter results
    total = len(items)  # approximate total from this fetch
    items = items[offset:]
    items = items[:limit]
    logger.debug("GET /api/search returned %d models", len(items))
    return {"models": items}


@router.get("/model/{repo_id:path}")
async def api_model_detail(repo_id: str):
    """Get model card, quantizations, capabilities, title, param_count, tags, license, author."""
    logger.debug("GET /api/model/%s", repo_id)
    gguf_files = hf_service.list_gguf_files(repo_id)
    file_sizes = hf_service.get_repo_file_sizes(repo_id)
    quantizations = hf_service.group_gguf_by_quantization(gguf_files, file_sizes=file_sizes)
    card_info = hf_service.get_model_card_info(repo_id)
    model_card = card_info["content"]
    model_card_html = ""
    if model_card:
        model_card_html = markdown.markdown(model_card, extensions=["extra", "nl2br"])
    logger.debug(
        "GET /api/model/%s: %d gguf files, %d quant groups, card_len=%d, title=%r",
        repo_id, len(gguf_files), len(quantizations), len(model_card), card_info.get("title"),
    )
    return {
        "repo_id": repo_id,
        "model_title": card_info.get("title", ""),
        "param_count": card_info.get("param_count", ""),
        "license": card_info.get("license", ""),
        "author": card_info.get("author", ""),
        "short_desc": card_info.get("short_desc", ""),
        "tags_mandatory": card_info.get("tags_mandatory", []),
        "tags_core": card_info.get("tags_core", []),
        "tags_optional": card_info.get("tags_optional", []),
        "vision": card_info.get("vision", False),
        "tools": card_info.get("tools", False),
        "thinking": card_info.get("thinking", False),
        "gguf_files": gguf_files,
        "quantizations": quantizations,
        "model_card": model_card,
        "model_card_html": model_card_html,
        # Keep capabilities dict for backward compat
        "capabilities": {
            "vision": card_info.get("vision", False),
            "tools": card_info.get("tools", False),
            "thinking": card_info.get("thinking", False),
        },
    }


@router.post("/download")
async def api_download(
    repo_id: str,
    filename: str | None = None,
    filenames: str | None = None,
    section_name: str | None = None,
    background_tasks: BackgroundTasks = None,  # FastAPI injects BackgroundTasks automatically
):
    """
    Start download of one or more GGUF files (multifile model). Pass filename= or filenames= comma-separated.
    Returns job_id. Poll GET /api/download/{job_id} for status.
    On success, adds/updates models.ini with path to the first file.
    """
    if filenames:
        to_download = [f.strip() for f in filenames.split(",") if f.strip()]
    elif filename:
        to_download = [filename.strip()]
    else:
        raise HTTPException(status_code=400, detail="Provide filename= or filenames=")
    
    # Validate inputs for security
    _validate_repo_id(repo_id)
    for f in to_download:
        _validate_filename(f)
        
    config = get_config()
    models_dir = Path(config["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    # Use a unique job_id per download attempt to avoid collision on re-download
    job_id = f"{repo_id}:{to_download[0]}:{int(time.time() * 1000)}"
    _download_jobs[job_id] = {"status": "running", "path": None, "error": None}
    logger.info(
        "Download started: repo=%s files=%s job_id=%s models_dir=%s",
        repo_id, to_download, job_id, models_dir,
    )

    def run_download():
        try:
            # Note: HF_HOME handling removed as it is not thread safe.
            # hf_service.download_model now takes models_dir as local_dir explicit argument.
            first_path = None
            for fn in to_download:
                logger.info("Downloading %s / %s …", repo_id, fn)
                # Pass models_dir as the explicit cache_dir to properly organize the HF hub files
                path = hf_service.download_model(
                    repo_id=repo_id, 
                    filename=fn, 
                    models_dir=models_dir,
                    job_id=job_id,
                    download_jobs_dict=_download_jobs
                )
                logger.info("Downloaded %s / %s → %s", repo_id, fn, path)
                if first_path is None:
                    first_path = path
            
            _download_jobs[job_id]["path"] = str(first_path)
            _download_jobs[job_id]["status"] = "completed"
            model_card = hf_service.get_model_card_content(repo_id)
            recommended = params_parser.recommended_params_with_defaults(model_card)
            section = section_name or _sanitize_section_name(repo_id, to_download[0])
            ini_path = get_models_ini_path(models_dir)
            recommended["LLAMA_ARG_MODEL"] = str(first_path)
            ini_manager.add_or_update_section(ini_path, section, recommended, merge=True)
            logger.info(
                "Download completed: job_id=%s section='%s' path=%s",
                job_id, section, first_path,
            )
        except Exception as e:
            _download_jobs[job_id]["status"] = "failed"
            _download_jobs[job_id]["error"] = str(e)
            logger.error("Download failed: job_id=%s error=%s", job_id, e, exc_info=True)

    if background_tasks is not None:
        background_tasks.add_task(run_download)
    else:
        run_download()
    return {"job_id": job_id, "status": "started"}


@router.get("/download/{job_id:path}")
async def api_download_status(job_id: str):
    """Get download job status. job_id may contain / and : (repo_id:filename)."""
    if job_id not in _download_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _download_jobs[job_id]


@router.get("/models")
async def api_list_models():
    """List models from models.ini."""
    config = get_config()
    path = get_models_ini_path(config["models_dir"])
    sections = ini_manager.list_sections(path)
    return {"models": sections}


@router.get("/models/{section_name}")
async def api_get_model(section_name: str):
    """Get one model section."""
    config = get_config()
    path = get_models_ini_path(config["models_dir"])
    params = ini_manager.get_section(path, section_name)
    if params is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"name": section_name, "params": params}

@router.post("/download/cancel/{job_id:path}")
async def api_download_cancel(job_id: str):
    """Cancel a running download job."""
    if job_id not in _download_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if _download_jobs[job_id]["status"] == "running":
        _download_jobs[job_id]["status"] = "cancelled"
    return {"status": "cancelling"}

@router.get("/models/check")
async def api_models_check(repo_id: str, filename: str):
    """Check if a specific model quantization is already downloaded."""
    config = get_config()
    models_dir = Path(config["models_dir"])
    
    parts = repo_id.split('/')
    if len(parts) == 2:
        author, model_name = parts[0], parts[1]
    else:
        author, model_name = "unknown", repo_id
        
    target_path = models_dir / author / model_name / filename
    
    exists = target_path.exists() and target_path.stat().st_size > 0
    return {"repo_id": repo_id, "filename": filename, "downloaded": exists}
