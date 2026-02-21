"""REST API: search, model detail, download, download status."""

import os
from pathlib import Path

import markdown
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.main import get_config
from app.config import get_models_ini_path
from app.services import hf_service, ini_manager, params_parser

router = APIRouter()

# In-memory download status: job_id -> {status, path?, error?}
_download_jobs: dict[str, dict] = {}


def _sanitize_section_name(repo_id: str, filename: str) -> str:
    """Derive a valid [section] name from repo and filename."""
    base = filename.removesuffix(".gguf").strip()
    if not base:
        base = repo_id.replace("/", "-")
    return (repo_id.replace("/", "-") + "-" + base).replace(" ", "_")[:80]


@router.get("/search")
async def api_search(
    q: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    """Search GGUF models on Hugging Face."""
    items = hf_service.search_models(query=q, limit=limit, offset=offset)
    return {"models": items}


@router.get("/model/{repo_id:path}")
async def api_model_detail(repo_id: str):
    """Get model card (markdown + HTML), GGUF quantizations (grouped), and capabilities."""
    gguf_files = hf_service.list_gguf_files(repo_id)
    quantizations = hf_service.group_gguf_by_quantization(gguf_files)
    model_card = hf_service.get_model_card_content(repo_id)
    capabilities = hf_service.get_model_capabilities(repo_id)
    model_card_html = ""
    if model_card:
        model_card_html = markdown.markdown(model_card, extensions=["extra", "nl2br"])
    return {
        "repo_id": repo_id,
        "gguf_files": gguf_files,
        "quantizations": quantizations,
        "model_card": model_card,
        "model_card_html": model_card_html,
        "capabilities": capabilities,
    }


@router.post("/download")
async def api_download(
    repo_id: str,
    filename: str | None = None,
    filenames: str | None = None,
    section_name: str | None = None,
    background_tasks: BackgroundTasks = None,  # FastAPI injects when no default
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
    for f in to_download:
        if not f.endswith(".gguf"):
            raise HTTPException(status_code=400, detail="Only .gguf files allowed")
    config = get_config()
    models_dir = Path(config["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"{repo_id}:{to_download[0]}"
    _download_jobs[job_id] = {"status": "running", "path": None, "error": None}

    def run_download():
        try:
            env_before = os.environ.get("HF_HOME")
            os.environ["HF_HOME"] = str(models_dir)
            try:
                first_path = None
                for fn in to_download:
                    path = hf_service.download_model(repo_id, fn, models_dir)
                    if first_path is None:
                        first_path = path
            finally:
                if env_before is None:
                    os.environ.pop("HF_HOME", None)
                else:
                    os.environ["HF_HOME"] = env_before
            _download_jobs[job_id]["path"] = str(first_path)
            _download_jobs[job_id]["status"] = "completed"
            model_card = hf_service.get_model_card_content(repo_id)
            recommended = params_parser.recommended_params_with_defaults(model_card)
            section = section_name or _sanitize_section_name(repo_id, to_download[0])
            ini_path = get_models_ini_path(models_dir)
            recommended["LLAMA_ARG_MODEL"] = str(first_path)
            ini_manager.add_or_update_section(ini_path, section, recommended, merge=True)
        except Exception as e:
            _download_jobs[job_id]["status"] = "failed"
            _download_jobs[job_id]["error"] = str(e)

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
