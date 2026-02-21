"""Discover: search GGUF models and model detail page."""

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse

from app.main import templates, get_config
from app.services import hf_service

router = APIRouter()


@router.get("/discover", response_class=HTMLResponse)
async def discover_page(
    request: Request,
    q: str | None = Query(None),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
):
    """Discover page: search and list GGUF models."""
    config = get_config()
    models = []
    models = hf_service.search_models(query=q, limit=limit, offset=offset)
    return templates.TemplateResponse(
        "discover.html",
        {"request": request, "models": models, "query": q or "", "config": config},
    )


@router.get("/model/{repo_id:path}", response_class=HTMLResponse)
async def model_detail_page(request: Request, repo_id: str):
    """Model detail: Hugging Face description and list of quantizations (GGUF files)."""
    config = get_config()
    gguf_files = hf_service.list_gguf_files(repo_id)
    model_card = hf_service.get_model_card_content(repo_id)
    return templates.TemplateResponse(
        "model_detail.html",
        {
            "request": request,
            "repo_id": repo_id,
            "gguf_files": gguf_files,
            "model_card": model_card,
            "config": config,
        },
    )
