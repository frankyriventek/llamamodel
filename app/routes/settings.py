"""Settings: show port and models dir (display only; config via config file / env)."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.main import templates, get_config

router = APIRouter()


@router.get("", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Show application settings (port, models directory)."""
    config = get_config()
    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "config": config},
    )
