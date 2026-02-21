"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import load_config, get_models_ini_path

app = FastAPI(title="LLM Manager", description="Manage GGUF models for llama.cpp")

_config: dict | None = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


# Templates and static files
ROOT = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
if (ROOT / "static").exists():
    app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Redirect to Discover (LMStudio-like entry)."""
    return templates.TemplateResponse("index.html", {"request": request})


# Routes will be registered by including routers
def setup_routes():
    from app.routes import discover, models_ini, settings, api
    app.include_router(discover.router, tags=["discover"])
    app.include_router(models_ini.router, prefix="/models", tags=["models"])
    app.include_router(settings.router, prefix="/settings", tags=["settings"])
    app.include_router(api.router, prefix="/api", tags=["api"])


setup_routes()
