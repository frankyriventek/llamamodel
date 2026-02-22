"""My Models: list and edit models.ini sections."""

import logging
from pathlib import Path

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from app.main import templates, get_config
from app.config import get_models_ini_path
from app.services import ini_manager
import os

logger = logging.getLogger(__name__)

router = APIRouter()


def _ini_path():
    config = get_config()
    return get_models_ini_path(config["models_dir"])


@router.get("", response_class=HTMLResponse)
async def my_models_page(request: Request):
    """List models defined in models.ini and dynamically scan for unconfigured .gguf files."""
    logger.debug("GET /models")
    config = get_config()
    path = _ini_path()
    sections = ini_manager.list_sections(path)
    logger.debug("GET /models: %d sections", len(sections))

    models_dir = Path(config["models_dir"])
    configured_paths = set()
    
    # We will enrich sections to have file size, card name, quant, file name
    from app.services.hf_service import _extract_quantization
    import os
    
    enriched_sections = []
    
    for s in sections:
        model_path_str = s["params"].get("LLAMA_ARG_MODEL")
        if model_path_str:
            resolved = Path(model_path_str).resolve()
            configured_paths.add(str(resolved))
            file_name = resolved.name
            size_mb = 0
            if resolved.exists():
                size_mb = os.path.getsize(resolved) / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.0f} MB"
            
            quant = _extract_quantization(file_name)
            # if we successfully extracted quant, remove it from the name to get card name
            card_name = file_name
            if quant:
                # Find the quant string and remove it (and preceding '-' or '.')
                import re
                esc_quant = re.escape(quant)
                # match dash or dot, then the quant (case insensitive), then .gguf
                # example: -Q4_K_M.gguf or .q4_k_m.gguf
                m = re.search(r"[-.]" + esc_quant + r"(?:\.gguf)?", file_name, flags=re.IGNORECASE)
                if m:
                    card_name = file_name[:m.start()]
                else:
                    card_name = file_name.replace(".gguf", "")
            else:
                card_name = file_name.replace(".gguf", "")
        else:
            file_name = "Unknown"
            size_str = "0 MB"
            card_name = s.name
            quant = "Unknown"
            
        s["display"] = {
            "card_name": card_name,
            "quantization": quant,
            "file_name": file_name,
            "file_size": size_str,
        }
        enriched_sections.append(s)

    unconfigured_files = []
    if models_dir.exists():
        for gf in models_dir.rglob("*.gguf"):
            rp = str(gf.resolve())
            if rp not in configured_paths:
                unconfigured_files.append({
                    "name": gf.name,
                    "path": rp,
                })
    unconfigured_files.sort(key=lambda x: x["name"])

    return templates.TemplateResponse(
        "my_models.html",
        {
            "request": request,
            "sections": enriched_sections,
            "unconfigured_files": unconfigured_files,
            "config": config
        },
    )


@router.post("/add_local")
async def add_local_model(request: Request, path: str = Form(...)):
    """Add a local unconfigured GGUF to models.ini."""
    logger.debug("POST /models/add_local: %s", path)
    gguf_path = Path(path)
    if not gguf_path.exists():
        raise HTTPException(status_code=400, detail="File not found")
    
    section_name = gguf_path.name.replace(".gguf", "")
    ini_path = _ini_path()
    params = {"LLAMA_ARG_MODEL": str(gguf_path.resolve())}
    ini_manager.add_or_update_section(ini_path, section_name, params, merge=True)
    return RedirectResponse(url="/models", status_code=303)


@router.get("/edit/{section_name}", response_class=HTMLResponse)
async def edit_model_page(request: Request, section_name: str):
    """Edit one model section."""
    logger.debug("GET /models/edit/%s", section_name)
    config = get_config()
    path = _ini_path()
    params = ini_manager.get_section(path, section_name)
    if params is None:
        logger.warning("GET /models/edit/%s: section not found", section_name)
        raise HTTPException(status_code=404, detail="Model not found")
    return templates.TemplateResponse(
        "model_edit.html",
        {"request": request, "section_name": section_name, "params": params, "config": config},
    )


@router.post("/edit/{section_name}")
async def save_model(section_name: str, request: Request):
    """Save model section from form. Form keys: param_<key> = value; optional new_param_key, new_param_value."""
    logger.debug("POST /models/edit/%s", section_name)
    path = _ini_path()
    form = await request.form()
    params = {}
    for k, v in form.items():
        if k.startswith("param_") and v is not None:
            arg_key = k[6:].strip()
            if arg_key:
                params[arg_key] = str(v).strip()
    new_key = (form.get("new_param_key") or "").strip()
    new_val = (form.get("new_param_value") or "").strip()
    if new_key:
        params[new_key] = new_val
    ini_manager.set_section(path, section_name, params)
    logger.info("POST /models/edit/%s: saved %d params", section_name, len(params))
    return RedirectResponse(url="/models", status_code=303)


@router.get("/delete/{section_name}")
async def delete_model(section_name: str):
    """Remove model section and associated file."""
    logger.debug("GET /models/delete/%s", section_name)
    path = _ini_path()
    
    # Check if there is a file associated
    params = ini_manager.get_section(path, section_name)
    file_deleted = False
    
    if params and "LLAMA_ARG_MODEL" in params:
        file_route = Path(params["LLAMA_ARG_MODEL"]).resolve()
        if file_route.exists() and file_route.is_file():
            try:
                os.remove(file_route)
                logger.info("GET /models/delete/%s: deleted file %s", section_name, file_route)
                file_deleted = True
            except Exception as e:
                logger.error("Failed to delete file %s: %s", file_route, e)
    
    ok = ini_manager.delete_section(path, section_name)
    if not ok:
        logger.warning("GET /models/delete/%s: section not found", section_name)
        raise HTTPException(status_code=404, detail="Model not found")
    logger.info("GET /models/delete/%s: section deleted", section_name)
    return RedirectResponse(url="/models", status_code=303)
