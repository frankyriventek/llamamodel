# Architecture: LLM Manager

## Overview
The LLM Manager is a Python-based web application providing an LMStudio-like interface designed to manage GGUF models for `llama.cpp`. It interacts with the Hugging Face Hub to discover, inspect, and download models, while maintaining a `models.ini` configuration file that defines loading parameters for the `llama.cpp` server.

## Graphical Architecture Block Diagram

```mermaid
flowchart LR
    subgraph Client [Client / Browser]
        UI[Web UI (HTML + JS)]
        CSS[styles.css]
    end

    subgraph Server [Backend / FastAPI]
        Router[API & Page Routers]
        Config[App Config (config.py)]
        Params[Params Parser]
        
        subgraph Services [Core Services]
            HF[HF Service (hf_service.py)]
            INI[INI Manager (ini_manager.py)]
        end
    end

    subgraph Data [Filesystem]
        CfgFile[(config.yaml)]
        ModelsDir[(models_dir)]
        IniFile[(models.ini)]
        GitCache[(HF Hub Cache)]
    end
    
    subgraph External [External Services]
        HuggingFace[Hugging Face API]
    end

    UI --> Router
    Router --> HF
    Router --> INI
    Router --> Config
    Router --> Params
    
    HF --> HuggingFace
    HF --> GitCache
    INI --> IniFile
    Config --> CfgFile
    
    GitCache -. inside .- ModelsDir
    IniFile -. inside .- ModelsDir
```

## Directory & File Structure

- **`app/main.py`**: The FastAPI application entrypoint. Configures Jinja2 templates, static files, and mounts the various API/page routers.
- **`app/config.py`**: Handles configuration logic, resolving defaults, `config.yaml`, and environment variables (`LLAMAMODEL_PORT`, `LLAMAMODEL_MODELS_DIR`).
- **`app/routes/`**: Contains sub-routers.
  - `api.py`: Search backend, model details, file downloads, job statuses, and APIs for local `models.ini` reads.
  - `discover.py`: Renders the `/discover` frontend page.
  - `models_ini.py`: Renders pages for managing locally downloaded models.
  - `settings.py`: Renders and accepts POST edits for the app settings.
- **`app/services/`**: Core business logic modules.
  - `hf_service.py`: Wraps `huggingface_hub` for querying models, reading model cards, enumerating quantizations, parsing capability metrics, and handling background downloads.
  - `ini_manager.py`: Uses `configparser` and OS-level file locking to read/write `models.ini` with compatibility for `llama.cpp`.
  - `params_parser.py`: Extracts model constraints (like `n_ctx`) out of Hugging Face model cards.
- **`static/`**: Static JS (`app.js`) and CSS (`style.css`), driving frontend dynamic behaviors like search-as-you-type, sorting, filtering, and toasts (flash messages).
- **`templates/`**: Jinja2 templates standardizing the UI.
- **`doc/`**: Documentation including `PLAN.md` and this `ARCHITECTURE.md`.

## Data Flows

1. **Discovery & Search**: User actions (typing a query, clicking a tag) route to `/api/search` via AJAX. The backend accesses Hugging Face using `HfApi()`, caching filters logically and mapping visual characteristics (vision, thinking, tools calling) back to UI-friendly payloads.
2. **Model Download**: Triggers a background task calling `hf_hub_download` asynchronously. An in-memory status dictionary tracking the job gets polled by the client for updates.
3. **models.ini Management**: On a successful download, the parameter parser evaluates the model card text and dynamically structures an INI configuration block ensuring the `LLAMA_ARG_MODEL` maps precisely to the newly grabbed disk file. Users optionally overwrite these mapped defaults in the "My Models" tab.

## Proposed Download Mechanism Architecture

### Requirements
1. **Default Directory:** `$HOME/.cache/huggingface/models`
2. **Directory Structure:** `author_name/model_name/` containing all quantizations for the model.
3. **UI Download Pane:** Progress bar, model name, file size, ETA, cancel button.
4. **Quantization Buttons UI:**
    - Downloaded: Italic label, tooltip "Quantized model already downloaded"
    - Not downloaded: Bold label, tooltip "Click to download Quantized Model"
5. **Re-download Confirmation:** Prompt if attempting to download an already existing quantization.

### Proposed Architecture

#### 1. Backend Changes (`app/services/download_service.py` & `api.py`)
- Natively `hf_hub_download` does not provide an async progress callback or an easy cancellation mechanism with chunk tracking.
- To achieve real-time progress, ETA, and cancellation, we will implement a custom chunked downloader using the `requests` library (which is available as a dependency of `huggingface_hub`).
- **URL Resolution:** We will use `huggingface_hub.hf_hub_url` to get the direct file URL for `repo_id` and `filename` or utilize `http_get` / HTTP requests wrapped with HF Hub tokens.
- **Target Path Construction:** 
    - The base download directory will be configured to default to `$HOME/.cache/huggingface/models`.
    - `author_name`, `model_name` will be parsed from `repo_id` (e.g., `author/model` -> `author`, `model`).
    - The final path will be `models_dir / author_name / model_name / filename`.
- **Download State Tracking:**
    - Maintain an in-memory dictionary `_download_jobs` to track global job states.
    - Fields: `job_id`, `repo_id`, `filename`, `status` (running, cancelled, completed, failed), `total_bytes`, `downloaded_bytes`, `start_time`, `speed`, `eta`, `error`.
    - An endpoint `/api/download/cancel/{job_id}` will transition the state to `cancelled`. The chunk loop will check this flag and abort cleanly, removing incomplete temporary files (`.download`).
- **File Exist Check endpoint:** Add an API endpoint `GET /api/models/check?repo_id=...&filename=...` to check if a target quantization is already downloaded (by computing the target path and checking `os.path.exists`). 

#### 2. Frontend Changes (`static/app.js` & `templates/`)
- **Quantization Buttons:** 
    - On model detail rendering, retrieve the list of already downloaded files.
    - Conditionally apply CSS classes for bold (un-downloaded) or italic/cursive (downloaded), updating the `title` attributes (tooltips) respectively.
    - Attach a click handler that checks the downloaded status. If downloaded, `confirm()` dialog pops up. If confirmed or not yet downloaded, initiate the POST to `/api/download`.
- **Download Pane Component:**
    - A fixed or floating pane detailing active downloads.
    - Polling `/api/download/{job_id}` every ~500ms to update progress bars, speed, ETA, and downloaded/total bytes.
    - Cancel button attached to each active job fires `/api/download/cancel/{job_id}`.

### Directory Resolution Overriding existing HF Cache
Instead of using Hugging Face's snapshot symlink structure (`models--author--model/snapshots/...`), we will write files linearly mapping the HF structure to standard user-friendly folders `/author/model/`.
