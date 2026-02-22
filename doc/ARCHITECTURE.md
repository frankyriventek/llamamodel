# Architecture: LlamaModel

## Overview
LlamaModel is a Python-based web application providing an LMStudio-like interface designed to manage GGUF models for `llama.cpp`. It interacts with the Hugging Face Hub to discover, inspect, and download models, while maintaining a `models.ini` configuration file that defines loading parameters for the `llama.cpp` server.

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
        Params[Params Parser (params_parser.py)]
        
        subgraph Services [Core Services]
            HF[HF Service (hf_service.py)]
            INI[INI Manager (ini_manager.py)]
        end
    end

    subgraph Data [Filesystem]
        CfgFile[(config.yaml)]
        ModelsDir[(models_dir)]
        IniFile[(models.ini)]
        HFCache[(HF Hub Direct Path Cache)]
    end
    
    subgraph External [External Services]
        HuggingFace[Hugging Face API & File Hub]
    end

    UI --> Router
    Router --> HF
    Router --> INI
    Router --> Config
    Router --> Params
    
    HF --> HuggingFace
    HF --> HFCache
    INI --> IniFile
    Config --> CfgFile
    
    HFCache -. inside .- ModelsDir
    IniFile -. inside .- ModelsDir
```

## Directory & File Structure

- **`app/main.py`**: The FastAPI application entrypoint. Configures Jinja2 templates, static files, and mounts the various API/page routers.
- **`app/config.py`**: Handles configuration logic, resolving defaults, `config.yaml`, and environment variables (`LLAMAMODEL_PORT`, `LLAMAMODEL_MODELS_DIR`).
- **`app/routes/`**: Contains sub-routers.
  - `api.py`: Search backend, model details, file downloads, job statuses, download cancellation, and APIs for local `models.ini` reads.
  - `discover.py`: Renders the `/discover` frontend page.
  - `models_ini.py`: Renders pages for managing locally downloaded models, including the Parameters Editor in "My Models".
  - `settings.py`: Renders and accepts POST edits for the app settings.
- **`app/services/`**: Core business logic modules.
  - `hf_service.py`: Wraps `huggingface_hub` and HTTP calls for querying models, reading model cards, enumerating quantizations, parsing capability metrics, and handling background chunked downloads (with ETA, speed, and real-time cancellation logic).
  - `ini_manager.py`: Uses `configparser` and OS-level file locking to safely read/write `models.ini` with full compatibility for `llama.cpp`.
  - `params_parser.py`: Extracts model constraints (like `n_ctx`) out of Hugging Face model cards for dynamically structuring default configurations.
- **`static/`**: Static JS (`app.js`) and CSS (`style.css`), driving frontend dynamic behaviors such as search-as-you-type, sorting, filtering, parameter editing interfaces, and download progress tracking.
- **`templates/`**: Jinja2 templates standardizing the UI (base, discover, model details, settings, and my models).
- **`doc/`**: Documentation including project plans and this `ARCHITECTURE.md`.

## Data Flows

1. **Discovery & Search**: User actions (typing a query, clicking a tag) route to `/api/search` via AJAX. The backend accesses Hugging Face using `HfApi()`, caching filters logically and mapping visual characteristics (e.g., vision, thinking, tool calling) to user interface components.
2. **Model Download Tracking**: Initiating a download triggers a background chunked request in `hf_service.py`. The service tracks job state and byte flow in memory. The UI component polls for real-time progress (`/api/download/status...`), updating rendering the progress bar, completion ETA, and download speed. Users can dynamically cancel jobs via `api.py`.
3. **models.ini Management**: On a successful download, the `params_parser.py` evaluates the model card text and dynamically evaluates configuration variables, mapping `LLAMA_ARG_MODEL` precisely to the newly pulled physical file path on the disk. This updates the main `models.ini` tracked by `ini_manager.py`.
4. **My Models Editor**: The "My Models" interface serves as a comprehensive editor for `models.ini` defaults. The user can add, delete, or modify specific constraints loaded on initialization by the LLama C++ framework (e.g. `n-gpu-layers`, `ctx-size`). All updates execute file locks in real-time, rewriting the target config file safely.

## Download Mechanism & File Storage

Instead of utilizing Hugging Face's default `snapshots` symlink cache structure, LlamaModel linearly constructs direct standardized folder hierarchies utilizing base parameters:

1. **Storage Path:** Defaults to `$HOME/.cache/huggingface/models` or as overridden in App Config.
2. **Standardized Tree Generation:** `author_name/model_name/quantization_file_name.gguf`. (e.g., `bartowski/Llama-3/Llama-Q4.gguf`)
3. **Execution Logic:**
   - URL resolution extracts direct Hugging Face Git file endpoints.
   - Using synchronous or asynchronous chunking mechanisms, files stream directly to the target directory.
   - Intermediate/running downloads get strictly partitioned utilizing `.download` extensions until completeness is securely verified by final chunk arrivals. 
   - Real-time client indicators accurately trace file size expansions.

## Feature Completeness
- Parameter search logic handles autocomplete functionality directly from `models.ini` accepted variables mapping parameters cleanly to LLM server constraints.
- Real-time download controls gracefully roll back disk caching on cancellation events to prevent storage bloat.
