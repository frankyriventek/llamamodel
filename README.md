# LLM Manager

A Python web application to manage large language models for **llama.cpp** servers: search and browse GGUF models on Hugging Face, download selected quantizations, and maintain a `models.ini` file compatible with the llama.cpp server.

## Features

- **Discover** – Search Hugging Face for GGUF-compatible models (LMStudio-like).
- **Model detail** – View the Hugging Face model card and list of available quantizations (GGUF files).
- **Download** – Download a chosen quantization via the Hugging Face API into your configured models directory.
- **models.ini** – After each download, the app adds an entry to `models.ini` with recommended parameters (parsed from the model card) and the path to the downloaded file. You can edit default parameters (e.g. context size, GPU layers) in **My Models**.
- **Configurable** – Port and models directory (and thus `models.ini` location) are configurable.

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt`

## Installation

```bash
cd /path/to/llamamodel
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Configuration

| Parameter           | Default                    | Description |
|--------------------|----------------------------|-------------|
| **port**           | `8081`                     | HTTP port for the web app. |
| **models_dir**     | `~/.cache/huggingface`      | Root directory for Hugging Face cache; downloads go under `models_dir/hub/`. The file `models.ini` is stored as `models_dir/models.ini`. |

Configure via:

1. **config.yaml** in the project root:

   ```yaml
   port: 8081
   models_dir: ~/.cache/huggingface
   ```

2. **Environment variables** (override config file):

   - `LLAMAMODEL_PORT` – port number
   - `LLAMAMODEL_MODELS_DIR` – path to models directory (e.g. `~/.cache/huggingface`)

## Running

```bash
python run.py
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8081
```

Then open **http://localhost:8081** (or the port you configured).

## models.ini and llama.cpp server

The app writes and updates `models.ini` in your configured models directory. The format is compatible with the **llama.cpp server**:

- First line: `LLAMA_CONFIG_VERSION = 1`
- One section per model: `[model_name]` with keys like `LLAMA_ARG_MODEL`, `LLAMA_ARG_N_CTX`, `LLAMA_ARG_N_GPU_LAYERS`, etc.

Use this file with the llama.cpp server (e.g. `--model-config path/to/models.ini` or the equivalent option in your build) to load models with the default parameters you set in the app.

## Project layout

- `app/` – FastAPI app, config, routes, services (Hugging Face, models.ini, params parsing).
- `templates/` – Jinja2 HTML (Discover, model detail, My Models, Settings).
- `config.yaml` – Default port and models directory.
- `run.py` – Entry point that reads config and runs uvicorn.
