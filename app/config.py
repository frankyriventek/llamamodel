"""Load application configuration from config.yaml and environment variables."""

import os
from pathlib import Path

import yaml

# Defaults
DEFAULT_PORT = 8081
DEFAULT_MODELS_DIR = "~/.cache/huggingface"

CONFIG_FILENAMES = ("config.yaml", "config.yml")


def _resolve_path(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def load_config() -> dict:
    """Load config from config.yaml (if present) and override with env vars."""
    root = Path(__file__).resolve().parent.parent
    config = {
        "port": DEFAULT_PORT,
        "models_dir": DEFAULT_MODELS_DIR,
    }
    for name in CONFIG_FILENAMES:
        path = root / name
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            if "port" in data:
                config["port"] = int(data["port"])
            if "models_dir" in data:
                config["models_dir"] = str(data["models_dir"])
            break
    if os.environ.get("LLAMAMODEL_PORT"):
        config["port"] = int(os.environ["LLAMAMODEL_PORT"])
    if os.environ.get("LLAMAMODEL_MODELS_DIR"):
        config["models_dir"] = os.environ["LLAMAMODEL_MODELS_DIR"]
    config["models_dir"] = str(_resolve_path(config["models_dir"]))
    return config


def get_models_ini_path(models_dir: Path | str) -> Path:
    """Path to models.ini inside the models directory."""
    return Path(models_dir) / "models.ini"
