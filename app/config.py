"""Load application configuration from config.yaml and environment variables."""

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_PORT = 8081
DEFAULT_MODELS_DIR = "~/.cache/huggingface/models"

CONFIG_FILENAMES = ("config.yaml", "config.yml")
_CONFIG_WRITE_FILENAME = "config.yaml"


def _resolve_path(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def load_config() -> dict:
    """Load config from config.yaml (if present) and override with env vars."""
    root = Path(__file__).resolve().parent.parent
    config = {
        "port": DEFAULT_PORT,
        "models_dir": DEFAULT_MODELS_DIR,
    }
    cfg_file_used: str | None = None
    for name in CONFIG_FILENAMES:
        path = root / name
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            if "port" in data:
                config["port"] = int(data["port"])
            if "models_dir" in data:
                config["models_dir"] = str(data["models_dir"])
            cfg_file_used = str(path)
            break
    if cfg_file_used:
        logger.debug("Config file loaded: %s", cfg_file_used)
    else:
        logger.debug("No config file found; using defaults")
    if os.environ.get("LLAMAMODEL_PORT"):
        config["port"] = int(os.environ["LLAMAMODEL_PORT"])
        logger.debug("Port overridden by LLAMAMODEL_PORT env var: %s", config["port"])
    if os.environ.get("LLAMAMODEL_MODELS_DIR"):
        config["models_dir"] = os.environ["LLAMAMODEL_MODELS_DIR"]
        logger.debug("models_dir overridden by LLAMAMODEL_MODELS_DIR env var: %s", config["models_dir"])
    config["models_dir"] = str(_resolve_path(config["models_dir"]))
    logger.info("Effective config: port=%s models_dir=%s", config["port"], config["models_dir"])
    return config


def save_config(port: int, models_dir: str) -> None:
    """Persist port and models_dir to config.yaml in the project root.

    Note: environment-variable overrides still take precedence at runtime;
    the caller should reload the in-memory config after saving.
    """
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / _CONFIG_WRITE_FILENAME
    data: dict = {}
    # Preserve any extra keys already in the file
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
    data["port"] = int(port)
    data["models_dir"] = str(models_dir)
    with open(cfg_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    logger.info("Config saved to %s: port=%s models_dir=%s", cfg_path, port, models_dir)


def get_models_ini_path(models_dir: Path | str) -> Path:
    """Path to models.ini inside the models directory."""
    return Path(models_dir) / "models.ini"
