"""Read/write models.ini compatible with llama.cpp server."""

import fcntl
from configparser import ConfigParser
from pathlib import Path
from typing import Any

LLAMA_CONFIG_VERSION = "1"
LLAMA_ARG_PREFIX = "LLAMA_ARG_"


def _ensure_models_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _lock_file(f, exclusive: bool = True) -> None:
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    except (OSError, AttributeError):
        pass  # Windows or unsupported


def _unlock_file(f) -> None:
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (OSError, AttributeError):
        pass


def read_ini(ini_path: Path) -> ConfigParser:
    """Read models.ini; create minimal one if missing. Expects LLAMA_CONFIG_VERSION at top or [section] blocks."""
    parser = ConfigParser()
    if ini_path.exists():
        with open(ini_path) as f:
            _lock_file(f, exclusive=False)
            try:
                content = f.read()
            finally:
                _unlock_file(f)
        # ConfigParser requires sections; prepend a dummy section if file starts with key=value
        if content.strip() and not content.strip().startswith("["):
            content = "[__top__]\n" + content
        parser.read_string(content)
        if parser.has_section("__top__"):
            parser.remove_section("__top__")
    return parser


def write_ini(ini_path: Path, parser: ConfigParser) -> None:
    """Write models.ini with file lock. Writes LLAMA_CONFIG_VERSION at top then model sections."""
    _ensure_models_dir(ini_path)
    with open(ini_path, "w") as f:
        _lock_file(f, exclusive=True)
        try:
            f.write(f"LLAMA_CONFIG_VERSION = {LLAMA_CONFIG_VERSION}\n\n")
            for section in parser.sections():
                if section == "DEFAULT":
                    continue
                f.write(f"[{section}]\n")
                for k, v in parser[section].items():
                    f.write(f"{k} = {v}\n")
                f.write("\n")
        finally:
            _unlock_file(f)


def list_sections(ini_path: Path) -> list[dict[str, Any]]:
    """Return list of model sections (excluding DEFAULT). Each item: {name, params}."""
    parser = read_ini(ini_path)
    result = []
    for section in parser.sections():
        if section == "DEFAULT":
            continue
        params = dict(parser[section])
        result.append({"name": section, "params": params})
    return result


def get_section(ini_path: Path, section_name: str) -> dict[str, str] | None:
    """Get one section's params or None if missing."""
    parser = read_ini(ini_path)
    if not parser.has_section(section_name):
        return None
    return dict(parser[section_name])


def set_section(ini_path: Path, section_name: str, params: dict[str, str]) -> None:
    """Set or replace one section. Keys should be LLAMA_ARG_* style."""
    parser = read_ini(ini_path)
    if parser.has_section(section_name):
        parser.remove_section(section_name)
    parser.add_section(section_name)
    for k, v in params.items():
        parser.set(section_name, k, str(v))
    write_ini(ini_path, parser)


def add_or_update_section(
    ini_path: Path,
    section_name: str,
    params: dict[str, str],
    *,
    merge: bool = True,
) -> None:
    """Add or update a section. If merge=True, existing keys are preserved if not in params."""
    parser = read_ini(ini_path)
    if parser.has_section(section_name) and merge:
        existing = dict(parser[section_name])
        for k, v in params.items():
            existing[k] = v
        params = existing
    if parser.has_section(section_name):
        parser.remove_section(section_name)
    parser.add_section(section_name)
    for k, v in params.items():
        parser.set(section_name, k, str(v))
    write_ini(ini_path, parser)


def delete_section(ini_path: Path, section_name: str) -> bool:
    """Remove a section. Returns True if it existed."""
    parser = read_ini(ini_path)
    if not parser.has_section(section_name):
        return False
    parser.remove_section(section_name)
    write_ini(ini_path, parser)
    return True
