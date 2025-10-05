"""Shared helpers for locating HomeSky configuration files."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List

ENVIRONMENT_VARIABLE = "HOMESKY_CONFIG"
APP_DIR_NAME = "HomeSky"
CONFIG_FILENAME = "config.toml"


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    """Return the given paths with duplicates removed while preserving order."""

    seen: set[Path] = set()
    ordered: List[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def environment_config_path() -> Path | None:
    """Return the config path specified via HOMESKY_CONFIG, if provided."""

    configured = os.environ.get(ENVIRONMENT_VARIABLE)
    if not configured:
        return None
    return Path(configured).expanduser()


def external_config_directory() -> Path:
    """Return the OS-appropriate directory for storing persistent config."""

    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / APP_DIR_NAME
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / APP_DIR_NAME
    return Path.home() / ".config" / APP_DIR_NAME


def external_config_path() -> Path:
    """Return the preferred external config path."""

    return external_config_directory() / CONFIG_FILENAME


def repository_config_paths() -> List[Path]:
    """Return config paths that live within the repository tree."""

    return [Path(CONFIG_FILENAME), Path("homesky") / CONFIG_FILENAME]


def candidate_config_paths() -> List[Path]:
    """Return config locations in priority order for loading."""

    env_path = environment_config_path()
    paths: List[Path] = []
    if env_path:
        paths.append(env_path)
    paths.append(external_config_path())
    paths.extend(repository_config_paths())
    return _unique_paths(paths)


def bootstrap_target_path() -> Path:
    """Return the path where a new config should be created when missing."""

    env_path = environment_config_path()
    if env_path:
        return env_path
    return external_config_path()


def ensure_parent_directory(path: Path) -> None:
    """Create the parent directory for *path* if it does not already exist."""

    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
