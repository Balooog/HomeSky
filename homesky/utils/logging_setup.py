"""Shared logging configuration for HomeSky."""

from __future__ import annotations

import datetime  # noqa: F401  # Imported for potential future timestamp helpers.
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import sys


LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "homesky", level: int = logging.INFO) -> logging.Logger:
    """Return a shared, rotating file logger for the given *name*."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_path = LOG_DIR / f"{name}.log"
    handler = TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=7, encoding="utf-8"
    )
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d – %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    logger.setLevel(level)
    logger.propagate = False
    logger.info("Logging initialized → %s", log_path)
    return logger


__all__ = ["get_logger"]

