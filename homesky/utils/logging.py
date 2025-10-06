"""Streamlit logging helpers for HomeSky."""

from __future__ import annotations

import datetime as _dt
import io
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional, TextIO
import warnings


class _StreamTee(io.TextIOBase):
    """Mirror writes to the original stream and a file handle."""

    def __init__(self, original: TextIO, log_handle: TextIO) -> None:
        super().__init__()
        self._original = original
        self._log_handle = log_handle
        self._lock = threading.Lock()

    def write(self, s: str) -> int:  # type: ignore[override]
        if not isinstance(s, str):
            s = str(s)
        with self._lock:
            self._original.write(s)
            self._original.flush()
            self._log_handle.write(s)
            self._log_handle.flush()
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        with self._lock:
            self._original.flush()
            self._log_handle.flush()

    def close(self) -> None:  # type: ignore[override]
        # Do not close the original stream; only flush both.
        with self._lock:
            self._log_handle.flush()
            self._original.flush()


_INITIALIZED = False
_LOG_HANDLE: Optional[TextIO] = None
_ORIGINAL_STDOUT: Optional[TextIO] = None
_ORIGINAL_STDERR: Optional[TextIO] = None
_ORIGINAL_EXCEPTHOOK = None
_ORIGINAL_SHOWWARNING = None


def _log_path(custom_path: Optional[str] = None) -> Path:
    if custom_path:
        path = Path(custom_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    base_dir = Path("data") / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "streamlit_error.log"


def _append_header(handle: TextIO, header: str) -> None:
    timestamp = _dt.datetime.now().isoformat()
    handle.write(f"\n[{timestamp}] {header}\n")
    handle.flush()


def setup_streamlit_logging(log_file: Optional[str] = None) -> None:
    """Mirror stdout/stderr and capture uncaught exceptions to a log file."""

    global _INITIALIZED, _LOG_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    global _ORIGINAL_EXCEPTHOOK, _ORIGINAL_SHOWWARNING

    if _INITIALIZED:
        return

    path = _log_path(log_file)
    log_handle = open(path, "a", encoding="utf-8", buffering=1)

    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    _LOG_HANDLE = log_handle

    sys.stdout = _StreamTee(sys.stdout, log_handle)
    sys.stderr = _StreamTee(sys.stderr, log_handle)

    _ORIGINAL_EXCEPTHOOK = sys.excepthook

    def _excepthook(exc_type, exc_value, exc_traceback):
        if _LOG_HANDLE is not None:
            _append_header(_LOG_HANDLE, "Uncaught Streamlit error")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=_LOG_HANDLE)
            _LOG_HANDLE.flush()
        if _ORIGINAL_EXCEPTHOOK is not None:
            _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)

    sys.excepthook = _excepthook

    _ORIGINAL_SHOWWARNING = warnings.showwarning

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        if _LOG_HANDLE is not None:
            _append_header(_LOG_HANDLE, f"Warning: {category.__name__}")
            _LOG_HANDLE.write(
                f"{filename}:{lineno}: {category.__name__}: {message}\n"
            )
            if line:
                _LOG_HANDLE.write(f"    {line.strip()}\n")
            _LOG_HANDLE.flush()
        if _ORIGINAL_SHOWWARNING is not None:
            _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file, line)

    warnings.showwarning = _showwarning

    _INITIALIZED = True


__all__ = ["setup_streamlit_logging"]
