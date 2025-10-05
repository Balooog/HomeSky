"""PySimpleGUI helper launcher for HomeSky."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import PySimpleGUI as sg

import ingest
from utils.config import (
    bootstrap_target_path,
    candidate_config_paths,
    ensure_parent_directory,
)

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore
from utils.db import DatabaseManager

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_EXAMPLE = PACKAGE_DIR / "config.example.toml"
CONFIG_TARGETS = candidate_config_paths()
STREAMLIT_ENTRY = PACKAGE_DIR / "visualize_streamlit.py"


def bootstrap_config_file() -> Path:
    for candidate in CONFIG_TARGETS:
        if candidate.exists():
            return candidate
    target = bootstrap_target_path()
    ensure_parent_directory(target)
    if CONFIG_EXAMPLE.exists():
        target.write_text(CONFIG_EXAMPLE.read_text(), encoding="utf-8")
    else:
        stub = (
            "[ambient]\n"
            "api_key = \"\"\n"
            "application_key = \"\"\n"
            "mac = \"\"\n\n"
            "[storage]\n"
            "root_dir = \"./data\"\n"
            "sqlite_path = \"./data/homesky.sqlite\"\n"
            "parquet_path = \"./data/homesky.parquet\"\n"
        )
        target.write_text(stub, encoding="utf-8")
    return target


def load_config() -> dict:
    return ingest.load_config()


def open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":  # pragma: no cover
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def should_launch_streamlit(config: dict) -> tuple[bool, str]:
    ui_settings = config.get("ui", {})
    if not ui_settings.get("launch_streamlit", False):
        return False, "Streamlit dashboard launch disabled. Set [ui] launch_streamlit = true to enable."
    if not STREAMLIT_ENTRY.exists():
        return False, f"Streamlit entrypoint missing at {STREAMLIT_ENTRY}."
    return True, ""


def run_streamlit(port: int | None = None) -> subprocess.Popen | None:
    try:
        command = ["streamlit", "run", str(STREAMLIT_ENTRY)]
        if port:
            command.extend(["--server.port", str(port)])
        return subprocess.Popen(command)
    except FileNotFoundError:
        return None


def main() -> None:
    sg.theme("DarkGrey9")
    post_init_messages: list[str] = []
    try:
        config = load_config()
    except FileNotFoundError:
        created_path = bootstrap_config_file()
        sg.popup_ok(
            "Missing config.toml. A starter file has been created at\n"
            f"{created_path.resolve()}\n\nUpdate your Ambient Weather credentials and relaunch HomeSky.",
            title="HomeSky configuration required",
        )
        return
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        existing = next((p for p in CONFIG_TARGETS if p.exists()), None)
        location = f"\n\nLocation: {existing.resolve()}" if existing else ""
        sg.popup_ok(
            "HomeSky found a configuration file but could not parse it."
            f"\n\nDetails: {exc}{location}\n\nReview the file for syntax errors or restore it from config.example.toml.",
            title="HomeSky configuration error",
        )
        return
    except Exception as exc:  # pragma: no cover - defensive guard
        sg.popup_error(
            "Unexpected error while loading configuration:\n\n"
            f"{exc}\n\nRun tools/ensure_config.ps1 to recreate a fresh config if the issue persists.",
            title="HomeSky configuration error",
        )
        return
    repaired = getattr(ingest.load_config, "last_was_repaired", False)
    normalized = getattr(ingest.load_config, "last_was_normalized", False)
    source_path = getattr(ingest.load_config, "last_path", None)
    if repaired and source_path:
        source = Path(source_path).resolve()
        backup = source.with_name("config.bak")
        backup_note = (
            f" A backup of the previous file is available at\n{backup.resolve()}"
            if backup.exists()
            else ""
        )
        sg.popup_ok(
            "HomeSky detected a corrupt configuration file and restored a fresh template.\n\n"
            f"Updated file: {source}.{backup_note}\n\nUpdate your Ambient Weather credentials and relaunch HomeSky.",
            title="HomeSky configuration restored",
        )
        return
    if normalized and source_path:
        post_init_messages.append(
            f"Config formatting normalized at {Path(source_path).resolve()}\n"
        )
    ingest.setup_logging(config)
    storage = config.get("storage", {})
    data_dir = Path(storage.get("root_dir", "./data")).resolve()
    sqlite_path = Path(storage.get("sqlite_path", "./data/homesky.sqlite"))
    parquet_path = Path(storage.get("parquet_path", "./data/homesky.parquet"))

    streamlit_allowed, streamlit_reason = should_launch_streamlit(config)

    layout = [
        [sg.Text("HomeSky Control", font=("Inter", 16))],
        [sg.Button("Fetch Now", key="fetch", size=(20, 1))],
        [sg.Button("Backfill (24h)", key="backfill24", size=(20, 1))],
        [
            sg.Button(
                "Open Dashboard",
                key="dashboard",
                size=(20, 1),
                disabled=not streamlit_allowed,
            )
        ],
        [sg.Button("Open Data Folder", key="data", size=(20, 1))],
        [sg.Button("View Logs", key="logs", size=(20, 1))],
        [sg.Multiline(size=(60, 10), key="log", autoscroll=True, disabled=True)],
        [sg.Button("Exit")],
    ]

    window = sg.Window("HomeSky", layout, finalize=True)

    if post_init_messages:
        for message in post_init_messages:
            window["log"].update(message, append=True)
    if not streamlit_allowed and streamlit_reason:
        window["log"].update(f"{streamlit_reason}\n", append=True)

    dashboard_process: subprocess.Popen | None = None
    db = DatabaseManager(sqlite_path, parquet_path)

    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        if event == "fetch":
            try:
                inserted = ingest.ingest_once(config, db)
                window["log"].update(f"Fetched {inserted} new rows\n", append=True)
            except Exception as exc:  # pragma: no cover
                window["log"].update(f"Error: {exc}\n", append=True)
        elif event == "backfill24":
            try:
                config = ingest.load_config()
                added = ingest.backfill(config, hours=24, db=db)
            except Exception as exc:
                sg.popup_error(
                    "Backfill failed.",
                    f"\nDetails: {exc}\n",
                    title="HomeSky backfill error",
                )
            else:
                message = (
                    f"Backfill added {added} rows\n"
                    if added
                    else "Backfill complete; no new rows\n"
                )
                window["log"].update(message, append=True)
        elif event == "dashboard":
            if dashboard_process and dashboard_process.poll() is None:
                window["log"].update("Dashboard already running\n", append=True)
            else:
                try:
                    config = ingest.load_config()
                except Exception as exc:
                    sg.popup_error(
                        "Streamlit could not start because the configuration could not be loaded.",
                        f"\nDetails: {exc}\n",
                        title="HomeSky dashboard error",
                    )
                    continue
                streamlit_allowed, streamlit_reason = should_launch_streamlit(config)
                if not streamlit_allowed:
                    if streamlit_reason:
                        window["log"].update(f"{streamlit_reason}\n", append=True)
                    continue
                port_value = config.get("ui", {}).get("dashboard_port")
                port_int: int | None = None
                if port_value:
                    try:
                        port_int = int(port_value)
                    except (TypeError, ValueError):
                        window["log"].update(
                            "Invalid dashboard_port value; launching Streamlit with default port.\n",
                            append=True,
                        )
                dashboard_process = run_streamlit(port=port_int)
                if dashboard_process is None:
                    window["log"].update(
                        "Streamlit CLI not found. Install streamlit in the active environment to enable the dashboard.\n",
                        append=True,
                    )
                else:
                    window["log"].update("Launching Streamlit dashboard...\n", append=True)
        elif event == "data":
            data_dir.mkdir(parents=True, exist_ok=True)
            open_path(data_dir)
            window["log"].update(f"Opened data folder at {data_dir}\n", append=True)
        elif event == "logs":
            log_path = Path(config.get("ingest", {}).get("log_path", "./data/logs/ingest.log")).resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            open_path(log_path if log_path.exists() else log_path.parent)
            window["log"].update(f"Opened log path {log_path}\n", append=True)

    if dashboard_process and dashboard_process.poll() is None:
        dashboard_process.terminate()

    window.close()


if __name__ == "__main__":
    main()
