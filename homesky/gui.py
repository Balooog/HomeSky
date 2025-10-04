"""PySimpleGUI helper launcher for HomeSky."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import PySimpleGUI as sg

import ingest
from utils.db import DatabaseManager

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_EXAMPLE = PACKAGE_DIR / "config.example.toml"
CONFIG_TARGETS = [Path("config.toml"), PACKAGE_DIR / "config.toml"]


def bootstrap_config_file() -> Path:
    for candidate in CONFIG_TARGETS:
        if candidate.exists():
            return candidate
    target = CONFIG_TARGETS[-1]
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


def run_streamlit() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "visualize_streamlit.py"])


def main() -> None:
    sg.theme("DarkGrey9")
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
    ingest.setup_logging(config)
    storage = config.get("storage", {})
    data_dir = Path(storage.get("root_dir", "./data")).resolve()
    sqlite_path = Path(storage.get("sqlite_path", "./data/homesky.sqlite"))
    parquet_path = Path(storage.get("parquet_path", "./data/homesky.parquet"))

    layout = [
        [sg.Text("HomeSky Control", font=("Inter", 16))],
        [sg.Button("Fetch Now", key="fetch", size=(20, 1))],
        [sg.Button("Open Dashboard", key="dashboard", size=(20, 1))],
        [sg.Button("Open Data Folder", key="data", size=(20, 1))],
        [sg.Button("View Logs", key="logs", size=(20, 1))],
        [sg.Multiline(size=(60, 10), key="log", autoscroll=True, disabled=True)],
        [sg.Button("Exit")],
    ]

    window = sg.Window("HomeSky", layout, finalize=True)

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
        elif event == "dashboard":
            if dashboard_process and dashboard_process.poll() is None:
                window["log"].update("Dashboard already running\n", append=True)
            else:
                dashboard_process = run_streamlit()
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
