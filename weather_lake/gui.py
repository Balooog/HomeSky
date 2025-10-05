"""PySimpleGUI helper launcher for Weather Lake."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import PySimpleGUI as sg

import ingest
from storage import StorageResult


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
        sg.popup_error("Missing config.toml. Copy config.example.toml and update your credentials.")
        return
    ingest.setup_logging(config)
    storage_cfg = config.get("storage", {})
    data_dir = Path(storage_cfg.get("root_dir", "./data")).resolve()
    storage_manager = ingest.get_storage_manager(config)

    layout = [
        [sg.Text("Weather Lake Control", font=("Inter", 16))],
        [sg.Button("Fetch Now", key="fetch", size=(20, 1))],
        [sg.Button("Open Dashboard", key="dashboard", size=(20, 1))],
        [sg.Button("Open Data Folder", key="data", size=(20, 1))],
        [sg.Button("View Logs", key="logs", size=(20, 1))],
        [sg.Multiline(size=(60, 10), key="log", autoscroll=True, disabled=True)],
        [sg.Button("Exit")],
    ]

    window = sg.Window("Weather Lake", layout, finalize=True)

    dashboard_process: subprocess.Popen | None = None

    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        if event == "fetch":
            try:
                result = ingest.ingest_once(config, storage_manager)
                if isinstance(result, StorageResult):
                    message = (
                        f"Fetched {result.inserted} new rows\n"
                        if result.inserted
                        else "No new rows\n"
                    )
                else:
                    message = f"Fetched {result} new rows\n"
                window["log"].update(message, append=True)
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
