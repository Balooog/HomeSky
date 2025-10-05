"""PySimpleGUI helper launcher for HomeSky."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import PySimpleGUI as sg

import ingest
from backfill import backfill_range
from import_offline import import_files
from storage import StorageResult

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

PACKAGE_DIR = Path(__file__).resolve().parent
STREAMLIT_ENTRY = PACKAGE_DIR / "visualize_streamlit.py"


def bootstrap_config_file() -> Path:
    return ingest.ensure_config()


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


def _format_timestamp(ts: object) -> str:
    if ts is None:
        return "n/a"
    if hasattr(ts, "to_pydatetime"):
        dt = ts.to_pydatetime()
    else:
        dt = ts  # type: ignore[assignment]
    if not isinstance(dt, datetime):
        return str(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def describe_result(action: str, result: StorageResult) -> str:
    if result.inserted:
        start = _format_timestamp(result.start)
        end = _format_timestamp(result.end)
        return f"{action}: {result.inserted} new rows ({start} – {end})\n"
    return f"{action}: no new rows\n"


def prompt_custom_backfill(
    *,
    default_limit: int,
    default_window: int,
) -> tuple[datetime, datetime, int, int] | None:
    layout = [
        [sg.Text("Start (UTC ISO)"), sg.Input(key="start")],
        [sg.Text("End (UTC ISO)"), sg.Input(key="end")],
        [sg.Text("Window minutes"), sg.Input(str(default_window), key="window")],
        [sg.Text("Limit per call"), sg.Input(str(default_limit), key="limit")],
        [sg.Button("Run", key="run"), sg.Button("Cancel")],
    ]
    dialog = sg.Window("Custom Backfill", layout, modal=True, keep_on_top=True)
    event, values = dialog.read()
    dialog.close()
    if event != "run":
        return None
    try:
        start = datetime.fromisoformat(values.get("start", "").strip())
        end = datetime.fromisoformat(values.get("end", "").strip())
    except ValueError:
        sg.popup_error("Enter ISO timestamps like 2024-05-29T00:00", title="Invalid datetime")
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)
    if start >= end:
        sg.popup_error("Start must be before end", title="Invalid range")
        return None
    try:
        window_minutes = int(values.get("window", default_window) or default_window)
        limit = int(values.get("limit", default_limit) or default_limit)
    except ValueError:
        sg.popup_error("Window and limit must be integers", title="Invalid settings")
        return None
    return start, end, window_minutes, limit


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
        expected = ingest.get_config_path()
        location = f"\n\nLocation: {expected.resolve()}" if expected else ""
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

    def resolve_paths(cfg: dict) -> tuple[Path, Path]:
        storage_cfg = cfg.get("storage", {})
        data = Path(storage_cfg.get("root_dir", "./data")).resolve()
        log = Path(cfg.get("ingest", {}).get("log_path", "./data/logs/ingest.log")).resolve()
        return data, log

    def resolve_limits(cfg: dict) -> tuple[int, int]:
        ingest_cfg = cfg.get("ingest", {})
        limit = int(ingest_cfg.get("backfill_limit", 288) or 288)
        window_default = int(ingest_cfg.get("backfill_window_minutes", 1440) or 1440)
        return limit, window_default

    storage_manager = ingest.get_storage_manager(config)
    data_dir, log_path = resolve_paths(config)
    limit_per_call, window_default = resolve_limits(config)

    streamlit_allowed, streamlit_reason = should_launch_streamlit(config)

    layout = [
        [sg.Text("HomeSky Control", font=("Inter", 16))],
        [sg.Button("Fetch Now", key="fetch", size=(20, 1))],
        [sg.Button("Backfill (24h)", key="backfill24", size=(20, 1))],
        [sg.Button("Backfill (custom)", key="backfill_custom", size=(20, 1))],
        [sg.Button("Import file(s)", key="import", size=(20, 1))],
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

    def reload_environment() -> bool:
        nonlocal config, storage_manager, data_dir, log_path, limit_per_call, window_default, streamlit_allowed, streamlit_reason
        try:
            config = ingest.load_config()
        except Exception as exc:
            sg.popup_error(
                "Unable to reload configuration.",
                f"\nDetails: {exc}\n",
                title="HomeSky configuration error",
            )
            return False
        storage_manager = ingest.get_storage_manager(config)
        data_dir, log_path = resolve_paths(config)
        limit_per_call, window_default = resolve_limits(config)
        streamlit_allowed, streamlit_reason = should_launch_streamlit(config)
        window["dashboard"].update(disabled=not streamlit_allowed)
        if not streamlit_allowed and streamlit_reason:
            window["log"].update(f"{streamlit_reason}\n", append=True)
        return True

    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        if event == "fetch":
            try:
                result = ingest.ingest_once(config, storage_manager)
            except Exception as exc:  # pragma: no cover
                window["log"].update(f"Error: {exc}\n", append=True)
            else:
                window["log"].update(describe_result("Fetch", result), append=True)
        elif event == "backfill24":
            if not reload_environment():
                continue
            mac = config.get("ambient", {}).get("mac")
            if not mac:
                sg.popup_error(
                    "Configure an Ambient Weather station MAC before running backfill.",
                    title="HomeSky backfill error",
                )
                continue
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(hours=24)
            try:
                result = backfill_range(
                    config=config,
                    storage=storage_manager,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    mac=mac,
                    limit_per_call=limit_per_call,
                )
            except Exception as exc:
                sg.popup_error(
                    "Backfill failed.",
                    f"\nDetails: {exc}\n",
                    title="HomeSky backfill error",
                )
            else:
                window["log"].update(describe_result("Backfill (24h)", result), append=True)
        elif event == "backfill_custom":
            if not reload_environment():
                continue
            mac = config.get("ambient", {}).get("mac")
            if not mac:
                sg.popup_error(
                    "Configure an Ambient Weather station MAC before running backfill.",
                    title="HomeSky backfill error",
                )
                continue
            prompt = prompt_custom_backfill(
                default_limit=limit_per_call,
                default_window=window_default,
            )
            if not prompt:
                continue
            start_dt, end_dt, window_minutes, limit = prompt
            try:
                result = backfill_range(
                    config=config,
                    storage=storage_manager,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    mac=mac,
                    window_minutes=window_minutes,
                    limit_per_call=limit,
                )
            except Exception as exc:
                sg.popup_error(
                    "Backfill failed.",
                    f"\nDetails: {exc}\n",
                    title="HomeSky backfill error",
                )
            else:
                window["log"].update(describe_result("Backfill (custom)", result), append=True)
        elif event == "import":
            file_selection = sg.popup_get_file(
                "Select export files",
                multiple_files=True,
                file_types=(
                    ("Weather exports", "*.csv;*.xlsx;*.xls"),
                    ("CSV", "*.csv"),
                    ("Excel", "*.xlsx;*.xls"),
                ),
            )
            if not file_selection:
                continue
            paths = [Path(path) for path in str(file_selection).split(";") if path]
            if not paths:
                continue
            if not reload_environment():
                continue
            try:
                report = import_files(paths, config=config, storage=storage_manager)
            except Exception as exc:
                sg.popup_error(
                    "Offline import failed.",
                    f"\nDetails: {exc}\n",
                    title="HomeSky import error",
                )
            else:
                summary = (
                    f"Imported {report['total_inserted']} rows from {len(paths)} file(s)."
                )
                if report.get("time_start") and report.get("time_end"):
                    summary += (
                        f" Range: {report['time_start']} – {report['time_end']}."
                    )
                summary += f" Report: {report['report_path']}"
                window["log"].update(summary + "\n", append=True)
                data_dir.mkdir(parents=True, exist_ok=True)
                try:
                    open_path(data_dir)
                except Exception:
                    pass
        elif event == "dashboard":
            if dashboard_process and dashboard_process.poll() is None:
                window["log"].update("Dashboard already running\n", append=True)
            else:
                if not reload_environment():
                    continue
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
            log_path.parent.mkdir(parents=True, exist_ok=True)
            open_path(log_path if log_path.exists() else log_path.parent)
            window["log"].update(f"Opened log path {log_path}\n", append=True)

    if dashboard_process and dashboard_process.poll() is None:
        dashboard_process.terminate()

    window.close()


if __name__ == "__main__":
    main()
