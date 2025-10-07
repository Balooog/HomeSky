"""PySimpleGUI helper launcher for HomeSky."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Allow both "python -m homesky.gui" and direct file execution.
import PySimpleGUI as sg

__HS_BOOTSTRAPPED__ = False
try:
    from . import ingest  # type: ignore[no-redef]
except Exception:  # pragma: no cover - fallback for direct runs
    import sys as _sys
    import pathlib as _pathlib

    _root = _pathlib.Path(__file__).resolve().parents[1]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    __HS_BOOTSTRAPPED__ = True
    from homesky import ingest  # type: ignore[no-redef]
from homesky.backfill import backfill_range
from homesky.import_offline import (
    TimestampDetectionError,
    TimestampOverride,
    import_files,
    save_timestamp_mapping,
)
from homesky.storage import StorageResult
from homesky.utils.logging_setup import get_logger

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

PACKAGE_DIR = Path(__file__).resolve().parent
STREAMLIT_ENTRY = PACKAGE_DIR / "visualize_streamlit.py"


def _warn_on_bare_imports() -> None:
    suspicious: list[str] = []
    for name, mod in list(sys.modules.items()):
        if isinstance(mod, types.ModuleType) and name in {
            "backfill",
            "ingest",
            "import_offline",
            "visualize_streamlit",
            "utils",
            "ambient",
            "db",
        }:
            suspicious.append(name)
    if suspicious:
        try:
            log = get_logger("streamlit")
            log.warning(
                "Detected bare intra-package imports: %s. Use 'from homesky.<module> import …'",
                suspicious,
            )
        except Exception:  # pragma: no cover - logging best effort
            pass


_warn_on_bare_imports()


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
    epoch_ms = int(dt.timestamp() * 1000)
    iso_utc = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()
    return iso_utc.replace("+00:00", "Z")


def _prompt_timestamp_override(
    path: Path,
    error: TimestampDetectionError,
    config: dict,
) -> tuple[TimestampOverride | None, bool]:
    columns = error.columns or list(error.rename_map.keys())
    if not columns:
        sg.popup_error(
            "HomeSky could not detect any columns in the selected file.",
            f"Check {error.sample_path} for a saved sample.",
            title="Timestamp detection failed",
        )
        return None, False
    preview = error.preview.copy()
    if preview.empty:
        preview_rows: list[list[str]] = []
        headings = columns
    else:
        preview = preview.fillna("").astype(str)
        headings = [str(col) for col in preview.columns]
        preview_rows = preview.values.tolist()
    default_tz = str(config.get("timezone", {}).get("local_tz") or "UTC")
    kind_options = [
        ("ISO 8601 (includes timezone)", "iso"),
        ("Epoch milliseconds", "epoch_ms"),
        ("Epoch seconds", "epoch"),
        (f"Locale datetime (convert from {default_tz})", "local"),
        ("Excel serial (days since 1899-12-30)", "excel"),
        ("Split date + time columns", "pair"),
    ]
    kind_labels = [label for label, _ in kind_options]
    kind_lookup = {label: value for label, value in kind_options}
    layout: list[list[sg.Element]] = [
        [sg.Text(f"Select the timestamp column for {path.name}")],
    ]
    if preview_rows:
        layout.append(
            [
                sg.Table(
                    values=preview_rows,
                    headings=headings,
                    auto_size_columns=True,
                    display_row_numbers=False,
                    justification="left",
                    num_rows=min(len(preview_rows), 10),
                    key="preview",
                )
            ]
        )
    else:
        layout.append([sg.Text("No preview rows available; columns listed below.")])
    layout.extend(
        [
            [
                sg.Text("Timestamp column"),
                sg.Combo(
                    columns,
                    default_value=columns[0],
                    readonly=True,
                    key="column",
                    size=(40, 1),
                ),
            ],
            [
                sg.Text("Interpretation"),
                sg.Combo(
                    kind_labels,
                    default_value=kind_labels[0],
                    readonly=True,
                    key="kind",
                    enable_events=True,
                    size=(45, 1),
                ),
            ],
            [
                sg.Text("Time column"),
                sg.Combo(
                    columns,
                    readonly=True,
                    key="time_column",
                    size=(40, 1),
                    disabled=True,
                ),
            ],
            [
                sg.Text("Timezone"),
                sg.Input(default_tz, key="timezone", disabled=True, size=(30, 1)),
            ],
            [
                sg.Checkbox(
                    "Remember this choice for files named like this",
                    default=True,
                    key="remember",
                )
            ],
            [
                sg.Button("Apply", key="apply"),
                sg.Button("Cancel"),
            ],
            [
                sg.Text(
                    f"Sample saved to {error.sample_path}",
                    text_color="gray",
                )
            ],
        ]
    )

    window = sg.Window(
        "Select timestamp column",
        layout,
        modal=True,
        keep_on_top=True,
        finalize=True,
    )
    try:
        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                return None, False
            if event == "kind":
                label = values.get("kind", kind_labels[0])
                kind_key = kind_lookup.get(label, "iso")
                enable_timezone = kind_key in {"local", "pair"}
                enable_time_column = kind_key == "pair"
                window["timezone"].update(disabled=not enable_timezone)
                if not enable_timezone:
                    window["timezone"].update(default_tz)
                window["time_column"].update(disabled=not enable_time_column)
                if not enable_time_column:
                    window["time_column"].update("")
            if event == "apply":
                column = values.get("column")
                label = values.get("kind", kind_labels[0])
                kind_key = kind_lookup.get(label, "iso")
                if not column:
                    sg.popup_error("Select a timestamp column before continuing.")
                    continue
                timezone_value = values.get("timezone") or default_tz
                time_column = values.get("time_column") or None
                if kind_key == "pair":
                    if not time_column or time_column == column:
                        sg.popup_error("Select a separate time column for the split date/time option.")
                        continue
                remember = bool(values.get("remember", False))
                override = TimestampOverride(
                    column=str(column),
                    kind=kind_key,
                    timezone=timezone_value if kind_key in {"local", "pair"} else None,
                    time_column=str(time_column) if kind_key == "pair" and time_column else None,
                )
                return override, remember
    finally:
        window.close()


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

    def resolve_timestamp(path: Path, error: TimestampDetectionError) -> Optional[TimestampOverride]:
        override, remember = _prompt_timestamp_override(path, error, config)
        if override and remember:
            save_timestamp_mapping(path.name, override)
        return override

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
            window["log"].update(
                "Backfill pacing ~1 req/s; duplicates skipped automatically.\n",
                append=True,
            )
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
            window["log"].update(
                "Backfill pacing ~1 req/s; duplicates skipped automatically.\n",
                append=True,
            )
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
                report = import_files(
                    paths,
                    config=config,
                    storage=storage_manager,
                    interactive=True,
                    resolver=resolve_timestamp,
                )
            except TimestampDetectionError as exc:
                sg.popup_error(
                    "Offline import canceled.",
                    f"\n{exc}\n\nSample: {exc.sample_path}",
                    title="HomeSky import error",
                )
            except Exception as exc:
                sg.popup_error(
                    "Offline import failed.",
                    f"\nDetails: {exc}\n",
                    title="HomeSky import error",
                )
            else:
                inserted = int(report.get("total_inserted") or 0)
                duplicates = int(report.get("total_duplicates") or 0)
                file_count = len(paths)
                if inserted == 0 and duplicates > 0:
                    summary = (
                        f"No new rows; {duplicates:,} duplicate rows already existed across {file_count} file(s)."
                    )
                elif inserted == 0:
                    summary = f"No rows imported from {file_count} file(s)."
                else:
                    summary = (
                        f"Imported {inserted:,} new row{'s' if inserted != 1 else ''} "
                        f"({duplicates:,} duplicates ignored) from {file_count} file(s)."
                    )
                if report.get("time_start") and report.get("time_end"):
                    summary += (
                        f" Range merged: {report['time_start']} – {report['time_end']}."
                    )
                summary += f" Report saved to {report['report_path']}"
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
