"""Long-running Ambient Weather ingest service."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from utils.ambient import AmbientClient
from utils.config import (
    candidate_config_paths,
    ensure_parent_directory,
    environment_config_path,
    external_config_path,
)
from utils.db import DatabaseManager
from utils.derived import compute_all_derived
from storage import StorageManager, StorageResult, canonicalize_records

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


CONFIG_TEMPLATE_PATH = Path(__file__).resolve().parent / "config.example.toml"
FALLBACK_TEMPLATE = """
[ambient]
api_key = ""
application_key = ""
mac = ""

[storage]
root_dir = "./data"
sqlite_path = "./data/homesky.sqlite"
parquet_path = "./data/homesky.parquet"
"""


def _split_comment(segment: str) -> Tuple[str, str]:
    in_string = False
    quote_char = ""
    for idx, char in enumerate(segment):
        if char in {'"', "'"}:
            if not in_string:
                in_string = True
                quote_char = char
            elif quote_char == char:
                in_string = False
        elif char == "#" and not in_string:
            prefix = segment[:idx]
            suffix = segment[idx:]
            trimmed = prefix.rstrip()
            whitespace = prefix[len(trimmed) :]
            return trimmed, f"{whitespace}{suffix}"
    return segment.rstrip(), ""


def _needs_quoting(value: str) -> bool:
    if not value:
        return False
    if value[0] in {'"', "'", "[", "{"}:
        return False
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return False
    try:
        tomllib.loads(f"value = {value}")
        return False
    except tomllib.TOMLDecodeError:
        return True


def _auto_quote_line(line: str) -> Tuple[str, bool]:
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#") or stripped.startswith("["):
        return line, False
    if "=" not in line:
        return line, False
    left, right = line.split("=", 1)
    value_segment, comment = _split_comment(right)
    value = value_segment.strip()
    if not _needs_quoting(value):
        return line, False
    quoted = f'"{value}"'
    new_line = f"{left.rstrip()} = {quoted}"
    if comment:
        new_line += comment
    return new_line, True


def _normalize_config_bytes(raw: bytes) -> Tuple[str, bool]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise
    leading_stripped = text.lstrip("\ufeff\r\n\t ")
    changed = leading_stripped != text or raw.startswith(b"\xef\xbb\xbf")
    text = leading_stripped
    lines = text.splitlines()
    quoted_lines: List[str] = []
    quoting_changed = False
    for line in lines:
        new_line, line_changed = _auto_quote_line(line)
        quoted_lines.append(new_line)
        if line_changed:
            quoting_changed = True
    normalized = "\n".join(quoted_lines)
    if text.endswith(("\r", "\n")):
        normalized += "\n"
    return normalized, changed or quoting_changed


def _template_text() -> str:
    if CONFIG_TEMPLATE_PATH.exists():
        return CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8")
    return FALLBACK_TEMPLATE.strip() + "\n"


def _backup_and_restore(candidate: Path) -> None:
    backup_path = candidate.with_name("config.bak")
    ensure_parent_directory(backup_path)
    try:
        if candidate.exists():
            backup_path.write_bytes(candidate.read_bytes())
            logger.warning("Existing config backed up to {}", backup_path)
    except OSError as exc:
        logger.warning("Unable to write backup config at {}: {}", backup_path, exc)
    try:
        ensure_parent_directory(candidate)
        candidate.write_text(_template_text(), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to write fresh config template at {}: {}", candidate, exc)
        raise


def get_config_path() -> Path:
    env_path = environment_config_path()
    if env_path:
        return env_path
    return external_config_path()


def ensure_config(path: Path | None = None) -> Path:
    target = Path(path or get_config_path())
    if target.exists():
        return target
    ensure_parent_directory(target)
    try:
        target.write_text(_template_text(), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to create configuration at {}: {}", target, exc)
        raise
    logger.info("Created starter configuration at {}", target)
    return target


def load_config(path: Path | None = None) -> Dict:
    load_config.last_path = None
    load_config.last_was_repaired = False
    load_config.last_was_normalized = False
    if path is None:
        try:
            ensure_config(get_config_path())
        except OSError:
            logger.warning("Unable to ensure external configuration exists")
    candidates = [path] if path else candidate_config_paths()
    preferred_external = external_config_path()
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            is_external = candidate.resolve() == preferred_external.resolve()
        except OSError:
            is_external = candidate == preferred_external
        attempts = 0
        repaired_candidate = False
        normalized_any = False
        while True:
            attempts += 1
            try:
                raw = candidate.read_bytes()
            except OSError as exc:
                logger.error("Unable to read config at {}: {}", candidate, exc)
                last_error = exc
                break
            try:
                normalized, changed = _normalize_config_bytes(raw)
            except UnicodeDecodeError as exc:
                logger.error("Config at {} is not valid UTF-8: {}", candidate, exc)
                last_error = exc
                if is_external and attempts == 1:
                    logger.warning(
                        "Attempting to restore {} from template due to invalid encoding.", candidate
                    )
                    _backup_and_restore(candidate)
                    repaired_candidate = True
                    continue
                break
            if changed:
                try:
                    candidate.write_text(normalized, encoding="utf-8")
                    logger.info("Normalized config formatting at {}", candidate)
                except OSError as exc:
                    logger.warning("Failed to write normalized config at {}: {}", candidate, exc)
                normalized_any = True
            try:
                config = tomllib.loads(normalized)
            except tomllib.TOMLDecodeError as exc:
                logger.error("Failed to parse config at {}: {}", candidate, exc)
                last_error = exc
                if is_external and attempts == 1:
                    logger.warning(
                        "Attempting to restore {} from template; previous version saved to config.bak.",
                        candidate,
                    )
                    try:
                        _backup_and_restore(candidate)
                        repaired_candidate = True
                    except OSError:
                        break
                    continue
                break
            print("[config] Loaded successfully")
            load_config.last_path = candidate
            load_config.last_was_repaired = repaired_candidate
            load_config.last_was_normalized = normalized_any
            return config
    if last_error:
        raise last_error
    raise FileNotFoundError(
        "config.toml not found. Run tools/ensure_config.ps1 or copy homesky/config.example.toml to homesky/config.toml and populate your credentials."
    )


load_config.last_path = None  # type: ignore[attr-defined]
load_config.last_was_repaired = False  # type: ignore[attr-defined]
load_config.last_was_normalized = False  # type: ignore[attr-defined]


def setup_logging(config: Dict) -> None:
    log_path = Path(config.get("ingest", {}).get("log_path", "./data/logs/ingest.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rotation_days = config.get("ingest", {}).get("log_rotate_days", 14)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        log_path,
        rotation=f"{rotation_days} days",
        retention=f"{rotation_days * 2} days",
        level="DEBUG",
        backtrace=True,
        enqueue=True,
    )
    logger.info("Logging initialized at {}", log_path)


def get_storage_manager(config: Dict) -> StorageManager:
    storage_cfg = config.get("storage", {})
    return StorageManager(
        sqlite_path=Path(storage_cfg.get("sqlite_path", "./data/homesky.sqlite")),
        parquet_path=Path(storage_cfg.get("parquet_path", "./data/homesky.parquet")),
        config=config,
    )


def get_database_manager(config: Dict) -> DatabaseManager:
    return get_storage_manager(config).database


def load_dataset(
    config: Dict | None = None,
    *,
    sqlite_path: Path | str | None = None,
    parquet_path: Path | str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load stored observations into a DataFrame."""

    cfg = config or {}
    storage = cfg.get("storage", {})
    tz_cfg = cfg.get("timezone", {})
    local_tz = str(tz_cfg.get("local_tz") or "UTC")
    sqlite_resolved = Path(sqlite_path or storage.get("sqlite_path", "./data/homesky.sqlite"))
    parquet_resolved = Path(parquet_path or storage.get("parquet_path", "./data/homesky.parquet"))
    db = DatabaseManager(sqlite_resolved, parquet_resolved)
    df = db.read_dataframe(limit=limit, local_tz=local_tz)
    if df.empty:
        return df
    if "s_time_local" in df.columns and "s_time_local" in df.index.names:
        df = df.reset_index(drop=True)
    if "s_time_local" not in df.columns:
        df = df.reset_index(drop=False)
        if "s_time_local" not in df.columns and "epoch_ms" in df.columns:
            timestamps = pd.to_datetime(df["epoch_ms"], unit="ms", errors="coerce", utc=True)
            try:
                zone = ZoneInfo(local_tz)
            except Exception:  # pragma: no cover - fallback to UTC
                zone = ZoneInfo("UTC")
            df["s_time_local"] = timestamps.dt.tz_convert(zone)
    df = df.drop_duplicates(subset=["s_time_local", "mac"], keep="last")
    df = df.sort_values("s_time_local").set_index("s_time_local")
    return df


def flatten_observations(records: List[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for device in records:
        last = device.get("lastData", {})
        if not last:
            continue
        row = {**last}
        row["mac"] = device.get("macAddress") or device.get("mac")
        if "dateutc" in row:
            dateutc = row["dateutc"]
            if isinstance(dateutc, (int, float)):
                row["obs_time_utc"] = pd.to_datetime(dateutc, unit="ms", errors="coerce", utc=True)
            else:
                row["obs_time_utc"] = pd.to_datetime(dateutc, errors="coerce", utc=True)
        if "date" in row:
            row["obs_time_local"] = pd.to_datetime(row["date"], errors="coerce")
        row["raw"] = last
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "obs_time_utc" in frame:
        frame = frame.sort_values("obs_time_utc")
        frame = frame.set_index("obs_time_utc")
    return frame


def filter_new_canonical(
    df: pd.DataFrame,
    storage: StorageManager,
    mac: str | None,
    enabled: bool,
) -> pd.DataFrame:
    if df.empty or not enabled:
        return df
    latest = storage.database.fetch_last_epoch_ms(mac)
    if latest is None:
        return df
    return df[df["epoch_ms"] > latest]


def detect_cadence_seconds(df: pd.DataFrame) -> int:
    if df.empty or "epoch" not in df:
        return 60
    epochs = pd.to_numeric(df["epoch"], errors="coerce").dropna()
    if epochs.empty:
        return 60
    diffs = epochs.diff().dropna()
    if diffs.empty:
        return 60
    cadence = int(diffs.median())
    return max(cadence, 60)


def drop_implausible(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if "tempf" in df:
        mask &= df["tempf"].between(-80, 150)
    if "windspeedmph" in df:
        mask &= df["windspeedmph"].between(0, 200)
    if "humidity" in df:
        mask &= df["humidity"].between(0, 100)
    return df.loc[mask]


def _history_records_to_frame(records: List[Dict], mac_fallback: Optional[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for entry in records:
        if not isinstance(entry, dict):
            continue
        raw = dict(entry)
        mac_value = raw.get("macAddress") or raw.get("mac") or mac_fallback or "unknown"
        dateutc = raw.get("dateutc")
        if isinstance(dateutc, (int, float)):
            obs_time_utc = pd.to_datetime(dateutc, unit="ms", errors="coerce", utc=True)
        else:
            obs_time_utc = pd.to_datetime(dateutc, errors="coerce", utc=True)
        if pd.isna(obs_time_utc):
            continue
        row = {**raw}
        row["mac"] = mac_value
        row["obs_time_utc"] = obs_time_utc
        if "date" in raw and raw["date"]:
            row["obs_time_local"] = pd.to_datetime(raw["date"], errors="coerce")
        elif "obs_time_local" in raw and raw["obs_time_local"]:
            row["obs_time_local"] = pd.to_datetime(raw["obs_time_local"], errors="coerce")
        row["raw"] = raw
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.dropna(subset=["obs_time_utc"])
    if frame.empty:
        return frame
    frame = frame.sort_values("obs_time_utc")
    frame = frame.set_index("obs_time_utc")
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def is_first_run(db: DatabaseManager) -> bool:
    try:
        snapshot = db.read_dataframe(limit=1)
    except FileNotFoundError:
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Unable to determine first-run status: {}", exc)
        return False
    return snapshot.empty


def backfill(
    config: Dict,
    hours: int,
    *,
    storage: Optional[StorageManager] = None,
    tz: Optional[str] = None,
) -> int:
    if hours <= 0:
        return 0
    storage_manager = storage or get_storage_manager(config)
    mac = config.get("ambient", {}).get("mac") or None
    if not mac:
        raise ValueError("A device MAC address is required for backfill operations")
    end = pd.Timestamp.utcnow().tz_localize("UTC")
    start = end - pd.Timedelta(hours=hours)
    try:
        from backfill import backfill_range
    except ImportError as exc:  # pragma: no cover - fallback for older setups
        raise RuntimeError("backfill module is unavailable") from exc
    result = backfill_range(
        config=config,
        storage=storage_manager,
        start_dt=start,
        end_dt=end,
        mac=mac,
        window_minutes=int(pd.Timedelta(days=1).total_seconds() // 60),
        limit_per_call=int(config.get("ingest", {}).get("backfill_limit", 288) or 288),
    )
    return result.inserted


def maybe_auto_backfill(config: Dict, storage: StorageManager) -> None:
    ingest_cfg = config.get("ingest", {})
    hours = int(ingest_cfg.get("backfill_hours", 0) or 0)
    if hours <= 0:
        return
    db = storage.database
    if not is_first_run(db):
        return
    try:
        from backfill import backfill_range
    except ImportError:
        logger.warning("Backfill module unavailable; skipping automatic backfill")
        return
    mac = config.get("ambient", {}).get("mac") or None
    if not mac:
        logger.warning("Cannot run automatic backfill without a configured MAC address")
        return
    end = pd.Timestamp.utcnow().tz_localize("UTC")
    start = end - pd.Timedelta(hours=hours)
    try:
        result = backfill_range(
            config=config,
            storage=storage,
            start_dt=start,
            end_dt=end,
            mac=mac,
            limit_per_call=ingest_cfg.get("backfill_limit", 288),
        )
    except Exception as exc:  # pragma: no cover - log and continue ingestion
        logger.warning("First-run backfill failed: {}", exc)
        return
    if result.inserted:
        logger.info(
            "First-run backfill added %s rows (%s to %s)",
            result.inserted,
            result.start,
            result.end,
        )
    else:
        logger.info("First-run backfill found no additional rows")


def ingest_once(config: Dict, storage: StorageManager) -> StorageResult:
    ambient_cfg = config.get("ambient", {})
    client = AmbientClient(
        api_key=ambient_cfg.get("api_key"),
        application_key=ambient_cfg.get("application_key"),
        mac=ambient_cfg.get("mac") or None,
    )
    mac = ambient_cfg.get("mac") or None
    limit = int(config.get("ingest", {}).get("fetch_limit", 288) or 288)
    records = client.get_device_data(mac=mac, limit=limit)
    frame = _history_records_to_frame(records, mac)
    if config.get("ingest", {}).get("drop_implausible_values", True):
        frame = drop_implausible(frame)
    canonical = canonicalize_records(frame.reset_index().to_dict(orient="records"), mac_hint=mac)
    canonical = filter_new_canonical(
        canonical,
        storage,
        mac,
        enabled=config.get("ingest", {}).get("deduplicate_by_timestamp", True),
    )
    if canonical.empty:
        logger.info("No new observations to ingest.")
        return StorageResult(0, None, None)
    result = storage.upsert_canonical(canonical)
    if result.inserted:
        logger.info(
            "Ingested %s new observation rows (%s to %s)",
            result.inserted,
            result.start,
            result.end,
        )
    else:
        logger.info("No new observations to ingest.")
    return result


def run_loop(config: Dict, storage: StorageManager) -> None:
    poll_seconds = 60
    while True:
        try:
            result = ingest_once(config, storage)
        except KeyboardInterrupt:
            logger.info("Ingest loop interrupted by user")
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("Ingest failed: {}", exc)
        else:
            if result.inserted:
                poll_seconds = detect_cadence_seconds(storage.database.read_dataframe(limit=10))
        logger.info("Sleeping for {} seconds", poll_seconds)
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HomeSky ingest service")
    parser.add_argument("--once", action="store_true", help="Run a single ingest cycle and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    setup_logging(config)
    logger.info("Starting HomeSky ingest service")
    storage = get_storage_manager(config)
    try:
        maybe_auto_backfill(config, storage)
        if args.once:
            ingest_once(config, storage)
        else:
            run_loop(config, storage)
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting.")


if __name__ == "__main__":
    main()
