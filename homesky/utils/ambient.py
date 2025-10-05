"""Ambient Weather Network client helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from loguru import logger

BASE_URL = "https://api.ambientweather.net/v1"


class AmbientAPIError(RuntimeError):
    """Raised when the Ambient Weather Network API returns an error."""


@dataclass(slots=True)
class AmbientClient:
    """Thin wrapper around the Ambient Weather Network REST API."""

    api_key: str
    application_key: str
    mac: Optional[str] = None
    session: Optional[requests.Session] = None
    retries: int = 3
    backoff: float = 5.0

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.api_key or not self.application_key:
            raise AmbientAPIError("API key and application key are required")

        payload = {
            "apiKey": self.api_key,
            "applicationKey": self.application_key,
        }
        if params:
            payload.update(params)
        if self.mac:
            payload.setdefault("macAddress", self.mac)

        session = self.session or requests.Session()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = session.get(f"{BASE_URL}/{path}", params=payload, timeout=30)
                if response.status_code >= 400:
                    raise AmbientAPIError(
                        f"Ambient API error {response.status_code}: {response.text}"
                    )
                return response.json()
            except (requests.RequestException, AmbientAPIError) as exc:  # pragma: no cover
                if attempt >= self.retries:
                    logger.error("Ambient API request failed after {} attempts", attempt)
                    raise
                sleep_for = self.backoff * attempt
                logger.warning(
                    "Ambient API request failed (attempt {}/{}): {}; retrying in {:.1f}s",
                    attempt,
                    self.retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)

    def get_devices(self) -> List[Dict[str, Any]]:
        """Return the list of devices bound to the API keys."""

        devices = self._request("devices")
        if not isinstance(devices, list):
            raise AmbientAPIError("Unexpected devices payload")
        if self.mac:
            devices = [device for device in devices if device.get("macAddress") == self.mac]
        logger.debug("Fetched {} devices from Ambient Weather", len(devices))
        return devices

    def get_device_data(self, limit: int = 288) -> List[Dict[str, Any]]:
        """Fetch recent observations for the configured device(s)."""

        params: Dict[str, Any] = {"limit": limit}
        data = self._request("devices", params=params)
        if not isinstance(data, list):
            raise AmbientAPIError("Unexpected observations payload")
        logger.debug("Fetched {} observation rows", len(data))
        return data


def get_devices(api_key: str, app_key: str, mac: str | None = None) -> List[Dict[str, Any]]:
    """Convenience function to retrieve device metadata."""

    client = AmbientClient(api_key=api_key, application_key=app_key, mac=mac)
    return client.get_devices()


def fetch_latest_observations(
    api_key: str,
    app_key: str,
    mac: str | None = None,
    limit: int = 288,
) -> List[Dict[str, Any]]:
    """Fetch latest observations for downstream ingestion."""

    client = AmbientClient(api_key=api_key, application_key=app_key, mac=mac)
    return client.get_device_data(limit=limit)


def _parse_dateutc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            numeric = None
        if numeric is not None:
            try:
                return datetime.fromtimestamp(numeric / 1000.0, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                pass
        iso_candidate = text
        if iso_candidate.endswith("Z"):
            iso_candidate = iso_candidate[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(iso_candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def fetch_history(
    api_key: str,
    app_key: str,
    mac: str | None,
    *,
    hours: int = 24,
    tz: str = "America/New_York",
    page_size: int = 288,
) -> List[Dict[str, Any]]:
    """Fetch historical observations spanning ``hours`` back from now."""

    if hours <= 0:
        return []
    if not mac:
        raise ValueError("A device MAC address is required to fetch history")

    end_cursor = datetime.now(timezone.utc)
    target_start = end_cursor - timedelta(hours=hours)
    seen: Set[Tuple[str, str]] = set()
    results: List[Dict[str, Any]] = []

    with requests.Session() as session:
        while True:
            remaining_hours = max(
                1.0, (end_cursor - target_start).total_seconds() / 3600.0
            )
            approx_samples = max(1, int(remaining_hours * 12))
            limit = max(1, min(page_size, approx_samples))
            params = {
                "apiKey": api_key,
                "applicationKey": app_key,
                "limit": limit,
                "endDate": end_cursor.strftime("%Y-%m-%d %H:%M:%S"),
                "tz": tz,
            }
            response = session.get(f"{BASE_URL}/devices/{mac}", params=params, timeout=30)
            if response.status_code >= 400:
                raise AmbientAPIError(
                    f"Ambient API error {response.status_code}: {response.text}"
                )
            payload = response.json()
            if not isinstance(payload, list):
                raise AmbientAPIError("Unexpected history payload")
            if not payload:
                break

            oldest_dt: Optional[datetime] = None
            unique_batch: List[Dict[str, Any]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                dt = _parse_dateutc(item.get("dateutc"))
                if dt and (oldest_dt is None or dt < oldest_dt):
                    oldest_dt = dt
                key = (str(item.get("dateutc")), str(item.get("macAddress") or mac))
                if key in seen:
                    continue
                seen.add(key)
                unique_batch.append(item)

            results.extend(unique_batch)

            if not oldest_dt:
                break
            if oldest_dt <= target_start:
                break
            if oldest_dt >= end_cursor:
                break

            end_cursor = oldest_dt - timedelta(seconds=1)

    return results
