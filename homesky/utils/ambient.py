"""Ambient Weather Network API helpers with rate limiting and backoff."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

BASE_URL = "https://api.ambientweather.net/v1"

_RATE_LOCK = threading.Lock()
_LAST_REQUEST_TS = 0.0
_MIN_REQUEST_INTERVAL = 1.0
_JITTER_RANGE = (0.0, 0.2)


class AmbientAPIError(RuntimeError):
    """Raised when the Ambient Weather API returns an error payload."""


def _throttle() -> None:
    global _LAST_REQUEST_TS
    with _RATE_LOCK:
        now = time.monotonic()
        target = _LAST_REQUEST_TS + _MIN_REQUEST_INTERVAL
        wait_seconds = max(0.0, target - now)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        jitter = random.uniform(*_JITTER_RANGE)
        if jitter > 0:
            time.sleep(jitter)
        _LAST_REQUEST_TS = time.monotonic()


def _should_retry(response: requests.Response) -> bool:
    if response.status_code == 429:
        return True
    if 500 <= response.status_code < 600:
        return True
    return False


@dataclass(slots=True)
class AmbientClient:
    """Thin wrapper around the Ambient Weather Network API."""

    api_key: str
    application_key: str
    mac: Optional[str] = None
    session: Optional[requests.Session] = None
    max_retries: int = 4

    def _request(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self.api_key or not self.application_key:
            raise AmbientAPIError("Ambient API credentials are required")

        payload = {
            "apiKey": self.api_key,
            "applicationKey": self.application_key,
        }
        if params:
            payload.update(params)
        if self.mac:
            payload.setdefault("macAddress", self.mac)

        attempt = 0
        session = self.session or requests.Session()
        while True:
            attempt += 1
            _throttle()
            try:
                response = session.get(f"{BASE_URL}/{path}", params=payload, timeout=30)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    logger.error("Ambient API request failed after {} attempts", attempt)
                    raise AmbientAPIError(str(exc)) from exc
                delay = min(8.0, 2 ** (attempt - 1))
                sleep_for = delay + random.uniform(0, 0.5)
                logger.warning(
                    "Ambient API request errored (attempt {}/{}): {}; retrying in {:.1f}s",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue

            if response.ok:
                try:
                    return response.json()
                except ValueError as exc:  # pragma: no cover - unexpected payload
                    logger.error("Ambient API returned invalid JSON: {}", exc)
                    raise AmbientAPIError("Invalid JSON response from Ambient API") from exc

            if not _should_retry(response) or attempt >= self.max_retries:
                logger.error(
                    "Ambient API error %s after %s attempts: %s",
                    response.status_code,
                    attempt,
                    response.text,
                )
                raise AmbientAPIError(
                    f"Ambient API error {response.status_code}: {response.text}"
                )

            delay = min(8.0, 2 ** (attempt - 1))
            sleep_for = delay + random.uniform(0, 0.5)
            logger.warning(
                "Ambient API rate limited/errored (attempt {}/{}); retrying in {:.1f}s",
                attempt,
                self.max_retries,
                sleep_for,
            )
            time.sleep(sleep_for)

    def get_devices(self) -> List[Dict[str, Any]]:
        payload = self._request("devices")
        if not isinstance(payload, list):
            raise AmbientAPIError("Unexpected response when listing devices")
        if self.mac:
            return [row for row in payload if row.get("macAddress") == self.mac]
        return payload

    def get_device_data(
        self,
        *,
        mac: Optional[str] = None,
        end_dt: Optional[datetime] = None,
        limit: int = 288,
        tz: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        mac_to_use = mac or self.mac
        if not mac_to_use:
            payload = self._request("devices", params={"limit": limit})
            if not isinstance(payload, list):
                raise AmbientAPIError("Unexpected payload when fetching device data")
            return payload
        params: Dict[str, Any] = {"limit": limit}
        if end_dt:
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            params["endDate"] = end_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if tz:
            params["tz"] = tz
        payload = self._request(f"devices/{mac_to_use}", params=params)
        if not isinstance(payload, list):
            raise AmbientAPIError("Unexpected payload when fetching device data")
        return payload


def get_devices(
    api_key: str,
    application_key: str,
    *,
    mac: Optional[str] = None,
) -> List[Dict[str, Any]]:
    client = AmbientClient(api_key=api_key, application_key=application_key, mac=mac)
    return client.get_devices()


def get_device_data(
    api_key: str,
    application_key: str,
    *,
    mac: Optional[str],
    end_dt: Optional[datetime] = None,
    limit: int = 288,
    tz: Optional[str] = None,
) -> List[Dict[str, Any]]:
    client = AmbientClient(api_key=api_key, application_key=application_key, mac=mac)
    return client.get_device_data(mac=mac, end_dt=end_dt, limit=limit, tz=tz)


def fetch_latest_observations(
    api_key: str,
    application_key: str,
    *,
    mac: Optional[str] = None,
    limit: int = 288,
) -> List[Dict[str, Any]]:
    client = AmbientClient(api_key=api_key, application_key=application_key, mac=mac)
    return client.get_device_data(mac=mac, limit=limit)


__all__ = [
    "AmbientAPIError",
    "AmbientClient",
    "get_devices",
    "get_device_data",
    "fetch_latest_observations",
]

