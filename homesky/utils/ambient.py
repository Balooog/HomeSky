"""Ambient Weather Network client helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

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
