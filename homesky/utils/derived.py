"""Derived metric calculations."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

DerivedFunc = Callable[[pd.DataFrame, Dict], pd.Series]

_DERIVED_REGISTRY: Dict[str, Tuple[DerivedFunc, Optional[str]]] = {}


def register_derived_metric(name: str, func: DerivedFunc, flag: Optional[str] = None) -> None:
    _DERIVED_REGISTRY[name] = (func, flag)


def compute_all_derived(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if df.empty:
        return df
    results = df.copy()
    derived_cfg = config.get("derived", {})
    for name, (func, flag) in _DERIVED_REGISTRY.items():
        if flag and not derived_cfg.get(flag, True):
            continue
        try:
            results[name] = func(df, config)
        except Exception as exc:  # pragma: no cover
            results[name] = np.nan
            print(f"Failed to compute derived metric {name}: {exc}")
    return results


# -- Derived metric implementations ------------------------------------


def _get_series(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(dtype="float64")


def _hdd(df: pd.DataFrame, config: Dict) -> pd.Series:
    base = config.get("derived", {}).get("hdd_base_f", 65)
    temps_f = _get_series(df, "tempf", "temp_f", "temperature")
    if temps_f.empty and "tempc" in df:
        temps_f = _get_series(df, "tempc") * 9 / 5 + 32
    return np.clip(base - temps_f, a_min=0, a_max=None)


def _cdd(df: pd.DataFrame, config: Dict) -> pd.Series:
    base = config.get("derived", {}).get("cdd_base_f", 65)
    temps_f = _get_series(df, "tempf", "temp_f", "temperature")
    if temps_f.empty and "tempc" in df:
        temps_f = _get_series(df, "tempc") * 9 / 5 + 32
    return np.clip(temps_f - base, a_min=0, a_max=None)


def _wetbulb(df: pd.DataFrame, config: Dict) -> pd.Series:
    temp_f = _get_series(df, "tempf", "temp_f")
    if temp_f.empty and "tempc" in df:
        temp_f = _get_series(df, "tempc") * 9 / 5 + 32
    humidity = _get_series(df, "humidity", "humidity_percent")
    temp_c = (temp_f - 32) * 5 / 9
    humidity = humidity.clip(lower=0, upper=100)
    wetbulb_c = (
        (0.151977 * (humidity + 8.313659) ** 0.5)
        + np.arctan(temp_c + humidity)
        - np.arctan(humidity - 1.676331)
        + 0.00391838 * humidity ** 1.5 * np.arctan(0.023101 * humidity)
        - 4.686035
    )
    wetbulb_f = wetbulb_c * 9 / 5 + 32
    return wetbulb_f


def _vpd(df: pd.DataFrame, config: Dict) -> pd.Series:
    temp_f = _get_series(df, "tempf", "temp_f")
    if temp_f.empty and "tempc" in df:
        temp_f = _get_series(df, "tempc") * 9 / 5 + 32
    humidity = _get_series(df, "humidity", "humidity_percent")
    temp_c = (temp_f - 32) * 5 / 9
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = humidity / 100 * es
    vpd_kpa = es - ea
    return vpd_kpa


def _wind_run(df: pd.DataFrame, config: Dict) -> pd.Series:
    wind_mph = _get_series(df, "windspeedmph", "wind_speed_mph")
    gust_mph = _get_series(df, "windgustmph", "wind_gust_mph")
    avg = pd.concat([wind_mph, gust_mph], axis=1).mean(axis=1, skipna=True)
    return avg / 60  # assuming 1-minute cadence → miles per minute


register_derived_metric("heating_degree_hours", _hdd)
register_derived_metric("cooling_degree_hours", _cdd)
register_derived_metric("wetbulb_temp_f", _wetbulb, flag="enable_wetbulb")
register_derived_metric(
    "vapor_pressure_deficit_kpa", _vpd, flag="enable_vpd"
)
register_derived_metric("wind_run_miles", _wind_run, flag="enable_wind_run")


__all__ = ["compute_all_derived", "register_derived_metric"]
