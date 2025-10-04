"""Dark-Sky-inspired theming helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class Theme:
    background: str
    surface: str
    primary: str
    accent: str
    text: str
    muted_text: str


DARK_THEME = Theme(
    background="#0B101B",
    surface="#141B2A",
    primary="#4C6FFF",
    accent="#F7B267",
    text="#F5F7FF",
    muted_text="#8792B0",
)

LIGHT_THEME = Theme(
    background="#F6F7FB",
    surface="#FFFFFF",
    primary="#3753E0",
    accent="#FF9F1C",
    text="#1B1F30",
    muted_text="#536080",
)


def get_theme(choice: str | None) -> Theme:
    if choice == "light":
        return LIGHT_THEME
    if choice == "dark":
        return DARK_THEME
    return DARK_THEME


def load_typography() -> Dict[str, str]:
    return {
        "font_family": "Inter, 'Segoe UI', sans-serif",
        "title_size": "1.8rem",
        "label_size": "0.9rem",
    }


__all__ = ["Theme", "get_theme", "load_typography"]
