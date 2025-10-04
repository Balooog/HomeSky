"""Diagnostic helper for HomeSky."""

from __future__ import annotations

import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

import ingest
from utils.db import DatabaseManager

console = Console()


def main() -> None:
    try:
        config = ingest.load_config()
    except FileNotFoundError:
        console.print("[bold red]config.toml missing[/bold red]")
        return

    console.print("[bold cyan]HomeSky Configuration[/bold cyan]")
    console.print_json(json.dumps(config))

    storage = config.get("storage", {})
    db = DatabaseManager(Path(storage.get("sqlite_path", "./data/homesky.sqlite")), Path(storage.get("parquet_path", "./data/homesky.parquet")))
    last = db.fetch_last_timestamp(config.get("ambient", {}).get("mac"))

    table = Table(title="Storage", box=box.ROUNDED)
    table.add_column("Resource")
    table.add_column("Path")
    table.add_row("SQLite", str(db.sqlite_path))
    table.add_row("Parquet", str(db.parquet_path))
    table.add_row("Last Observation UTC", str(last or "None"))
    console.print(table)


if __name__ == "__main__":
    main()
