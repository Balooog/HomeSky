# HomeSky
single-user Windows desktop app for Ambient Weather station with long-term capture, Dark-Sky-style visuals, and rich exports (web-ready later)

### Daily dev run (fast feedback)

```pwsh
Set-Location "$env:USERPROFILE\Desktop"

if (-not (Test-Path ".\HomeSky")) {
  git clone https://github.com/Balooog/HomeSky.git
}

Set-Location ".\HomeSky"

git fetch origin
git reset --hard origin/main
# (Keep .venv for faster runs; avoid: git clean -xfd)

.\tools\ensure_venv_and_deps.ps1
.\tools\ensure_config.ps1

python .\homesky\gui.py
# Test Build Complete
```

> Tip: `git clean -xfd` deletes `.venv/` and `homesky/config.toml`. Skip it unless you intentionally want a factory reset.

### Configure & relaunch (first run)

> The launcher writes `homesky\config.toml` on demand.  Add your Ambient Weather keys before continuing; the file is gitignored so secrets stay local.

```pwsh
notepad .\homesky\config.toml
python .\homesky\gui.py
```

### FAQ — "Fetched 1 new rows"

- **What it means** – HomeSky only appends observations whose `dateutc` timestamp is newer than the last stored
  record. If Ambient publishes a single fresh sample you will see `Fetched 1 new rows` in the GUI log. A run with no
  newer data prints `Fetched 0 new rows` and keeps the database unchanged.
- **Where the row lives** – Open the data folder (`.\data` from the repo, or the path configured under `[storage]`).
  `homesky.sqlite` is the canonical append-only table, while `homesky.parquet` (when enabled) mirrors the same rows in a
  columnar format for external tooling.
- **See it on the dashboard** – Re-enable Streamlit autolaunch by adding the block below to your roaming config
  (usually `%APPDATA%\HomeSky\config.toml`). Restart the GUI and click **Open Dashboard** to load the charts.

  ```toml
  [ui]
  launch_streamlit = true
  dashboard_port = 8501
  ```
- **Fetching more history** – Click **Backfill (24h)** in the GUI to pull a full day of history on demand. Use
  **Backfill (custom)** to enter wider ranges; progress is checkpointed to `data/state/backfill.json` so interrupted jobs
  resume safely. The ingest service also backfills automatically on the first run using the `[ingest].backfill_hours`
  setting in `config.toml`, so there is no need to delete the existing database to "force" older data.
- **API limits respected** – Ambient Weather limits each user key to ~1 request/sec (and each application key to 3/sec).
  HomeSky enforces these ceilings with a shared throttle, jitter, and exponential backoff (1s → 2s → 4s → 8s) so long
  backfills finish without `429` storms.
- **Offline imports** – Fill historical gaps without touching the API. Drop CSV/XLSX exports in the GUI via
  **Import file(s)** (or run `python -m homesky.import_offline data/import/*.csv`) and HomeSky will normalize the columns,
  dedupe by `(mac, epoch_ms)`, append to SQLite/Parquet, and write a JSON report to `data/logs/import_*.json`.

### Known-good config template

Paste this into `homesky\config.toml` (UTF-8 without BOM) and fill in your credentials:

```toml
[ambient]
api_key = "<YOUR_API_KEY>"
application_key = "<YOUR_APP_KEY>"
mac = "C4:5B:BE:6E:93:3D"

[storage]
root_dir = "./data"
sqlite_path = "./data/homesky.sqlite"
parquet_path = "./data/homesky.parquet"
```

### Build EXE (when packaging)

```pwsh
Set-Location -Path "$env:USERPROFILE\Desktop"

if (Test-Path ".\HomeSky\.git") {
  Set-Location ".\HomeSky"
} else {
  git clone https://github.com/Balooog/HomeSky.git
  Set-Location ".\HomeSky"
}

.\tools\ensure_venv_and_deps.ps1
pip install -r dev-requirements.txt
python -m PyInstaller --onefile --clean --name HomeSky --distpath dist homesky\gui.py

Write-Host "# Test Build Complete" -ForegroundColor Green
```

# Test Build Complete
