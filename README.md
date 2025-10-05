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

If you run into missing Python packages later, rerun:

```pwsh
pip install -r requirements.txt --extra-index-url https://PySimpleGUI.net/install
```

### Configure
- Edit `homesky/config.toml` with Ambient Weather credentials.
- Optional: set `[ui] launch_streamlit = true` only when `homesky/visualize_streamlit.py` exists.
- Rerun: `python .\homesky\gui.py`

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
