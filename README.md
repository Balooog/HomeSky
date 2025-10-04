# HomeSky
single-user Windows desktop app for Ambient Weather station with long-term capture, Dark-Sky-style visuals, and rich exports (web-ready later)

### Build-test (clean pull on Windows)

> You can re-test each merged build locally without editing the repo.  This section clones fresh, resets to the latest `main`, installs deps, runs a quick smoke test, and builds an EXE.  Two spaces between sentences.

**Prereqs:** Git for Windows, Python 3.11+ (recommended), PowerShell.

```pwsh
# 0) Start in a clean spot
cd $env:USERPROFILE\Desktop

# 1) Fresh clone or refresh
if (Test-Path .\HomeSky) {
  cd .\HomeSky
  git fetch origin
  git reset --hard origin/main
  git clean -xfd
} else {
  git clone https://github.com/Balooog/HomeSky.git
  cd .\HomeSky
}

# 2) Clean venv and deps
if (Test-Path .\.venv) { Remove-Item -Recurse -Force .\.venv }
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Quick smoke (adjust files as they land in repo)
# Pull one cycle then open the UI (if present)
# These are no-ops if files aren't present yet.
if (Test-Path .\homesky\ingest.py) { python .\homesky\ingest.py --once }
if (Test-Path .\homesky\visualize_streamlit.py) { Start-Process -FilePath "streamlit" -ArgumentList "run homesky/visualize_streamlit.py" }

# 4) Build EXE (console ON for debugging)
# Replace homesky\gui.py with your actual entry-point if different
python -m PyInstaller --onefile --clean --name HomeSky --distpath dist homesky\gui.py

Write-Host "Build artifact:"
Get-ChildItem .\dist
```

**Notes**
- Keep console builds while iterating.  Switch to `--noconsole` once stable.
- If the repo adds data files (icons, themes), include them with `--add-data` flags in the build command.
- If you encounter errors, paste the log into our Codex update template in `/codex/updates/`.
