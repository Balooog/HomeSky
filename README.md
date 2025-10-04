# HomeSky
single-user Windows desktop app for Ambient Weather station with long-term capture, Dark-Sky-style visuals, and rich exports (web-ready later)

### Run locally or build fast

> Use the helper scripts in `/tools` to reuse your virtual environment and skip dependency reinstalls unless `requirements.txt` changes.  They also take care of the PySimpleGUI private index and bootstrap a config file if it does not exist yet.

```pwsh
# Clean refresh or first clone
cd %USERPROFILE%\Desktop
if (Test-Path .\HomeSky) { cd .\HomeSky; git fetch origin; git reset --hard origin/main; git clean -xfd } else { git clone https://github.com/Balooog/HomeSky.git; cd .\HomeSky }

# Dev run (fast; reuses .venv; installs only if requirements changed)
.\tools\ensure_venv_and_deps.ps1
.\tools\ensure_config.ps1
python .\homesky\gui.py

# --- or, when you want an EXE ---
.\tools\ensure_venv_and_deps.ps1
pip install -r dev-requirements.txt
python -m PyInstaller --onefile --clean --name HomeSky --distpath dist homesky\gui.py

# Test Build Complete
```
