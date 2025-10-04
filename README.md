# HomeSky
single-user Windows desktop app for Ambient Weather station with long-term capture, Dark-Sky-style visuals, and rich exports (web-ready later)

### Run locally (recommended while iterating)

> Recommended flow while developing.  Start clean, install dependencies, then launch the GUI or dashboard directly without building an EXE.

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

# 2) Virtual environment and dependencies
if (Test-Path .\.venv) { Remove-Item -Recurse -Force .\.venv }
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the app (pick the entry point you need)
python .\homesky\gui.py
# or
# streamlit run homesky/visualize_streamlit.py
```

Keep the console window open while running so you can capture tracebacks or `[HomeSky]` log lines for debugging.

### Build EXE (only for releases or EXE-specific debugging)

> Build only after you are happy with local testing.  Installs PyInstaller from the development requirements and produces a console-enabled executable for easier troubleshooting.

```pwsh
# 0) Ensure your virtual environment is active
. .\.venv\Scripts\Activate.ps1

# 1) Install build tooling
pip install -r dev-requirements.txt

# 2) Build (console on for debugging)
python -m PyInstaller --onefile --clean --name HomeSky --distpath dist homesky\gui.py
.\dist\HomeSky.exe
```

**Tips**
- Keep console builds while iterating; switch to `--noconsole` once the GUI is stable.
- Add assets with `--add-data` flags when icons, themes, or config files need to ship inside the EXE.
- Copy any build or runtime errors into `/codex/updates/` so we can prep the next Codex patch quickly.
