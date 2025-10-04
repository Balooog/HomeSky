# run_gui.ps1 — ensure venv+deps, ensure config, then launch GUI
.\tools\ensure_venv_and_deps.ps1
.\tools\ensure_config.ps1
python .\homesky\gui.py
