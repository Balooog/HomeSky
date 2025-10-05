# build_exe.ps1 — package EXE using existing venv; installs deps only when changed
.\tools\ensure_venv_and_deps.ps1
pip install -r dev-requirements.txt
python -m PyInstaller --onefile --clean --name HomeSky --distpath dist homesky\gui.py
Write-Host "# Test Build Complete" -ForegroundColor Green
