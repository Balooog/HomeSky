# HomeSky
single-user Windows desktop app for Ambient Weather station with long-term capture, Dark-Sky-style visuals, and rich exports (web-ready later)

### Fix-Now (if paths drift)

> These commands reset you to a clean repo root, avoid double `HomeSky\HomeSky` nesting, and recreate helper scripts if they were deleted locally.

```pwsh
# 0) Always start from the Desktop using PowerShell syntax
Set-Location -Path "$env:USERPROFILE\Desktop"

# 1) If you accidentally nested into HomeSky\HomeSky, move back up once
if (Test-Path ".\HomeSky\HomeSky\.git") { Set-Location ".\HomeSky" }

# 2) Fresh clone or refresh without double-cd
if (Test-Path ".\HomeSky\.git") {
  Set-Location ".\HomeSky"
  git fetch origin
  git reset --hard origin/main
  git clean -xfd
} else {
  git clone https://github.com/Balooog/HomeSky.git
  Set-Location ".\HomeSky"
}

# 3) Recreate helper scripts if they are missing
if (-not (Test-Path ".\tools")) { New-Item -ItemType Directory -Path ".\tools" | Out-Null }

@'
param(
  [string]$ReqPath = ".\requirements.txt",
  [string]$ExtraIndex = "https://PySimpleGUI.net/install"
)

if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) {
  Write-Host "[venv] creating..." -ForegroundColor Cyan
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip | Out-Null

if (-not (Test-Path $ReqPath)) {
  Write-Error "requirements.txt not found at $ReqPath"
  exit 1
}

$hashFile = ".\.venv\.req.hash"
$current = (Get-FileHash $ReqPath -Algorithm SHA256).Hash
$prior = (Test-Path $hashFile) ? (Get-Content $hashFile -Raw) : ""

if ($current -ne $prior) {
  Write-Host "[deps] installing/refreshing..." -ForegroundColor Yellow
  pip install -r $ReqPath --extra-index-url $ExtraIndex
  Set-Content -Path $hashFile -Value $current -Encoding UTF8
} else {
  Write-Host "[deps] requirements unchanged -> skipping installs." -ForegroundColor Green
}
"@ | Out-File -Encoding utf8 ".\tools\ensure_venv_and_deps.ps1"

@'
param(
  [string]$Config = ".\homesky\config.toml",
  [string]$Example = ".\homesky\config.example.toml"
)

if (-not (Test-Path $Config)) {
  if (Test-Path $Example) {
    Write-Host "[config] creating from example..." -ForegroundColor Cyan
    Copy-Item $Example $Config
  } else {
    Write-Host "[config] creating minimal stub..." -ForegroundColor Yellow
@"
[ambient]
api_key = ""
application_key = ""
mac = ""

[storage]
root_dir = "./data"
sqlite_path = "./data/homesky.sqlite"
parquet_path = "./data/homesky.parquet"
"@ | Out-File -Encoding utf8 $Config
  }
  Write-Host "[config] Edit $Config to add credentials." -ForegroundColor Yellow
} else {
  Write-Host "[config] present." -ForegroundColor Green
}
"@ | Out-File -Encoding utf8 ".\tools\ensure_config.ps1"
```

### Daily dev run (fast feedback)

```pwsh
Set-Location -Path "$env:USERPROFILE\Desktop"

if (Test-Path ".\HomeSky\.git") {
  Set-Location ".\HomeSky"
  git fetch origin
  git reset --hard origin/main
  git clean -xfd
} else {
  git clone https://github.com/Balooog/HomeSky.git
  Set-Location ".\HomeSky"
}

.\tools\ensure_venv_and_deps.ps1
.\tools\ensure_config.ps1
python .\homesky\gui.py
```

### Configure & relaunch (first run)

> The launcher writes `homesky\config.toml` on demand.  Add your Ambient Weather keys before continuing; the file is gitignored so secrets stay local.

```pwsh
notepad .\homesky\config.toml
python .\homesky\gui.py
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
