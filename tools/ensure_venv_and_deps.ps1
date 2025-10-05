param(
  [string]$ReqPath = ".\requirements.txt",
  [string]$ExtraIndex = "https://PySimpleGUI.net/install"
)

# Ensure venv exists
if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) {
  Write-Host "[venv] creating..." -ForegroundColor Cyan
  python -m venv .venv
}

# Activate venv
. .\.venv\Scripts\Activate.ps1

# Upgrade pip quietly
python -m pip install --upgrade pip | Out-Null

# Compute current requirements hash
if (-not (Test-Path $ReqPath)) {
  Write-Error "requirements.txt not found at $ReqPath"
  exit 1
}
$hashFile = ".\.venv\.req.hash"
$current = (Get-FileHash $ReqPath -Algorithm SHA256).Hash
$prior = ""
if (Test-Path $hashFile) { $prior = Get-Content $hashFile -Raw }

if ($current -ne $prior) {
  Write-Host "[deps] installing/refreshing..." -ForegroundColor Yellow
  pip install -r $ReqPath --extra-index-url $ExtraIndex
  Set-Content -Path $hashFile -Value $current -Encoding UTF8
} else {
  Write-Host "[deps] requirements unchanged -> skipping installs." -ForegroundColor Green
  Write-Host "[deps] If you hit missing module errors, run: pip install -r requirements.txt --extra-index-url $ExtraIndex" -ForegroundColor DarkYellow
}

# Return the venv activated in the caller shell
