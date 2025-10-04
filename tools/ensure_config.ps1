param(
  [string]$Config = ".\homesky\config.toml",
  [string]$Example = ".\homesky\config.example.toml"
)

if (-not (Test-Path $Config)) {
  if (Test-Path $Example) {
    Write-Host "[config] Creating config.toml from example..." -ForegroundColor Cyan
    Copy-Item $Example $Config
  } else {
    Write-Host "[config] No example found; creating a minimal stub..." -ForegroundColor Yellow
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
  Write-Host "[config] config.toml present." -ForegroundColor Green
}
