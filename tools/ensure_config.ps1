param(
  [string]$Config,
  [string]$Example = ".\homesky\config.example.toml",
  [bool]$PreserveConfig = $true
)

function Get-PreferredConfigPath {
  if ($env:HOMESKY_CONFIG) {
    return $env:HOMESKY_CONFIG
  }
  if ($env:APPDATA) {
    return (Join-Path $env:APPDATA "HomeSky\config.toml")
  }
  if ($env:XDG_CONFIG_HOME) {
    return (Join-Path $env:XDG_CONFIG_HOME "HomeSky/config.toml")
  }
  return "$HOME/.config/HomeSky/config.toml"
}

$preferredPath = Get-PreferredConfigPath

if (-not $PSBoundParameters.ContainsKey('Config') -or [string]::IsNullOrWhiteSpace($Config)) {
  $Config = $preferredPath
} elseif ($PreserveConfig) {
  try {
    $resolvedConfig = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Config))
  } catch {
    $resolvedConfig = $Config
  }
  $repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
  if ($resolvedConfig.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "[config] preserve_config active; redirecting to persistent path $preferredPath" -ForegroundColor Yellow
    $Config = $preferredPath
  }
}

$destinationDir = Split-Path -Parent $Config
if (-not (Test-Path $destinationDir)) {
  New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
}

if (-not (Test-Path $Config)) {
  if (Test-Path $Example) {
    Write-Host "[config] Creating config.toml from example..." -ForegroundColor Cyan
    Copy-Item $Example $Config
  } else {
    Write-Host "[config] No example found; creating a minimal stub..." -ForegroundColor Yellow
    $stub = @"
[ambient]
api_key = ""
application_key = ""
mac = ""

[storage]
root_dir = "./data"
sqlite_path = "./data/homesky.sqlite"
parquet_path = "./data/homesky.parquet"
"@
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Config, $stub, $utf8)
  }
  Write-Host "[config] Edit $Config to add credentials." -ForegroundColor Yellow
} else {
  Write-Host "[config] config.toml present." -ForegroundColor Green
}
