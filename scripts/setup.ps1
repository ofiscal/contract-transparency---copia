# One-time setup for Windows: installs uv if needed, then creates .venv with the
# tested versions from uv.lock.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1            # core + dashboard + dev
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 mlflow     # + MLflow
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found; installing it for the current user (https://docs.astral.sh/uv/)"
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

$extraArgs = @()
foreach ($e in $args) { $extraArgs += @("--extra", $e) }
uv sync @extraArgs
New-Item -ItemType Directory -Force -Path data, models, .secrets | Out-Null

Write-Host ""
Write-Host "Done. Run commands through uv:"
Write-Host "  uv run ct demo-data --with-records"
Write-Host "  uv run ct run -c configs/demo.yaml"
Write-Host "  uv run ct app -c configs/demo.yaml      # http://localhost:8501"
