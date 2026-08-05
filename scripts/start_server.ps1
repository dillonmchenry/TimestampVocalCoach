# start_server.ps1 - Launch the SecondPass FastAPI server.
# Usage: .\scripts\start_server.ps1 [OPTIONS]
#
# Optional parameters:
#   -Port     Port to bind (default: 8000)
#   -Device   "cuda" or "cpu" (default: cuda)
#   -Reload   Pass --reload flag for development hot-reload
#
# The server binds to 127.0.0.1 only. Expose externally via Cloudflare Tunnel
# (start_tunnel.ps1) - do not bind 0.0.0.0 on a shared/public network.

param(
    [int]$Port = 8000,
    [string]$Device = "cuda",
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

# Change to the repo root regardless of where this script is called from.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Load .env so OPENAI_API_KEY is available (python-dotenv handles this at
# runtime too, but setting it here makes it visible to the uvicorn process).
$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.+)$") {
            $key = $Matches[1].Trim()
            $val = $Matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
    Write-Host "[start_server] Loaded .env" -ForegroundColor DarkGray
}

$reloadFlag = if ($Reload) { "--reload" } else { "" }

Write-Host ""
Write-Host "  SecondPass - starting server" -ForegroundColor Cyan
Write-Host "  http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host ""

$uvicornArgs = @(
    "-m", "uvicorn",
    "web.api.main:app",
    "--host", "127.0.0.1",
    "--port", "$Port"
)
if ($Reload) {
    $uvicornArgs += "--reload"
}

Write-Host "[start_server] python $($uvicornArgs -join ' ')" -ForegroundColor DarkGray
& python @uvicornArgs
