# start_tunnel.ps1 - Start a Cloudflare Tunnel pointing at the local server.
#
# Prerequisites:
#   1. cloudflared must be installed:
#      winget install cloudflare.cloudflared
#   2. The SecondPass server must already be running on $Port (start_server.ps1).
#
# Usage:
#   .\scripts\start_tunnel.ps1                  # quick tunnel, random *.trycloudflare.com URL
#   .\scripts\start_tunnel.ps1 -Named myapp     # named tunnel (requires: cloudflared tunnel login)
#   .\scripts\start_tunnel.ps1 -Port 8001       # if server is on a different port
#
# Named tunnel notes:
#   - Run once: cloudflared tunnel login
#   - Run once: cloudflared tunnel create <name>
#   - Run once: cloudflared tunnel route dns <name> <hostname>
#   - Then this script uses: cloudflared tunnel run <name>
#   Named tunnels give you a stable URL that persists across restarts.

param(
    [int]$Port = 8000,
    [string]$Named = ""
)

$ErrorActionPreference = "Stop"

# Resolve cloudflared.exe. Prefer PATH, then common winget install location.
# Existing terminals often miss PATH updates from winget until they are restarted.
$Cloudflared = $null
$cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
if ($cmd) {
    $Cloudflared = $cmd.Source
} else {
    $fallback = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
    if (Test-Path $fallback) {
        $Cloudflared = $fallback
        Write-Host "[start_tunnel] Using $Cloudflared (PATH not refreshed yet)" -ForegroundColor DarkGray
    }
}

if (-not $Cloudflared) {
    Write-Error @"
cloudflared not found. Install it with:
    winget install cloudflare.cloudflared

Then close and reopen this terminal (or refresh PATH) and re-run this script.
"@
    exit 1
}

# Check the server is actually up before opening the tunnel.
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
    Write-Host "[start_tunnel] Server is up - CUDA: $($health.cuda), songs: $($health.songs_available)" -ForegroundColor Green
} catch {
    Write-Warning "[start_tunnel] Server health check failed at http://127.0.0.1:$Port/api/health"
    Write-Warning "Make sure start_server.ps1 is running before starting the tunnel."
    Write-Warning "Continuing anyway..."
}

Write-Host ""

if ($Named -ne "") {
    Write-Host "  Starting NAMED tunnel: $Named" -ForegroundColor Cyan
    Write-Host "  Your URL is configured in Cloudflare DNS." -ForegroundColor Cyan
    Write-Host ""
    & $Cloudflared tunnel run $Named
} else {
    Write-Host "  Starting QUICK tunnel -> http://127.0.0.1:$Port" -ForegroundColor Cyan
    Write-Host "  A random https://*.trycloudflare.com URL will appear below." -ForegroundColor Cyan
    Write-Host "  Share that URL with testers. It changes every time you restart." -ForegroundColor Yellow
    Write-Host ""
    & $Cloudflared tunnel --url "http://127.0.0.1:$Port"
}
