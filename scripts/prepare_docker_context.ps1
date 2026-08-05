# prepare_docker_context.ps1 - Stage files needed by the Dockerfile that live
# outside the TimestampVocalCoach repo (NanoPitch sibling) or are gitignored.
#
# Run this ONCE before each `docker build`. It is safe to re-run.
#
# What it does:
#   1. Copies NanoPitch model.py + best.pth into ./nanopitch/ (gitignored)
#   2. Verifies rmvpe/model.pt exists (you must have it already)
#
# Usage:
#   .\scripts\prepare_docker_context.ps1
#   .\scripts\prepare_docker_context.ps1 -NanoPitchDir "D:\other\NanoPitch"

param(
    [string]$NanoPitchDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# ---------------------------------------------------------------------------
# Resolve NanoPitch directory
# ---------------------------------------------------------------------------
if ($NanoPitchDir -eq "") {
    # Try NANOPITCH_DIR env var first
    $envDir = [System.Environment]::GetEnvironmentVariable("NANOPITCH_DIR")
    if ($envDir -and (Test-Path $envDir)) {
        $NanoPitchDir = $envDir
    } else {
        # NanoPitch lives at GitHub\NanoPitch; TimestampVocalCoach is at
        # GitHub\STARS\TimestampVocalCoach -- so we need to go up two levels.
        $candidate = Join-Path $RepoRoot "..\..\NanoPitch"
        $resolved = Resolve-Path $candidate -ErrorAction SilentlyContinue
        if ($resolved) { $NanoPitchDir = $resolved.Path }
    }
}

if (-not $NanoPitchDir -or -not (Test-Path $NanoPitchDir)) {
    Write-Error @"
NanoPitch repo not found at '$NanoPitchDir'.
Pass the path explicitly:
    .\scripts\prepare_docker_context.ps1 -NanoPitchDir "C:\path\to\NanoPitch"
"@
    exit 1
}

Write-Host "[prepare] NanoPitch source: $NanoPitchDir" -ForegroundColor DarkGray

# ---------------------------------------------------------------------------
# 1. Copy NanoPitch runtime files into ./nanopitch/
# ---------------------------------------------------------------------------
$NpDest = Join-Path $RepoRoot "nanopitch"
$CheckpointName = "best_150+late_clean_112gru_model"
$CheckpointSrc  = Join-Path $NanoPitchDir "training\runs\$CheckpointName\checkpoints\best.pth"
$CheckpointDest = Join-Path $NpDest "training\runs\$CheckpointName\checkpoints\best.pth"
$ModelSrc       = Join-Path $NanoPitchDir "training\model.py"
$ModelDest      = Join-Path $NpDest "training\model.py"

# model.py
New-Item -ItemType Directory -Path (Split-Path $ModelDest) -Force | Out-Null
Copy-Item -Path $ModelSrc -Destination $ModelDest -Force
Write-Host "[prepare] Copied model.py   -> nanopitch\training\model.py" -ForegroundColor Green

# best.pth checkpoint
New-Item -ItemType Directory -Path (Split-Path $CheckpointDest) -Force | Out-Null
Copy-Item -Path $CheckpointSrc -Destination $CheckpointDest -Force
$sizeMB = [math]::Round((Get-Item $CheckpointDest).Length / 1MB, 2)
Write-Host "[prepare] Copied best.pth   -> nanopitch\...checkpoints\best.pth ($sizeMB MB)" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 2. Verify rmvpe/model.pt exists
# ---------------------------------------------------------------------------
$RmvpePath = Join-Path $RepoRoot "rmvpe\model.pt"
if (Test-Path $RmvpePath) {
    $rmvpeMB = [math]::Round((Get-Item $RmvpePath).Length / 1MB, 1)
    Write-Host "[prepare] Found rmvpe\model.pt ($rmvpeMB MB)" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[prepare] ERROR: rmvpe\model.pt not found at $RmvpePath" -ForegroundColor Red
    Write-Host "  Download it from HuggingFace:" -ForegroundColor Yellow
    Write-Host "    pip install huggingface_hub" -ForegroundColor Yellow
    Write-Host "    python -c `"from huggingface_hub import hf_hub_download; hf_hub_download('verstar/STARS', 'rmvpe/model.pt', local_dir='.')`"" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  Build context is ready. Run:" -ForegroundColor Cyan
Write-Host "    docker build -t secondpass:latest ." -ForegroundColor Cyan
Write-Host ""
