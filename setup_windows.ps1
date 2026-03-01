# Zhang Exact Replication - Windows Setup Script
# Run this in PowerShell in your C:\Projects\zhang2025-reproduction directory

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "ZHANG EXACT REPLICATION - SETUP" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

# 1. Check Python version
Write-Host "`n[1/4] Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version
Write-Host "  ✓ Found: $pythonVersion" -ForegroundColor Green

# 2. Install required packages
Write-Host "`n[2/4] Installing required packages..." -ForegroundColor Yellow
Write-Host "  Installing: pandas, requests, pyarrow" -ForegroundColor Gray

python -m pip install --upgrade pip --quiet
python -m pip install pandas requests pyarrow --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Packages installed successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Package installation failed" -ForegroundColor Red
    exit 1
}

# 3. Create directory structure
Write-Host "`n[3/4] Creating directories..." -ForegroundColor Yellow

$dirs = @(
    "data",
    "data\gdelt_zhang_exact",
    "scripts",
    "results"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✓ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  → Exists: $dir" -ForegroundColor Gray
    }
}

# 4. Verify setup
Write-Host "`n[4/4] Verifying setup..." -ForegroundColor Yellow

# Test imports
$testScript = @"
import pandas as pd
import requests
import sys
print('✓ All packages working')
print(f'  Python: {sys.version.split()[0]}')
print(f'  Pandas: {pd.__version__}')
"@

$testScript | python

Write-Host "`n" + "=" * 80 -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Download the Python scripts from Claude (already saved in outputs)" -ForegroundColor White
Write-Host "  2. Run: python download_gdelt_zhang_exact.py" -ForegroundColor White
Write-Host "`nEstimated time:" -ForegroundColor Yellow
Write-Host "  Full data (2015-2025): ~3-4 hours" -ForegroundColor White
Write-Host "  Sample (2023-2024):     ~30-45 minutes" -ForegroundColor White






































