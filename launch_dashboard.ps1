# Quick Start Script for Spam Classifier Dashboard
# Activates virtual environment and launches Streamlit

Write-Host "=" * 60
Write-Host "Spam Classifier Dashboard Launcher" -ForegroundColor Cyan
Write-Host "=" * 60

# Check if virtual environment exists
if (-Not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: py -3.10 -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Green
& .venv\Scripts\Activate.ps1

# Check Python version
$pythonVersion = python --version
Write-Host "Using: $pythonVersion" -ForegroundColor Green

# Launch Streamlit
Write-Host "`nLaunching Streamlit dashboard..." -ForegroundColor Green
Write-Host "Dashboard will open in your default browser" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Yellow

streamlit run app\streamlit_app.py
