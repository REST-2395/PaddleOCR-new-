param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Project virtual environment Python was not found: $PythonExe"
}

if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $ProjectRoot "build")
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $ProjectRoot "dist")
}

& $PythonExe -m pip install pyinstaller
& $PythonExe -m PyInstaller --noconfirm --clean (Join-Path $ProjectRoot "DigitOCR_GUI.spec")

Write-Host ""
Write-Host "Build completed. Output directory:" -ForegroundColor Green
Write-Host (Join-Path $ProjectRoot "dist\DigitOCR_GUI")
