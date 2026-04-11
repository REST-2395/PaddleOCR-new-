@echo off
setlocal
for %%I in ("%~dp0.") do set "SCRIPT_DIR=%%~fI"
set "VENV_PYTHON=%SCRIPT_DIR%\.venv\Scripts\python.exe"
set "VENV_PYTHONW=%SCRIPT_DIR%\.venv\Scripts\pythonw.exe"

if exist "%VENV_PYTHON%" (
    if not exist "%VENV_PYTHONW%" (
        set "VENV_PYTHONW=%VENV_PYTHON%"
    )
    "%VENV_PYTHONW%" "%SCRIPT_DIR%\gui_app.pyw"
    exit /b %errorlevel%
)

where py >nul 2>&1
if not errorlevel 1 (
    py -3 "%SCRIPT_DIR%\gui_app.pyw"
    exit /b %errorlevel%
)

where python >nul 2>&1
if not errorlevel 1 (
    python "%SCRIPT_DIR%\gui_app.pyw"
    exit /b %errorlevel%
)

echo Python 3.10+ is required to run the project.
exit /b 1
