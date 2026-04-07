@echo off
setlocal
for %%I in ("%~dp0.") do set "SCRIPT_DIR=%%~fI"
set "BOOTSTRAP_SCRIPT=%SCRIPT_DIR%\bootstrap_env.py"
set "VENV_PYTHON=%SCRIPT_DIR%\.venv\Scripts\python.exe"
set "VENV_PYTHONW=%SCRIPT_DIR%\.venv\Scripts\pythonw.exe"
set "BOOTSTRAP_PYTHON="
set "BOOTSTRAP_PYTHON_ARGS="

if exist "%VENV_PYTHON%" (
    if not exist "%VENV_PYTHONW%" (
        set "VENV_PYTHONW=%VENV_PYTHON%"
    )
    "%VENV_PYTHONW%" "%SCRIPT_DIR%\gui_app.pyw"
    exit /b %errorlevel%
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "BOOTSTRAP_PYTHON=py"
        set "BOOTSTRAP_PYTHON_ARGS=-3"
    ) else (
        where python >nul 2>&1
        if not errorlevel 1 (
            set "BOOTSTRAP_PYTHON=python"
        )
    )
)

if not defined BOOTSTRAP_PYTHON (
    echo Python 3.10+ is required to bootstrap the project.
    exit /b 1
)

"%BOOTSTRAP_PYTHON%" %BOOTSTRAP_PYTHON_ARGS% "%BOOTSTRAP_SCRIPT%" --project-root "%SCRIPT_DIR%" --skip-installed --skip-bootstrap-packages
if errorlevel 1 exit /b %errorlevel%

if not exist "%VENV_PYTHONW%" (
    set "VENV_PYTHONW=%VENV_PYTHON%"
)

"%VENV_PYTHONW%" "%SCRIPT_DIR%\gui_app.pyw"
