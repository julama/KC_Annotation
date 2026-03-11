@echo off
echo ========================================
echo KC Annotation Tool - Setup Script
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Attempting automatic installation...
    echo.
    
    REM Check if winget is available
    winget --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: winget is not available on this system.
        echo Please install Python 3.11 manually from python.org or the Microsoft Store.
        pause
        exit /b 1
    )
    
    echo Installing Python 3.11 via winget (user install, no admin needed)...
    winget install Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
    
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Python.
        echo Please install Python 3.11 manually from python.org or the Microsoft Store.
        pause
        exit /b 1
    )
    
    echo.
    echo Python installed successfully!
    echo.
    echo IMPORTANT: Please close this window and run setup.bat again.
    echo This is needed for Windows to recognize the new Python installation.
    pause
    exit /b 0
)

echo Found Python:
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "kcannot" (
    echo Creating virtual environment 'kcannot'...
    python -m venv kcannot
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment 'kcannot' already exists.
)
echo.

REM Activate and install dependencies
echo Activating virtual environment...
call kcannot\Scripts\activate.bat

echo.
echo Installing/updating dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo If you're behind a corporate proxy, you may need to configure pip.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup complete!
echo.
echo To run the application:
echo   Double-click 'run.bat'
echo   Or run: run.bat --mat-file "path\to\your\file.mat"
echo ========================================
pause
