@echo off
REM KC Annotation Tool - Run Script
REM Double-click this file to launch the application

REM Check if virtual environment exists
if not exist "kcannot\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call kcannot\Scripts\activate.bat

REM Run the application (pass any command line arguments)
python main_panel.py %*

REM Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)
