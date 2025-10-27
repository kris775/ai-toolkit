@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM AI-Toolkit Multi-Model Training Setup Script
REM Version: 2.1 (Updated October 2025)
REM 
REM CHANGES IN v2.1:
REM - Fixed Modal CLI authentication check (works with new Modal CLI)
REM - Added simple token paste option (like AINxtGen)
REM - Better error handling for missing files
REM ============================================================================

echo.
echo ============================================
echo   AI-Toolkit Multi-Model Training Setup
echo   Version 2.1 - October 2025
echo ============================================
echo.

REM ============================================================================
REM CHECK 1: Administrator Privileges
REM ============================================================================

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] This script requires administrator privileges
    echo.
    echo Please:
    echo 1. Right-click this file
    echo 2. Select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo [OK] Running with administrator privileges
echo.

REM Change to the script's directory (where the .bat file is located)
cd /d "%~dp0"

echo Current directory: %CD%
echo.

REM ============================================================================
REM CHECK 2: Verify we're in the correct directory
REM ============================================================================

if not exist "run_modal.py" (
    echo [ERROR] Cannot find run_modal.py
    echo.
    echo This script must be run from the ai-toolkit folder!
    echo Current directory: %CD%
    echo.
    echo Please:
    echo 1. Copy this script to C:\ai-toolkit
    echo 2. Run it from there
    echo.
    pause
    exit /b 1
)

echo [OK] Found run_modal.py - correct directory
echo.

REM ============================================================================
REM CHECK 3: Python Installation
REM ============================================================================

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo.
    echo Please install Python 3.10 or 3.11 from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM ============================================================================
REM CHECK 4: Git Installation
REM ============================================================================

echo Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed!
    echo.
    echo Please install Git from:
    echo https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo [OK] Git found:
git --version
echo.

REM ============================================================================
REM CHECK 5: requirements.txt exists
REM ============================================================================

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    echo.
    echo Your repository may be incomplete.
    echo Please make sure you're in the correct ai-toolkit directory.
    echo.
    pause
    exit /b 1
)

echo [OK] requirements.txt found
echo.

REM ============================================================================
REM STEP 6: Virtual Environment
REM ============================================================================

if not exist "venv\" (
    echo Creating Python virtual environment...
    echo This will take about 30-60 seconds...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    echo.
) else (
    echo [OK] Virtual environment already exists
    echo.
)

REM ============================================================================
REM STEP 7: Activate Virtual Environment
REM ============================================================================

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

REM ============================================================================
REM STEP 8: Update Python Packages
REM ============================================================================

REM --- Modal CLI & pip install (skip after first)
where modal >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Modal CLI not found, ensuring installation in venv...
    pip install modal
    if %errorlevel% neq 0 (
        echo [ERROR] Modal CLI failed to install!
        pause
        exit /b 1
    )
)
if not exist "venv\lib\site-packages\modal" (
    echo [INFO] Installing requirements...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [ERROR] Requirements install failed!
        pause
        exit /b 1
    )
    echo [OK] All Python requirements installed
) else (
    echo [INFO] Requirements and Modal CLI already installed, skipping reinstalls!
)
echo.
REM ============================================================================
REM CHECK 9: Modal.com Authentication (NEW METHOD)
REM ============================================================================

echo Checking Modal.com authentication...
echo.

REM Check if .modal.toml exists
if exist "%USERPROFILE%\.modal.toml" (
    echo [OK] Modal token file found
    echo.
    set MODAL_AUTHENTICATED=1
) else (
    set MODAL_AUTHENTICATED=0
)

REM --- Modal token check/guidance (improved)
:CHECK_MODAL_TOKEN
modal token list >nul 2>&1
if %errorlevel% neq 0 (
    echo ============================================================
    echo [ACTION] Modal API token is NOT set in this environment.
    echo.
    echo 1. Open https://modal.com/settings/tokens in your browser
    echo 2. Click "New Token" and copy BOTH token id and secret
    echo 3. You will be prompted for BOTH below
    echo.
    set /p WANT_OPEN="Open Modal token site in your browser? (y/n): "
    if /i "!WANT_OPEN!"=="y" start https://modal.com/settings/tokens
    echo.
    set /p TOKEN_ID="Enter your Modal Token ID (starts with ak-): "
    set /p TOKEN_SECRET="Enter your Modal Token Secret (starts with as-): "
    echo Running: modal token set --token-id !TOKEN_ID! --token-secret !TOKEN_SECRET!
    modal token set --token-id !TOKEN_ID! --token-secret !TOKEN_SECRET!
    if !errorlevel! neq 0 (
        echo [ERROR] Modal token set failed, try again.
        goto :CHECK_MODAL_TOKEN
    )
    echo [OK] Modal API token set
)

echo.

REM ============================================================================
REM CHECK 10: HuggingFace Token
REM ============================================================================

if not exist ".env" (
    echo ========================================
    echo   HUGGINGFACE TOKEN REQUIRED
    echo ========================================
    echo.
    echo You need a HuggingFace token to download models.
    echo.
    echo To get your token:
    echo 1. Go to https://huggingface.co/settings/tokens
    echo 2. Click "New token"
    echo 3. Select "Read" permission
    echo 4. Copy the token
    echo.
    set /p HF_TOKEN="Paste your HuggingFace token here: "
    echo HF_TOKEN=!HF_TOKEN!> .env
    echo.
    echo [OK] Token saved to .env file
    echo.
) else (
    echo [OK] .env file with HuggingFace token exists
    echo.
)

REM ============================================================================
REM STEP 11: Model Selection Menu
REM ============================================================================

echo ============================================
echo   SELECT MODEL TYPE TO TRAIN
echo ============================================
echo.
echo  1. FLUX.1          - Text-to-Image (24GB, ~45 min)
echo  2. Qwen-Image      - Character/Style (27GB, ~60 min)
echo  3. Qwen-Image-Edit - Image Editing (27GB, ~60 min)
echo  4. WAN 2.2 T2I     - Text-to-Image (60GB+, H100)
echo  5. WAN 2.2 I2V     - Image-to-Video (60GB+, 8+ hours)
echo  6. OmniGen2        - Multi-modal (H100)
echo  7. HiDream         - Image Generation
echo  8. SD/SDXL         - Stable Diffusion
echo  9. Custom          - Custom config
echo.
set /p MODEL_CHOICE="Enter your choice (1-9): "

if "%MODEL_CHOICE%"=="1" set MODEL_NAME=FLUX.1
if "%MODEL_CHOICE%"=="2" set MODEL_NAME=Qwen-Image
if "%MODEL_CHOICE%"=="3" set MODEL_NAME=Qwen-Image-Edit  
if "%MODEL_CHOICE%"=="4" set MODEL_NAME=WAN 2.2 T2I
if "%MODEL_CHOICE%"=="5" set MODEL_NAME=WAN 2.2 I2V
if "%MODEL_CHOICE%"=="6" set MODEL_NAME=OmniGen2
if "%MODEL_CHOICE%"=="7" set MODEL_NAME=HiDream
if "%MODEL_CHOICE%"=="8" set MODEL_NAME=Stable Diffusion
if "%MODEL_CHOICE%"=="9" set MODEL_NAME=Custom

echo.
echo Selected: %MODEL_NAME%
echo.

REM ============================================================================
REM STEP 12: Config File
REM ============================================================================

echo ============================================
echo   CONFIG FILE SETUP
echo ============================================
echo.
set /p CONFIG_FILE="Enter your config filename (e.g., my_training.yml): "

if not exist "config\%CONFIG_FILE%" (
    echo.
    echo [ERROR] Config file not found: config\%CONFIG_FILE%
    echo.
    echo Please create your config file first!
    echo Example configs: config\examples\
    echo.
    pause
    exit /b 1
)

echo [OK] Config file found
echo.

REM ============================================================================
REM STEP 13: Dataset Check
REM ============================================================================

set /p DATASET_OK="Dataset prepared in datasets\ folder? (Y/N): "
if /i not "%DATASET_OK%"=="Y" (
    echo.
    echo Please prepare your dataset first!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   READY TO START TRAINING
echo ============================================
echo.
echo Config: config\%CONFIG_FILE%
echo Model:  %MODEL_NAME%
echo.
echo FIRST RUN:
echo  - Downloads base model (24-60GB depending on model)
echo  - Fast downloads (2-3 min) via HF_HUB_ENABLE_HF_TRANSFER
echo  - Model cached for future runs
echo.
echo SUBSEQUENT RUNS:
echo  - Skip model download (uses cache)
echo  - Start training immediately
echo  - Saves time and credits!
echo.
echo Monitor at: https://modal.com/apps
echo.
pause

echo.
echo ============================================
echo   LAUNCHING TRAINING ON MODAL.COM
echo ============================================
echo.

modal run run_modal.py::main --config-file-list-str /root/ai-toolkit/config/%CONFIG_FILE%

echo.
echo ============================================
echo   TRAINING LAUNCHED
echo ============================================
echo.
echo Download results:
echo   modal volume ls flux-lora-models
echo   modal volume get flux-lora-models your_model_name
echo.
echo Monitor: https://modal.com/apps
echo.
pause
