@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM AI-Toolkit Multi-Model Training Setup Script
REM Version: 2.0 (October 2025)
REM 
REM WHAT THIS DOES:
REM - Checks if Python and Git are installed
REM - Sets up Python virtual environment
REM - Installs/updates all dependencies
REM - Checks Modal.com authentication
REM - Checks HuggingFace token
REM - Guides you through model selection
REM - Launches training on Modal
REM 
REM HOW TO USE:
REM 1. Right-click this file
REM 2. Select "Run as administrator"
REM 3. Follow the prompts
REM ============================================================================

echo.
echo ============================================
echo   AI-Toolkit Multi-Model Training Setup
echo   Version 2.0 - October 2025
echo ============================================
echo.

REM ============================================================================
REM CHECK 1: Administrator Privileges
REM ============================================================================
REM WHY: Some operations need admin rights (creating folders, installing packages)

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

REM ============================================================================
REM CHECK 2: Python Installation
REM ============================================================================
REM WHY: Python is required to run the training code

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo.
    echo Please install Python 3.10 or 3.11 from:
    echo https://www.python.org/downloads/
    echo.
    echo After installing, run this script again.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM ============================================================================
REM CHECK 3: Git Installation
REM ============================================================================
REM WHY: Git is used for version control and syncing with ostris updates

echo Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed!
    echo.
    echo Please install Git from:
    echo https://git-scm.com/download/win
    echo.
    echo After installing, run this script again.
    echo.
    pause
    exit /b 1
)

echo [OK] Git found:
git --version
echo.

REM ============================================================================
REM CHECK 4: Virtual Environment
REM ============================================================================
REM WHY: Keeps Python packages isolated from other projects

if not exist "venv\" (
    echo Creating Python virtual environment...
    echo This will take about 30-60 seconds...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo.
        echo Possible causes:
        echo - Python not properly installed
        echo - Insufficient disk space
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully
    echo.
) else (
    echo [OK] Virtual environment already exists
    echo.
)

REM ============================================================================
REM STEP 5: Activate Virtual Environment
REM ============================================================================

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    echo.
    echo Try deleting the 'venv' folder and running this script again.
    echo.
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

REM ============================================================================
REM STEP 6: Update Python Packages
REM ============================================================================
REM WHY: Ensures you have the latest versions of all required packages

echo Updating Python packages...
echo This may take 3-5 minutes on first run...
echo.

REM Upgrade pip itself
python -m pip install --upgrade pip --quiet

REM Install requirements from requirements.txt
pip install -r requirements.txt --quiet

REM Install Modal
pip install --upgrade modal --quiet

echo [OK] All packages updated successfully
echo.

REM ============================================================================
REM CHECK 7: Modal.com Authentication
REM ============================================================================
REM WHY: Modal.com provides free GPU access for training

echo Checking Modal.com authentication...
modal token list >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo   MODAL.COM AUTHENTICATION REQUIRED
    echo ========================================
    echo.
    echo Modal.com provides FREE H100 GPU access!
    echo You get $30 free credits every month.
    echo.
    echo To authenticate:
    echo 1. Go to https://modal.com and sign up (no credit card needed)
    echo 2. After signing up, run this command:
    echo    modal setup
    echo 3. Follow the instructions to get your token
    echo 4. Then run this setup script again
    echo.
    pause
    exit /b 1
)

echo [OK] Modal.com is authenticated
echo.

REM ============================================================================
REM CHECK 8: HuggingFace Token
REM ============================================================================
REM WHY: Needed to download model weights from HuggingFace

if not exist ".env" (
    echo.
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
    echo HF_TOKEN=!HF_TOKEN! > .env
    echo.
    echo [OK] Token saved to .env file
    echo.
    echo IMPORTANT: Also login to HuggingFace CLI:
    echo Run this command: huggingface-cli login
    echo And paste your token again.
    echo.
    pause
) else (
    echo [OK] .env file with HuggingFace token exists
    echo.
)

REM ============================================================================
REM STEP 9: Model Selection Menu
REM ============================================================================

echo ============================================
echo   SELECT MODEL TYPE TO TRAIN
echo ============================================
echo.
echo  1. FLUX.1          - Text-to-Image (24GB model, ~45 min training)
echo  2. Qwen-Image      - Character/Style (27GB model, ~60 min training)
echo  3. Qwen-Image-Edit - Image Editing (27GB model, ~60 min training)
echo  4. WAN 2.2 T2I     - Text-to-Image (60GB+ model, needs H100)
echo  5. WAN 2.2 I2V     - Image-to-Video (60GB+ model, 8+ hours)
echo  6. OmniGen2        - Multi-modal (requires H100)
echo  7. HiDream         - Image Generation
echo  8. Stable Diffusion/SDXL - Classic models
echo  9. Custom          - I have my own config ready
echo.
set /p MODEL_CHOICE="Enter your choice (1-9): "

REM Set model name based on choice
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
REM STEP 10: Config File Check
REM ============================================================================

echo ============================================
echo   CONFIG FILE SETUP
echo ============================================
echo.
echo Your config file should be in the config\ folder
echo Example configs are in: config\examples\
echo.
set /p CONFIG_FILE="Enter your config filename (e.g., my_training.yml): "

if not exist "config\%CONFIG_FILE%" (
    echo.
    echo [ERROR] Config file not found: config\%CONFIG_FILE%
    echo.
    echo Please:
    echo 1. Look in config\examples\ for example configs
    echo 2. Copy an example to config\
    echo 3. Edit it with your settings
    echo 4. Run this script again
    echo.
    pause
    exit /b 1
)

echo [OK] Config file found: config\%CONFIG_FILE%
echo.

REM ============================================================================
REM STEP 11: Dataset Check
REM ============================================================================

set /p DATASET_READY="Have you prepared your dataset in the datasets\ folder? (Y/N): "
if /i not "%DATASET_READY%"=="Y" (
    echo.
    echo Please prepare your dataset first:
    echo.
    echo 1. Create a folder: datasets\my_project_name\
    echo 2. Add your training images to that folder
    echo 3. If needed, add caption files (.txt with same name as images)
    echo 4. Update the folder_path in your config file
    echo.
    echo Example structure:
    echo   datasets\
    echo     my_character\
    echo       image001.jpg
    echo       image001.txt
    echo       image002.jpg
    echo       image002.txt
    echo.
    pause
    exit /b 1
)

echo.

REM ============================================================================
REM STEP 12: Final Confirmation and Launch
REM ============================================================================

echo ============================================
echo   READY TO START TRAINING
echo ============================================
echo.
echo Config File: config\%CONFIG_FILE%
echo Model Type:  %MODEL_NAME%
echo.
echo IMPORTANT INFORMATION:
echo.
echo  FIRST RUN:
echo    - Will download base model (24-60GB depending on model)
echo    - Downloads are FAST (2-3 min) thanks to HF_HUB_ENABLE_HF_TRANSFER
echo    - Model is cached for future runs
echo.
echo  SUBSEQUENT RUNS:
echo    - Skip model download (uses cache)
echo    - Start training immediately
echo    - Saves time and Modal credits!
echo.
echo  GPU USAGE:
echo    - Training runs on Modal H100 GPU (very fast!)
echo    - Cost: ~$5 per training (~45 min for Flux)
echo    - You have $30 free credits per month
echo.
echo  MONITORING:
echo    - After files upload, you can close this window
echo    - Monitor progress at: https://modal.com/apps
echo    - Check logs in real-time on Modal dashboard
echo.
echo Press any key to start training...
pause >nul

echo.
echo ============================================
echo   LAUNCHING TRAINING ON MODAL.COM
echo ============================================
echo.
echo Uploading files and starting training...
echo This may take 1-2 minutes...
echo.

REM Launch the training
modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/%CONFIG_FILE%

echo.
echo ============================================
echo   TRAINING LAUNCHED
echo ============================================
echo.
echo Your training is now running on Modal!
echo.
echo TO DOWNLOAD RESULTS:
echo.
echo   1. Check which volume your model is in:
echo      modal volume ls
echo.
echo   2. List files in the volume:
echo      modal volume ls flux-lora-models
echo      (or qwen-lora-models, wan-lora-models, etc.)
echo.
echo   3. Download your trained model:
echo      modal volume get flux-lora-models your_model_name
echo.
echo TO MONITOR PROGRESS:
echo   Visit: https://modal.com/apps
echo   Click on your running job to see live logs
echo.
echo TO TRAIN MORE MODELS:
echo   Just run this script again!
echo   Models are cached, so subsequent runs are faster.
echo.
pause
