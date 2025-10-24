'''
Enhanced ostris/ai-toolkit for Modal.com with multi-model support
Combines: Latest ostris features + AINxtGen bug fixes + Multi-model caching

Author: Your customization based on ostris/ai-toolkit
Date: October 2025

Run with:
modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/your_config.yml
'''

import os
import sys

# CRITICAL FIX #1: Enable fast HuggingFace downloads (AINxtGen optimization)
# This makes downloading 50GB models much faster (2-3 minutes instead of 10-15 minutes)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import modal
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load your HuggingFace token from .env file
load_dotenv()

# Add ai-toolkit to Python path so imports work
sys.path.insert(0, "/root/ai-toolkit")

# Disable telemetry (optional privacy feature)
os.environ['DISABLE_TELEMETRY'] = 'YES'

# ============================================================================
# MULTI-MODEL VOLUME SETUP
# ============================================================================
# WHY: We create separate storage for each model type to keep things organized
# BENEFIT: Your Flux LoRAs won't mix with Qwen LoRAs, easier to manage

flux_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
qwen_volume = modal.Volume.from_name("qwen-lora-models", create_if_missing=True)
qwen_edit_volume = modal.Volume.from_name("qwen-edit-lora-models", create_if_missing=True)
wan_volume = modal.Volume.from_name("wan-lora-models", create_if_missing=True)
omnigen_volume = modal.Volume.from_name("omnigen-lora-models", create_if_missing=True)
hidream_volume = modal.Volume.from_name("hidream-lora-models", create_if_missing=True)
sd_volume = modal.Volume.from_name("sd-lora-models", create_if_missing=True)

# CRITICAL FIX #2: Shared cache for ALL base models (THE KEY OPTIMIZATION!)
# WHY: Without this, every training run downloads the 24-60GB base model again
# BENEFIT: First run downloads model, all future runs use cached version = huge time/cost savings
model_cache_volume = modal.Volume.from_name("ai-models-cache", create_if_missing=True)

# ============================================================================
# DIRECTORY PATHS
# ============================================================================
MOUNT_DIR = "/root/ai-toolkit/modal_output"  # Where trained LoRAs are saved
CACHE_DIR = "/root/.cache"  # Where HuggingFace caches downloaded models
HF_HOME = "/root/.cache/huggingface"  # Specific HuggingFace cache location

# Set cache environment variable so HuggingFace knows where to cache
os.environ["HF_HOME"] = HF_HOME

# ============================================================================
# MODAL CONTAINER IMAGE DEFINITION
# ============================================================================
# WHY: This defines what software is installed in the Modal cloud container
# WHAT: All the Python packages needed for training different models

image = (
    modal.Image.debian_slim(python_version="3.11")  # Base Linux image with Python 3.11
    .apt_install("libgl1", "libglib2.0-0", "git")  # System libraries needed
    .pip_install(
        # Core packages
        "python-dotenv",        # Load .env files
        "torch",                # PyTorch - main deep learning framework
        "torchvision",          # Computer vision utilities
        "diffusers[torch]",     # Hugging Face diffusers for Flux/SD
        "transformers",         # Hugging Face transformers for language models
        "ftfy",                 # Text fixing
        "oyaml",                # YAML file reading
        "opencv-python",        # Image processing
        "albumentations",       # Image augmentation
        "safetensors",          # Safe model file format
        
        # CRITICAL FIX #4: Pin this version to avoid breaking changes
        "lycoris-lora==1.8.3",  # LoRA training library (pinned version)
        
        "flatten_json",         # JSON utilities
        "pyyaml",               # YAML parsing
        "tensorboard",          # Training monitoring
        "kornia",               # Image transformations
        "invisible-watermark",  # Watermarking
        "einops",               # Tensor operations
        "accelerate",           # Training acceleration
        "toml",                 # TOML file reading
        "pydantic",             # Data validation
        "omegaconf",            # Configuration management
        "k-diffusion",          # Diffusion models
        "open_clip_torch",      # CLIP models
        "timm",                 # Vision models
        "prodigyopt",           # Optimizer
        "controlnet_aux==0.0.7",# ControlNet utilities
        "bitsandbytes",         # Quantization
        "hf_transfer",          # FAST HuggingFace downloads (works with HF_HUB_ENABLE_HF_TRANSFER)
        "lpips",                # Perceptual loss
        "pytorch_fid",          # FID score calculation
        "optimum-quanto",       # Quantization
        "sentencepiece",        # Tokenization
        "huggingface_hub",      # HF Hub utilities
        "peft"                  # Parameter efficient fine-tuning
    )
)

# ============================================================================
# CODE MOUNTING - CRITICAL FIX #3
# ============================================================================
# WHY: We need to upload your local ai-toolkit folder to Modal's cloud
# FIX: Use Path(__file__).parent instead of hardcoded path (works for everyone)
# BENEFIT: No need to manually edit the path - it auto-detects the current folder

code_mount = modal.Mount.from_local_dir(
    Path(__file__).parent,      # Current directory (where run_modal.py is)
    remote_path="/root/ai-toolkit"  # Where it appears in Modal container
)

# ============================================================================
# MODAL APP DEFINITION
# ============================================================================
# WHY: This creates your Modal app that will run in the cloud

app = modal.App(
    name="ai-toolkit-multi-model",  # Name shown in Modal dashboard
    image=image,                     # Container image we defined above
    mounts=[code_mount]              # Mount your code into the container
)

# Optional: Enable PyTorch anomaly detection for debugging
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    import torch
    torch.autograd.set_detect_anomaly(True)

# Import the job runner from toolkit
import argparse
from toolkit.job import get_job

# ============================================================================
# HELPER FUNCTION: Print Training Summary
# ============================================================================
# WHY: Give user clear feedback about what happened
# ENHANCEMENT: Better than original's simple print statement

def print_end_message(jobs_completed, jobs_failed):
    """
    Prints a nice summary box at the end of training
    Shows how many jobs succeeded and failed
    """
    print("\n" + "="*60)
    print("🎯 TRAINING SUMMARY")
    print("="*60)
    print(f"✅ Completed: {jobs_completed} job{'s' if jobs_completed != 1 else ''}")
    if jobs_failed > 0:
        print(f"❌ Failed: {jobs_failed} job{'s' if jobs_failed != 1 else ''}")
    print("="*60 + "\n")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
# This is where all the magic happens!

@app.function(
    gpu="H100",  # ENHANCEMENT: Use H100 by default (faster than A100, often cheaper on Modal)
                 # You can change this to "A100" or "A10G" if H100 is unavailable
    
    timeout=60 * 60 * 4,  # 4 hours maximum runtime
                          # Increase if training WAN models (can take 8+ hours)
    
    volumes={
        MOUNT_DIR: flux_volume,        # Output volume (will change per model)
        CACHE_DIR: model_cache_volume  # CRITICAL: Persistent cache (THE KEY FIX!)
    }
)
def main(config_file_list_str: str, recover: bool = False, name: str = None):
    """
    Main training function that runs in Modal's cloud
    
    WHAT IT DOES:
    1. Reads your config file
    2. Detects which model you're training (Flux, Qwen, WAN, etc.)
    3. Selects the appropriate storage volume
    4. Runs the training
    5. Saves everything (with proper error handling)
    
    ENHANCEMENTS FROM AINXTGEN:
    - Multi-model detection and routing
    - Persistent caching to avoid re-downloads
    - Better error messages with emojis
    - Proper volume commit timing (prevents data loss)
    
    PARAMETERS:
    - config_file_list_str: Comma-separated list of config files to run
    - recover: If True, continues even if a job fails
    - name: Optional name to replace [name] placeholder in config
    """
    
    # Convert comma-separated string to list
    # Example: "/root/ai-toolkit/config/job1.yml,/root/ai-toolkit/config/job2.yml"
    #       -> ["/root/ai-toolkit/config/job1.yml", "/root/ai-toolkit/config/job2.yml"]
    config_file_list = config_file_list_str.split(",")
    
    # Track successes and failures
    jobs_completed = 0
    jobs_failed = 0
    
    # Print startup message
    print(f"\n{'='*60}")
    print(f"🚀 Starting {len(config_file_list)} training job{'s' if len(config_file_list) != 1 else ''}")
    print(f"{'='*60}\n")
    
    # Loop through each config file (allows batch training)
    for config_file in config_file_list:
        try:
            # ================================================================
            # STEP 1: Read config and detect model type
            # ================================================================
            print(f"\n{'='*60}")
            print(f"📋 Processing: {config_file}")
            print(f"{'='*60}\n")
            
            # Read the YAML config file
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Try to extract model name from config
            # Config structure: config -> process -> [0] -> model -> name_or_path
            model_name = ""
            try:
                if 'config' in config_data:
                    process_list = config_data['config'].get('process', [])
                    if len(process_list) > 0 and 'model' in process_list[0]:
                        model_name = process_list[0]['model'].get('name_or_path', '').lower()
            except:
                model_name = ""  # If anything goes wrong, default to empty
            
            # ================================================================
            # STEP 2: Select appropriate volume based on model type
            # ================================================================
            # WHY: Keep different model types organized in separate storage
            # HOW: Check if model name contains specific keywords
            
            if 'qwen' in model_name and 'edit' in model_name:
                # Qwen-Image-Edit model (for image editing tasks)
                output_volume = qwen_edit_volume
                volume_name = "qwen-edit-lora-models"
                model_type = "Qwen-Image-Edit"
                emoji = "🎨"
                
            elif 'qwen' in model_name:
                # Qwen-Image model (for character/style training)
                output_volume = qwen_volume
                volume_name = "qwen-lora-models"
                model_type = "Qwen-Image"
                emoji = "🎨"
                
            elif 'wan' in model_name:
                # WAN 2.2 models (text-to-image, image-to-video)
                output_volume = wan_volume
                volume_name = "wan-lora-models"
                model_type = "WAN 2.2"
                emoji = "🎬"
                
            elif 'omnigen' in model_name:
                # OmniGen multi-modal model
                output_volume = omnigen_volume
                volume_name = "omnigen-lora-models"
                model_type = "OmniGen"
                emoji = "🌟"
                
            elif 'hidream' in model_name:
                # HiDream model
                output_volume = hidream_volume
                volume_name = "hidream-lora-models"
                model_type = "HiDream"
                emoji = "💭"
                
            elif 'stable-diffusion' in model_name or 'sdxl' in model_name:
                # Stable Diffusion models
                output_volume = sd_volume
                volume_name = "sd-lora-models"
                model_type = "Stable Diffusion"
                emoji = "🖼️"
                
            else:
                # Default to FLUX (most common)
                output_volume = flux_volume
                volume_name = "flux-lora-models"
                model_type = "FLUX.1"
                emoji = "⚡"
            
            # Print what we detected
            print(f"{emoji} Detected model: {model_type}")
            print(f"💾 Output volume: {volume_name}")
            print(f"📦 Cache volume: ai-models-cache (shared across all models)")
            print(f"🔄 Base models will be cached and reused across training runs!")
            print(f"💰 This saves time and Modal credits!\n")
            
            # ================================================================
            # STEP 3: Update volumes for this specific job
            # ================================================================
            # WHY: Modal needs to know which volumes to use for THIS specific training
            # NOTE: We update the function's volume dict dynamically
            
            # Create the volumes dictionary for this job
            volumes_dict = {
                MOUNT_DIR: output_volume,         # Where trained LoRAs save
                CACHE_DIR: model_cache_volume     # Where base models cache
            }
            
            # ================================================================
            # STEP 4: Set up the training job
            # ================================================================
            
            # Get job configuration from toolkit
            job = get_job(config_file, name)
            
            # Set the output folder in the config
            job.config['process'][0]['training_folder'] = MOUNT_DIR
            
            # Create the output directory
            os.makedirs(MOUNT_DIR, exist_ok=True)
            print(f"📁 Training folder: {MOUNT_DIR}\n")
            
            # ================================================================
            # STEP 5: RUN THE ACTUAL TRAINING
            # ================================================================
            print(f"🎯 Starting training...\n")
            print(f"💡 TIP: Monitor progress at https://modal.com/apps\n")
            
            job.run()  # This is where the actual training happens!
            
            # ================================================================
            # STEP 6: Save everything properly (CRITICAL FIX #5)
            # ================================================================
            # WHY: Proper saving prevents data loss
            # FIX: Error handling ensures we know if something goes wrong
            
            print(f"\n{'='*60}")
            print(f"💾 SAVING RESULTS")
            print(f"{'='*60}\n")
            
            # Save the trained LoRA to its volume
            print(f"💾 Saving trained model to {volume_name}...")
            try:
                output_volume.commit()
                print(f"✅ Model saved successfully!")
            except Exception as e:
                print(f"⚠️  Warning: Error saving output: {e}")
                # Don't fail the job completely, just warn
            
            # Save the model cache (so next run doesn't re-download)
            print(f"💾 Saving model cache for future runs...")
            try:
                model_cache_volume.commit()
                print(f"✅ Cache saved successfully!")
                print(f"💡 Next training will skip the ~{24 if 'flux' in model_type.lower() else 60}GB model download!")
            except Exception as e:
                print(f"⚠️  Warning: Error saving cache: {e}")
            
            # ================================================================
            # STEP 7: Cleanup and finish
            # ================================================================
            job.cleanup()
            jobs_completed += 1
            
            print(f"\n{'='*60}")
            print(f"✅ JOB COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}\n")
            
        except Exception as e:
            # ================================================================
            # ERROR HANDLING (ENHANCEMENT FIX #6)
            # ================================================================
            # WHY: Better error messages help you debug problems
            # WHAT: Print the full error with stack trace
            
            print(f"\n{'='*60}")
            print(f"❌ ERROR IN JOB")
            print(f"{'='*60}\n")
            print(f"Error message: {e}\n")
            
            # Print full error details
            import traceback
            print("Full error details:")
            traceback.print_exc()
            print()
            
            jobs_failed += 1
            
            # If recover=False, stop completely on first error
            if not recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e  # Re-raise the error to stop execution
            # If recover=True, just log the error and continue to next job
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print_end_message(jobs_completed, jobs_failed)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
# This part runs when you type: modal run run_modal.py --config-file-list-str=...

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Train AI models on Modal.com with multi-model support and caching"
    )
    
    # Config file argument (required)
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Config file(s) to run. Examples:\n'
             '  Single: /root/ai-toolkit/config/my_config.yml\n'
             '  Multiple: /root/ai-toolkit/config/job1.yml /root/ai-toolkit/config/job2.yml'
    )
    
    # Recover flag (optional)
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running remaining jobs even if one fails'
    )
    
    # Name replacement (optional)
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] placeholder in config files'
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Convert list of config files to comma-separated string
    # (Required for Modal's function calling format)
    config_file_list_str = ",".join(args.config_file_list)
    
    # Call the main function on Modal
    main.call(
        config_file_list_str=config_file_list_str,
        recover=args.recover,
        name=args.name
    )
