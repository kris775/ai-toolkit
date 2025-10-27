'''
FINAL PRODUCTION VERSION - WITH MODAL SECRETS
Fully compliant with AINxtGen + Ostris + Multi-Model
Uses Modal Secrets for HF_TOKEN (REQUIRED for HuggingFace authentication)
'''

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load .env file (optional, for local testing)
load_dotenv()

import modal

# Add toolkit to path
sys.path.insert(0, "/root/ai-toolkit")

# ============================================================================
# VOLUMES - SEPARATE BASE MODELS + LORA OUTPUTS
# ============================================================================

# Model base volumes (for downloaded models)
flux_model_volume = modal.Volume.from_name("flux-base-model", create_if_missing=True)
qwen_model_volume = modal.Volume.from_name("qwen-base-model", create_if_missing=True)
qwen_edit_model_volume = modal.Volume.from_name("qwen-edit-base-model", create_if_missing=True)
wan_model_volume = modal.Volume.from_name("wan-base-model", create_if_missing=True)
omnigen_model_volume = modal.Volume.from_name("omnigen-base-model", create_if_missing=True)
hidream_model_volume = modal.Volume.from_name("hidream-base-model", create_if_missing=True)
sd_model_volume = modal.Volume.from_name("sd-base-model", create_if_missing=True)

# LoRA output volumes (for trained models)
flux_lora_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
qwen_lora_volume = modal.Volume.from_name("qwen-lora-models", create_if_missing=True)
qwen_edit_lora_volume = modal.Volume.from_name("qwen-edit-lora-models", create_if_missing=True)
wan_lora_volume = modal.Volume.from_name("wan-lora-models", create_if_missing=True)
omnigen_lora_volume = modal.Volume.from_name("omnigen-lora-models", create_if_missing=True)
hidream_lora_volume = modal.Volume.from_name("hidream-lora-models", create_if_missing=True)
sd_lora_volume = modal.Volume.from_name("sd-lora-models", create_if_missing=True)

# Cache volume
hf_cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Paths
MOUNT_DIR = "/root/ai-toolkit/modal_output"
CACHE_DIR = "/cache/huggingface"
FLUX_MODEL_PATH = "/root/FLUX.1-dev"
QWEN_MODEL_PATH = "/root/qwen-model"
QWEN_EDIT_MODEL_PATH = "/root/qwen-edit-model"
WAN_MODEL_PATH = "/root/wan-model"
OMNIGEN_MODEL_PATH = "/root/omnigen-model"
HIDREAM_MODEL_PATH = "/root/hidream-model"
SD_MODEL_PATH = "/root/sd-model"

# ============================================================================
# MODAL IMAGE
# ============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git", "curl")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "CUDA_HOME": "/usr/local/cuda-12",  # For torchao/bitsandbytes
        "HF_HOME": CACHE_DIR,
        'DISABLE_TELEMETRY': 'YES'
    })
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "torchao==0.9.0",
        "git+https://github.com/huggingface/diffusers@main",
        "transformers==4.49.0",
        "peft",
        "huggingface_hub>=0.34.0",
        "hf_transfer",
        "accelerate",
        "lycoris-lora==1.8.3",
        "safetensors",
        "opencv-python",
        "albumentations==1.4.15",
        "albucore==0.0.16",
        "Pillow",
        "pytorch-wavelets==1.3.0",
        "kornia",
        "einops",
        "invisible-watermark",
        "lpips",
        "pytorch_fid",
        "controlnet_aux==0.0.10",
        "git+https://github.com/jaretburkett/easy_dwpose.git",
        "pyyaml",
        "oyaml",
        "tensorboard",
        "toml",
        "pydantic",
        "omegaconf",
        "python-dotenv",
        "flatten_json",
        "python-slugify",
        "prodigyopt",
        "bitsandbytes",
        "k-diffusion",
        "open_clip_torch",
        "timm",
        "optimum-quanto==0.2.4",
        "matplotlib==3.10.1",
        "sentencepiece",
        "requests",
        "gradio",
    )
    .add_local_dir(
        str(Path(__file__).parent),
        remote_path="/root/ai-toolkit"
    )
)

# ============================================================================
# MODAL APP
# ============================================================================

app = modal.App(name="ai-toolkit-multi-model")

# Optional debug mode
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    import torch
    torch.autograd.set_detect_anomaly(True)

from toolkit.job import get_job

def print_end_message(jobs_completed, jobs_failed):
    print("\n" + "="*60)
    print("🎯 TRAINING SUMMARY")
    print("="*60)
    print(f"✅ Completed: {jobs_completed} job{'s' if jobs_completed != 1 else ''}")
    if jobs_failed > 0:
        print(f"❌ Failed: {jobs_failed} job{'s' if jobs_failed != 1 else ''}")
    print("="*60 + "\n")

@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("huggingface-secret")],  # CRITICAL: HF_TOKEN from Modal Secret
    volumes={
        # Output volumes for trained models
        MOUNT_DIR: flux_lora_volume,
        CACHE_DIR: hf_cache_volume,
        # Model base volumes (will be selected per model)
        FLUX_MODEL_PATH: flux_model_volume,
        QWEN_MODEL_PATH: qwen_model_volume,
        QWEN_EDIT_MODEL_PATH: qwen_edit_model_volume,
        WAN_MODEL_PATH: wan_model_volume,
        OMNIGEN_MODEL_PATH: omnigen_model_volume,
        HIDREAM_MODEL_PATH: hidream_model_volume,
        SD_MODEL_PATH: sd_model_volume,
    }
)
def main(config_file_list_str: str, recover: bool = False, name: str = None):
    # ADD THIS LINE - Read HF_TOKEN from Modal Secret at runtime
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"✅ HF_TOKEN loaded from Modal Secret")
    else:
        print(f"❌ WARNING: HF_TOKEN not found!")
    config_file_list = config_file_list_str.split(",")
    jobs_completed = 0
    jobs_failed = 0
    
    print(f"\n{'='*60}")
    print(f"🚀 AI-Toolkit Modal Training (Production Ready)")
    print(f"✅ Multi-Model | HF Secret | CUDA 12 | Torch 2.6.0")
    print(f"{'='*60}\n")
    
    for config_file in config_file_list:
        try:
            print(f"\n📋 Processing: {config_file}\n")
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            model_name = ""
            try:
                if 'config' in config_data:
                    process_list = config_data['config'].get('process', [])
                    if len(process_list) > 0 and 'model' in process_list[0]:
                        model_name = process_list[0]['model'].get('name_or_path', '').lower()
            except:
                model_name = ""
            
            # Model routing with volumes
            if 'qwen' in model_name and 'edit' in model_name:
                output_lora_volume = qwen_edit_lora_volume
                model_volume = qwen_edit_model_volume
                model_path = QWEN_EDIT_MODEL_PATH
                volume_name = "qwen-edit-lora-models"
            elif 'qwen' in model_name:
                output_lora_volume = qwen_lora_volume
                model_volume = qwen_model_volume
                model_path = QWEN_MODEL_PATH
                volume_name = "qwen-lora-models"
            elif 'wan' in model_name:
                output_lora_volume = wan_lora_volume
                model_volume = wan_model_volume
                model_path = WAN_MODEL_PATH
                volume_name = "wan-lora-models"
            elif 'omnigen' in model_name:
                output_lora_volume = omnigen_lora_volume
                model_volume = omnigen_model_volume
                model_path = OMNIGEN_MODEL_PATH
                volume_name = "omnigen-lora-models"
            elif 'hidream' in model_name:
                output_lora_volume = hidream_lora_volume
                model_volume = hidream_model_volume
                model_path = HIDREAM_MODEL_PATH
                volume_name = "hidream-lora-models"
            elif 'stable-diffusion' in model_name or 'sdxl' in model_name:
                output_lora_volume = sd_lora_volume
                model_volume = sd_model_volume
                model_path = SD_MODEL_PATH
                volume_name = "sd-lora-models"
            else:
                output_lora_volume = flux_lora_volume
                model_volume = flux_model_volume
                model_path = FLUX_MODEL_PATH
                volume_name = "flux-lora-models"
            
            print(f"💾 Output LoRA Volume: {volume_name}")
            print(f"📦 HuggingFace Cache: {CACHE_DIR}")
            print(f"🏗️ Model Path: {model_path}\n")
            
            job = get_job(config_file, name)
            job.config['process'][0]['training_folder'] = MOUNT_DIR
            os.makedirs(MOUNT_DIR, exist_ok=True)
            # CRITICAL FIX: Ensure model path directory exists and is writable
            os.makedirs(model_path, exist_ok=True)
            
            print(f"🎯 Starting training...\n")
            job.run()
            
            print(f"\n💾 SAVING\n")
            try:
                output_lora_volume.commit()
                model_volume.commit()
                hf_cache_volume.commit()
                print(f"✅ All volumes saved!")
            except Exception as e:
                print(f"⚠️ {e}")
            
            job.cleanup()
            jobs_completed += 1
            print(f"\n✅ DONE!\n")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            jobs_failed += 1
            if not recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
    
    print_end_message(jobs_completed, jobs_failed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file_list', nargs='+', type=str)
    parser.add_argument('-r', '--recover', action='store_true')
    parser.add_argument('-n', '--name', type=str, default=None)
    args = parser.parse_args()
    config_file_list_str = ",".join(args.config_file_list)
    main.remote(
        config_file_list_str=config_file_list_str,
        recover=args.recover,
        name=args.name
    )
