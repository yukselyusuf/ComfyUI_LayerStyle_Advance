import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def get_hf_cache_dir():
    # If not set, use default path
    home = str(Path.home())
    cache_dir = os.path.join(home, '.cache', 'huggingface', 'hub')
    return cache_dir

def copy_from_cache(cache_dir, repo_id, filename, dest_dir):
    # Convert repo_id to cache path format
    repo_cache_path = os.path.join(cache_dir, 'models--' + repo_id.replace('/', '--'), 'snapshots')
    if os.path.exists(repo_cache_path):
        # Get the latest snapshot (should be the only one or the most recent)
        snapshots = sorted(os.listdir(repo_cache_path))  # Sort to get the latest
        if snapshots:
            latest_snapshot = snapshots[-1]
            src_file = os.path.join(repo_cache_path, latest_snapshot, filename)
            if os.path.exists(src_file):
                dest_file = os.path.join(dest_dir, filename)
                print(f"Copying {filename} from cache...")
                shutil.copy2(src_file, dest_file)
                return True
    return False

def download_florence2_model():
    # Get the ComfyUI models directory
    # You may need to modify this path according to your ComfyUI installation
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Create Florence2 directory if it doesn't exist
    florence_path = os.path.join(models_dir, "florence2")
    os.makedirs(florence_path, exist_ok=True)
    
    # Model paths
    base_ft_path = os.path.join(florence_path, "base-ft")
    promptgen_path = os.path.join(florence_path, "base-PromptGen")
    os.makedirs(base_ft_path, exist_ok=True)
    
    # HuggingFace repo IDs
    base_ft_repo = "microsoft/Florence-2-base-ft"
    promptgen_repo = "MiaoshouAI/Florence-2-base-PromptGen"
    
    # Get Hugging Face cache directory
    cache_dir = get_hf_cache_dir()
    print(f"Using Hugging Face cache directory: {cache_dir}")
    
    # Required files from base-ft
    base_ft_files = ["configuration_florence2.py", "modeling_florence2.py", "processing_florence2.py"]
    
    print(f"Starting download of Florence2 base-ft files...")
    print(f"Files will be downloaded to: {base_ft_path}")
    
    try:
        # First try to copy from cache
        for file in base_ft_files:
            if not copy_from_cache(cache_dir, base_ft_repo, file, base_ft_path):
                print(f"Downloading {file} from repository...")
                # If not in cache, download individual file
                downloaded_file = hf_hub_download(
                    repo_id=base_ft_repo,
                    filename=file,
                    cache_dir=cache_dir,
                    resume_download=True
                )
                # Copy downloaded file to destination
                shutil.copy2(downloaded_file, os.path.join(base_ft_path, file))
        
        print("Base-ft files downloaded successfully!")
        
        # Now download the PromptGen model
        print(f"\nStarting download of Florence2 base-PromptGen model...")
        print(f"Model will be downloaded to: {promptgen_path}")
        
        snapshot_download(
            repo_id=promptgen_repo,
            local_dir=promptgen_path,
            ignore_patterns=["*.md", "*.txt"]
        )
        print("PromptGen model downloaded successfully!")
        print(f"\nAll files have been downloaded:")
        print(f"Base-ft files: {base_ft_path}")
        print(f"PromptGen model: {promptgen_path}")
        
    except Exception as e:
        print(f"Error occurred during download: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_florence2_model() 