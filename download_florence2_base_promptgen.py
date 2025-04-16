import os
import sys
from huggingface_hub import snapshot_download

def download_florence2_model():
    # Get the ComfyUI models directory
    # You may need to modify this path according to your ComfyUI installation
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Create Florence2 directory if it doesn't exist
    florence_path = os.path.join(models_dir, "florence2")
    os.makedirs(florence_path, exist_ok=True)
    
    # Model path
    model_path = os.path.join(florence_path, "base-PromptGen")
    
    # HuggingFace repo ID
    repo_id = "MiaoshouAI/Florence-2-base-PromptGen"
    
    print(f"Starting download of Florence2 base-PromptGen model...")
    print(f"Model will be downloaded to: {model_path}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            ignore_patterns=["*.md", "*.txt"]
        )
        print("Download completed successfully!")
        print(f"Model files are saved in: {model_path}")
        
    except Exception as e:
        print(f"Error occurred during download: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_florence2_model() 