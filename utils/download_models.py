import os
from huggingface_hub import snapshot_download

model_map = {
    "bloom": "bigscience/bloom-7b1",  
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma2": "google/gemma-2-9b-it"  
    
}


os.makedirs("models", exist_ok=True)


for folder_name, model_id in model_map.items():
    save_path = os.path.join("models", folder_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading {model_id} to {save_path}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=save_path,
        local_dir_use_symlinks=False
    )
    print(f"Finished downloading {model_id}.\n")
