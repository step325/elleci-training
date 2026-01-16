import os
from huggingface_hub import hf_hub_download

# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

print("Starting download to checkpoints/...")
model_path = hf_hub_download(
    repo_id="Stepkrep/elleci",
    filename="elleci_v1_final.pth",
    local_dir="checkpoints",
    token=os.environ.get("HF_TOKEN")
)
print(f"Model downloaded to {model_path}")
