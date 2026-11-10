import os

# Disable hf_transfer to avoid the os error 32, since it struggles with locks
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Completely bypass the locked file by using a local cache directory
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "cache", "hf_bypass")

print("Downloading BGE-M3 model (using hf-mirror to bypass locked files)...")
from huggingface_hub import snapshot_download  # noqa: E402 - cache environment is set above

model_path = snapshot_download(
    repo_id="BAAI/bge-m3",
    local_files_only=False,
    resume_download=False, 
    force_download=True,
    ignore_patterns=["*.onnx", "*.onnx_data", "onnx/*", "imgs/*", "*.DS_Store"] 
)

print(f"BGE-M3 model downloaded to {model_path}")

print("Downloading BGE-Reranker-V2-M3 model...")
reranker_path = snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_files_only=False,
    resume_download=False,
    force_download=True,
    ignore_patterns=["*.onnx", "*.onnx_data", "onnx/*", "imgs/*", "*.DS_Store"]
)

print(f"BGE-Reranker-V2-M3 model downloaded to {reranker_path}")
print("Finished setting up BGE models.")

