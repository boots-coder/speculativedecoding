HF_TOKEN = ""  # Set your Hugging Face token via environment or local config, do not commit real tokens

# Base Qwen3 8B model on Hugging Face
BASE_MODEL_ID = "Qwen/Qwen3-8B"

# Local path where the pruned 4B model will be written by mergekit
PRUNED_MODEL_PATH = "models/qwen3-4b-passthrough"

# Device configuration for inference
DEVICE = "cuda"  # or "cpu" if you only have CPU

