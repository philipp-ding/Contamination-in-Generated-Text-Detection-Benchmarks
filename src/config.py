import os
from dotenv import load_dotenv

# take environment variables from ".env" file in src folder
load_dotenv(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".env"))

# TODO: if required, set your custome cache directory
custom_cache_dir = "/mnt/hdd-baracuda/pdingfelder/tmp"
os.environ["HF_HOME"] = custom_cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(custom_cache_dir, "datasets")
os.environ["HF_METRICS_CACHE"] = os.path.join(custom_cache_dir, "metrics")

# TODO: adjust CUDA setup depending on your setup
# Disable NCCL features incompatible with RTX 40xx
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Restrict to only GPU 0 (CUDA:0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
