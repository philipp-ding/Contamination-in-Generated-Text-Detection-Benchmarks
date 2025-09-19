import os
import json
import random
from train_roberta import run
from types import SimpleNamespace

# Disable NCCL features incompatible with RTX 40xx
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Restrict to only GPU 0 (CUDA:0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_DIR = "/mnt/hdd-baracuda/pdingfelder/Masterarbeit/DetectRL"

task1_path = f"{BASE_DIR}/Benchmark/Tasks/Task1/"


# --- Set parameters in a dict ---
args_dict = {
    "model_name": "roberta-base",
    "save_model_path": "roberta_base_classifier",
    "train_data_path": f"{task1_path}/multi_llms_ChatGPT_train.json",
    "test_data_path": f"{task1_path}/multi_llms_ChatGPT_test.json",
    "transfer_test_data_path": f"{task1_path}/multi_llms_Claude-instant_test.json",
    "train_df": None,
    "transfer_df": None,
    "test_df": None,
    # "transfer_test_data_path": "../data/arxiv_transfer_test.json",
    "epochs": 3,
    "learning_rate": 1e-6,
    "batch_size": 32,
    "seed": 2023,
    "mode": "train",
    "DEVICE": "cuda"
}

# Convert dict to namespace-like object
args = SimpleNamespace(**args_dict)

# Call the run function
run(args)
