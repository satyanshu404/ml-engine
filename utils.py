import os
import torch
import yaml
from typing import Tuple, Dict


def set_device(device=None) -> Tuple[torch.device, str]:
    """Set device and autocast device type
    Returns:
        device: device for training and
        autocast_device_type: device type for autocast
    """
    if device is not None:
        return torch.device(device), device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        autocast_device_type = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        autocast_device_type = "mps"
    else:
        device = torch.device("cpu")
        autocast_device_type = "cpu"
    return device, autocast_device_type


def read_file(file_type: str, file_path: str):
    """Load file based on file type"""
    assert os.path.exists(file_path), f"File {file_path} not found"
    if file_type == "yaml":
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    elif file_type == "json":
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def set_gloabl_seed(seed: int) -> None:
    """Set global seed for deterministic training"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":5120:8"

    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
