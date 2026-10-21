"""Device selection and reproducibility helpers.

AMD GPUs (e.g. RX 9070 XT) running a ROCm build of PyTorch appear through the
same ``torch.cuda`` API as NVIDIA cards, so "cuda" covers both.
"""
import os
import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Pick a device: explicit ``prefer`` > $LAB_DEVICE > GPU if present > CPU."""
    choice = prefer or os.environ.get("LAB_DEVICE")
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable description, noting whether a GPU is ROCm (AMD) or CUDA."""
    if device.type == "cuda":
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        return f"{torch.cuda.get_device_name(device)} ({backend})"
    return "CPU"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch so a run is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
