import torch
import logging

def setup_device():
    """Detects and returns the available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("NVIDIA GPU detected! Using: cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Apple Silicon detected! Using: mps")
    else:
        device = torch.device("cpu")
        logging.info("No GPU detected. Using: cpu")
    return device

def set_seed(seed=42):
    """Sets the seed for reproducibility across runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)