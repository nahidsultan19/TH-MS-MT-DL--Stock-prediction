"""Shared utilities: seeding, logging, config loading, path helpers."""
import os
import random
import yaml
import numpy as np
import tensorflow as tf
import torch


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int = 42):
    """Set seeds for full reproducibility across numpy, TF, torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_metrics(metrics: dict, path: str):
    import pandas as pd
    pd.DataFrame(metrics).to_excel(path, index=False)