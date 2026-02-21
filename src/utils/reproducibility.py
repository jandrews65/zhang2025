"""Reproducibility utilities.

Sets random seeds and provides environment snapshot for reproducible runs.
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
from loguru import logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info(f"Random seed set to {seed}")


def log_environment() -> dict[str, str]:
    """Log the current environment for reproducibility.

    Returns:
        Dictionary of environment information.
    """
    env_info: dict[str, str] = {
        "python_version": sys.version,
        "platform": sys.platform,
    }

    try:
        import torch

        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda or "N/A"
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        env_info["torch_version"] = "not installed"

    try:
        import xgboost

        env_info["xgboost_version"] = xgboost.__version__
    except ImportError:
        env_info["xgboost_version"] = "not installed"

    for key, value in env_info.items():
        logger.info(f"{key}: {value}")

    return env_info
