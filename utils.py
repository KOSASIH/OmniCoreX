"""
OmniCoreX Utilities Module

Helper functions, logging setup, configuration parsing,
and common utilities used throughout the OmniCoreX system.

Features:
- Robust logging setup with configurable formats and levels.
- Configuration loader supporting YAML and JSON with overrides.
- Seed setting for reproducibility.
- Timing and benchmarking decorators.
- Various small utilities for system use.
"""

import os
import sys
import yaml
import json
import logging
import random
import time
import numpy as np
import torch

# ----------------------- Logging Setup ----------------------- #

def setup_logging(log_level=logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Sets up a logger with console and optional file handlers.

    Args:
        log_level: Logging level (e.g., logging.INFO).
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("OmniCoreX")
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# Global logger instance
logger = setup_logging()

# ----------------------- Configuration Loading ----------------------- #

def load_config_file(config_path: str) -> dict:
    """
    Loads a YAML or JSON configuration file.

    Args:
        config_path: Path to the config file.

    Returns:
        Dictionary of configuration parameters.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            cfg = yaml.safe_load(f)
        elif ext == ".json":
            cfg = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    return cfg

def merge_dicts(base: dict, override: dict) -> dict:
    """
    Deep merges two dictionaries, with the override taking precedence.

    Args:
        base: Base dictionary.
        override: Dictionary with override values.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result

# ----------------------- Seed Setting ----------------------- #

def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy and torch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

# ----------------------- Timing Utilities ----------------------- #

def timeit(func):
    """
    Decorator to measure and log function execution time.

    Usage:
        @timeit
        def my_function(...):
            ...
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Function {func.__name__!r} executed in {(end - start):.4f}s")
        return result
    return wrapper

# ----------------------- Other Utility Functions ----------------------- #

def ensure_dir(dirname: str):
    """
    Creates directory if it does not exist.

    Args:
        dirname: Directory path to create.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        logger.debug(f"Directory created: {dirname}")

def to_device(batch: dict, device: torch.device) -> dict:
    """
    Moves all tensor elements in batch dict to specified device.

    Args:
        batch: Dictionary with tensors.
        device: Target torch device.

    Returns:
        Dictionary with tensors on device.
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

if __name__ == "__main__":
    # Demo usage of utilities

    set_seed(1234)
    logger.info("This is a test log message.")

    # Create dummy config files and test merging
    base_cfg = {"model": {"layers": 12, "embed_dim": 256}, "training": {"batch_size": 32}}
    override_cfg = {"model": {"layers": 24}, "training": {"learning_rate": 0.001}}

    merged_cfg = merge_dicts(base_cfg, override_cfg)
    logger.info(f"Merged config: {merged_cfg}")

    # Test directory creation
    test_dir = "./tmp_test_dir"
    ensure_dir(test_dir)

    # Test timing decorator
    @timeit
    def dummy_work():
        import time; time.sleep(0.5)

    dummy_work()

