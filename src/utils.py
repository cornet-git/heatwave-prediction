"""
utils.py — Shared helpers: logging configuration and project-level path constants.
"""

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (one level above this file's directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

try:
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
except OSError:
    pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
