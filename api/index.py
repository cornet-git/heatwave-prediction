"""
api/index.py — Vercel Serverless Function entrypoint.
Exposes the FastAPI application instance `app`.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path so that `src` modules resolve correctly
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.api import app
