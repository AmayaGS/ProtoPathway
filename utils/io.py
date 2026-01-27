"""
I/O utilities for file handling.
"""

import os
import logging


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logging.debug(f"Created directory: {path}")


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))