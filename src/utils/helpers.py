"""Shared helper utilities used by training and inference modules."""

from pathlib import Path


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
