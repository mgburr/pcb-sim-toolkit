"""Resolve paths to bundled resources in both dev and PyInstaller-packaged modes."""

import sys
from pathlib import Path


def get_base_path() -> Path:
    """Return the base path for resolving bundled resources.

    When running from a PyInstaller bundle, returns sys._MEIPASS.
    Otherwise, returns the project root directory.
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent


def get_examples_dir() -> Path:
    """Return path to the examples directory."""
    return get_base_path() / "examples"


def get_configs_dir() -> Path:
    """Return path to the configs directory."""
    return get_base_path() / "configs"
