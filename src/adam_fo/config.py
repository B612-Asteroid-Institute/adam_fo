"""Configuration and paths for find_orb."""
import os
from pathlib import Path


def get_data_dir() -> Path:
    """Get the XDG data directory for adam_fo."""
    xdg_data_home = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return Path(xdg_data_home) / "adam_fo"


def get_cache_dir() -> Path:
    """Get the XDG cache directory for adam_fo."""
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(xdg_cache_home) / "adam_fo"


def get_find_orb_home() -> Path:
    """Get the find_orb home directory."""
    return get_data_dir() / "find_orb/.find_orb"


# Standard paths
BUILD_DIR = get_data_dir()
FO_BINARY_DIR = BUILD_DIR / "find_orb/find_orb/"
LINUX_JPL_PATH = get_find_orb_home() / "linux_p1550p2650.440t"
BC405_FILENAME = get_find_orb_home() / "bc405.dat"


def check_build_exists() -> None:
    """Check if the build directory exists and raise an error if it doesn't."""
    if not BUILD_DIR.exists():
        raise RuntimeError(
            "Build directory not found. Please run 'build-fo' command first to install find_orb."
        ) 