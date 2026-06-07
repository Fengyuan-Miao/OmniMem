"""OmniMem project utilities and command-line entry points."""

from .config import (
    PROJECT_ROOT,
    default_gme_model,
    default_memgallery_dir,
    default_minilm_model,
    default_siglip_model,
    require_memgallery_dir,
)

__all__ = [
    "PROJECT_ROOT",
    "default_gme_model",
    "default_memgallery_dir",
    "default_minilm_model",
    "default_siglip_model",
    "require_memgallery_dir",
]

__version__ = "0.1.0"
