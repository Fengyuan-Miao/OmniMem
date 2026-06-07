"""Portable project, dataset, and model path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_valid_memgallery(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if (path / "data" / "dialog").is_dir():
            return path
    return None


def default_memgallery_dir() -> Path:
    """Find a local Mem-Gallery checkout without requiring a monorepo layout."""
    env_value = os.getenv("OMNIMEM_MEMGALLERY_DIR")
    candidates = []
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            PROJECT_ROOT / "benchmarks" / "Mem-Gallery",
            PROJECT_ROOT.parent / "benchmark" / "Mem-Gallery",
            Path.cwd() / "benchmarks" / "Mem-Gallery",
            Path.cwd() / "benchmark" / "Mem-Gallery",
        ]
    )
    return _first_valid_memgallery(candidates) or candidates[0]


def require_memgallery_dir(value: str | Path | None = None) -> Path:
    """Resolve and validate a Mem-Gallery dataset directory."""
    path = Path(value).expanduser().resolve() if value else default_memgallery_dir()
    dialog_dir = path / "data" / "dialog"
    if not dialog_dir.is_dir():
        raise FileNotFoundError(
            "Mem-Gallery data was not found. Pass --data-dir, set "
            "OMNIMEM_MEMGALLERY_DIR, or place the dataset at "
            f"{PROJECT_ROOT / 'benchmarks' / 'Mem-Gallery'}. "
            f"Expected directory: {dialog_dir}"
        )
    return path


def _model_default(env_name: str, local_path: str, public_id: str) -> str:
    override = os.getenv(env_name)
    if override:
        return override
    local = Path(local_path)
    return str(local) if local.exists() else public_id


def default_minilm_model() -> str:
    return _model_default(
        "OMNIMEM_MINILM_MODEL",
        "/home/miaofy/models/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


def default_siglip_model() -> str:
    return _model_default(
        "OMNIMEM_SIGLIP_MODEL",
        "/home/miaofy/models/SigLIP-Base-Patch16-384",
        "google/siglip-base-patch16-384",
    )


def default_gme_model() -> str:
    return _model_default(
        "OMNIMEM_GME_MODEL",
        "/home/miaofy/models/GME-Qwen2-VL-2B-Instruct",
        "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    )
