"""Small utilities shared by the SVI implementation."""

from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_iso_time(value: Optional[Any]) -> str:
    if value is None:
        return utcnow_iso()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    return str(value)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def tokenize(text: Any) -> List[str]:
    normalized = normalize_text(text)
    # Keep CJK spans and latin/number terms.
    return re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalized)


def unique_list(values: Iterable[Any], max_items: Optional[int] = None) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        item = str(value).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if max_items is not None and len(result) >= max_items:
            break
    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_first_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    candidate = text.strip()
    if "```" in candidate:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate)
        if match:
            block_obj = extract_first_json_object(match.group(1))
            if block_obj is not None:
                return block_obj

    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(candidate[start:], start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_first_json_value(text: str) -> Optional[Any]:
    """Extract the first JSON object or array from a model response."""
    obj = extract_first_json_object(text)
    if obj is not None:
        return obj
    if not text:
        return None
    candidate = text.strip()
    if "```" in candidate:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate)
        if match:
            block_value = extract_first_json_value(match.group(1))
            if block_value is not None:
                return block_value

    start = candidate.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(candidate[start:], start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def image_to_base64_jpeg(image: Any) -> str:
    from PIL import Image

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, bytes):
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
    elif isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    elif hasattr(image, "__array__"):
        import numpy as np

        pil_image = Image.fromarray(np.array(image)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image type for base64 conversion: {type(image)}")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def rough_token_trim(text: str, max_tokens: int) -> str:
    max_chars = max(1, max_tokens) * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
