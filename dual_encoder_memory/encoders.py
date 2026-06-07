"""Local MiniLM and SigLIP encoders for dual-encoder memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
from PIL import Image

from omnimem.config import default_minilm_model, default_siglip_model


def _l2_normalize(values: Iterable[float]) -> List[float]:
    arr = np.asarray(list(values), dtype="float32")
    if arr.size == 0:
        return []
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return arr.tolist()
    return (arr / norm).tolist()


def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if hasattr(image, "__array__"):
        return Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")
    raise TypeError(f"Unsupported image input: {type(image)!r}")


class MiniLMTextEncoder:
    """SentenceTransformer wrapper for all-MiniLM-L6-v2 text embeddings."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = "cpu",
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or default_minilm_model()
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=device)
        self.dim: Optional[int] = None

    def encode(self, text: str) -> List[float]:
        embedding = self.model.encode(
            str(text or "")[:8000],
            normalize_embeddings=True,
        )
        values = embedding.tolist()
        self.dim = len(values)
        return values

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        embeddings = self.model.encode(
            [str(text or "")[:8000] for text in texts],
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        results = [embedding.tolist() for embedding in embeddings]
        if results:
            self.dim = len(results[0])
        return results


class SigLIPVisionEncoder:
    """SigLIP image/text tower wrapper for visual retrieval."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = "cpu",
    ):
        self.model_name = model_name or default_siglip_model()
        self.device = device or "cpu"
        self._processor = None
        self._model = None
        self.dim: Optional[int] = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        import torch
        from transformers import SiglipModel, SiglipProcessor

        local_only = Path(self.model_name).expanduser().exists()
        self._processor = SiglipProcessor.from_pretrained(
            self.model_name,
            local_files_only=local_only,
        )
        self._model = SiglipModel.from_pretrained(
            self.model_name,
            local_files_only=local_only,
        ).to(self.device)
        self._model.eval()

    @staticmethod
    def _feature_tensor(outputs: Any) -> Any:
        if hasattr(outputs, "detach"):
            return outputs
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            return hidden[:, 0]
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        return outputs

    def encode_image(self, image: Any) -> List[float]:
        self._load()
        import torch

        pil_image = _to_pil_image(image)
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(self._model, "get_image_features"):
                features = self._feature_tensor(self._model.get_image_features(**inputs))
            else:
                outputs = self._model.vision_model(**inputs)
                features = self._feature_tensor(outputs)
        values = features.detach().cpu().numpy().astype("float32").reshape(-1)
        result = _l2_normalize(values)
        self.dim = len(result)
        return result

    def encode_images(self, images: List[Any]) -> List[List[float]]:
        self._load()
        import torch

        if not images:
            return []
        pil_images = [_to_pil_image(image) for image in images]
        inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(self._model, "get_image_features"):
                features = self._feature_tensor(self._model.get_image_features(**inputs))
            else:
                outputs = self._model.vision_model(**inputs)
                features = self._feature_tensor(outputs)
        arr = features.detach().cpu().numpy().astype("float32")
        results = [_l2_normalize(row) for row in arr]
        if results:
            self.dim = len(results[0])
        return results

    def encode_text(self, text: str) -> List[float]:
        self._load()
        import torch

        inputs = self._processor(
            text=[str(text or "")],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(self._model, "get_text_features"):
                features = self._feature_tensor(self._model.get_text_features(**inputs))
            else:
                outputs = self._model.text_model(**inputs)
                features = self._feature_tensor(outputs)
        values = features.detach().cpu().numpy().astype("float32").reshape(-1)
        result = _l2_normalize(values)
        self.dim = len(result)
        return result
