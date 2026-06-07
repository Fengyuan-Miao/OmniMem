"""GME-Qwen2-VL unified embedding wrapper."""

from __future__ import annotations

import sys
import types
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from omnimem.config import default_gme_model

from .models import GmeEntryEmbedding, GmeMemoryRecord


DEFAULT_QUERY_INSTRUCTION: Optional[str] = None


def l2_normalize(values: Iterable[float]) -> List[float]:
    arr = np.asarray(list(values), dtype="float32")
    if arr.ndim != 1 or arr.size == 0:
        return []
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return arr.tolist()
    return (arr / norm).astype("float32").tolist()


def select_entry_embedding_image(record: GmeMemoryRecord) -> Tuple[Optional[str], str]:
    """Return the first image path and id used for the entry embedding."""
    for image in record.images:
        if image.path:
            return image.path, image.image_id
    return None, ""


def entry_embedding_text(record: GmeMemoryRecord) -> str:
    """Text passed to GME for entry encoding, matching MuRAG observation semantics."""
    value = record.metadata.get("embedding_text") if isinstance(record.metadata, dict) else None
    if value is None:
        value = record.text
    return str(value or "")


class GmeQwen2VLEncoder:
    """Thin wrapper around the local GME inference helper."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:1",
        min_image_tokens: int = 256,
        max_image_tokens: int = 1280,
        max_length: int = 1800,
        query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_path = str(model_path or default_gme_model())
        self.device = device
        self.query_instruction = query_instruction or None
        model_dir = Path(self.model_path)
        if (model_dir / "gme_inference.py").exists() and str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        import transformers

        if not hasattr(transformers, "AutoModelForVision2Seq"):
            if hasattr(transformers, "AutoModelForImageTextToText"):
                transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText
            elif hasattr(transformers, "Qwen2VLForConditionalGeneration"):
                transformers.AutoModelForVision2Seq = transformers.Qwen2VLForConditionalGeneration
        try:
            from gme_inference import GmeQwen2VL
        except ImportError:
            from ._gme_runtime import GmeQwen2VL

        self.model = GmeQwen2VL(
            model_path=self.model_path,
            device=device,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )
        self._patch_transformers5_forward_if_needed()

    def _patch_transformers5_forward_if_needed(self) -> None:
        if hasattr(self.model.base.model, "embed_tokens"):
            return

        def forward_compat(
            gme_self: Any,
            input_ids: Optional[Any] = None,
            attention_mask: Optional[Any] = None,
            position_ids: Optional[Any] = None,
            past_key_values: Optional[Any] = None,
            inputs_embeds: Optional[Any] = None,
            pixel_values: Optional[Any] = None,
            image_grid_thw: Optional[Any] = None,
            pooling_mask: Optional[Any] = None,
            **kwargs: Any,
        ) -> Any:
            outputs = gme_self.base.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=kwargs.get("mm_token_type_ids"),
                return_dict=True,
            )
            pooling_mask = attention_mask if pooling_mask is None else pooling_mask
            left_padding = pooling_mask[:, -1].sum() == pooling_mask.shape[0]
            if left_padding:
                embeddings = outputs.last_hidden_state[:, -1]
            else:
                import torch

                sequence_lengths = pooling_mask.sum(dim=1) - 1
                batch_size = outputs.last_hidden_state.shape[0]
                embeddings = outputs.last_hidden_state[
                    torch.arange(batch_size, device=outputs.last_hidden_state.device),
                    sequence_lengths.to(outputs.last_hidden_state.device),
                ]
            if gme_self.normalize:
                import torch

                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.contiguous()

        self.model.forward = types.MethodType(forward_compat, self.model)

    @staticmethod
    def _load_image(image_path_or_url: str | Image.Image | None) -> Image.Image:
        if isinstance(image_path_or_url, Image.Image):
            return image_path_or_url.convert("RGB")
        value = str(image_path_or_url or "")
        try:
            if value.startswith("http://") or value.startswith("https://"):
                with urllib.request.urlopen(value, timeout=10) as response:
                    return Image.open(BytesIO(response.read())).convert("RGB")
            if value.startswith("file://"):
                value = value[7:]
            if not value or not Path(value).exists():
                raise FileNotFoundError(value)
            return Image.open(value).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), color="white")

    @staticmethod
    def _vectors(tensor: Any) -> List[List[float]]:
        if hasattr(tensor, "detach"):
            array = tensor.detach().cpu().float().numpy()
        else:
            array = np.asarray(tensor, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return [l2_normalize(row.tolist()) for row in array]

    def encode_entries(
        self,
        records: List[GmeMemoryRecord],
        batch_size: int = 4,
        show_progress_bar: bool = False,
    ) -> List[GmeEntryEmbedding]:
        outputs: List[Optional[GmeEntryEmbedding]] = [None] * len(records)
        text_items = []
        image_items = []
        fused_items = []
        for index, record in enumerate(records):
            image_path, image_id = select_entry_embedding_image(record)
            text = entry_embedding_text(record)
            has_text = bool(text.strip())
            if image_path and has_text:
                fused_items.append((index, record, text, image_path, image_id))
            elif image_path:
                image_items.append((index, record, image_path, image_id))
            else:
                text_items.append((index, record, text if has_text else " "))

        for start in range(0, len(text_items), batch_size):
            batch = text_items[start : start + batch_size]
            vectors = self._vectors(
                self.model.get_text_embeddings(
                    texts=[item[2] for item in batch],
                    is_query=False,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                )
            )
            for item, vector in zip(batch, vectors):
                index, record, _text = item
                outputs[index] = GmeEntryEmbedding(
                    memory_id=record.memory_id,
                    vector=vector,
                    embedding_mode="text",
                )

        for start in range(0, len(image_items), batch_size):
            batch = image_items[start : start + batch_size]
            vectors = self._vectors(
                self.model.get_image_embeddings(
                    images=[self._load_image(item[2]) for item in batch],
                    is_query=False,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                )
            )
            for item, vector in zip(batch, vectors):
                index, record, image_path, image_id = item
                outputs[index] = GmeEntryEmbedding(
                    memory_id=record.memory_id,
                    vector=vector,
                    embedding_mode="image",
                    image_id=image_id,
                    image_path=image_path,
                )

        for start in range(0, len(fused_items), batch_size):
            batch = fused_items[start : start + batch_size]
            vectors = self._vectors(
                self.model.get_fused_embeddings(
                    texts=[item[2] for item in batch],
                    images=[self._load_image(item[3]) for item in batch],
                    is_query=False,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                )
            )
            for item, vector in zip(batch, vectors):
                index, record, _text, image_path, image_id = item
                outputs[index] = GmeEntryEmbedding(
                    memory_id=record.memory_id,
                    vector=vector,
                    embedding_mode="image_text_pair",
                    image_id=image_id,
                    image_path=image_path,
                )

        return [item for item in outputs if item is not None]

    def encode_query(
        self,
        query: str,
        question_image: Optional[str] = None,
        batch_size: int = 1,
    ) -> Tuple[List[float], str]:
        query_text = query if str(query or "").strip() else " "
        instruction_kwargs = (
            {"instruction": self.query_instruction} if self.query_instruction else {}
        )
        if question_image:
            image = self._load_image(question_image)
            if str(query or "").strip():
                tensor = self.model.get_fused_embeddings(
                    texts=[query_text],
                    images=[image],
                    batch_size=batch_size,
                    show_progress_bar=False,
                    **instruction_kwargs,
                )
                return self._vectors(tensor)[0], "query_image_text_pair"
            tensor = self.model.get_image_embeddings(
                images=[image],
                batch_size=batch_size,
                show_progress_bar=False,
                **instruction_kwargs,
            )
            return self._vectors(tensor)[0], "query_image"
        tensor = self.model.get_text_embeddings(
            texts=[query_text],
            batch_size=batch_size,
            show_progress_bar=False,
            **instruction_kwargs,
        )
        return self._vectors(tensor)[0], "query_text"
