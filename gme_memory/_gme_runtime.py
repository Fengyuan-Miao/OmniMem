"""Bundled GME-Qwen2-VL inference runtime.

This keeps OmniMem usable when the model checkout does not include a top-level
``gme_inference.py`` on ``sys.path``.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch
from PIL import Image
import transformers


AutoProcessor = transformers.AutoProcessor
if hasattr(transformers, "AutoModelForVision2Seq"):
    AutoModelForVision2Seq = transformers.AutoModelForVision2Seq
elif hasattr(transformers, "AutoModelForImageTextToText"):
    AutoModelForVision2Seq = transformers.AutoModelForImageTextToText
else:
    AutoModelForVision2Seq = transformers.Qwen2VLForConditionalGeneration


class GmeQwen2VL:
    """Minimal inference wrapper matching the public GME helper API."""

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_image_tokens: int = 256,
        max_image_tokens: int = 1280,
        max_length: int = 1800,
        **kwargs: Any,
    ) -> None:
        source = model_path or model_name
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.base = AutoModelForVision2Seq.from_pretrained(
            source,
            torch_dtype=dtype,
            **kwargs,
        )
        self.base.eval()
        self.normalize = True
        self.device = device
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(
            source,
            min_pixels=min_image_tokens * 28 * 28,
            max_pixels=max_image_tokens * 28 * 28,
            **kwargs,
        )
        self.processor.tokenizer.padding_side = "right"
        self.default_instruction = "You are a helpful assistant."

    def forward(
        self,
        input_ids: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        position_ids: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[Any] = None,
        pixel_values: Optional[Any] = None,
        image_grid_thw: Optional[Any] = None,
        pooling_mask: Optional[Any] = None,
        **_: Any,
    ) -> Any:
        if inputs_embeds is None:
            inputs_embeds = self.base.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.base.visual.get_dtype())
                image_embeds = self.base.visual(
                    pixel_values,
                    grid_thw=image_grid_thw,
                ).to(inputs_embeds.device)
                image_mask = input_ids == self.base.config.image_token_id
                inputs_embeds[image_mask] = image_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.base.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        pooling_mask = attention_mask if pooling_mask is None else pooling_mask
        left_padding = pooling_mask[:, -1].sum() == pooling_mask.shape[0]
        if left_padding:
            embeddings = outputs.last_hidden_state[:, -1]
        else:
            sequence_lengths = pooling_mask.sum(dim=1) - 1
            embeddings = outputs.last_hidden_state[
                torch.arange(
                    outputs.last_hidden_state.shape[0],
                    device=outputs.last_hidden_state.device,
                ),
                sequence_lengths,
            ]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def embed(
        self,
        texts: List[Optional[str]],
        images: List[Optional[Image.Image]],
        is_query: bool = True,
        instruction: Optional[str] = None,
        **_: Any,
    ) -> Any:
        self.base.to(self.device)
        input_texts = []
        input_images = []
        has_images = any(image is not None for image in images)
        active_instruction = (
            instruction if is_query and instruction is not None else self.default_instruction
        )
        for text, image in zip(texts, images):
            content = ""
            if image is not None:
                content += "<|vision_start|><|image_pad|><|vision_end|>"
                input_images.append(image.convert("RGB"))
            if text is not None:
                content += text
            input_texts.append(
                f"<|im_start|>system\n{active_instruction}<|im_end|>\n"
                f"<|im_start|>user\n{content}<|im_end|>\n"
                "<|im_start|>assistant\n<|endoftext|>"
            )

        inputs = self.processor(
            text=input_texts,
            images=input_images if has_images else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            return self.forward(**inputs)

    def get_fused_embeddings(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Image.Image]] = None,
        **kwargs: Any,
    ) -> Any:
        batch_size = max(1, int(kwargs.pop("batch_size", 32)))
        if texts is None and images is None:
            raise ValueError("texts or images must be provided")
        item_count = len(texts) if texts is not None else len(images or [])
        outputs = []
        for start in range(0, item_count, batch_size):
            stop = min(item_count, start + batch_size)
            text_batch: List[Optional[str]] = (
                list(texts[start:stop]) if texts is not None else [None] * (stop - start)
            )
            image_batch: List[Optional[Image.Image]] = (
                list(images[start:stop]) if images is not None else [None] * (stop - start)
            )
            outputs.append(
                self.embed(
                    text_batch,
                    image_batch,
                    is_query=kwargs.get("is_query", True),
                    instruction=kwargs.get("instruction"),
                ).cpu()
            )
        return torch.cat(outputs, dim=0)

    def get_text_embeddings(self, texts: List[str], **kwargs: Any) -> Any:
        return self.get_fused_embeddings(texts=texts, **kwargs)

    def get_image_embeddings(self, images: List[Image.Image], **kwargs: Any) -> Any:
        return self.get_fused_embeddings(images=images, **kwargs)
