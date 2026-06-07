"""One-pass structured visual extraction for SVI-OmniMem."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import SVIConfig
from .models import (
    IndexedAttribute,
    OCRObservation,
    RetrievalAnchor,
    StructuredVisualCard,
)
from .utils import (
    extract_first_json_object,
    image_to_base64_jpeg,
    rough_token_trim,
    tokenize,
    unique_list,
)

logger = logging.getLogger(__name__)


class StructuredVisualExtractor:
    """Extract compact searchable visual fields from one image.

    The output is intentionally unverified. It is a coarse retrieval index that
    points back to the retained raw image.
    """

    def __init__(self, orchestrator: Any, config: Optional[SVIConfig] = None):
        self.orchestrator = orchestrator
        self.config = config or SVIConfig()

    def extract(
        self,
        image: Any,
        image_mau_id: str,
        raw_pointer: str,
        global_caption: str,
        text_context: Optional[str] = None,
        timestamp: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> StructuredVisualCard:
        card = StructuredVisualCard.new(
            image_mau_id=image_mau_id,
            raw_pointer=raw_pointer,
            global_caption=global_caption,
            session_id=session_id,
            turn_id=turn_id,
            observed_at=timestamp,
            tags=tags or [],
            source_text_context=rough_token_trim(text_context or "", 260),
            schema_version=self.config.schema_version,
        )

        data = self._call_extractor(image, global_caption, text_context, card.observed_at)
        if not data:
            self._populate_fallback_card(card, global_caption, text_context)
            return card

        parsed = self._card_from_payload(
            data=data,
            image_mau_id=image_mau_id,
            raw_pointer=raw_pointer,
            fallback_caption=global_caption,
            session_id=session_id,
            turn_id=turn_id,
            timestamp=card.observed_at,
            tags=tags or [],
            source_text_context=rough_token_trim(text_context or "", 260),
        )
        if self._is_empty_card(parsed):
            self._populate_fallback_card(parsed, global_caption, text_context)
        return parsed

    def _call_extractor(
        self,
        image: Any,
        global_caption: str,
        text_context: Optional[str],
        timestamp: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            base64_image = image_to_base64_jpeg(image)
        except Exception as exc:
            logger.warning("SVI image encoding failed: %s", exc)
            return None

        try:
            client = self.orchestrator._get_llm_client()
            model = getattr(self.orchestrator.config.llm, "caption_model", None)
            max_tokens = max(900, self.config.extraction_budget.max_card_tokens * 4)
            prompt = self._build_prompt(global_caption, text_context, timestamp)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]
            response = self._create_json_completion(
                client=client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("SVI structured extraction failed: %s", exc)
            return None

        payload = extract_first_json_object(text)
        if not payload:
            payload = self._repair_non_json_response(
                client=client,
                model=model,
                raw_text=text,
                global_caption=global_caption,
                text_context=text_context,
                timestamp=timestamp,
                max_tokens=max_tokens,
            )
        if not payload:
            logger.warning(
                "SVI extractor returned non-JSON response: %s",
                rough_token_trim(text.replace("\n", " "), 80),
            )
        return payload

    def _repair_non_json_response(
        self,
        client: Any,
        model: str,
        raw_text: str,
        global_caption: str,
        text_context: Optional[str],
        timestamp: str,
        max_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        repair_prompt = f"""Convert the text below into ONE valid JSON object for a visual retrieval index.

Use only information present in the text, caption, or dialogue context. If unsure, produce a compact caption-level index.
Do not include markdown.

Required keys:
- global_caption: string
- retrieval_anchors: list
- ocr_observations: list

Caption: {rough_token_trim(global_caption, 80)}
Dialogue context: {rough_token_trim(text_context or "", 120)}
Observation timestamp: {timestamp}

Text to convert:
{rough_token_trim(raw_text, 500)}
"""
        messages = [{"role": "user", "content": repair_prompt}]
        try:
            response = self._create_json_completion(
                client=client,
                model=model,
                messages=messages,
                max_tokens=min(max_tokens, 700),
            )
        except Exception as exc:
            logger.debug("SVI JSON repair failed: %s", exc)
            return None
        return extract_first_json_object(response.choices[0].message.content or "")

    def _create_json_completion(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
    ) -> Any:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            return client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.debug("SVI JSON-mode extraction unavailable, retrying plain: %s", exc)
            return client.chat.completions.create(**kwargs)

    def _is_empty_card(self, card: StructuredVisualCard) -> bool:
        return not (
            card.retrieval_anchors
            or card.ocr_observations
        )

    def _populate_fallback_card(
        self,
        card: StructuredVisualCard,
        global_caption: str,
        text_context: Optional[str],
    ) -> None:
        seed_text = " ".join(part for part in [global_caption, text_context or ""] if part)
        keywords = self._fallback_keywords(seed_text)
        if not keywords:
            keywords = ["image"]

        card.extraction_scope = "caption_dialogue_fallback"
        if global_caption and global_caption != "Image captured":
            card.global_caption = global_caption

        card.retrieval_anchors = [
            RetrievalAnchor(
                anchor_id="fallback_caption",
                category=keywords[0],
                salient_attributes={},
            )
        ]

    def _fallback_keywords(self, text: str) -> List[str]:
        stop = {
            "the",
            "and",
            "with",
            "for",
            "that",
            "this",
            "from",
            "into",
            "showing",
            "image",
            "caption",
            "assistant",
            "user",
            "a",
            "an",
            "of",
            "to",
            "in",
            "on",
            "is",
            "are",
            "was",
            "were",
        }
        return unique_list(
            token for token in tokenize(text) if token not in stop and len(token) > 1
        )[:8]

    def _build_prompt(
        self,
        global_caption: str,
        text_context: Optional[str],
        timestamp: str,
    ) -> str:
        budget = self.config.extraction_budget
        context_text = text_context or ""
        return f"""Return ONLY one minified JSON object. No markdown, no prose.

Build a compact searchable visual memory index. The original image is retained for later verification, so this JSON is only a retrieval hint.

Rules:
- Do not infer hidden facts.
- Do not claim permanence or current truth; record observations at the supplied timestamp only.
- Keep values short.
- Prefer brands, logos, readable text, distinctive colors, counts, spatial relations.
- Strict hard budgets:
  anchors<={budget.max_retrieval_anchors}
  attributes<={budget.max_retrieval_anchors * budget.max_attributes_per_anchor}
  ocr<={budget.max_ocr_observations}

Existing short caption: {rough_token_trim(global_caption, 80)}
Dialogue context: {rough_token_trim(context_text, 160)}
Observation timestamp: {timestamp}

Use this SHORT schema exactly:
{{"global_caption":"short caption","anchors":["brand","object","logo"],"attributes":["brand.color=red","object.count=2"],"ocr":["readable text"]}}"""

    def _card_from_payload(
        self,
        data: Dict[str, Any],
        image_mau_id: str,
        raw_pointer: str,
        fallback_caption: str,
        session_id: Optional[str],
        turn_id: Optional[str],
        timestamp: str,
        tags: List[str],
        source_text_context: str = "",
    ) -> StructuredVisualCard:
        card = StructuredVisualCard.new(
            image_mau_id=image_mau_id,
            raw_pointer=raw_pointer,
            global_caption=str(data.get("global_caption") or fallback_caption),
            session_id=session_id,
            turn_id=turn_id,
            observed_at=timestamp,
            tags=tags,
            source_text_context=source_text_context,
            schema_version=self.config.schema_version,
        )

        data = self._normalize_payload_shape(data)
        card.retrieval_anchors = self._parse_anchors(data.get("retrieval_anchors") or [])
        card.ocr_observations = self._parse_ocr(data.get("ocr_observations") or [])
        return card

    def _normalize_payload_shape(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get("retrieval_anchors") is not None:
            return data

        normalized = dict(data)
        anchors = []
        for index, item in enumerate(data.get("anchors") or [], start=1):
            if isinstance(item, str):
                anchors.append(
                    {
                        "anchor_id": f"a{index}",
                        "category": item,
                        "salient_attributes": {},
                    }
                )
            elif isinstance(item, dict):
                anchors.append(
                    {
                        "anchor_id": str(item.get("id") or item.get("anchor_id") or f"a{index}"),
                        "category": str(item.get("category") or item.get("name") or "object"),
                        "salient_attributes": {},
                    }
                )

        anchor_by_category = {
            str(anchor.get("category", "")).lower(): anchor for anchor in anchors
        }
        for raw_attr in data.get("attributes") or []:
            if not isinstance(raw_attr, str) or "=" not in raw_attr:
                continue
            left, value = raw_attr.split("=", 1)
            if "." in left:
                subject, attr_name = left.split(".", 1)
            else:
                subject = anchors[0]["category"] if anchors else "image"
                attr_name = left
            subject = subject.strip() or "image"
            attr_name = attr_name.strip() or "attribute"
            anchor = anchor_by_category.get(subject.lower())
            if anchor is None:
                anchor = {
                    "anchor_id": f"a{len(anchors) + 1}",
                    "category": subject,
                    "salient_attributes": {},
                }
                anchors.append(anchor)
                anchor_by_category[subject.lower()] = anchor
            anchor["salient_attributes"][attr_name] = {
                "value": value.strip(),
            }

        normalized["retrieval_anchors"] = anchors
        normalized["ocr_observations"] = [
            {"text": str(item)}
            for item in data.get("ocr") or []
            if str(item).strip()
        ]
        return normalized

    def _parse_anchors(self, raw_items: List[Any]) -> List[RetrievalAnchor]:
        budget = self.config.extraction_budget
        anchors: List[RetrievalAnchor] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            anchor = RetrievalAnchor.from_dict(item)
            filtered_attrs: Dict[str, IndexedAttribute] = {}
            for name, attr in anchor.salient_attributes.items():
                if attr.value:
                    filtered_attrs[str(name)] = attr
                if len(filtered_attrs) >= budget.max_attributes_per_anchor:
                    break
            anchor.salient_attributes = filtered_attrs
            anchors.append(anchor)
            if len(anchors) >= budget.max_retrieval_anchors:
                break
        return anchors

    def _parse_ocr(self, raw_items: List[Any]) -> List[OCRObservation]:
        budget = self.config.extraction_budget
        observations: List[OCRObservation] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            obs = OCRObservation.from_dict(item)
            if not obs.text.strip():
                continue
            observations.append(obs)
            if len(observations) >= budget.max_ocr_observations:
                break
        return observations
