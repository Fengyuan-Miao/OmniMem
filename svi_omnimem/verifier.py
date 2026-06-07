"""Raw-image evidence verification for SVI-OmniMem."""

from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import SVIConfig
from .models import RetrievalCandidate, StructuredVisualCard, VerificationResult, VisualQueryRequirement
from .stores import StructuredVisualStore
from .utils import extract_first_json_object, extract_first_json_value, safe_float

logger = logging.getLogger(__name__)


class RawEvidenceVerifier:
    """Verify candidate cards by reopening retained raw images.

    The preferred path verifies a set of candidate images in one VLM call. This
    keeps QA-time compute light while letting the verifier return multiple
    supporting visual memories instead of stopping at a single top candidate.
    """

    def __init__(
        self,
        orchestrator: Any,
        visual_store: StructuredVisualStore,
        config: Optional[SVIConfig] = None,
    ):
        self.orchestrator = orchestrator
        self.visual_store = visual_store
        self.config = config or SVIConfig()

    def verify(
        self,
        query: str,
        requirement: VisualQueryRequirement,
        candidates: List[RetrievalCandidate],
        budget: int,
    ) -> List[VerificationResult]:
        if budget <= 0:
            return []
        selected = candidates[:budget]
        if not selected:
            return []

        if self.config.batch_verification:
            batch_results = self.verify_batch(query, requirement, selected)
            if batch_results is not None:
                return batch_results
            return self._abstain_batch_results(selected, "batch_verifier_failed")

        results: List[VerificationResult] = []
        for candidate in selected:
            card = self.visual_store.get_by_card_id(candidate.card_id)
            if not card:
                continue
            results.append(self.verify_card(query, requirement, card))
        return results

    def _abstain_batch_results(
        self,
        candidates: List[RetrievalCandidate],
        error: str,
    ) -> List[VerificationResult]:
        results: List[VerificationResult] = []
        for candidate in candidates:
            card = self.visual_store.get_by_card_id(candidate.card_id)
            if not card:
                continue
            results.append(
                VerificationResult(
                    supports=False,
                    source_card_id=card.card_id,
                    source_image_mau_id=card.image_mau_id,
                    raw_pointer=card.raw_pointer,
                    observation_time=card.observed_at,
                    error=error,
                    abstained=True,
                )
            )
        return results

    def verify_batch(
        self,
        query: str,
        requirement: VisualQueryRequirement,
        candidates: List[RetrievalCandidate],
    ) -> Optional[List[VerificationResult]]:
        prepared: List[Tuple[int, RetrievalCandidate, StructuredVisualCard, str]] = []
        early_results: List[VerificationResult] = []
        for index, candidate in enumerate(candidates, start=1):
            card = self.visual_store.get_by_card_id(candidate.card_id)
            if not card:
                continue
            raw_bytes = self.orchestrator.cold_storage.retrieve(card.raw_pointer)
            if not raw_bytes:
                early_results.append(
                    VerificationResult(
                        supports=False,
                        source_card_id=card.card_id,
                        source_image_mau_id=card.image_mau_id,
                        raw_pointer=card.raw_pointer,
                        observation_time=card.observed_at,
                        error="raw_pointer_missing",
                        abstained=True,
                    )
                )
                continue
            image_b64 = base64.b64encode(raw_bytes).decode("utf-8")
            prepared.append((index, candidate, card, image_b64))

        if not prepared:
            return early_results

        prompt = self._build_batch_prompt(query, requirement, prepared)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, candidate, card, image_b64 in prepared:
            public_id = self._public_image_id(card) or card.card_id
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Candidate {index} metadata:\n"
                        f"public_image_id: {public_id}\n"
                        f"card_id: {card.card_id}\n"
                        f"observed_at: {card.observed_at}\n"
                        f"retrieval_score: {candidate.score:.4f}\n"
                        "structured_hints_not_evidence:\n"
                        f"{card.to_mirror_text()}"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )

        client = self.orchestrator._get_llm_client()
        model = getattr(self.orchestrator.config.llm, "caption_model", None)
        try:
            response = self._create_json_completion(
                client=client,
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=900,
            )
            text = response.choices[0].message.content or ""
            data = self._coerce_verification_json(extract_first_json_value(text))
            if not data:
                logger.debug("SVI batch verifier returned non-JSON: %s", text[:500])
                return self._verify_batch_compact(query, requirement, prepared, early_results)
            results = self._parse_batch_results(data, prepared, early_results)
            missing_count = sum(
                1 for item in results if item.error == "batch_verifier_missing_result"
            )
            if missing_count == len(prepared):
                logger.debug(
                    "SVI batch verifier returned no alignable results; retrying compact batch verification"
                )
                return self._verify_batch_compact(query, requirement, prepared, early_results)
            return results
        except Exception as exc:
            logger.warning("SVI batch verification failed: %s", exc)
            return None

    def _verify_batch_compact(
        self,
        query: str,
        requirement: VisualQueryRequirement,
        prepared: List[Tuple[int, RetrievalCandidate, StructuredVisualCard, str]],
        early_results: List[VerificationResult],
    ) -> Optional[List[VerificationResult]]:
        prompt = self._build_batch_prompt(query, requirement, prepared)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, _candidate, _card, image_b64 in prepared:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Candidate image {index}. Return this exact candidate_index "
                        f"for the next image."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )

        client = self.orchestrator._get_llm_client()
        model = getattr(self.orchestrator.config.llm, "caption_model", None)
        try:
            response = self._create_json_completion(
                client=client,
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=900,
            )
            text = response.choices[0].message.content or ""
            data = self._coerce_verification_json(extract_first_json_value(text))
            if not data:
                return None
            return self._parse_batch_results(data, prepared, early_results)
        except Exception as exc:
            logger.warning("SVI compact batch verification failed: %s", exc)
            return None

    def _parse_batch_results(
        self,
        data: Dict[str, Any],
        prepared: List[Tuple[int, RetrievalCandidate, StructuredVisualCard, str]],
        early_results: List[VerificationResult],
    ) -> List[VerificationResult]:
        raw_results = data.get("results") or data.get("candidates") or data.get("verifications")
        if raw_results is None and "supports" in data:
            raw_results = [data]
        if not isinstance(raw_results, list):
            raw_results = []

        by_index: Dict[int, Dict[str, Any]] = {}
        by_card_id: Dict[str, Dict[str, Any]] = {}
        by_public_id: Dict[str, Dict[str, Any]] = {}
        sequential_items: List[Dict[str, Any]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            sequential_items.append(item)
            candidate_index = (
                item.get("candidate_index")
                or item.get("candidate_id")
                or item.get("index")
                or item.get("candidate")
                or item.get("id")
            )
            if isinstance(candidate_index, str):
                match = re.search(r"\d+", candidate_index)
                candidate_index = match.group(0) if match else candidate_index
            try:
                by_index[int(candidate_index)] = item
            except (TypeError, ValueError):
                pass
            card_id = str(item.get("card_id") or "").strip()
            if card_id:
                by_card_id[card_id] = item
            public_id = str(
                item.get("public_image_id")
                or item.get("image_id")
                or item.get("source_image_id")
                or ""
            ).strip()
            if public_id:
                by_public_id[public_id] = item

        results = list(early_results)
        for seq_pos, (index, _candidate, card, _image_b64) in enumerate(prepared):
            public_id = self._public_image_id(card) or ""
            item = (
                by_index.get(index)
                or by_card_id.get(card.card_id)
                or by_public_id.get(public_id)
            )
            if item is None and len(sequential_items) == len(prepared):
                item = sequential_items[seq_pos]
            if item is None:
                results.append(
                    VerificationResult(
                        supports=False,
                        source_card_id=card.card_id,
                        source_image_mau_id=card.image_mau_id,
                        raw_pointer=card.raw_pointer,
                        observation_time=card.observed_at,
                        error="batch_verifier_missing_result",
                        abstained=True,
                    )
                )
                continue
            result = VerificationResult.from_dict(
                item,
                source_card_id=card.card_id,
                source_image_mau_id=card.image_mau_id,
                raw_pointer=card.raw_pointer,
                observation_time=card.observed_at,
            )
            result.confidence = safe_float(item.get("confidence"), result.confidence)
            self._apply_consistency_guard(result)
            if not result.supports:
                result.answer_fragment = ""
                result.visible_evidence = ""
                result.verified_facts = []
            results.append(result)
        return results

    def verify_card(
        self,
        query: str,
        requirement: VisualQueryRequirement,
        card: StructuredVisualCard,
    ) -> VerificationResult:
        raw_bytes = self.orchestrator.cold_storage.retrieve(card.raw_pointer)
        if not raw_bytes:
            return VerificationResult(
                supports=False,
                source_card_id=card.card_id,
                source_image_mau_id=card.image_mau_id,
                raw_pointer=card.raw_pointer,
                observation_time=card.observed_at,
                error="raw_pointer_missing",
                abstained=True,
            )

        image_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        prompt = self._build_prompt(query, requirement, card)
        client = self.orchestrator._get_llm_client()
        model = getattr(self.orchestrator.config.llm, "caption_model", None)

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ]
            response = self._create_json_completion(
                client=client,
                model=model,
                messages=messages,
                max_tokens=500,
            )
            text = response.choices[0].message.content or ""
            data = extract_first_json_object(text)
            if not data:
                return VerificationResult(
                    supports=False,
                    source_card_id=card.card_id,
                    source_image_mau_id=card.image_mau_id,
                    raw_pointer=card.raw_pointer,
                    observation_time=card.observed_at,
                    error="verifier_non_json",
                    abstained=True,
                )
            result = VerificationResult.from_dict(
                data,
                source_card_id=card.card_id,
                source_image_mau_id=card.image_mau_id,
                raw_pointer=card.raw_pointer,
                observation_time=card.observed_at,
            )
            self._apply_consistency_guard(result)
            if not result.supports:
                result.answer_fragment = ""
                result.visible_evidence = ""
                result.verified_facts = []
            return result
        except Exception as exc:
            logger.warning("SVI raw verification failed: %s", exc)
            return VerificationResult(
                supports=False,
                source_card_id=card.card_id,
                source_image_mau_id=card.image_mau_id,
                raw_pointer=card.raw_pointer,
                observation_time=card.observed_at,
                error=str(exc),
                abstained=True,
            )

    def _coerce_verification_json(self, value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"results": value}
        return None

    def _apply_consistency_guard(self, result: VerificationResult) -> None:
        """Reject verifier outputs that contradict their own support decision."""
        text = f"{result.answer_fragment} {result.visible_evidence}".lower()
        if not result.supports:
            positive_markers = [
                "more similar to",
                "matches",
                "consistent with",
                "same as",
                "resembles",
                "related to",
                "shares",
                "same salient",
                "same logo",
                "same product",
                "same person",
                "same place",
                "same scene",
                "same activity",
                "same visual concept",
            ]
            contradiction_markers = [
                "does not match",
                "doesn't match",
                "do not match",
                "not match",
                "mismatch",
                "not consistent",
                "cannot determine",
                "not enough evidence",
                "insufficient evidence",
            ]
            if (
                any(marker in text for marker in positive_markers)
                and not any(marker in text for marker in contradiction_markers)
            ):
                result.supports = True
                result.error = None
            return
        contradiction_markers = [
            "does not show",
            "doesn't show",
            "do not show",
            "not show",
            "does not contain",
            "doesn't contain",
            "do not contain",
            "not contain",
            "does not match",
            "doesn't match",
            "do not match",
            "not match",
            "mismatch",
            "not consistent",
            "cannot determine",
            "not enough evidence",
            "insufficient evidence",
        ]
        if any(marker in text for marker in contradiction_markers):
            result.supports = False
            result.abstained = True
            result.error = "verifier_self_contradiction"

    def _create_json_completion(
        self,
        client: Any,
        model: str,
        messages: List[dict],
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
            logger.debug("SVI JSON-mode verification unavailable, retrying plain: %s", exc)
            return client.chat.completions.create(**kwargs)

    def _public_image_id(self, card: StructuredVisualCard) -> Optional[str]:
        for tag in card.tags:
            if tag.startswith("image_id:"):
                return tag.split(":", 1)[1]
        return None

    def _build_batch_prompt(
        self,
        query: str,
        requirement: VisualQueryRequirement,
        prepared: List[Tuple[int, RetrievalCandidate, StructuredVisualCard, str]],
    ) -> str:
        candidate_lines = []
        for index, _candidate, card, _image_b64 in prepared:
            public_id = self._public_image_id(card) or card.card_id
            candidate_lines.append(
                f"- candidate_index={index}; public_image_id={public_id}; "
                f"card_id={card.card_id}; observed_at={card.observed_at}"
            )
        candidate_list = "\n".join(candidate_lines)
        return f"""You are verifying a set of candidate visual memories against one user question.

Question: {query}
Query type: {requirement.query_type}
Candidate identifiers:
{candidate_list}

General verification policy:
- Inspect each provided image independently. The structured hints are retrieval hints, not evidence.
- Return one result for every candidate image. Do not stop after the first match.
- Every result must include the exact candidate_index, public_image_id, and card_id from the candidate identifiers list.
- If multiple candidate images visibly satisfy the question, mark all of them as supports=true.
- A candidate supports the question only when visible image content is sufficient evidence for the visual part of the answer or for identifying the requested image.
- Do not use the image to answer conversational, temporal, causal, preference, contradiction, or discussion-history questions. For those, return supports=false unless the question explicitly asks to find or inspect an image.
- If the question asks what a user/person mentioned, said, discussed, asked, chose, preferred, decided, remembered, or did earlier/later in the conversation, visible pixels alone cannot prove that fact. Mark supports=false even when the image contains a visually related object, brand, or scene.
- For comparison questions involving a current question image description, compare each candidate image to that current image description. Do not mark a candidate as supporting merely because it matches its own retrieval hint or a named option.
- If the question asks to find a remembered image that relates to a current question image description, support can be semantic visual relatedness: the same salient entity, object type, logo, product, place, person, scene, activity, or visual concept. It does not need to be the identical photo or composition.
- The answer_fragment must be an extractive visual evidence fragment, not a persuasive caption, not a story, and not a final conversational answer.
- Do not infer brands, names, counts, or relations that are not visually supported.
- Return concise JSON only.

Return this JSON shape:
{{
  "results": [
    {{
      "candidate_index": 1,
      "public_image_id": "image id from candidate identifiers",
      "card_id": "card id from candidate identifiers",
      "supports": true,
      "answer_fragment": "short answer grounded in this image",
      "visible_evidence": "what visible evidence supports the answer",
      "verified_facts": [
        {{
          "subject": "object or entity",
          "predicate": "attribute/relation/text/count/state",
          "value": "verified value",
          "evidence_description": "visible evidence",
          "evidence_scope": "full_image"
        }}
      ],
      "confidence": 0.0,
      "abstained": false
    }}
  ]
}}"""

    def _build_prompt(
        self,
        query: str,
        requirement: VisualQueryRequirement,
        card: StructuredVisualCard,
    ) -> str:
        return f"""You are verifying a visual memory question against the original image.

Question: {query}
Query type: {requirement.query_type}
Structured retrieval hints, not evidence:
{card.to_mirror_text()}

Rules:
- Inspect only the provided image.
- If the image does not contain enough evidence, abstain.
- Do not rely on the retrieval hints as final truth.
- If the question asks about conversation history, text-only facts, order, contradiction, preference, or whether something was mentioned/said/discussed/chosen, an image alone cannot prove it. Return supports=false unless the question explicitly asks to inspect or identify the image.
- For comparison questions involving a current question image description, compare the candidate image to that current image description instead of trusting retrieval hints.
- If the question asks to find a remembered image related to a current question image description, support can be semantic visual relatedness through a shared salient entity, object type, logo, product, place, person, scene, activity, or visual concept.
- Return concise JSON only.

Return this JSON shape:
{{
  "supports": true,
  "answer_fragment": "short answer grounded in the image",
  "visible_evidence": "what visible evidence supports the answer",
  "verified_facts": [
    {{
      "subject": "object or entity",
      "predicate": "attribute/relation/text/count/state",
      "value": "verified value",
      "evidence_description": "visible evidence",
      "evidence_scope": "full_image"
    }}
  ],
  "confidence": 0.0,
  "abstained": false
}}"""
