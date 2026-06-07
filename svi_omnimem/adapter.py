"""External adapter that adds SVI APIs to an OmniSimpleMem orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import SVIConfig
from .executor import VisualMemoryPlanExecutor
from .extractor import StructuredVisualExtractor
from .models import StructuredVisualQueryResult
from .planner import VisualMemoryPlanner
from .promoter import VerifiedFactPromoter
from .query_parser import VisualQueryParser
from .retriever import StructuredVisualRetriever
from .stores import StructuredVisualStore, VerifiedFactStore
from .utils import ensure_iso_time, unique_list
from .verifier import RawEvidenceVerifier

logger = logging.getLogger(__name__)


class SVIOmniMemAdapter:
    """Attach Structured Visual Indexing to an existing orchestrator.

    The adapter avoids mutating baseline OmniSimpleMem behavior. It reuses the
    existing image ingestion path for entropy filtering, raw-image cold storage,
    captioning, and Image MAU creation; then writes compact SVI cards and, when
    configured, a searchable text mirror.
    """

    def __init__(
        self,
        orchestrator: Any,
        config: Optional[SVIConfig] = None,
        storage_dir: Optional[str] = None,
    ):
        self.orchestrator = orchestrator
        self.config = config or SVIConfig()
        base_dir = Path(storage_dir or self._default_storage_dir())
        base_dir.mkdir(parents=True, exist_ok=True)

        self.visual_store = StructuredVisualStore(str(base_dir))
        self.verified_fact_store = VerifiedFactStore(str(base_dir))
        self.extractor = StructuredVisualExtractor(orchestrator, self.config)
        self.query_parser = VisualQueryParser()
        embedding_service = getattr(orchestrator, "_embedding_service", None)
        if embedding_service is None:
            embedding_service = getattr(getattr(orchestrator, "retriever", None), "_embedding_service", None)
        self.retriever = StructuredVisualRetriever(
            self.visual_store,
            self.verified_fact_store,
            self.config,
            embedding_service=embedding_service,
        )
        self.planner = VisualMemoryPlanner(
            orchestrator=orchestrator,
            config=self.config,
            visual_store=self.visual_store,
            verified_fact_store=self.verified_fact_store,
        )
        self.plan_executor = VisualMemoryPlanExecutor(
            self.visual_store,
            self.verified_fact_store,
            self.retriever,
            self.config,
        )
        self.verifier = RawEvidenceVerifier(
            orchestrator,
            self.visual_store,
            self.config,
        )
        self.promoter = VerifiedFactPromoter(
            self.verified_fact_store,
            self.config,
        )

    def _default_storage_dir(self) -> str:
        try:
            index_dir = self.orchestrator.config.storage.index_dir
            return str(Path(index_dir) / "svi_omnimem")
        except Exception:
            return "./svi_omnimem_index"

    def add_image_structured(
        self,
        image: Any,
        text_context: Optional[str] = None,
        seed_caption: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> Any:
        """Add an image through OmniSimpleMem and attach an SVI card.

        Returns the baseline ProcessingResult with extra metadata:
        ``svi_card_id`` and optional ``svi_mirror_mau_id``.
        """
        session_id = session_id or getattr(self.orchestrator, "session_id", None)
        tags = unique_list(tags or [])

        result, already_stored = self._process_image_base(
            image,
            session_id=session_id,
            tags=tags,
            force=force,
        )
        if not getattr(result, "success", False) or not getattr(result, "mau", None):
            return result

        mau = result.mau
        observed_at = ensure_iso_time(timestamp or getattr(mau, "timestamp", None))
        raw_pointer = getattr(mau, "raw_pointer", "") or ""
        global_caption = seed_caption or getattr(mau, "summary", "") or "Image captured"

        card = self.extractor.extract(
            image=image,
            image_mau_id=mau.id,
            raw_pointer=raw_pointer,
            global_caption=global_caption,
            text_context=text_context,
            timestamp=observed_at,
            session_id=session_id,
            turn_id=turn_id,
            tags=tags,
        )

        if card.global_caption and card.global_caption != "Image captured":
            mau.summary = card.global_caption

        if not already_stored:
            self._attach_svi_metadata(mau, card.card_id, None)
            self._clear_unstorable_visual_embedding(mau)
            try:
                self.orchestrator._store_mau(mau, tags)
            except Exception as exc:
                result.success = False
                result.error = f"{type(exc).__name__}: {exc}"
                logger.warning("SVI image MAU storage failed: %s", result.error)
                return result

        self.visual_store.append(card)

        mirror_mau_id = None
        if self.config.index_text_mirror:
            mirror_mau_id = self._store_text_mirror(card, session_id, tags)

        if already_stored or mirror_mau_id:
            self._annotate_image_mau(mau, card.card_id, mirror_mau_id)

        result.metadata = getattr(result, "metadata", {}) or {}
        result.metadata["svi_card_id"] = card.card_id
        if mirror_mau_id:
            result.metadata["svi_mirror_mau_id"] = mirror_mau_id
        return result

    def _clear_unstorable_visual_embedding(self, mau: Any) -> None:
        embedding = getattr(mau, "embedding", None)
        if not embedding:
            return
        try:
            embedding_dim = len(embedding)
        except TypeError:
            mau.embedding = []
            return

        cfg = getattr(self.orchestrator, "config", None)
        embedding_cfg = getattr(cfg, "embedding", None)
        expected_dims = {
            int(getattr(embedding_cfg, "embedding_dim", 0) or 0),
            int(getattr(embedding_cfg, "visual_embedding_dim", 0) or 0),
        }
        expected_dims.discard(0)
        if embedding_dim in expected_dims:
            return

        mau.details = mau.details or {}
        mau.details["svi_embedding_note"] = (
            f"cleared incompatible visual embedding dim {embedding_dim}; "
            f"expected one of {sorted(expected_dims)}"
        )
        mau.embedding = []

    def _process_image_base(
        self,
        image: Any,
        session_id: Optional[str],
        tags: List[str],
        force: bool,
    ) -> Tuple[Any, bool]:
        """Process image once, skipping baseline caption when possible.

        Returns ``(ProcessingResult, already_stored)``. The preferred path uses
        ImageProcessor directly with ``generate_caption=False`` so SVI pays for
        only one structured VLM call at ingestion. The fallback keeps the adapter
        compatible with mock or older orchestrators that only expose add_image().
        """
        if hasattr(self.orchestrator, "image_processor") and hasattr(
            self.orchestrator, "_store_mau"
        ):
            result = self.orchestrator.image_processor.process(
                image,
                session_id=session_id,
                force=force,
                generate_caption=False,
            )
            return result, False

        result = self.orchestrator.add_image(
            image,
            session_id=session_id,
            tags=tags,
            force=force,
        )
        return result, True

    def _store_text_mirror(
        self,
        card,
        session_id: Optional[str],
        tags: List[str],
    ) -> Optional[str]:
        mirror_text = card.to_mirror_text()
        mirror_tags = unique_list([*tags, "svi_mirror", f"svi_card:{card.card_id}"])
        try:
            result = self.orchestrator.add_text(
                mirror_text,
                session_id=session_id,
                tags=mirror_tags,
                force=True,
            )
        except Exception as exc:
            logger.warning("SVI text mirror storage failed: %s", exc)
            return None
        if not getattr(result, "success", False) or not getattr(result, "mau", None):
            return None
        mirror_mau = result.mau
        try:
            image_mau = self.orchestrator.mau_store.get(card.image_mau_id)
            if image_mau:
                image_mau.add_related(mirror_mau.id)
                self.orchestrator.mau_store.update(image_mau)
            mirror_mau.add_related(card.image_mau_id)
            mirror_mau.details = mirror_mau.details or {}
            mirror_mau.details["svi_card_id"] = card.card_id
            mirror_mau.details["source_image_mau_id"] = card.image_mau_id
            self.orchestrator.mau_store.update(mirror_mau)
        except Exception as exc:
            logger.debug("SVI mirror linking failed: %s", exc)
        return mirror_mau.id

    def _annotate_image_mau(
        self,
        mau: Any,
        card_id: str,
        mirror_mau_id: Optional[str],
    ) -> None:
        try:
            stored = self.orchestrator.mau_store.get(mau.id) or mau
            self._attach_svi_metadata(stored, card_id, mirror_mau_id)
            self.orchestrator.mau_store.update(stored)
        except Exception as exc:
            logger.debug("SVI image MAU annotation failed: %s", exc)

    def _attach_svi_metadata(
        self,
        mau: Any,
        card_id: str,
        mirror_mau_id: Optional[str],
    ) -> None:
        mau.details = mau.details or {}
        mau.details["svi_card_id"] = card_id
        if mirror_mau_id:
            mau.details["svi_mirror_mau_id"] = mirror_mau_id
            mau.add_related(mirror_mau_id)
        mau.metadata.keywords = unique_list(
            [
                *(mau.metadata.keywords or []),
                "structured_visual_index",
                "raw_verification_required",
            ]
        )

    def query_structured_visual(
        self,
        query: str,
        top_k: int = 10,
        verify: bool = True,
        verification_budget: int = 3,
        writeback_verified_fact: bool = True,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[str, str]] = None,
    ) -> StructuredVisualQueryResult:
        """Retrieve structured visual memories and optionally verify raw images."""
        requirement = self.query_parser.parse(query)
        retrieval_budget = max(top_k, verification_budget) if verify else top_k
        candidates, retrieval_claims = self.retriever.retrieve(
            query=query,
            requirement=requirement,
            top_k=retrieval_budget,
            tags_filter=tags_filter,
            time_range=time_range,
            current_session_id=getattr(self.orchestrator, "session_id", None),
        )
        execution_trace = [
            {
                "op": "generic_card_retrieval",
                "count": len(candidates),
                "planner": "disabled",
            }
        ]

        verified_evidence = []
        if verify and self.config.verification_enabled and requirement.requires_raw_verification:
            verified_evidence = self.verifier.verify(
                query=query,
                requirement=requirement,
                candidates=candidates,
                budget=max(0, verification_budget),
            )

        promoted_facts = []
        if writeback_verified_fact and verified_evidence:
            promoted_facts = self.promoter.promote(verified_evidence, requirement)

        result = StructuredVisualQueryResult(
            query=query,
            requirement=requirement,
            candidates=candidates[:top_k],
            verified_evidence=verified_evidence,
            promoted_facts=promoted_facts,
            retrieval_claims=retrieval_claims,
            answer_context="",
            plan=None,
            execution_trace=execution_trace,
        )
        result.answer_context = self._format_answer_context(result)
        return result

    def _format_answer_context(self, result: StructuredVisualQueryResult) -> str:
        lines: List[str] = []
        supported = [
            evidence
            for evidence in result.verified_evidence
            if evidence.supports
            and not evidence.abstained
            and not self._looks_like_negative_visual_evidence(evidence)
        ]
        if supported:
            lines.append("Verified visual evidence:")
            for evidence in supported:
                card = self.visual_store.get_by_card_id(evidence.source_card_id)
                public_image_id = self._public_image_id_for_card(evidence.source_card_id)
                lines.append(
                    "- "
                    + "; ".join(
                        part
                        for part in [
                            f"image_id: {public_image_id}" if public_image_id else "",
                            f"date: {card.observed_at}" if card and card.observed_at else "",
                            f"session: {card.session_id}" if card and card.session_id else "",
                            f"turn: {card.turn_id}" if card and card.turn_id else "",
                            evidence.answer_fragment,
                            f"evidence: {evidence.visible_evidence}" if evidence.visible_evidence else "",
                            f"confidence: {evidence.confidence:.2f}",
                        ]
                        if part
                    )
                )
        elif result.verified_evidence:
            return ""
        else:
            lines.append(
                "Unverified visual retrieval hints only; inspect raw images "
                "before treating visual details as facts:"
            )
            for candidate in result.candidates[:5]:
                card = self.visual_store.get_by_card_id(candidate.card_id)
                public_image_id = self._public_image_id_for_card(candidate.card_id)
                lines.append(
                    "- "
                    + "; ".join(
                        part
                        for part in [
                            f"image_id: {public_image_id}" if public_image_id else "",
                            f"date: {card.observed_at}" if card and card.observed_at else "",
                            f"session: {card.session_id}" if card and card.session_id else "",
                            f"turn: {card.turn_id}" if card and card.turn_id else "",
                            f"caption: {card.global_caption}"
                            if card and card.global_caption
                            else "",
                        ]
                        if part
                    )
                )
            return "\n".join(lines)

        if supported:
            lines.append("Candidate images:")
            supported_ids = {evidence.source_image_mau_id for evidence in supported}
            candidates = [
                candidate
                for candidate in result.candidates
                if candidate.image_mau_id in supported_ids
            ]
            for candidate in candidates[:5]:
                card = self.visual_store.get_by_card_id(candidate.card_id)
                public_image_id = self._public_image_id_for_card(candidate.card_id)
                candidate_label = public_image_id or candidate.card_id
                meta = ", ".join(
                    part
                    for part in [
                        f"date={card.observed_at}" if card and card.observed_at else "",
                        f"session={card.session_id}" if card and card.session_id else "",
                        f"turn={card.turn_id}" if card and card.turn_id else "",
                    ]
                    if part
                )
                meta_text = f" {meta}" if meta else ""
                lines.append(
                    f"- {candidate_label}{meta_text} score={candidate.score:.3f} routes={','.join(candidate.routes)}"
                )
        return "\n".join(lines)

    def _looks_like_negative_visual_evidence(self, evidence: Any) -> bool:
        text = f"{evidence.answer_fragment} {evidence.visible_evidence}".lower()
        negative_markers = [
            "does not show",
            "doesn't show",
            "do not show",
            "not show",
            "does not contain",
            "doesn't contain",
            "do not contain",
            "not contain",
            "not a ",
            "not an ",
            "not the ",
            "instead of",
        ]
        return any(marker in text for marker in negative_markers)

    def _public_image_id_for_card(self, card_id: Optional[str]) -> Optional[str]:
        if not card_id:
            return None
        card = self.visual_store.get_by_card_id(card_id)
        if not card:
            return None
        for tag in card.tags:
            if tag.startswith("image_id:"):
                return tag.split(":", 1)[1]
        return None

    def stats(self) -> dict:
        return {
            "structured_visual_cards": self.visual_store.count(),
            "verified_visual_facts": self.verified_fact_store.count(),
            "schema_version": self.config.schema_version,
        }
