"""Fallback query requirement construction for SVI-OmniMem."""

from __future__ import annotations

from .models import VisualQueryRequirement


class VisualQueryParser:
    """Compatibility fallback used when no planner-derived requirement exists."""

    def parse(self, query: str) -> VisualQueryRequirement:
        del query
        return VisualQueryRequirement(
            requires_visual_evidence=True,
            entities=[],
            requested_attributes=[],
            relation_constraints=[],
            ocr_terms=[],
            state_slots=[],
            temporal_scope="any",
            target_time=None,
            requires_raw_verification=True,
            query_type="visual",
        )
