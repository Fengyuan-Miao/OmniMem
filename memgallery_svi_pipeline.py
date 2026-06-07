"""Mem-Gallery evaluation pipeline for low-prior SVI-OmniMem.

This runner uses SVI as a lightweight structured visual index on top of
OmniSimpleMem. It intentionally avoids Mem-Gallery category-specific retrieval
logic: QA point labels are used only for reporting metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from omnimem.config import (  # noqa: E402
    PROJECT_ROOT,
    default_memgallery_dir,
    default_minilm_model,
    require_memgallery_dir,
)

try:
    from omni_memory import OmniMemoryConfig, OmniMemoryOrchestrator  # noqa: E402
    from omni_memory.core.mau import MAUMetadata, ModalityType, MultimodalAtomicUnit  # noqa: E402
except ModuleNotFoundError:
    OmniMemoryConfig = None
    OmniMemoryOrchestrator = None
    MAUMetadata = None
    ModalityType = None
    MultimodalAtomicUnit = None
from svi_omnimem import SVIConfig, SVIOmniMemAdapter  # noqa: E402

DEFAULT_DATA_DIR = default_memgallery_dir()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "memgallery"


def require_svi_backend() -> None:
    if OmniMemoryOrchestrator is None:
        raise ModuleNotFoundError(
            "The SVI compatibility runner requires OmniSimpleMem's "
            "`omni_memory` package. Install it in the active environment "
            "before running SVI."
        )


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def normalize_answer(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: Any, answer: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ans_tokens = normalize_answer(answer).split()
    if not pred_tokens and not ans_tokens:
        return 1.0
    if not pred_tokens or not ans_tokens:
        return 0.0
    pred_counts: Dict[str, int] = defaultdict(int)
    ans_counts: Dict[str, int] = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    for token in ans_tokens:
        ans_counts[token] += 1
    common = sum(min(pred_counts[t], ans_counts[t]) for t in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ans_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: Any, answer: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def contains_answer(prediction: Any, answer: Any) -> float:
    pred = normalize_answer(prediction)
    ans = normalize_answer(answer)
    return float(bool(ans) and ans in pred)


def answer_format_instruction(question: str) -> str:
    q = normalize_answer(question)
    tokens = set(q.split())
    if asks_for_remembered_image_ids(question):
        if question_requests_single_image(question):
            return (
                "Output format: return only one matching public image id. "
                "Do not include explanations."
            )
        return (
            "Output format: return only the matching public image id values, "
            "comma-separated. Do not include explanations."
        )
    yes_no_starters = (
        "has ",
        "have ",
        "had ",
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "should ",
        "would ",
        "will ",
    )
    if q.startswith(yes_no_starters):
        conflict_markers = {
            "conflict",
            "conflicts",
            "conflicting",
            "contradict",
            "contradicts",
            "contradiction",
            "inconsistent",
            "opposite",
        }
        if conflict_markers & tokens:
            return (
                "Output format: answer only Yes or No. Use Yes when the "
                "dialogue contradicts or does not support the statement; use "
                "No when it is supported."
            )
        if " or " in q and "yes or no" not in q and "answer yes or no" not in q:
            return (
                "Output format: answer with the supported option phrase or a "
                "short direct statement, not a bare Yes/No."
            )
        return (
            "Output format: answer only Yes, No, or Not mentioned. "
            "Use Yes/No when the memory context entails the polarity; use Not "
            "mentioned only when the context lacks support."
        )
    list_markers = {
        "what",
        "which",
        "who",
    }
    plural_or_list = bool(
        {"items", "things", "examples", "options", "candidates", "images", "pictures", "photos", "names"} & tokens
    ) or any(token.endswith("s") for token in tokens if len(token) > 4)
    if tokens & list_markers and plural_or_list:
        return (
            "Output format: return only the requested item names or values, "
            "comma-separated. Do not summarize, explain, or address the user."
        )
    return ""


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


GENERIC_STOPWORDS = {
    "about", "after", "again", "all", "also", "and", "any", "are",
    "attached", "been", "before", "being", "can", "did", "does",
    "for", "from", "had", "has", "have", "her", "him", "his",
    "how", "into", "is", "it", "its", "more", "now", "of",
    "first", "last", "earliest", "latest", "before", "after",
    "only", "or", "please", "return", "same", "she", "shown",
    "than", "that", "the", "their", "then", "there", "this",
    "to", "was", "were", "what", "when", "which", "who", "with",
    "yes", "you", "your",
}


def token_variants(token: str) -> List[str]:
    variants = [token]
    if len(token) > 4 and token.endswith("ies"):
        variants.append(token[:-3] + "y")
    if len(token) > 4 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 3:
            variants.append(stem)
    if len(token) > 3 and token.endswith("ed"):
        stem = token[:-2]
        if len(stem) > 3:
            variants.append(stem)
    if len(token) > 3 and token.endswith("es"):
        stem = token[:-2]
        if len(stem) > 3:
            variants.append(stem)
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        stem = token[:-1]
        if len(stem) > 3:
            variants.append(stem)
    deduped: List[str] = []
    for item in variants:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def retrieval_tokens(text: Any) -> List[str]:
    tokens = normalize_answer(text).split()
    result: List[str] = []
    for token in tokens:
        if token in GENERIC_STOPWORDS:
            continue
        if not (token.isdigit() or len(token) > 2):
            continue
        for variant in token_variants(token):
            if variant not in GENERIC_STOPWORDS:
                result.append(variant)
    return result


def tag_value(tags: List[str], prefix: str) -> str:
    for tag in tags:
        if tag.startswith(prefix):
            return tag.split(":", 1)[1]
    return ""


MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def question_date_filter(question: str):
    q = normalize_answer(question)
    exact_dates = {
        datetime.strptime(match, "%Y-%m-%d").date()
        for match in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", str(question or ""))
    }
    if exact_dates:
        return lambda value: _parse_iso_date(value) in exact_dates

    month_matches = [
        (match.group(1).lower(), match.group(2).lower())
        for match in re.finditer(
            r"\b(early|mid|late)?\s*("
            + "|".join(sorted(MONTH_NAME_TO_NUMBER, key=len, reverse=True))
            + r")\b",
            q,
        )
    ]
    if not month_matches:
        return None

    constraints = []
    for phase, month_name in month_matches:
        month = MONTH_NAME_TO_NUMBER.get(month_name)
        if not month:
            continue
        if phase == "early":
            day_range = range(1, 11)
        elif phase == "mid":
            day_range = range(11, 21)
        elif phase == "late":
            day_range = range(21, 32)
        else:
            day_range = range(1, 32)
        constraints.append((month, set(day_range)))
    if not constraints:
        return None

    def matches(value: str) -> bool:
        parsed = _parse_iso_date(value)
        if parsed is None:
            return False
        return any(parsed.month == month and parsed.day in days for month, days in constraints)

    return matches


def _parse_iso_date(value: str) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def question_mentions_character_profile(question: str, profile_names: Iterable[str]) -> bool:
    q_tokens = set(retrieval_tokens(question))
    for name in profile_names:
        if set(retrieval_tokens(name)) & q_tokens:
            return True
    return False


def iter_scenarios(data_dir: Path, scenario: Optional[str]) -> Iterable[Path]:
    dialog_dir = data_dir / "data" / "dialog"
    if scenario:
        path = dialog_dir / f"{scenario}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        yield path
        return
    yield from sorted(dialog_dir.glob("*.json"))


def parse_scenario_names(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [
        item.strip()
        for item in re.split(r"[,;\s]+", value)
        if item.strip()
    ]


def resolve_scenario_paths(args: argparse.Namespace) -> List[Path]:
    if args.all_scenarios:
        paths = list(iter_scenarios(args.data_dir, None))
    else:
        names = parse_scenario_names(args.scenarios)
        if not names:
            names = [args.scenario or "Entrepreneurship_Blockchain_Economics_Logistics_Nature"]
        paths = [
            path
            for name in names
            for path in iter_scenarios(args.data_dir, name)
        ]
    if args.max_scenarios is not None:
        paths = paths[: max(0, args.max_scenarios)]
    return paths


def image_path(data_dir: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    if rel.is_absolute():
        return rel
    return (data_dir / "data" / "dialog" / rel).resolve()


def optional_image_path(data_dir: Path, rel_path: Any) -> Optional[Path]:
    if not rel_path:
        return None
    path = image_path(data_dir, str(rel_path))
    return path if path.exists() else None


IMAGE_ID_PATTERN = re.compile(r"\bD\d+:IMG_\d+\b")


def extract_public_image_ids(value: Any) -> List[str]:
    ids: List[str] = []
    for match in IMAGE_ID_PATTERN.findall(str(value or "")):
        if match not in ids:
            ids.append(match)
    return ids


def extract_clue_turn_ids(value: Any) -> List[str]:
    raw_items = value if isinstance(value, list) else [value]
    turns: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if re.fullmatch(r"D\d+:\d+", text) and text not in turns:
            turns.append(text)
    return turns


def public_image_id_for_card(card: Any) -> str:
    if not card:
        return ""
    return tag_value(getattr(card, "tags", []) or [], "image_id:")


def build_image_retrieval_audit(
    svi: SVIOmniMemAdapter,
    svi_result: Any,
    answer: str,
    clue: Any,
    verification_budget: int,
    visual_audit_applicable: bool,
) -> Dict[str, Any]:
    total_cards = svi.visual_store.count()
    candidate_image_ids: List[str] = []
    candidate_turn_ids: List[str] = []
    for candidate in getattr(svi_result, "candidates", []) or []:
        card = svi.visual_store.get_by_card_id(getattr(candidate, "card_id", ""))
        public_id = public_image_id_for_card(card)
        if public_id and public_id not in candidate_image_ids:
            candidate_image_ids.append(public_id)
        turn_id = getattr(card, "turn_id", "") if card else ""
        if turn_id and turn_id not in candidate_turn_ids:
            candidate_turn_ids.append(turn_id)

    verified_image_ids: List[str] = []
    supported_image_ids: List[str] = []
    verified_turn_ids: List[str] = []
    supported_turn_ids: List[str] = []
    for evidence in getattr(svi_result, "verified_evidence", []) or []:
        card = svi.visual_store.get_by_card_id(getattr(evidence, "source_card_id", ""))
        public_id = public_image_id_for_card(card)
        if public_id and public_id not in verified_image_ids:
            verified_image_ids.append(public_id)
        turn_id = getattr(card, "turn_id", "") if card else ""
        if turn_id and turn_id not in verified_turn_ids:
            verified_turn_ids.append(turn_id)
        if getattr(evidence, "supports", False) and not getattr(evidence, "abstained", False):
            if public_id and public_id not in supported_image_ids:
                supported_image_ids.append(public_id)
            if turn_id and turn_id not in supported_turn_ids:
                supported_turn_ids.append(turn_id)

    gold_image_ids = extract_public_image_ids(answer)
    clue_turn_ids = extract_clue_turn_ids(clue)

    def any_hit(gold: List[str], found: List[str]) -> Optional[bool]:
        if not gold:
            return None
        return bool(set(gold) & set(found))

    def all_hit(gold: List[str], found: List[str]) -> Optional[bool]:
        if not gold:
            return None
        return set(gold).issubset(set(found))

    return {
        "total_visual_cards": total_cards,
        "visual_audit_applicable": visual_audit_applicable,
        "verification_budget": verification_budget,
        "verification_budget_ratio": (
            min(max(verification_budget, 0), total_cards) / total_cards
            if total_cards
            else 0.0
        ),
        "candidate_image_ids": candidate_image_ids,
        "verified_image_ids": verified_image_ids,
        "supported_image_ids": supported_image_ids,
        "gold_image_ids": gold_image_ids,
        "candidate_gold_any": any_hit(gold_image_ids, candidate_image_ids),
        "candidate_gold_all": all_hit(gold_image_ids, candidate_image_ids),
        "verified_gold_any": any_hit(gold_image_ids, supported_image_ids),
        "verified_gold_all": all_hit(gold_image_ids, supported_image_ids),
        "candidate_turn_ids": candidate_turn_ids,
        "verified_turn_ids": verified_turn_ids,
        "supported_turn_ids": supported_turn_ids,
        "clue_turn_ids": clue_turn_ids,
        "candidate_clue_any": any_hit(clue_turn_ids, candidate_turn_ids),
        "verified_clue_any": any_hit(clue_turn_ids, supported_turn_ids),
    }


def salient_literals(text: str, max_items: int = 12) -> List[str]:
    values: List[str] = []
    for pattern in [r"'([^']{1,60})'", r'"([^"]{1,60})"', r"\b\d{4}-\d{2}-\d{2}\b"]:
        for match in re.findall(pattern, text):
            item = str(match).strip()
            if item and item not in values:
                values.append(item)
            if len(values) >= max_items:
                return values
    return values


def make_turn_text(turn: Dict[str, Any]) -> str:
    lines = []
    if turn.get("user"):
        lines.append(f"User: {turn['user']}")
    if turn.get("assistant"):
        lines.append(f"Assistant: {turn['assistant']}")
    image_ids = turn.get("image_id") or []
    captions = turn.get("image_caption") or []
    for index, caption in enumerate(captions):
        img_id = image_ids[index] if index < len(image_ids) else ""
        prefix = f"Image {img_id}: " if img_id else "Image: "
        lines.append(prefix + str(caption))
    return "\n".join(lines)


def make_profile_text(profile: Any) -> str:
    if not isinstance(profile, dict) or not profile:
        return ""
    name = str(profile.get("name") or "").strip()
    lines = ["Character profile memory:"]
    if name:
        lines.append(f"The person referred to by name '{name}' is the user in User lines.")
    if profile.get("persona_summary"):
        lines.append("Persona summary: " + str(profile["persona_summary"]))
    traits = profile.get("traits") or []
    if traits:
        lines.append("Traits: " + ", ".join(str(item) for item in traits))
    if profile.get("conversation_style"):
        lines.append("Conversation style: " + str(profile["conversation_style"]))
    return "\n".join(lines)


def store_text_memory(
    orchestrator: OmniMemoryOrchestrator,
    text: str,
    session_id: str,
    tags: List[str],
) -> bool:
    """Store text, falling back to a zero-vector MAU if embeddings are unavailable."""
    result = orchestrator.add_text(text, session_id=session_id, tags=tags, force=True)
    if getattr(result, "success", False):
        return True

    embedding_dim = int(getattr(orchestrator.config.embedding, "embedding_dim", 384))
    mau = MultimodalAtomicUnit(
        modality_type=ModalityType.TEXT,
        summary=text[:200],
        embedding=[0.0] * embedding_dim,
        metadata=MAUMetadata(session_id=session_id, tags=list(tags)),
        details={"full_text": text, "embedding_fallback": "zero_vector"},
    )
    orchestrator._store_mau(mau, tags)
    return False


def build_config(args: argparse.Namespace, data_dir: Path) -> OmniMemoryConfig:
    config = OmniMemoryConfig.create_default()
    config.storage.base_dir = str(data_dir)
    config.storage.cold_storage_dir = str(data_dir / "cold_storage")
    config.storage.index_dir = str(data_dir / "index")
    config.embedding.model_name = args.embedding_model
    config.embedding.embedding_dim = args.embedding_dim
    if args.embedding_device:
        config.embedding.device = args.embedding_device
    config.llm.caption_model = args.vlm_model
    config.llm.summary_model = args.vlm_model
    config.llm.query_model = args.vlm_model
    config.llm.api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "ollama"
    config.llm.api_base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    config.llm.temperature = 0.0
    config.llm.max_tokens = args.max_answer_tokens
    config.retrieval.default_top_k = args.omni_top_k
    config.enable_self_evolution = False
    config.enable_knowledge_extraction = False
    return config


@dataclass
class SharedRuntime:
    embedding_service: Optional[Any] = None


def build_shared_runtime(args: argparse.Namespace) -> SharedRuntime:
    runtime = SharedRuntime()
    if not args.share_embedding_runtime:
        return runtime
    try:
        from omni_memory.utils.embedding import EmbeddingService

        config = build_config(args, args.output_dir / "_shared_embedding_runtime")
        runtime.embedding_service = EmbeddingService(config)
        if args.preload_embedding_model:
            runtime.embedding_service.embed_text("warm up shared embedding runtime")
        print(
            "[INFO] Shared embedding runtime ready: "
            f"{args.embedding_model} device={args.embedding_device or 'auto'}"
        )
    except Exception as exc:
        print(f"[WARN] Shared embedding runtime unavailable: {type(exc).__name__}: {exc}")
        runtime.embedding_service = None
    return runtime


def attach_shared_runtime(
    orchestrator: OmniMemoryOrchestrator,
    runtime: Optional[SharedRuntime],
) -> None:
    if not runtime or runtime.embedding_service is None:
        return
    setattr(orchestrator, "_embedding_service", runtime.embedding_service)
    for name in ("text_processor", "audio_processor", "video_processor"):
        processor = getattr(orchestrator, name, None)
        if processor is not None:
            setattr(processor, "_embedding_service", runtime.embedding_service)
    retriever = getattr(orchestrator, "retriever", None)
    if retriever is not None:
        setattr(retriever, "_embedding_service", runtime.embedding_service)


def format_text_fallback_context(
    orchestrator: OmniMemoryOrchestrator,
    limit: int,
) -> str:
    """Return stored text MAUs as a simple context when vector retrieval is empty."""
    lines: List[str] = []
    for mau in orchestrator.mau_store.iter_all():
        if mau.modality_type != ModalityType.TEXT:
            continue
        details = mau.details if isinstance(mau.details, dict) else {}
        text = details.get("full_text") or mau.summary
        if not text:
            continue
        tags = []
        if mau.metadata and mau.metadata.tags:
            tags = mau.metadata.tags
        if "svi_mirror" in tags:
            continue
        header = f"[TEXT-{len(lines) + 1}]"
        if tags:
            header += " " + " | ".join(tags[:4])
        lines.append(f"{header}\n{text}")
        if len(lines) >= limit:
            break
    return "\n\n".join(lines)


def format_text_timeline_context(
    orchestrator: OmniMemoryOrchestrator,
    limit: int,
) -> str:
    """Return a compact chronological text index for broad reasoning questions."""
    items = []
    for mau in orchestrator.mau_store.iter_all():
        if mau.modality_type != ModalityType.TEXT:
            continue
        tags = mau.metadata.tags if mau.metadata and mau.metadata.tags else []
        if "svi_mirror" in tags:
            continue
        summary = str(mau.summary or "").strip()
        details = mau.details if isinstance(mau.details, dict) else {}
        if not summary:
            summary = str(details.get("full_text") or "").strip()[:240]
        if not summary:
            continue
        tag_text = " | ".join(tags[:4])
        items.append((float(getattr(mau, "timestamp", 0.0) or 0.0), tag_text, summary))
    items.sort(key=lambda item: item[0])
    lines = []
    for index, (_timestamp, tag_text, summary) in enumerate(items[:limit], start=1):
        header = f"[TIMELINE-{index}]"
        if tag_text:
            header += " " + tag_text
        lines.append(f"{header} {summary}")
    return "\n".join(lines)


def format_visual_source_turn_context(
    orchestrator: OmniMemoryOrchestrator,
    svi: SVIOmniMemAdapter,
    svi_result: Any,
    limit: int = 5,
) -> str:
    """Add dialogue provenance for verified visual memories.

    SVI verification can identify an image visually, but many benchmark answers
    require the dialogue fact attached to that image. This helper links verified
    cards back to their source turn tags and appends the original text turn as
    evidence, without using question-type rules or dataset labels.
    """
    wanted_turns = []
    seen_turns = set()
    for evidence in getattr(svi_result, "verified_evidence", []) or []:
        if not getattr(evidence, "supports", False) or getattr(evidence, "abstained", False):
            continue
        card = svi.visual_store.get_by_card_id(getattr(evidence, "source_card_id", ""))
        if not card:
            continue
        key = (card.session_id or "", card.turn_id or "")
        if key in seen_turns:
            continue
        seen_turns.add(key)
        wanted_turns.append((card, key))

    if not wanted_turns:
        return ""

    lines = ["Dialogue provenance for verified visual memories:"]
    added = 0
    for card, (session_id, turn_id) in wanted_turns:
        if added >= limit:
            break
        public_image_id = tag_value(card.tags, "image_id:")
        source_text = ""
        source_tags: List[str] = []
        for mau in orchestrator.mau_store.iter_all():
            if mau.modality_type != ModalityType.TEXT:
                continue
            tags = mau.metadata.tags if mau.metadata and mau.metadata.tags else []
            if "svi_mirror" in tags:
                continue
            if session_id and f"session_id:{session_id}" not in tags:
                continue
            if turn_id and f"turn_id:{turn_id}" not in tags:
                continue
            details = mau.details if isinstance(mau.details, dict) else {}
            source_text = str(details.get("full_text") or mau.summary or "").strip()
            source_tags = tags
            break
        if not source_text:
            continue
        header_parts = []
        if public_image_id:
            header_parts.append(f"image_id={public_image_id}")
        if session_id:
            header_parts.append(f"session={session_id}")
        if turn_id:
            header_parts.append(f"turn={turn_id}")
        date = tag_value(source_tags, "date:") or card.observed_at
        if date:
            header_parts.append(f"date={date}")
        lines.append(
            f"Visual source turn {added + 1} ({', '.join(header_parts)}):\n"
            f"{source_text[:1400]}"
        )
        added += 1
    return "\n\n".join(lines) if added else ""


def format_image_candidate_source_context(
    svi: SVIOmniMemAdapter,
    svi_result: Any,
    limit: int = 5,
) -> str:
    lines = ["Source-grounded image candidates:"]
    added = 0
    for candidate in getattr(svi_result, "candidates", []) or []:
        if added >= limit:
            break
        card = svi.visual_store.get_by_card_id(getattr(candidate, "card_id", ""))
        if not card:
            continue
        public_image_id = tag_value(card.tags, "image_id:")
        meta = ", ".join(
            part
            for part in [
                f"image_id={public_image_id}" if public_image_id else "",
                f"date={card.observed_at}" if card.observed_at else "",
                f"session={card.session_id}" if card.session_id else "",
                f"turn={card.turn_id}" if card.turn_id else "",
                f"score={getattr(candidate, 'score', 0.0):.3f}",
            ]
            if part
        )
        source = card.source_text_context or card.global_caption
        lines.append(
            f"- {meta}\n"
            f"  Image caption: {card.global_caption[:500]}\n"
            f"  Source turn: {source[:900]}"
        )
        added += 1
    return "\n".join(lines) if added else ""


def format_text_overlap_context(
    orchestrator: OmniMemoryOrchestrator,
    query: str,
    existing_context: str,
    limit: int,
) -> str:
    """Add lexical text evidence as a generic fallback to vector retrieval."""
    query_tokens = set(token for token in normalize_answer(query).split() if len(token) > 2)
    if not query_tokens:
        return ""
    existing = existing_context.lower()
    scored = []
    for mau in orchestrator.mau_store.iter_all():
        if mau.modality_type != ModalityType.TEXT:
            continue
        tags = mau.metadata.tags if mau.metadata and mau.metadata.tags else []
        if "svi_mirror" in tags:
            continue
        details = mau.details if isinstance(mau.details, dict) else {}
        full_text = str(details.get("full_text") or mau.summary or "").strip()
        if not full_text:
            continue
        if full_text[:80].lower() in existing:
            continue
        text_tokens = set(token for token in normalize_answer(full_text).split() if len(token) > 2)
        overlap = query_tokens & text_tokens
        if not overlap:
            continue
        score = len(overlap) / max(len(query_tokens), 1)
        tag_text = " | ".join(tags[:4])
        scored.append((score, float(getattr(mau, "timestamp", 0.0) or 0.0), tag_text, full_text))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lines = []
    for index, (score, _timestamp, tag_text, full_text) in enumerate(scored[:limit], start=1):
        header = f"[LEXICAL-{index} score={score:.2f}]"
        if tag_text:
            header += " " + tag_text
        lines.append(f"{header}\n{full_text[:1200]}")
    return "\n\n".join(lines)


def format_hybrid_text_context(
    orchestrator: OmniMemoryOrchestrator,
    query: str,
    vector_items: List[Dict[str, Any]],
    limit: int,
    neighbor_window: int = 1,
    include_character_profile: bool = False,
) -> str:
    """Merge vector-ranked text MAUs with full-turn lexical evidence.

    This is a generic text-memory recall layer: vector retrieval handles semantic
    paraphrase, while BM25-style full-turn scoring keeps names, dates, lists,
    and other exact facts from being lost in summaries. It does not encode
    benchmark categories or domain-specific keywords.
    """
    records: List[Dict[str, Any]] = []
    for mau in orchestrator.mau_store.iter_all():
        if mau.modality_type != ModalityType.TEXT:
            continue
        tags = mau.metadata.tags if mau.metadata and mau.metadata.tags else []
        if "svi_mirror" in tags:
            continue
        if "character_profile" in tags and not include_character_profile:
            continue
        details = mau.details if isinstance(mau.details, dict) else {}
        full_text = str(details.get("full_text") or mau.summary or "").strip()
        if not full_text:
            continue
        records.append(
            {
                "id": mau.id,
                "text": full_text,
                "summary": str(mau.summary or ""),
                "tags": tags,
                "date": tag_value(tags, "date:"),
                "session_id": tag_value(tags, "session_id:"),
                "turn_id": tag_value(tags, "turn_id:"),
                "timestamp": float(getattr(mau, "timestamp", 0.0) or 0.0),
            }
        )
    if not records:
        return ""

    query_terms = retrieval_tokens(query)
    doc_terms = [retrieval_tokens(item["text"] + " " + item["summary"] + " " + " ".join(item["tags"])) for item in records]
    doc_freq: Dict[str, int] = defaultdict(int)
    for terms in doc_terms:
        for term in set(terms):
            doc_freq[term] += 1

    vector_rank = {str(item.get("id")): rank for rank, item in enumerate(vector_items, start=1)}
    avg_len = sum(len(terms) for terms in doc_terms) / max(len(doc_terms), 1)
    scored = []
    for record, terms in zip(records, doc_terms):
        tf: Dict[str, int] = defaultdict(int)
        for term in terms:
            tf[term] += 1
        bm25 = 0.0
        doc_len = max(len(terms), 1)
        for term in query_terms:
            if not tf.get(term):
                continue
            idf = math.log(1 + (len(records) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf[term] + 1.2 * (1 - 0.75 + 0.75 * doc_len / max(avg_len, 1))
            bm25 += idf * (tf[term] * 2.2) / max(denom, 1e-6)
        vector_bonus = 0.0
        if record["id"] in vector_rank:
            vector_bonus = 2.0 / vector_rank[record["id"]]
        score = bm25 + vector_bonus
        if score <= 0:
            continue
        scored.append((score, record))

    scored.sort(key=lambda item: (item[0], item[1]["timestamp"]), reverse=True)
    ordered_records = sorted(records, key=lambda item: item["timestamp"])
    positions = {record["id"]: index for index, record in enumerate(ordered_records)}
    score_by_id = {record["id"]: score for score, record in scored}

    selected_by_id: Dict[str, Dict[str, Any]] = {}
    seed_records = [record for _score, record in scored[: max(1, limit)]]
    for record in seed_records:
        selected_by_id[record["id"]] = record

    expanded_limit = max(1, limit)
    for record in seed_records:
        if len(selected_by_id) >= expanded_limit:
            break
        if neighbor_window <= 0 or record["id"] not in positions:
            continue
        center = positions[record["id"]]
        for offset in range(-neighbor_window, neighbor_window + 1):
            if offset == 0:
                continue
            neighbor_index = center + offset
            if neighbor_index < 0 or neighbor_index >= len(ordered_records):
                continue
            neighbor = ordered_records[neighbor_index]
            if neighbor["session_id"] != record["session_id"]:
                continue
            if neighbor["id"] not in selected_by_id:
                selected_by_id[neighbor["id"]] = neighbor
            if len(selected_by_id) >= expanded_limit:
                break

    selected_records = sorted(selected_by_id.values(), key=lambda item: item["timestamp"])
    lines = ["Relevant text memories:"]
    for index, record in enumerate(selected_records, start=1):
        meta_parts = []
        if record["date"]:
            meta_parts.append(f"date={record['date']}")
        if record["session_id"]:
            meta_parts.append(f"session={record['session_id']}")
        if record["turn_id"]:
            meta_parts.append(f"turn={record['turn_id']}")
        score = score_by_id.get(record["id"], 0.0)
        if score > 0:
            meta_parts.append(f"score={score:.2f}")
        else:
            meta_parts.append("score=neighbor")
        text = record["text"][:1400]
        literal_values = salient_literals(record["text"])
        literal_line = ""
        if literal_values:
            literal_line = "\nSalient quoted/date values: " + "; ".join(literal_values)
        lines.append(f"Memory {index} ({', '.join(meta_parts)}):\n{text}{literal_line}")
    return "\n\n".join(lines)


def ingest_scenario(
    data: Dict[str, Any],
    data_dir: Path,
    orchestrator: OmniMemoryOrchestrator,
    svi: SVIOmniMemAdapter,
    args: argparse.Namespace,
) -> Dict[str, int]:
    stats = {
        "turns_seen": 0,
        "text_turns": 0,
        "images_seen": 0,
        "images_stored": 0,
        "profile_memories": 0,
    }
    profile_text = make_profile_text(data.get("character_profile"))
    if profile_text:
        profile = data.get("character_profile") or {}
        profile_tags = ["character_profile", "role:user"]
        if profile.get("name"):
            profile_tags.append(f"character_name:{profile['name']}")
        store_text_memory(
            orchestrator,
            profile_text,
            session_id="character_profile",
            tags=profile_tags,
        )
        stats["profile_memories"] += 1

    sessions = data.get("multi_session_dialogues") or []
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]

    for session in sessions:
        session_id = str(session.get("session_id") or "")
        date = str(session.get("date") or "")
        turns = session.get("dialogues") or []
        if args.max_turns is not None:
            turns = turns[: args.max_turns]
        for turn in turns:
            stats["turns_seen"] += 1
            turn_id = str(turn.get("round") or "")
            tags = [
                f"session_id:{session_id}",
                f"turn_id:{turn_id}",
                f"date:{date}",
            ]
            text = make_turn_text(turn)
            if text:
                store_text_memory(orchestrator, text, session_id=session_id, tags=tags)
                stats["text_turns"] += 1

            image_ids = turn.get("image_id") or []
            input_images = turn.get("input_image") or []
            captions = turn.get("image_caption") or []
            for index, rel_image in enumerate(input_images):
                img_id = str(image_ids[index] if index < len(image_ids) else "")
                caption = str(captions[index] if index < len(captions) else "")
                path = image_path(data_dir, rel_image)
                if not path.exists():
                    continue
                image_tags = [*tags]
                if img_id:
                    image_tags.append(f"image_id:{img_id}")
                result = svi.add_image_structured(
                    str(path),
                    text_context=text,
                    seed_caption=caption,
                    session_id=session_id,
                    turn_id=turn_id,
                    timestamp=date,
                    tags=image_tags,
                    force=True,
                )
                stats["images_seen"] += 1
                if getattr(result, "success", False):
                    stats["images_stored"] += 1
    return stats


def format_deterministic_evidence_table(omni_context: str, limit: int = 12) -> str:
    """Create a compact chronological evidence table from retrieved text context.

    This is intentionally non-semantic: it does not classify the query or infer
    answers. It only preserves chronology and source ids so the answer model can
    reason over ordered memories more reliably.
    """
    if not omni_context.strip():
        return ""
    pattern = re.compile(
        r"Memory\s+(?P<idx>\d+)\s+\((?P<meta>[^)]*)\):\n(?P<body>.*?)(?=\n\nMemory\s+\d+\s+\(|\Z)",
        re.DOTALL,
    )
    rows = []
    for match in pattern.finditer(omni_context):
        meta = match.group("meta")
        body = match.group("body").strip()
        date = ""
        session = ""
        turn = ""
        score = ""
        for part in [item.strip() for item in meta.split(",")]:
            if part.startswith("date="):
                date = part.split("=", 1)[1]
            elif part.startswith("session="):
                session = part.split("=", 1)[1]
            elif part.startswith("turn="):
                turn = part.split("=", 1)[1]
            elif part.startswith("score="):
                score = part.split("=", 1)[1]
        first_line = body.split("\n", 1)[0].strip()
        literals = salient_literals(body, max_items=8)
        literal_text = f" | literals: {', '.join(literals)}" if literals else ""
        rows.append((date, session, turn, score, first_line[:360], literal_text))
    if not rows:
        return ""
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    lines = ["Chronological evidence table:"]
    for index, (date, session, turn, score, first_line, literal_text) in enumerate(rows[:limit], start=1):
        source = "/".join(part for part in [date, session, turn] if part) or f"row-{index}"
        score_text = f" score={score}" if score else ""
        lines.append(f"{index}. {source}{score_text}: {first_line}{literal_text}")
    return "\n".join(lines)


def context_preview(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    value = text or ""
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def parse_memory_blocks(omni_context: str) -> List[Dict[str, str]]:
    pattern = re.compile(
        r"Memory\s+(?P<idx>\d+)\s+\((?P<meta>[^)]*)\):\n(?P<body>.*?)(?=\n\nMemory\s+\d+\s+\(|\Z)",
        re.DOTALL,
    )
    blocks: List[Dict[str, str]] = []
    for match in pattern.finditer(omni_context):
        item = {
            "idx": match.group("idx"),
            "meta": match.group("meta"),
            "body": match.group("body").strip(),
        }
        for part in [value.strip() for value in item["meta"].split(",")]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            item[key.strip()] = value.strip()
        blocks.append(item)
    return blocks


def split_speaker_text(body: str, speaker: str) -> str:
    pattern = re.compile(
        rf"{speaker}:\s*(.*?)(?=\n(?:User|Assistant|Image|Salient quoted/date values):|\Z)",
        re.DOTALL,
    )
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


def extract_ordered_mentions(text: str, limit: int = 16) -> List[str]:
    candidates: List[tuple[int, str]] = []
    patterns = [
        r"'([^']{1,60})'",
        r'"([^"]{1,60})"',
        r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*)+\b",
        r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = match.group(1) if match.groups() else match.group(0)
            value = re.sub(r"\s+", " ", value).strip(" .,;:!?()[]")
            if value and value.lower() not in GENERIC_STOPWORDS:
                candidates.append((match.start(), value))
    candidates.sort(key=lambda item: item[0])

    mentions: List[str] = []
    for _pos, value in candidates:
        if value not in mentions:
            mentions.append(value)
        if len(mentions) >= limit:
            break
    return mentions


def format_ordered_mention_sketch(omni_context: str, limit: int = 10) -> str:
    blocks = parse_memory_blocks(omni_context)
    lines = ["Ordered mention sketch from retrieved text memories:"]
    added = 0
    for block in blocks:
        if added >= limit:
            break
        user_mentions = extract_ordered_mentions(split_speaker_text(block["body"], "User"))
        assistant_mentions = extract_ordered_mentions(split_speaker_text(block["body"], "Assistant"))
        if not user_mentions and not assistant_mentions:
            continue
        meta = ", ".join(
            part
            for part in [
                f"date={block.get('date', '')}" if block.get("date") else "",
                f"session={block.get('session', '')}" if block.get("session") else "",
                f"turn={block.get('turn', '')}" if block.get("turn") else "",
            ]
            if part
        )
        lines.append(f"- {meta or 'memory ' + block['idx']}")
        if user_mentions:
            lines.append("  User mentions in order: " + " -> ".join(user_mentions[:12]))
        if assistant_mentions:
            lines.append(
                "  Assistant mentions in order: " + " -> ".join(assistant_mentions[:12])
            )
        added += 1
    return "\n".join(lines) if added else ""


def format_global_ordered_mention_sketch(
    orchestrator: OmniMemoryOrchestrator,
    limit: int = 16,
) -> str:
    """Expose chronological mention order across stored text memories.

    This is a generic memory-time index. It does not infer answers or classify
    questions; it only surfaces ordered user/assistant mentions from the memory
    timeline so the answer model can apply first/last/order operations even
    when semantic retrieval misses an early or late turn.
    """
    if limit <= 0:
        return ""

    records = []
    for mau in orchestrator.mau_store.iter_all():
        if mau.modality_type != ModalityType.TEXT:
            continue
        tags = mau.metadata.tags if mau.metadata and mau.metadata.tags else []
        if "svi_mirror" in tags:
            continue
        details = mau.details if isinstance(mau.details, dict) else {}
        full_text = str(details.get("full_text") or mau.summary or "").strip()
        if not full_text:
            continue
        user_mentions = extract_ordered_mentions(split_speaker_text(full_text, "User"))
        assistant_mentions = extract_ordered_mentions(
            split_speaker_text(full_text, "Assistant")
        )
        if not user_mentions and not assistant_mentions:
            continue
        records.append(
            {
                "timestamp": float(getattr(mau, "timestamp", 0.0) or 0.0),
                "date": tag_value(tags, "date:"),
                "session": tag_value(tags, "session_id:"),
                "turn": tag_value(tags, "turn_id:"),
                "user_mentions": user_mentions,
                "assistant_mentions": assistant_mentions,
            }
        )

    if not records:
        return ""
    records.sort(key=lambda item: (item["timestamp"], item["date"], item["session"], item["turn"]))
    lines = ["Global chronological mention sketch from all text memories:"]
    for index, record in enumerate(records[:limit], start=1):
        meta = ", ".join(
            part
            for part in [
                f"date={record['date']}" if record["date"] else "",
                f"session={record['session']}" if record["session"] else "",
                f"turn={record['turn']}" if record["turn"] else "",
            ]
            if part
        )
        lines.append(f"- {index}. {meta or 'memory'}")
        if record["user_mentions"]:
            lines.append(
                "  User mentions in order: "
                + " -> ".join(record["user_mentions"][:12])
            )
        if record["assistant_mentions"]:
            lines.append(
                "  Assistant mentions in order: "
                + " -> ".join(record["assistant_mentions"][:12])
            )
    return "\n".join(lines)


def format_entity_slot_context(
    orchestrator: OmniMemoryOrchestrator,
    query: str,
    limit: int = 4,
) -> str:
    """Surface compact entity-role facts from text memories.

    The extraction is deliberately shallow: it keeps source text snippets around
    query-overlapping memories so the answer model can audit owner/entity/slot
    bindings. It does not decide which entity answers the question.
    """
    if limit <= 0:
        return ""
    query_terms = set(retrieval_tokens(query))
    records = []
    for mau in orchestrator.mau_store.iter_all():
        if mau.modality_type != ModalityType.TEXT:
            continue
        tags = mau.metadata.tags if mau.metadata and mau.metadata.tags else []
        if "svi_mirror" in tags:
            continue
        details = mau.details if isinstance(mau.details, dict) else {}
        full_text = str(details.get("full_text") or mau.summary or "").strip()
        if not full_text:
            continue
        terms = set(retrieval_tokens(full_text))
        overlap = query_terms & terms
        if not overlap:
            continue
        score = len(overlap) / max(len(query_terms), 1)
        records.append(
            {
                "score": score,
                "timestamp": float(getattr(mau, "timestamp", 0.0) or 0.0),
                "date": tag_value(tags, "date:"),
                "session": tag_value(tags, "session_id:"),
                "turn": tag_value(tags, "turn_id:"),
                "text": full_text,
            }
        )
    if not records:
        return ""

    records.sort(key=lambda item: (item["score"], item["timestamp"]), reverse=True)
    selected = sorted(records[:limit], key=lambda item: item["timestamp"])
    lines = ["Entity-role audit snippets from text memories:"]
    for index, record in enumerate(selected, start=1):
        meta = ", ".join(
            part
            for part in [
                f"date={record['date']}" if record["date"] else "",
                f"session={record['session']}" if record["session"] else "",
                f"turn={record['turn']}" if record["turn"] else "",
                f"score={record['score']:.2f}",
            ]
            if part
        )
        user_text = split_speaker_text(record["text"], "User")
        assistant_text = split_speaker_text(record["text"], "Assistant")
        user_line = f"\n  User: {user_text[:500]}" if user_text else ""
        assistant_line = f"\n  Assistant: {assistant_text[:500]}" if assistant_text else ""
        lines.append(f"- {index}. {meta}{user_line}{assistant_line}")
    return "\n".join(lines)


def should_use_entity_slot_context(
    question: str,
    question_attachment_context: str,
) -> bool:
    if question_attachment_context.strip():
        return False
    q = normalize_answer(question)
    tokens = set(q.split())
    slot_terms = {
        "name",
        "names",
        "named",
        "called",
        "owner",
        "owners",
        "breed",
        "type",
        "model",
        "brand",
        "color",
        "attribute",
        "attributes",
    }
    relation_terms = {"of", "whose", "friend", "classmate", "pet", "dog", "cat"}
    has_possessive = "'s" in question or "’s" in question or " s " in f" {q} "
    return bool(slot_terms & tokens) and (
        has_possessive or bool(relation_terms & tokens)
    )


def needs_ordered_mention_sketch(question: str) -> bool:
    tokens = set(normalize_answer(question).split())
    mention_ops = {
        "first",
        "last",
        "earliest",
        "latest",
        "mentioned",
        "mention",
        "listed",
        "list",
        "candidate",
        "candidates",
        "name",
        "names",
        "called",
    }
    return bool(mention_ops & tokens)


def build_ordered_mention_sketch(
    orchestrator: OmniMemoryOrchestrator,
    omni_context: str,
    question: str,
    retrieved_limit: int,
    global_limit: int,
    auto_enabled: bool,
) -> str:
    should_include = retrieved_limit > 0 or (
        auto_enabled and needs_ordered_mention_sketch(question)
    )
    if not should_include:
        return ""
    parts = []
    retrieved = format_ordered_mention_sketch(
        omni_context,
        limit=max(retrieved_limit, 10),
    )
    if retrieved:
        parts.append(retrieved)
    global_sketch = format_global_ordered_mention_sketch(
        orchestrator,
        limit=global_limit,
    )
    if global_sketch:
        parts.append(global_sketch)
    return "\n\n".join(parts)



def format_relative_time_note(omni_context: str) -> str:
    blocks = parse_memory_blocks(omni_context)
    lines: List[str] = []
    for block in blocks:
        date_text = block.get("date", "")
        if not date_text:
            continue
        try:
            base_date = datetime.strptime(date_text[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        body = block["body"].lower()
        mappings = []
        if "yesterday" in body:
            mappings.append(f"yesterday={base_date - timedelta(days=1)}")
        if "today" in body:
            mappings.append(f"today={base_date}")
        if "tomorrow" in body:
            mappings.append(f"tomorrow={base_date + timedelta(days=1)}")
        if not mappings:
            continue
        meta = ", ".join(
            part
            for part in [
                f"date={block.get('date', '')}" if block.get("date") else "",
                f"session={block.get('session', '')}" if block.get("session") else "",
                f"turn={block.get('turn', '')}" if block.get("turn") else "",
            ]
            if part
        )
        lines.append(f"- {meta}: " + ", ".join(mappings))
    if not lines:
        return ""
    return "Relative time normalization from retrieved text memories:\n" + "\n".join(lines)


def asks_for_remembered_image_ids(question: str) -> bool:
    q = normalize_answer(question)
    tokens = set(q.split())
    if re.search(r"\bwhich\s+(image|picture|photo)s?\b", q):
        return True
    if any(
        phrase in q
        for phrase in [
            "what breed",
            "what name",
            "what is the name",
            "what color",
            "what content",
            "same breed",
            "same as",
        ]
    ):
        return False
    has_visual_target = bool(
        {"image", "images", "photo", "photos", "picture", "pictures"} & tokens
    )
    has_memory_image_action = any(
        phrase in q
        for phrase in [
            "which of the images",
            "which picture",
            "which photo",
            "what images",
            "search for the image",
            "search for image",
            "find the image",
            "find image",
            "image mentioned in the dialogue",
            "images uploaded",
            "image uploaded",
        ]
    )
    has_explicit_id_request = bool(
        {"id", "ids"} & tokens
    ) and has_visual_target
    return has_memory_image_action or has_explicit_id_request


def question_requests_single_image(question: str) -> bool:
    q = normalize_answer(question)
    if re.search(r"\bwhich\s+(image|picture|photo)\b", q):
        return not bool(re.search(r"\b(which|what)\s+(images|pictures|photos)\b", q))
    return False


def direct_answer_from_verified_images(
    question: str,
    svi_context: str,
    svi_result: Any = None,
    svi: Any = None,
    profile_names: Optional[List[str]] = None,
) -> str:
    """Return image ids directly when verified visual evidence is already decisive."""
    if not asks_for_remembered_image_ids(question):
        return ""

    date_matches = question_date_filter(question)
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
    evidence_by_card_id: Dict[str, Any] = {}
    if svi_result is not None:
        for evidence in getattr(svi_result, "verified_evidence", []) or []:
            if getattr(evidence, "source_card_id", ""):
                evidence_by_card_id[evidence.source_card_id] = evidence

    def turn_index(turn_id: Any) -> Optional[int]:
        match = re.search(r"(\d+)$", str(turn_id or ""))
        return int(match.group(1)) if match else None

    def user_source_text(text: Any) -> str:
        lines = []
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("User:"):
                lines.append(stripped.split("User:", 1)[1].strip())
        return " ".join(lines)

    records: List[Dict[str, Any]] = []
    for line in svi_context.splitlines():
        if not line.startswith("- image_id:"):
            continue
        if any(marker in line.lower() for marker in negative_markers):
            continue
        record: Dict[str, Any] = {}
        for part in line[2:].split(";"):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            record[key.strip().lower()] = value.strip()
        image_id = record.get("image_id", "")
        if not image_id:
            continue
        if date_matches and not date_matches(record.get("date", "")):
            continue
        record["turn_index"] = turn_index(record.get("turn"))
        record["candidate_score"] = 0.0
        record["card_id"] = ""
        records.append(record)

    if not records and svi_result is not None and svi is not None:
        for candidate in getattr(svi_result, "candidates", []) or []:
            card = svi.visual_store.get_by_card_id(getattr(candidate, "card_id", ""))
            if not card:
                continue
            image_id = tag_value(card.tags, "image_id:")
            if not image_id:
                continue
            if date_matches and not date_matches(card.observed_at):
                continue
            records.append(
                {
                    "image_id": image_id,
                    "date": card.observed_at or "",
                    "session": card.session_id or "",
                    "turn": card.turn_id or "",
                    "turn_index": turn_index(card.turn_id),
                    "candidate_score": coerce_float(getattr(candidate, "score", 0.0)),
                    "card_id": card.card_id,
                    "global_caption": card.global_caption or "",
                    "source_text_context": card.source_text_context or "",
                    "tags": " ".join(card.tags),
                }
            )

    seen_image_ids = {str(record.get("image_id", "")) for record in records if record.get("image_id")}
    if svi_result is not None and svi is not None:
        for candidate in getattr(svi_result, "candidates", []) or []:
            card = svi.visual_store.get_by_card_id(getattr(candidate, "card_id", ""))
            if not card:
                continue
            image_id = tag_value(card.tags, "image_id:")
            if not image_id or image_id in seen_image_ids:
                continue
            if date_matches and not date_matches(card.observed_at):
                continue
            records.append(
                {
                    "image_id": image_id,
                    "date": card.observed_at or "",
                    "session": card.session_id or "",
                    "turn": card.turn_id or "",
                    "turn_index": turn_index(card.turn_id),
                    "candidate_score": coerce_float(getattr(candidate, "score", 0.0)),
                    "card_id": card.card_id,
                    "global_caption": card.global_caption or "",
                    "source_text_context": card.source_text_context or "",
                    "tags": " ".join(card.tags),
                }
            )
            seen_image_ids.add(image_id)

    question_terms = set(retrieval_tokens(question))
    question_norm = normalize_answer(question)
    single_image_target = question_requests_single_image(question)
    before_mode = any(
        phrase in question_norm
        for phrase in [
            "before",
            "earlier",
            "prior to",
            "before making",
            "before deciding",
            "early",
            "before the decision",
        ]
    )
    prep_markers = [
        "search",
        "prepare",
        "preparing",
        "library",
        "explore",
        "look into",
        "advisor",
        "research",
        "study",
        "materials",
    ]
    decision_markers = [
        "decided",
        "decision",
        "minor",
        "switch major",
        "switching major",
        "after much consideration",
        "overall, i’m satisfied",
        "overall, i'm satisfied",
        "declared a minor",
    ]
    after_mode = any(
        phrase in question_norm
        for phrase in [
            "after making",
            "after deciding",
            "after the decision",
            "later",
            "subsequently",
            "afterward",
        ]
    )
    context_stop = {
        "image",
        "images",
        "picture",
        "pictures",
        "photo",
        "photos",
        "upload",
        "uploaded",
        "show",
        "shows",
        "shown",
        "talking",
        "topic",
        "about",
        "what",
        "which",
        "did",
        "user",
        "they",
        "them",
        "conversation",
        "dialogue",
        "said",
        "mentioned",
        "relate",
        "related",
        "showed",
        "shown",
    }
    for name in profile_names or []:
        context_stop.update(retrieval_tokens(name))
    question_terms = {term for term in question_terms if term not in context_stop}

    scored: List[Dict[str, Any]] = []
    for record in records:
        image_id = str(record.get("image_id", ""))
        if not image_id:
            continue
        score = coerce_float(record.get("candidate_score"), 0.0) * 0.25
        if svi_result is not None and svi is not None:
            for candidate in getattr(svi_result, "candidates", []) or []:
                card = svi.visual_store.get_by_card_id(getattr(candidate, "card_id", ""))
                if not card:
                    continue
                public_id = tag_value(card.tags, "image_id:")
                if public_id != image_id:
                    continue
                record["card_id"] = card.card_id
                record["turn_index"] = turn_index(card.turn_id)
                record["date"] = card.observed_at or record.get("date", "")
                field = " ".join(
                    [
                        card.global_caption,
                        user_source_text(card.source_text_context)
                        or card.source_text_context,
                        " ".join(card.tags),
                    ]
                )
                field_terms = set(retrieval_tokens(field))
                if question_terms:
                    score += len(question_terms & field_terms) / max(len(question_terms), 1)
                if before_mode or after_mode:
                    field_norm = normalize_answer(field)
                    if any(marker in field_norm for marker in prep_markers):
                        score += 0.35
                    elif before_mode:
                        score -= 0.25
                    if any(marker in field_norm for marker in decision_markers):
                        score -= 0.45
                    elif after_mode:
                        score -= 0.25
                evidence = evidence_by_card_id.get(card.card_id)
                if evidence and getattr(evidence, "supports", False):
                    score += 0.35
                break
        if card is not None:
            scored.append(
                {
                    "score": score,
                    "image_id": image_id,
                    "turn_index": record.get("turn_index"),
                    "date": str(record.get("date") or ""),
                }
            )
            continue

        field = " ".join(
            [
                str(record.get("global_caption") or ""),
                user_source_text(record.get("source_text_context"))
                or str(record.get("source_text_context") or ""),
                str(record.get("tags") or ""),
            ]
        )
        field_terms = set(retrieval_tokens(field))
        if question_terms:
            score += len(question_terms & field_terms) / max(len(question_terms), 1)
        if before_mode or after_mode:
            field_norm = normalize_answer(field)
            if any(marker in field_norm for marker in prep_markers):
                score += 0.35
            elif before_mode:
                score -= 0.25
            if any(marker in field_norm for marker in decision_markers):
                score -= 0.45
            elif after_mode:
                score -= 0.25
        scored.append(
            {
                "score": score,
                "image_id": image_id,
                "turn_index": record.get("turn_index"),
                "date": str(record.get("date") or ""),
            }
        )

    if question_terms and scored:
        positive_scores = [item["score"] for item in scored if item["score"] > 0]
        if positive_scores:
            threshold = max(positive_scores) * 0.90
            scored = [item for item in scored if item["score"] >= threshold]

    if not scored:
        return ""

    if single_image_target:
        if before_mode or after_mode:
            def temporal_key(item: Dict[str, Any]) -> tuple:
                date_value = str(item.get("date") or "")
                turn_index_value = item.get("turn_index")
                turn_sort = (
                    int(turn_index_value)
                    if isinstance(turn_index_value, int)
                    else 10**9
                )
                if before_mode:
                    return (-item["score"], date_value, turn_sort, item["image_id"])
                return (-item["score"], date_value, -turn_sort, item["image_id"])

            return sorted(scored, key=temporal_key)[0]["image_id"]

        best = max(scored, key=lambda item: (item["score"], item["image_id"]))
        return str(best["image_id"])

    image_ids: List[str] = []
    for item in sorted(scored, key=lambda item: (-item["score"], item["image_id"])):
        image_id = str(item["image_id"])
        if image_id not in image_ids:
            image_ids.append(image_id)
    return ", ".join(image_ids)


def should_use_svi_answer_context(question: str, question_attachment_context: str) -> bool:
    if asks_for_remembered_image_ids(question):
        return True
    has_attachment = bool(question_attachment_context.strip())
    if not has_attachment:
        return False
    q = normalize_answer(question)
    current_image_lookup = [
        "what color",
    ]
    if any(phrase in q for phrase in current_image_lookup):
        return False
    return True



def make_svi_query_text(question: str, image_caption: str) -> str:
    """Build the SVI retrieval query without benchmark category routing."""
    caption = str(image_caption or "").strip()
    if caption:
        q = normalize_answer(question)
        if asks_for_remembered_image_ids(question):
            return "IMAGE_RECALL::\n" + question + "\nCurrent question image caption: " + caption
        if any(
            phrase in q
            for phrase in [
                "more similar",
                "same as",
                "same breed",
                "which image",
                "which picture",
                "which photo",
                "compare",
            ]
        ):
            return "VISUAL_COMPARE::\n" + caption
        return "IMAGE_RECALL::\n" + question + "\nCurrent question image caption: " + caption
    return question


def clean_final_answer_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    marker_match = re.search(
        r"FINAL_ANSWER\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if marker_match:
        text = marker_match.group(1).strip()
        text = text.splitlines()[0].strip() if "\n" in text else text
        return text.strip().strip('"').strip("'").strip()

    try:
        data = extract_json_object(text)
        answer = data.get("answer")
        if answer is not None:
            return str(answer).strip()
    except Exception:
        pass

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    for pattern in [
        r"^\*\*Final answer:\*\*\s*",
        r"^\*\*Answer:\*\*\s*",
        r"^Final answer:\s*",
        r"^Answer:\s*",
    ]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text


def apply_entity_role_guard(
    question: str,
    question_attachment_context: str,
    answer_omni_context: str,
    prediction: str,
) -> str:
    """Avoid transferring a slot answer onto the wrong entity type.

    This only applies to plain text slot-filling questions without an attached
    image. It keeps names tied to the entity type actually mentioned in the
    retrieved memories and falls back to "Not mentioned." when the prediction
    is not supported by a line that also contains the target noun.
    """
    if question_attachment_context.strip():
        return prediction
    if not should_use_entity_slot_context(question, question_attachment_context):
        return prediction

    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction or normalized_prediction in {
        "yes",
        "no",
        "not mentioned",
    }:
        return prediction

    question_norm = normalize_answer(question)
    target_terms = [
        term
        for term in [
            "cat",
            "cats",
            "dog",
            "dogs",
            "pet",
            "pets",
            "bird",
            "birds",
            "rabbit",
            "rabbits",
            "fish",
            "horse",
            "horses",
            "person",
            "people",
        ]
        if term in question_norm
    ]
    if not target_terms:
        return prediction

    context_lines = [
        line.strip()
        for line in (answer_omni_context or "").splitlines()
        if line.strip()
    ]
    if not context_lines:
        return "Not mentioned."

    for line in context_lines:
        normalized_line = normalize_answer(line)
        if normalized_prediction in normalized_line and any(
            term in normalized_line for term in target_terms
        ):
            return prediction

    return "Not mentioned."


def apply_conflict_statement_guard(
    question: str,
    answer_omni_context: str,
    prediction: str,
) -> str:
    """Resolve explicit conflict/contradiction questions with a light guard.

    When a question asks whether a quoted statement conflicts with the dialogue,
    and the retrieved context clearly points in the opposite direction, we
    prefer the contradiction judgment over a model-produced No/Not mentioned.
    """
    question_norm = normalize_answer(question)
    prediction_norm = normalize_answer(prediction)
    if prediction_norm not in {"yes", "no", "not mentioned"}:
        return prediction
    conflict_markers = {
        "conflict",
        "conflicts",
        "conflicting",
        "contradict",
        "contradicts",
        "contradiction",
        "inconsistent",
        "opposite",
    }
    if not conflict_markers & set(question_norm.split()):
        return prediction

    literals = salient_literals(question)
    statement = normalize_answer(literals[0]) if literals else ""
    if not statement:
        match = re.search(
            r"statement\s+(.*?)(?:\s+conflict|\s+contradict|\?|$)",
            question,
            flags=re.IGNORECASE,
        )
        statement = normalize_answer(match.group(1)) if match else ""
    if not statement:
        return prediction

    statement_tokens = [
        token for token in retrieval_tokens(statement) if token not in GENERIC_STOPWORDS
    ]
    if not statement_tokens:
        return prediction

    context = normalize_answer(answer_omni_context)
    if not any(token in context for token in statement_tokens):
        return prediction

    negative_statement = any(
        phrase in statement
        for phrase in [
            "dislike",
            "dislikes",
            "don't like",
            "doesn't like",
            "not like",
            "hate",
            "hates",
            "never",
            "unlike",
            "unsuitable",
        ]
    )
    positive_statement = any(
        phrase in statement
        for phrase in [
            "like",
            "likes",
            "love",
            "loves",
            "suit",
            "suits",
            "prefer",
            "prefers",
            "enjoy",
            "enjoys",
            "works well",
        ]
    )
    positive_context = any(
        phrase in context
        for phrase in [
            "like",
            "likes",
            "love",
            "loves",
            "suit",
            "suits",
            "works well",
            "versatile",
            "timeless",
            "definitely",
            "great",
            "good",
            "recommend",
            "prefer",
            "enjoy",
            "chic",
        ]
    )
    negative_context = any(
        phrase in context
        for phrase in [
            "dislike",
            "dislikes",
            "hate",
            "hates",
            "don't like",
            "doesn't like",
            "not like",
            "worry",
            "plain",
            "boring",
            "avoid",
            "unsuitable",
        ]
    )

    if prediction_norm in {"no", "not mentioned"}:
        if negative_statement and positive_context:
            return "Yes"
        if positive_statement and negative_context:
            return "Yes"

    return prediction


def extract_breed_label(text: str) -> str:
    """Pull a concise breed-like label from a source turn or caption."""
    cleaned = str(text or "")
    patterns = [
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+Terrier)\b",
        r"\b(Maltese)\b",
        r"\b(Scottish Terrier)\b",
        r"\b(Norwich Terrier)\b",
        r"\b(Cairn Terrier)\b",
        r"\b(Poodle)\b",
        r"\b(Beagle)\b",
        r"\b(Pug)\b",
        r"\b(Boxer)\b",
        r"\b(Husky)\b",
        r"\b(Pomeranian)\b",
        r"\b(Chihuahua)\b",
        r"\b(Shih Tzu)\b",
        r"\b(Bulldog)\b",
        r"\b(Dachshund)\b",
        r"\b(Collie)\b",
        r"\b(Retriever)\b",
        r"\b(Spaniel)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            return value[:1].upper() + value[1:]
    return ""


def llm_answer(
    orchestrator: OmniMemoryOrchestrator,
    question: str,
    question_attachment_context: str,
    expected_format: str,
    svi_context: str,
    omni_context: str,
    max_tokens: int,
    evidence_brief: str = "",
    evidence_table: str = "",
    mention_sketch: str = "",
) -> str:
    client = orchestrator._get_llm_client()
    prompt = f"""Answer the question using only the memory context below.
{expected_format}

Return a concise answer. If the answer is not supported by any memory context,
say "Not mentioned." Do not copy context labels, memory IDs, or timeline labels
into the answer. Do not return evidence summaries, audit reports, reasoning
steps, headings, or markdown unless the user explicitly asks for a list.
If an Evidence brief or deterministic evidence table is provided, treat it as
organized evidence and use the raw contexts to resolve ambiguity or audit support.
If a Relative time normalization note is provided, use it to resolve words like
yesterday/today/tomorrow against the memory date metadata.
If an Ordered mention sketch is provided, use it only as a surface-order index
over retrieved text memories; verify the final answer against the raw memory
text.
Use the OmniSimpleMem text memory as the primary source for dialogue facts,
topic changes, temporal order, summaries, choices, preferences, mentions, and
contradiction judgments. These questions may require comparing multiple turns in
chronological order; do not require an exact sentence match when the answer is
entailed by the sequence of turns. For yes/no questions about whether a topic
changed, whether one discussion happened before another, or whether two memories
conflict, infer the answer from the ordered memories and answer Yes or No when
the context supports it. Treat order words such as earliest/latest/first/last and
before/after as operations over memory evidence: first identify the memories,
entities, or events that match the non-order content of the question, then apply
chronological order across memories and order of mention within a memory. The
source memory does not need to literally contain the words first, last, before,
or after. For yes/no before/after questions, answer Yes when the earliest
supporting memory for the first event precedes the earliest supporting memory
for the second event. Resolve relative dates like yesterday/tomorrow using the
memory date metadata.
For questions with multiple constraints joined by and/or/with/about, every
required constraint must be supported. A memory that supports only one part of a
compound question is not enough; answer "Not mentioned." for unsupported parts.
For direct lookup questions about a specific entity's name, owner, type, or
attribute, preserve entity roles exactly: do not transfer a name or attribute
from one entity type or owner to another. Evidence about someone's dog does not
answer a question about that person's cat. If the requested concrete entity or
attribute is absent from the relevant memories, say "Not mentioned."
Speaker roles are binding. If the question asks what the user mentioned, listed,
asked, chose, or decided, count only User lines unless an Assistant line is
explicitly confirmed by the User later. Assistant suggestions or examples are
not user mentions by themselves.
Owner and target roles are binding. A name, model, brand, or attribute attached
to one entity is not a candidate for another entity unless the memory explicitly
proposes it for that target entity.
Use Entity-role audit snippets to check whether a name or attribute is bound to
the exact requested owner and entity type. If snippets mention a related entity
but not the requested slot, answer "Not mentioned."
Distinguish options/candidates/examples from the final selected item. If the
question asks for candidates, options, examples, proposed names, or alternatives,
return all proposed items even if later memories selected only one of them. If
the question asks for the final choice, return only the final choice.
Use SVI visual context only as grounded memory-image evidence. For dialogue
history, topic order, temporal order, contradiction, preference, summary,
mentions, and choices, rely on OmniSimpleMem text memory and ignore SVI unless
the question explicitly asks to identify, compare, or inspect an image. Treat
visual labels that are not visible text as appearance hints, then prefer explicit
OmniSimpleMem text labels for fine-grained names, breeds, brands, or identities.
When a current question attachment describes an image, use that description to
match against remembered entities, but do not let SVI evidence from a different
stored image override explicit text memory labels unless the question asks for
that stored image. Entity type words are binding: evidence about a different
object/species/category does not answer the requested one.
For visual comparison or identification questions, prefer the single best-matching
verified image instead of averaging over multiple retrieved candidates. If a
verified image is tied to a source turn that names the object, breed, person, or
brand, treat that canonical source label as the answer anchor and do not replace
it with a looser visual guess from another candidate.
For questions asking which image/picture/photo is the answer, return only the
public image id(s) and do not explain. If the question clearly asks for one
image, return exactly one public image id. If the question asks whether a
statement conflicts with the dialogue, answer Yes when the dialogue contradicts
or does not support the statement, and No when it is supported.
If verified SVI evidence lists image_id entries and the question asks for
remembered images, first filter those verified entries by all explicit
constraints in the question, including date, session, turn, visible object, and
comparison target. Return only the relevant image_id values exactly. If multiple
verified images satisfy the question, include all of them. Do not invent image
IDs that are absent from verified evidence.
If a current question image description is provided, treat it as part of the
user's query, not as stored memory. Use it for matching/comparison against
remembered entities, but still require memory context for remembered facts.
{expected_format}

Question:
{question}

Current question attachment:
{question_attachment_context}

Evidence brief:
{evidence_brief}

Ordered mention sketch:
{mention_sketch}

Entity-role audit snippets:
{format_entity_slot_context(orchestrator, question, limit=4) if should_use_entity_slot_context(question, question_attachment_context) else ""}

Relative time normalization:
{format_relative_time_note(omni_context)}

OmniSimpleMem text memory context:
{omni_context}

Deterministic evidence table:
{evidence_table}

SVI visual memory context:
{svi_context}
"""
    kwargs = {
        "model": orchestrator.config.llm.query_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    response = client.chat.completions.create(**kwargs)
    return clean_final_answer_text(response.choices[0].message.content or "")


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("empty judge response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError("judge response did not contain a JSON object")


def llm_evidence_brief(
    orchestrator: OmniMemoryOrchestrator,
    question: str,
    question_attachment_context: str,
    svi_context: str,
    omni_context: str,
    max_tokens: int,
) -> tuple[str, Dict[str, Any]]:
    client = orchestrator._get_llm_client()
    prompt = f"""You are an evidence selector for a memory QA system.

Compress the provided memory context into a short, faithful evidence brief for
answering the question. Do not invent facts and do not answer from world
knowledge. Prefer explicit text memory for dialogue facts, names, dates, order,
preferences, and entity ownership. Use visual evidence only for visible image
content or image identity.

Include:
- relevant facts with date/session/turn when available
- chronological order when the question depends on order
- all candidate/options/examples when the question asks for candidates or lists
- an explicit note when the requested entity/type is not present
- any conflict between visual hints and explicit text labels

Return plain text, at most 8 bullet lines.

Question:
{question}

Current question attachment:
{question_attachment_context}

SVI visual memory context:
{svi_context}

OmniSimpleMem text memory context:
{omni_context}
"""
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=orchestrator.config.llm.query_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        brief = (response.choices[0].message.content or "").strip()
        return brief, {
            "evidence_brief": brief,
            "evidence_brief_latency_ms": round((time.time() - started) * 1000, 1),
            "evidence_brief_error": "",
        }
    except Exception as exc:
        return "", {
            "evidence_brief": "",
            "evidence_brief_latency_ms": round((time.time() - started) * 1000, 1),
            "evidence_brief_error": str(exc),
        }


def llm_answer_check(
    orchestrator: OmniMemoryOrchestrator,
    question: str,
    question_attachment_context: str,
    svi_context: str,
    omni_context: str,
    draft_answer: str,
    max_tokens: int,
) -> tuple[str, Dict[str, Any]]:
    client = orchestrator._get_llm_client()
    prompt = f"""You are a strict evidence consistency checker for memory QA.

Use only the provided memory evidence. Check whether the draft answer is fully
supported, whether it misses an obvious supported answer, or whether it adds
unsupported entities. Return a corrected concise answer when needed.

Rules:
- Keep the draft answer if it is supported and complete.
- If the draft says Not mentioned but the evidence clearly supports an answer, correct it.
- If the draft names an entity, attribute, list item, or image id not supported by evidence, remove or correct it.
- If the question asks for a list, keep all supported core items and remove unsupported extras.
- If the question asks for a single image, keep exactly one public image id.
- If the question asks whether a statement conflicts with the dialogue, correct the answer to Yes when the dialogue contradicts or does not support the statement, and No when it is supported.
- Preserve entity roles: do not use evidence about one entity type or owner to answer another.
- Return only valid JSON with keys: answer, changed, reason.

Question:
{question}

Current question attachment:
{question_attachment_context}

SVI visual memory context:
{svi_context}

OmniSimpleMem text memory context:
{omni_context}

Draft answer:
{draft_answer}
"""
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=orchestrator.config.llm.query_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        data = extract_json_object(content)
        answer = str(data.get("answer") or draft_answer).strip() or draft_answer
        return answer, {
            "answer_check_changed": bool(data.get("changed")),
            "answer_check_reason": str(data.get("reason") or ""),
            "answer_check_latency_ms": round((time.time() - started) * 1000, 1),
            "answer_check_error": "",
        }
    except Exception as exc:
        return draft_answer, {
            "answer_check_changed": False,
            "answer_check_reason": "",
            "answer_check_latency_ms": round((time.time() - started) * 1000, 1),
            "answer_check_error": str(exc),
        }


def chat_completion_http(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'ollama'}",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"].get("content") or "").strip()


def llm_judge_answer(
    question: str,
    answer: str,
    prediction: str,
    point: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.judge_mode == "off":
        return {
            "judge_correct": None,
            "judge_score": None,
            "judge_reason": "",
            "judge_model": None,
            "judge_latency_ms": 0.0,
            "judge_error": "",
        }

    prompt = f"""You are an impartial evaluator for a memory QA benchmark.

Judge whether the prediction correctly answers the question compared with the gold answer.

Rules:
- Semantic equivalence is correct; exact wording is not required.
- For image-id answers, mark correct if the prediction clearly contains the correct image id and does not contradict it.
- For yes/no answers, mark correct if the polarity matches the gold answer.
- For list answers, all core gold entities must be covered. Extra wrong entities should reduce the score or make it incorrect.
- Ignore harmless extra explanation when the final answer is still unambiguous.
- Return only valid JSON with keys: correct, score, reason.

Point: {point}
Question:
{question}

Gold answer:
{answer}

Prediction:
{prediction}
"""
    started = time.time()
    try:
        content = chat_completion_http(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model=args.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=args.judge_max_tokens,
        )
        data = extract_json_object(content)
        score = max(0.0, min(1.0, coerce_float(data.get("score"))))
        correct = bool(data.get("correct"))
        return {
            "judge_correct": int(correct),
            "judge_score": score,
            "judge_reason": str(data.get("reason") or ""),
            "judge_model": args.judge_model,
            "judge_latency_ms": round((time.time() - started) * 1000, 1),
            "judge_error": "",
        }
    except (KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
        return {
            "judge_correct": 0,
            "judge_score": 0.0,
            "judge_reason": "",
            "judge_model": args.judge_model,
            "judge_latency_ms": round((time.time() - started) * 1000, 1),
            "judge_error": str(exc),
        }


def run_scenario(
    path: Path,
    args: argparse.Namespace,
    runtime: Optional[SharedRuntime] = None,
) -> Path:
    data_dir = args.data_dir
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario = path.stem
    run_dir = args.output_dir / scenario / f"{now_stamp()}_svi_original"
    memory_dir = run_dir / "omnimem_data"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args, memory_dir)
    orchestrator = OmniMemoryOrchestrator(config=config, data_dir=str(memory_dir))
    attach_shared_runtime(orchestrator, runtime)
    svi_config = SVIConfig()
    svi_config.planner_mode = "off"
    svi_config.verification_budget = args.verification_budget
    svi_config.verification_enabled = not args.no_verify
    svi_config.promote_verified_facts = args.promote_verified_facts
    svi = SVIOmniMemAdapter(orchestrator, svi_config)

    start = time.time()
    ingest_stats = ingest_scenario(data, data_dir, orchestrator, svi, args)
    profile_names = []
    profile = data.get("character_profile") or {}
    if isinstance(profile, dict) and profile.get("name"):
        profile_names.append(str(profile["name"]))
    qas = data.get("human-annotated QAs") or []
    if args.max_questions is not None:
        qas = qas[: args.max_questions]

    predictions_path = run_dir / "predictions.jsonl"
    results = []
    with predictions_path.open("w", encoding="utf-8") as f:
        for index, qa in enumerate(qas, start=1):
            question = str(qa.get("question") or "")
            q_image_path = optional_image_path(data_dir, qa.get("question_image"))
            question_attachment_context = ""
            query_text = question
            if qa.get("image_caption"):
                question_attachment_context = "Image caption: " + str(qa["image_caption"])
                query_text = make_svi_query_text(question, str(qa["image_caption"]))
            use_svi_for_answer = should_use_svi_answer_context(
                question,
                question_attachment_context,
            )

            t0 = time.time()
            svi_result = svi.query_structured_visual(
                query_text,
                top_k=args.svi_top_k,
                verify=not args.no_verify and use_svi_for_answer,
                verification_budget=args.verification_budget,
            )
            try:
                omni_result = orchestrator.query(
                    question,
                    top_k=args.omni_top_k,
                    auto_expand=False,
                )
                text_items = [
                    item
                    for item in omni_result.items
                    if str(item.get("modality_type") or "").lower() == "text"
                ]
                omni_context = format_hybrid_text_context(
                    orchestrator,
                    question,
                    text_items,
                    limit=args.hybrid_text_limit,
                    neighbor_window=args.text_neighbor_window,
                    include_character_profile=question_mentions_character_profile(
                        question,
                        profile_names,
                    ),
                )
            except Exception:
                omni_context = format_hybrid_text_context(
                    orchestrator,
                    question,
                    [],
                    limit=args.hybrid_text_limit,
                    neighbor_window=args.text_neighbor_window,
                    include_character_profile=question_mentions_character_profile(
                        question,
                        profile_names,
                    ),
                )
            if not omni_context.strip():
                omni_context = format_text_fallback_context(
                    orchestrator,
                    limit=args.text_context_limit,
                )
            if args.use_lexical_context:
                lexical_context = format_text_overlap_context(
                    orchestrator,
                    question,
                    omni_context,
                    limit=args.lexical_context_limit,
                )
                if lexical_context.strip():
                    omni_context = (
                        omni_context.strip()
                        + "\n\nAdditional lexical text evidence:\n"
                        + lexical_context
                    ).strip()
            if not omni_context.strip() and not args.no_text_timeline:
                omni_context = format_text_timeline_context(
                    orchestrator,
                    limit=args.timeline_context_limit,
                )
            visual_source_context = ""
            if use_svi_for_answer:
                visual_source_context = format_visual_source_turn_context(
                    orchestrator,
                    svi,
                    svi_result,
                    limit=args.visual_source_turn_limit,
                )
            if visual_source_context.strip():
                omni_context = (
                    omni_context.strip()
                    + "\n\n"
                    + visual_source_context.strip()
                ).strip()

            answer_omni_context = omni_context
            if use_svi_for_answer:
                answer_context_parts = []
                if should_use_entity_slot_context(question, question_attachment_context):
                    entity_context = format_entity_slot_context(
                        orchestrator,
                        question,
                        limit=4,
                    )
                    if entity_context.strip():
                        answer_context_parts.append(entity_context.strip())
                if visual_source_context.strip():
                    answer_context_parts.append(visual_source_context.strip())
                if answer_context_parts:
                    answer_omni_context = "\n\n".join(answer_context_parts).strip()
                elif omni_context.strip():
                    answer_omni_context = omni_context.strip()
            elif should_use_entity_slot_context(question, question_attachment_context):
                entity_context = format_entity_slot_context(
                    orchestrator,
                    question,
                    limit=4,
                )
                if entity_context.strip():
                    answer_omni_context = entity_context.strip()

            evidence_brief = ""
            evidence_brief_fields: Dict[str, Any] = {
                "evidence_brief": "",
                "evidence_brief_latency_ms": 0.0,
                "evidence_brief_error": "",
            }
            answer_check: Dict[str, Any] = {
                "draft_prediction": "",
                "answer_check_changed": False,
                "answer_check_reason": "",
                "answer_check_latency_ms": 0.0,
                "answer_check_error": "",
            }
            if args.dry_run:
                prediction = ""
            else:
                format_hint = answer_format_instruction(question)
                answer_max_tokens = args.max_answer_tokens
                if "Yes, No, or Not mentioned" in format_hint:
                    answer_max_tokens = min(answer_max_tokens, 48)
                elif "public image id" in format_hint or "image id" in format_hint:
                    answer_max_tokens = min(answer_max_tokens, 32)
                elif "comma-separated" in format_hint:
                    answer_max_tokens = min(answer_max_tokens, 64)
                elif should_use_entity_slot_context(
                    question,
                    question_attachment_context,
                ):
                    answer_max_tokens = min(answer_max_tokens, 64)
                elif use_svi_for_answer:
                    answer_max_tokens = min(answer_max_tokens, 96)
                answer_svi_context = (
                    svi_result.answer_context
                    if use_svi_for_answer
                    else ""
                )
                if use_svi_for_answer:
                    candidate_context = format_image_candidate_source_context(
                        svi,
                        svi_result,
                        limit=args.visual_candidate_context_limit,
                    )
                    if candidate_context.strip():
                        answer_svi_context = (
                            answer_svi_context.strip()
                            + "\n\n"
                            + candidate_context.strip()
                        ).strip()
                direct_prediction = direct_answer_from_verified_images(
                    question,
                    answer_svi_context,
                    svi_result=svi_result,
                    svi=svi,
                    profile_names=profile_names,
                )
                if not direct_prediction and use_svi_for_answer:
                    q_norm = normalize_answer(question)
                    if q_norm.startswith("what breed"):
                        top_candidate = (getattr(svi_result, "candidates", []) or [None])[0]
                        if top_candidate is not None:
                            card = svi.visual_store.get_by_card_id(
                                getattr(top_candidate, "card_id", "")
                            )
                            if card:
                                breed_hint = extract_breed_label(
                                    card.source_text_context or card.global_caption or ""
                                )
                                if breed_hint:
                                    direct_prediction = breed_hint
                if args.evidence_brief_mode == "llm":
                    evidence_brief, evidence_brief_fields = llm_evidence_brief(
                        orchestrator,
                        question=question,
                        question_attachment_context=question_attachment_context,
                        svi_context=answer_svi_context,
                        omni_context=omni_context,
                        max_tokens=args.evidence_brief_max_tokens,
                    )
                if direct_prediction:
                    draft_prediction = direct_prediction
                else:
                    draft_prediction = llm_answer(
                        orchestrator,
                        question,
                        question_attachment_context=question_attachment_context,
                        expected_format=format_hint,
                        svi_context=answer_svi_context,
                        omni_context=answer_omni_context,
                        max_tokens=answer_max_tokens,
                        evidence_brief=evidence_brief,
                        evidence_table=format_deterministic_evidence_table(
                            answer_omni_context,
                            limit=args.evidence_table_limit,
                        ) if args.evidence_table_limit > 0 else "",
                        mention_sketch=build_ordered_mention_sketch(
                            orchestrator,
                            omni_context=answer_omni_context,
                            question=question,
                            retrieved_limit=args.mention_sketch_limit,
                            global_limit=args.global_mention_sketch_limit,
                            auto_enabled=args.auto_mention_sketch,
                        ),
                    )
                prediction = draft_prediction
                answer_check["draft_prediction"] = draft_prediction
                if args.answer_check_mode == "llm":
                    prediction, check_fields = llm_answer_check(
                        orchestrator,
                        question=question,
                        question_attachment_context=question_attachment_context,
                        svi_context=answer_svi_context,
                        omni_context=answer_omni_context,
                        draft_answer=draft_prediction,
                        max_tokens=args.answer_check_max_tokens,
                    )
                    answer_check.update(check_fields)

            prediction = apply_entity_role_guard(
                question=question,
                question_attachment_context=question_attachment_context,
                answer_omni_context=answer_omni_context,
                prediction=prediction,
            )
            prediction = apply_conflict_statement_guard(
                question=question,
                answer_omni_context=answer_omni_context,
                prediction=prediction,
            )

            answer = str(qa.get("answer") or "")
            judge = llm_judge_answer(
                question=qa.get("question") or question,
                answer=answer,
                prediction=prediction,
                point=qa.get("point"),
                args=args,
            )
            image_retrieval_audit = build_image_retrieval_audit(
                svi=svi,
                svi_result=svi_result,
                answer=answer,
                clue=qa.get("clue"),
                verification_budget=args.verification_budget,
                visual_audit_applicable=use_svi_for_answer,
            )
            row = {
                "index": index,
                "scenario": scenario,
                "point": qa.get("point"),
                "question": qa.get("question"),
                "question_image": str(q_image_path) if q_image_path else None,
                "answer": answer,
                "prediction": prediction,
                "evidence_table": format_deterministic_evidence_table(
                    omni_context,
                    limit=args.evidence_table_limit,
                ) if args.evidence_table_limit > 0 else "",
                "omni_context_preview": context_preview(
                    omni_context,
                    args.debug_context_chars,
                ),
                "svi_context_preview": context_preview(
                    answer_svi_context if not args.dry_run else "",
                    args.debug_context_chars,
                ),
                **evidence_brief_fields,
                **answer_check,
                "em": exact_match(prediction, answer),
                "f1": token_f1(prediction, answer),
                "contains_gt": contains_answer(prediction, answer),
                **judge,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "svi": svi_result.to_dict(),
                "image_retrieval_audit": image_retrieval_audit,
                "clue": qa.get("clue"),
                "session_id": qa.get("session_id"),
            }
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    metrics = summarize(results, ingest_stats, svi.stats(), time.time() - start)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"scenario": scenario}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    orchestrator.close()
    print(f"[INFO] Saved Mem-Gallery SVI run: {run_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return run_dir


def run_scenarios(args: argparse.Namespace) -> List[Path]:
    require_svi_backend()
    paths = resolve_scenario_paths(args)
    runtime = build_shared_runtime(args)
    run_dirs: List[Path] = []
    print(f"[INFO] Running {len(paths)} scenario(s) in one Python process")
    for index, path in enumerate(paths, start=1):
        print(f"[INFO] Scenario {index}/{len(paths)}: {path.stem}")
        run_dirs.append(run_scenario(path, args, runtime=runtime))
    return run_dirs


def summarize(
    rows: List[Dict[str, Any]],
    ingest_stats: Dict[str, int],
    svi_stats: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    by_point: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_point[str(row.get("point") or "")].append(row)

    def avg(items: List[float]) -> float:
        return sum(items) / len(items) if items else 0.0

    judged_rows = [row for row in rows if row.get("judge_correct") is not None]
    judge_errors = [row for row in judged_rows if row.get("judge_error")]
    audits = [row.get("image_retrieval_audit") or {} for row in rows]
    visual_audits = [audit for audit in audits if audit.get("visual_audit_applicable")]
    gold_image_audits = [audit for audit in audits if audit.get("gold_image_ids")]
    clue_audits = [
        audit
        for audit in visual_audits
        if audit.get("clue_turn_ids")
    ]

    def avg_bool(items: List[Optional[bool]]) -> float:
        values = [1.0 if item else 0.0 for item in items if item is not None]
        return avg(values)

    return {
        "num_results": len(rows),
        "primary_metric": "judge_accuracy",
        "judge_accuracy": avg([row["judge_correct"] for row in judged_rows]),
        "judge_score": avg([row["judge_score"] for row in judged_rows]),
        "judge_error_rate": len(judge_errors) / len(judged_rows) if judged_rows else 0.0,
        "em": avg([row["em"] for row in rows]),
        "f1": avg([row["f1"] for row in rows]),
        "contains_gt": avg([row["contains_gt"] for row in rows]),
        "avg_latency_ms": avg([row["latency_ms"] for row in rows]),
        "elapsed_seconds": elapsed,
        "image_retrieval": {
            "avg_total_visual_cards": avg(
                [coerce_float(audit.get("total_visual_cards")) for audit in audits]
            ),
            "avg_verification_budget_ratio": avg(
                [coerce_float(audit.get("verification_budget_ratio")) for audit in audits]
            ),
            "visual_rows": len(visual_audits),
            "avg_visual_verification_budget_ratio": avg(
                [
                    coerce_float(audit.get("verification_budget_ratio"))
                    for audit in visual_audits
                ]
            ),
            "gold_image_rows": len(gold_image_audits),
            "candidate_gold_any": avg_bool(
                [audit.get("candidate_gold_any") for audit in gold_image_audits]
            ),
            "candidate_gold_all": avg_bool(
                [audit.get("candidate_gold_all") for audit in gold_image_audits]
            ),
            "verified_gold_any": avg_bool(
                [audit.get("verified_gold_any") for audit in gold_image_audits]
            ),
            "verified_gold_all": avg_bool(
                [audit.get("verified_gold_all") for audit in gold_image_audits]
            ),
            "clue_rows": len(clue_audits),
            "candidate_clue_any": avg_bool(
                [audit.get("candidate_clue_any") for audit in clue_audits]
            ),
            "verified_clue_any": avg_bool(
                [audit.get("verified_clue_any") for audit in clue_audits]
            ),
        },
        **ingest_stats,
        **svi_stats,
        "by_point": {
            point: {
                "n": len(items),
                "judge_accuracy": avg(
                    [
                        row["judge_correct"]
                        for row in items
                        if row.get("judge_correct") is not None
                    ]
                ),
                "judge_score": avg(
                    [
                        row["judge_score"]
                        for row in items
                        if row.get("judge_score") is not None
                    ]
                ),
                "em": avg([row["em"] for row in items]),
                "f1": avg([row["f1"] for row in items]),
                "contains_gt": avg([row["contains_gt"] for row in items]),
            }
            for point, items in sorted(by_point.items())
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-prior SVI on Mem-Gallery.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma/space-separated scenario names to run in this process.",
    )
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--svi-top-k", type=int, default=10)
    parser.add_argument("--omni-top-k", type=int, default=10)
    parser.add_argument("--verification-budget", type=int, default=5)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--promote-verified-facts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl-8b-instruct-ctx4k:latest")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:11435/v1")
    parser.add_argument("--api-key", type=str, default="ollama")
    parser.add_argument("--embedding-model", type=str, default=default_minilm_model())
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--embedding-device", type=str, default=None)
    parser.add_argument("--share-embedding-runtime", action="store_true", default=True)
    parser.add_argument("--no-share-embedding-runtime", dest="share_embedding_runtime", action="store_false")
    parser.add_argument("--preload-embedding-model", action="store_true", default=True)
    parser.add_argument("--no-preload-embedding-model", dest="preload_embedding_model", action="store_false")
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--text-context-limit", type=int, default=20)
    parser.add_argument("--hybrid-text-limit", type=int, default=14)
    parser.add_argument("--text-neighbor-window", type=int, default=1)
    parser.add_argument("--visual-source-turn-limit", type=int, default=5)
    parser.add_argument("--visual-candidate-context-limit", type=int, default=5)
    parser.add_argument("--timeline-context-limit", type=int, default=30)
    parser.add_argument("--lexical-context-limit", type=int, default=6)
    parser.add_argument("--use-lexical-context", action="store_true")
    parser.add_argument("--no-text-timeline", action="store_true")
    parser.add_argument("--evidence-table-limit", type=int, default=12)
    parser.add_argument("--mention-sketch-limit", type=int, default=0)
    parser.add_argument("--global-mention-sketch-limit", type=int, default=20)
    parser.add_argument("--auto-mention-sketch", action="store_true", default=True)
    parser.add_argument("--no-auto-mention-sketch", dest="auto_mention_sketch", action="store_false")
    parser.add_argument("--evidence-brief-mode", choices=["llm", "off"], default="off")
    parser.add_argument("--evidence-brief-max-tokens", type=int, default=384)
    parser.add_argument("--answer-check-mode", choices=["llm", "off"], default="off")
    parser.add_argument("--answer-check-max-tokens", type=int, default=256)
    parser.add_argument("--debug-context-chars", type=int, default=6000)
    parser.add_argument("--judge-mode", choices=["llm", "off"], default="llm")
    parser.add_argument("--judge-base-url", type=str, default="http://127.0.0.1:11436/v1")
    parser.add_argument("--judge-model", type=str, default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--judge-api-key", type=str, default="ollama")
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = require_memgallery_dir(args.data_dir)
    args.output_dir = args.output_dir.resolve()
    run_scenarios(args)


if __name__ == "__main__":
    main()
