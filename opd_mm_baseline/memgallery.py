"""Mem-Gallery adaptation for the OPD-MM hidden memory schema."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import MemoryRecord, OPDSample
from .retrieval import DenseEncoder, HiddenMemoryStore, VisionEncoder


IMAGE_ID_PATTERN = re.compile(r"\bD\d+:IMG_\d+\b")
TURN_ID_PATTERN = re.compile(r"\bD\d+:\d+\b")


def resolve_image_path(data_dir: Path, value: str) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return (data_dir / "data" / "dialog" / path).resolve()


def optional_image_path(data_dir: Path, value: Any) -> Optional[Path]:
    if not value:
        return None
    path = resolve_image_path(data_dir, str(value))
    return path if path.is_file() else None


def observed_at(date_value: str, global_turn: int, offset: int = 0) -> str:
    try:
        base = datetime.strptime(str(date_value)[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        value = base + timedelta(seconds=max(0, global_turn) * 10 + offset)
        return value.isoformat().replace("+00:00", "Z")
    except ValueError:
        return f"order:{global_turn:06d}:{offset:02d}"


def scenario_records(
    data: Dict[str, Any],
    data_dir: Path,
    max_sessions: Optional[int] = None,
    max_turns: Optional[int] = None,
) -> List[MemoryRecord]:
    records: List[MemoryRecord] = []
    sessions = data.get("multi_session_dialogues") or []
    if max_sessions is not None:
        sessions = sessions[: max(0, max_sessions)]
    global_turn = 0
    for session_index, session in enumerate(sessions, start=1):
        session_id = str(session.get("session_id") or f"D{session_index}")
        date_value = str(session.get("date") or "")
        turns = session.get("dialogues") or []
        if max_turns is not None:
            turns = turns[: max(0, max_turns)]
        for turn_index, turn in enumerate(turns, start=1):
            global_turn += 1
            turn_id = str(turn.get("round") or f"{session_id}:{turn_index}")
            common_metadata = {
                "session_id": session_id,
                "session_date": date_value,
                "session_index": session_index,
                "turn_index": turn_index,
                "global_turn_index": global_turn,
            }
            user_text = str(turn.get("user") or "").strip()
            assistant_text = str(turn.get("assistant") or "").strip()
            turn_content = "\n".join(
                line
                for line in [
                    f"User: {user_text}" if user_text else "",
                    f"Assistant: {assistant_text}" if assistant_text else "",
                ]
                if line
            )
            if turn_content:
                records.append(
                    MemoryRecord(
                        memory_id=f"{turn_id}:turn",
                        turn_id=turn_id,
                        timestamp=observed_at(date_value, global_turn, 0),
                        author="user" if user_text else "assistant",
                        modality="text",
                        source_type="conversation",
                        summary=user_text or assistant_text,
                        content=turn_content,
                        metadata=dict(common_metadata),
                    )
                )
            image_ids = turn.get("image_id") or []
            image_paths = turn.get("input_image") or []
            captions = turn.get("image_caption") or []
            for image_index, relative_path in enumerate(image_paths):
                path = resolve_image_path(data_dir, str(relative_path))
                if not path.is_file():
                    continue
                image_id = str(
                    image_ids[image_index]
                    if image_index < len(image_ids)
                    else f"{turn_id}:IMG_{image_index + 1:03d}"
                )
                caption = str(
                    captions[image_index] if image_index < len(captions) else ""
                )
                metadata = dict(common_metadata)
                metadata.update(
                    {
                        "public_image_id": image_id,
                        "relative_path": str(relative_path),
                        "image_index": image_index,
                    }
                )
                records.append(
                    MemoryRecord(
                        memory_id=f"{turn_id}:image:{image_index + 1}",
                        turn_id=turn_id,
                        timestamp=observed_at(
                            date_value,
                            global_turn,
                            1 + image_index,
                        ),
                        author="user",
                        modality="image",
                        source_type="uploaded_image",
                        summary=caption,
                        content=f"Public image id: {image_id}",
                        raw_pointer=str(path),
                        metadata=metadata,
                    )
                )
    return records


def scenario_samples(
    data: Dict[str, Any],
    store: HiddenMemoryStore,
    data_dir: Path,
    scenario: str,
    max_questions: Optional[int] = None,
    include_oracle_profile: bool = False,
) -> List[OPDSample]:
    qas = data.get("human-annotated QAs") or []
    if max_questions is not None:
        qas = qas[: max(0, max_questions)]
    samples = []
    for index, qa in enumerate(qas, start=1):
        question_image = optional_image_path(data_dir, qa.get("question_image"))
        samples.append(
            OPDSample(
                sample_id=f"{scenario}:{index}",
                query=str(qa.get("question") or ""),
                gold_answer=str(qa.get("answer") or ""),
                memory_store=store,
                metadata={
                    "scenario": scenario,
                    "index": index,
                    "point": qa.get("point"),
                    "clue": qa.get("clue"),
                    "gold_clue_turn_ids": list(
                        dict.fromkeys(
                            TURN_ID_PATTERN.findall(
                                str(qa.get("clue") or "")
                            )
                        )
                    ),
                    "session_id": qa.get("session_id"),
                    "question_image": str(question_image) if question_image else None,
                    "gold_image_ids": IMAGE_ID_PATTERN.findall(
                        str(qa.get("answer") or "")
                    ),
                },
            )
        )
        clue_turn_ids = samples[-1].metadata["gold_clue_turn_ids"]
        if clue_turn_ids:
            profile = store.abstract_support_profile(clue_turn_ids)
            if include_oracle_profile:
                retrieval_profile = store.oracle_retrieval_profile(
                    samples[-1].query,
                    clue_turn_ids,
                    question_image=(
                        str(question_image) if question_image else None
                    ),
                )
                profile["retrieval_ranks_for_original_query"] = retrieval_profile
                profile["verified_action_advice"] = store.oracle_action_advice(
                    retrieval_profile
                )
            samples[-1].metadata["teacher_privileged_context"] = profile
        else:
            samples[-1].metadata["teacher_privileged_context"] = {}
    return samples


def build_scenario_store(
    data: Dict[str, Any],
    data_dir: Path,
    dense_encoder: Optional[DenseEncoder] = None,
    vision_encoder: Optional[VisionEncoder] = None,
    max_sessions: Optional[int] = None,
    max_turns: Optional[int] = None,
) -> tuple[HiddenMemoryStore, List[MemoryRecord]]:
    records = scenario_records(
        data,
        data_dir=data_dir,
        max_sessions=max_sessions,
        max_turns=max_turns,
    )
    return (
        HiddenMemoryStore(
            records,
            dense_encoder=dense_encoder,
            vision_encoder=vision_encoder,
        ),
        records,
    )


def iter_scenario_paths(
    data_dir: Path,
    names: Optional[Iterable[str]] = None,
) -> Iterable[Path]:
    dialog_dir = data_dir / "data" / "dialog"
    if names:
        for name in names:
            yield dialog_dir / f"{name}.json"
        return
    yield from sorted(dialog_dir.glob("*.json"))
