"""Mem-Gallery runner for GME-Qwen2-VL unified-entry memory."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from gme_memory import (  # noqa: E402
    GmeImagePointer,
    GmeMemoryRecord,
    GmeMemoryRetriever,
    GmeMemoryStore,
    GmeQwen2VLEncoder,
)
from omnimem.config import (  # noqa: E402
    PROJECT_ROOT,
    default_gme_model,
    default_memgallery_dir,
    require_memgallery_dir,
)


DEFAULT_DATA_DIR = default_memgallery_dir()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "memgallery_gme"
IMAGE_ID_PATTERN = re.compile(r"\bD\d+:IMG_\d+\b")
IMAGE_QUERY_PATTERN = re.compile(
    r"\b(image|images|picture|pictures|photo|photos|visual|figure|figures|"
    r"shown|attached|screenshot|diagram|chart)\b",
    re.IGNORECASE,
)
SYSTEM_PROMPT = """You are an AI assistant evaluated on multimodal long-term conversational memory.
For the given question-answering task, your responses must be concise, yet complete enough to accurately answer the questions.
If multiple pieces of information about the same event appear in the conversation, always rely on the most recent information.

The question-answering evaluation will contain several multimodal task types:

Factual Retrieval: Retrieve explicit facts mentioned in the conversation for the answer.

Multi-entity Reasoning: Combine the retrieved information to reason and infer an answer.

Temporal Reasoning: Resolve time-dependent questions.

Visual-centric Reasoning: Besides textual information, answer questions using visual images in the conversation.

Test-time Learning: Learn new visual knowledge from provided images within historical dialogue and use it in question-answering.

Visual-centric Search: Find the image(s) that match the information in a given query and return their image ID(s).

Conflict Detection: Detect contradictions between the conversation history and the information provided in the question.

Knowledge Resolution: Resolve knowledge conflicts or updates by prioritizing the most recent information.

Answer Refusal: Decline to answer when the information does not exist in the conversation history.

Follow all instructions strictly. Only answer using information contained within the multimodal conversation. Do not hallucinate. Always remain consistent and grounded in the dialogue history."""
MSG_START_PROMPT_WO_MEMORY = """
The retrieved memory contents are as follows:

"""
MMMEMORY_DIALOGUE_PROMPT = """
{textual_context}
image:
image_id: {image_id}
image_content:
"""
QUESTION_IMAGE_PROMPT = """
Here is the attached image of the question:
"""
DIALOGUE_AGENT_PROMPT = """
Your task is to answer the question about the conversation between {speaker_a} and {speaker_b} in a concise manner with the help of memory content.
Please only provide the content of the answer, without including introductory phrases like 'answer:'.
For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.
Generate answers primarily concise, yet complete enough to accurately answer the questions.

The current question is as follows:
{observation} {format_constraint}
"""
POINT_FORMAT_CONSTRAINTS = {
    "AR": "Provide your answer based on the information in the conversation. Only if the information about the question is not present in the conversation, reply with: \"Not mentioned.\"",
    "CD": "Please check whether this information conflicts with the conversation, and reply strictly with either \"Yes.\" or \"No.\"",
    "VS": "Return the image_id of the image(s). If there are multiple images, sort them in ascending order and separate them by commas. Format example: \"D2:IMG_003, D2:IMG_010, D10:IMG_002\" (for format reference only).",
}


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class ProgressBar:
    def __init__(self, total: int, label: str, enabled: bool = True, width: int = 28):
        self.total = max(0, int(total))
        self.label = label
        self.enabled = enabled and self.total > 0
        self.width = max(10, int(width))
        self.current = 0
        self._closed = False

    def update(self, current: Optional[int] = None, message: str = "") -> None:
        if not self.enabled:
            return
        if current is not None:
            self.current = max(0, min(int(current), self.total))
        fraction = self.current / self.total if self.total else 1.0
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" {message}" if message else ""
        sys.stderr.write(
            f"\r[{self.label}] [{bar}] {self.current}/{self.total} "
            f"{fraction * 100:5.1f}%{suffix}"
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled and not self._closed:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self._closed = True


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
    common = sum(min(pred_counts[token], ans_counts[token]) for token in pred_counts)
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


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
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
                return json.loads(cleaned[start : index + 1])
    raise ValueError("unterminated JSON object")


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


def image_to_data_url(path: str | Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def make_manual_observed_at(
    date_value: str,
    session_index: int,
    turn_index: int,
    global_turn_index: int,
) -> str:
    try:
        base = datetime.strptime(str(date_value)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        offset = timedelta(seconds=max(0, session_index - 1) * 1000 + max(0, turn_index))
        return (base + offset).isoformat().replace("+00:00", "Z")
    except ValueError:
        return f"order:{global_turn_index:06d}"


def extract_public_image_ids(value: Any) -> List[str]:
    ids: List[str] = []
    for match in IMAGE_ID_PATTERN.findall(str(value or "")):
        if match not in ids:
            ids.append(match)
    return ids


def make_turn_text(
    turn: Dict[str, Any],
    speaker_a: str = "user",
    speaker_b: str = "assistant",
) -> str:
    lines = []
    if turn.get("user"):
        lines.append(f"{speaker_a}: {turn['user']}")
    if turn.get("assistant"):
        lines.append(f"{speaker_b}: {turn['assistant']}")
    return "\n".join(lines)


def speaker_names(data: Dict[str, Any]) -> tuple[str, str]:
    profile = data.get("character_profile") or {}
    name = profile.get("name") if isinstance(profile, dict) else None
    speaker_a = f"user ({name})" if name else "user"
    return speaker_a, "assistant"


def shorten(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def first_existing_image(images: List[GmeImagePointer]) -> Optional[GmeImagePointer]:
    for image in images:
        if image.path and Path(image.path).exists():
            return image
    return None


def format_memory_text(index: int, memory: GmeMemoryRecord, args: argparse.Namespace) -> str:
    timestamp = memory.metadata.get("timestamp") or memory.date or ""
    text = shorten(memory.text, args.evidence_text_chars)
    return f"[Memory {index}] timestamp: {timestamp}\n{text}"


def build_murag_memory_dicts(
    result: Any,
    args: argparse.Namespace,
    include_images: bool,
) -> List[Dict[str, Any]]:
    memory_dicts: List[Dict[str, Any]] = []
    image_budget = args.answer_image_top_n if args.answer_image_top_n > 0 else 10**9
    image_count = 0
    entries = (
        result.entries
        if args.answer_context_limit <= 0
        else result.entries[: args.answer_context_limit]
    )
    for index, entry in enumerate(entries):
        memory = entry.memory
        image_obj = None
        image = first_existing_image(memory.images) if include_images and image_count < image_budget else None
        if image:
            image_obj = {
                "path": image.path,
                "caption": shorten(image.caption, args.evidence_caption_chars),
                "img_id": image.image_id,
            }
            image_count += 1
        memory_dicts.append(
            {
                "text": format_memory_text(index, memory, args),
                "image": image_obj,
                "timestamp": memory.metadata.get("timestamp") or memory.date,
                "dialogue_id": memory.metadata.get("dialogue_id") or memory.turn_id,
                "memory_id": memory.memory_id,
                "score": entry.score,
                "rank": entry.rank,
                "embedding_mode": entry.embedding_mode,
                "matched_image_ids": entry.matched_image_ids,
                "all_image_ids": entry.retrieved_image_ids(),
            }
        )
    return memory_dicts


def render_memory_context_preview(memory_dicts: List[Dict[str, Any]]) -> str:
    if not memory_dicts:
        return "No retrieved evidence."
    lines = ["Retrieved memory contents:"]
    for memory in memory_dicts:
        lines.append(str(memory.get("text") or ""))
        image = memory.get("image")
        if image:
            caption = image.get("caption") or ""
            lines.append(f"image_id: {image.get('img_id') or ''}")
            if caption:
                lines.append(f"image_caption: {caption}")
        lines.append("")
    return "\n".join(lines).strip()


def memory_dict_answer_images(memory_dicts: List[Dict[str, Any]]) -> List[GmeImagePointer]:
    images: List[GmeImagePointer] = []
    for memory in memory_dicts:
        image = memory.get("image")
        if not image:
            continue
        images.append(
            GmeImagePointer(
                image_id=str(image.get("img_id") or ""),
                path=str(image.get("path") or ""),
                caption=str(image.get("caption") or ""),
            )
        )
    return images


def should_attach_answer_images(
    question: str,
    has_question_image: bool,
    result: Any,
    mode: str,
) -> bool:
    if mode == "off":
        return False
    if mode == "always":
        return True
    if has_question_image or IMAGE_QUERY_PATTERN.search(question or ""):
        return True
    return False


def format_constraint_for_point(point: Any) -> str:
    point_key = str(point or "").upper()
    constraint = POINT_FORMAT_CONSTRAINTS.get(point_key, "")
    return f"\n\n{constraint}" if constraint else ""


def build_query_prompt(
    question: str,
    point: Any,
    speaker_a: str,
    speaker_b: str,
) -> str:
    return DIALOGUE_AGENT_PROMPT.format(
        observation=question,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        format_constraint=format_constraint_for_point(point),
    )


def chat_completion_http(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
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
    with opener.open(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"].get("content") or "").strip()


def llm_answer(
    question: str,
    point: Any,
    memory_dicts: List[Dict[str, Any]],
    question_image: Optional[str],
    speaker_a: str,
    speaker_b: str,
    args: argparse.Namespace,
) -> tuple[str, Dict[str, Any]]:
    query_prompt = build_query_prompt(
        question=question,
        point=point,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )
    answer_images = memory_dict_answer_images(memory_dicts)
    if memory_dicts:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": MSG_START_PROMPT_WO_MEMORY}
        ]
        for memory in memory_dicts:
            image = memory.get("image")
            if image:
                content.append(
                    {
                        "type": "text",
                        "text": MMMEMORY_DIALOGUE_PROMPT.format(
                            textual_context=memory.get("text", ""),
                            image_id=image.get("img_id", ""),
                        ),
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image.get("path", ""))},
                    }
                )
            else:
                content.append({"type": "text", "text": str(memory.get("text") or "")})
        content.append({"type": "text", "text": query_prompt})
    else:
        content = [{"type": "text", "text": query_prompt}]

    if question_image:
        content.append({"type": "text", "text": QUESTION_IMAGE_PROMPT})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(question_image)},
            }
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    started = time.time()
    try:
        answer = chat_completion_http(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.vlm_model,
            messages=messages,
            temperature=0.0,
            max_tokens=args.max_answer_tokens,
        )
        return answer.strip(), {
            "answer_latency_ms": round((time.time() - started) * 1000, 1),
            "answer_error": "",
            "qa_vlm_calls": 1,
            "answer_image_count": len(answer_images),
            "answer_image_ids": [image.image_id for image in answer_images],
            "answer_image_paths": [image.path for image in answer_images],
            "question_image_attached": bool(question_image),
        }
    except Exception as exc:
        return "", {
            "answer_latency_ms": round((time.time() - started) * 1000, 1),
            "answer_error": str(exc),
            "qa_vlm_calls": 1,
            "answer_image_count": len(answer_images),
            "answer_image_ids": [image.image_id for image in answer_images],
            "answer_image_paths": [image.path for image in answer_images],
            "question_image_attached": bool(question_image),
        }


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


def build_memory_records(
    data: Dict[str, Any],
    data_dir: Path,
    args: argparse.Namespace,
) -> tuple[List[GmeMemoryRecord], Dict[str, int]]:
    stats = {
        "turns_seen": 0,
        "memories_stored": 0,
        "images_seen": 0,
        "image_backed_entries": 0,
        "entries_embedded": 0,
    }
    records: List[GmeMemoryRecord] = []
    speaker_a, speaker_b = speaker_names(data)
    sessions = data.get("multi_session_dialogues") or []
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]

    for session_index, session in enumerate(sessions, start=1):
        session_id = str(session.get("session_id") or "")
        date = str(session.get("date") or "")
        turns = session.get("dialogues") or []
        if args.max_turns is not None:
            turns = turns[: args.max_turns]
        for turn_index, turn in enumerate(turns, start=1):
            stats["turns_seen"] += 1
            global_turn_index = stats["turns_seen"]
            turn_id = str(turn.get("round") or "")
            text = make_turn_text(turn, speaker_a=speaker_a, speaker_b=speaker_b)
            if not text:
                continue
            images: List[GmeImagePointer] = []
            image_ids = turn.get("image_id") or []
            input_images = turn.get("input_image") or []
            captions = turn.get("image_caption") or []
            if input_images:
                rel_image = input_images[0]
                img_id = str(image_ids[0] if image_ids else "")
                caption = str(captions[0] if captions else "")
                path = image_path(data_dir, rel_image)
                if path.exists():
                    images.append(
                        GmeImagePointer(
                            image_id=img_id,
                            path=str(path),
                            caption=caption,
                            metadata={"relative_path": str(rel_image)},
                        )
                    )
                    stats["images_seen"] += 1
            if not text and not images:
                continue
            manual_observed_at = make_manual_observed_at(
                date_value=date,
                session_index=session_index,
                turn_index=turn_index,
                global_turn_index=global_turn_index,
            )
            records.append(
                GmeMemoryRecord(
                    memory_id=f"{session_id}:{turn_id}",
                    text=text,
                    session_id=session_id,
                    turn_id=turn_id,
                    date=date,
                    images=images,
                    metadata={
                        "source": "memgallery_murag_observation",
                        "embedding_text": text,
                        "timestamp": date,
                        "dialogue_id": turn_id,
                        "counter_id": len(records),
                        "manual_observed_at": manual_observed_at,
                        "session_index": session_index,
                        "turn_index": turn_index,
                        "global_turn_index": global_turn_index,
                    },
                )
            )
    return records, stats


def ingest_scenario(
    data: Dict[str, Any],
    data_dir: Path,
    store: GmeMemoryStore,
    encoder: GmeQwen2VLEncoder,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    records, stats = build_memory_records(data, data_dir, args)
    started = time.time()
    embeddings = encoder.encode_entries(
        records,
        batch_size=args.gme_batch_size,
        show_progress_bar=False,
    )
    by_memory = {embedding.memory_id: embedding for embedding in embeddings}
    for record in records:
        embedding = by_memory.get(record.memory_id)
        if embedding is None:
            continue
        store.add_memory(
            record,
            entry_embedding=embedding.vector,
            embedding_mode=embedding.embedding_mode,
            embedding_image_id=embedding.image_id,
        )
        stats["memories_stored"] += 1
        stats["entries_embedded"] += 1
        if embedding.embedding_mode in {"image_text_pair", "image"}:
            stats["image_backed_entries"] += 1
    store.set_meta("gme_model", encoder.model_path)
    store.set_meta("gme_device", encoder.device)
    store.set_meta("gme_query_instruction", encoder.query_instruction)
    store.save_index()
    stats["gme_ingest_embedding_seconds"] = time.time() - started
    return stats


def build_image_retrieval_audit(result: Any, answer: str, answer_images: List[GmeImagePointer]) -> Dict[str, Any]:
    gold_ids = extract_public_image_ids(answer)
    retrieved_ids: List[str] = []
    embedding_image_ids: List[str] = []
    for entry in result.entries:
        for image_id in entry.matched_image_ids:
            if image_id and image_id not in embedding_image_ids:
                embedding_image_ids.append(image_id)
        for image in entry.memory.images:
            if image.image_id and image.image_id not in retrieved_ids:
                retrieved_ids.append(image.image_id)
    answer_image_ids = []
    for image in answer_images:
        if image.image_id and image.image_id not in answer_image_ids:
            answer_image_ids.append(image.image_id)

    def any_hit(found: List[str]) -> Optional[bool]:
        return bool(set(gold_ids) & set(found)) if gold_ids else None

    return {
        "gold_image_ids": gold_ids,
        "embedding_image_ids": embedding_image_ids,
        "retrieved_entry_image_ids": retrieved_ids,
        "answer_image_ids": answer_image_ids,
        "embedding_image_gold_any": any_hit(embedding_image_ids),
        "retrieved_entry_gold_any": any_hit(retrieved_ids),
        "answer_image_gold_any": any_hit(answer_image_ids),
    }


def retrieved_dialogue_ids(result: Any) -> List[str]:
    ids: List[str] = []
    for entry in result.entries:
        value = entry.memory.metadata.get("dialogue_id") or entry.memory.turn_id
        value = str(value or "")
        if value and value not in ids:
            ids.append(value)
    return ids


def run_scenario(
    path: Path,
    args: argparse.Namespace,
    encoder: Optional[GmeQwen2VLEncoder] = None,
) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario = path.stem
    speaker_a, speaker_b = speaker_names(data)
    run_dir = args.output_dir / scenario / f"{now_stamp()}_gme_qwen2vl_unified"
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder = encoder or GmeQwen2VLEncoder(
        model_path=args.gme_model,
        device=args.gme_device,
        min_image_tokens=args.gme_min_image_tokens,
        max_image_tokens=args.gme_max_image_tokens,
        max_length=args.gme_max_length,
        query_instruction=args.gme_query_instruction or None,
        trust_remote_code=args.gme_trust_remote_code,
    )
    store = GmeMemoryStore(run_dir)
    started = time.time()
    ingest_stats = ingest_scenario(data, args.data_dir, store, encoder, args)
    retriever = GmeMemoryRetriever(store, encoder)

    qas = data.get("human-annotated QAs") or []
    if args.max_questions is not None:
        qas = qas[: args.max_questions]

    rows: List[Dict[str, Any]] = []
    predictions_path = run_dir / "predictions.jsonl"
    qa_progress = ProgressBar(
        total=len(qas),
        label=f"{scenario} QA",
        enabled=not args.no_progress,
    )
    qa_progress.update(0)
    with predictions_path.open("w", encoding="utf-8") as f:
        for index, qa in enumerate(qas, start=1):
            question = str(qa.get("question") or "")
            q_image_path = optional_image_path(args.data_dir, qa.get("question_image"))
            t0 = time.time()
            retrieval_started = time.time()
            result = retriever.retrieve(
                question,
                question_image=str(q_image_path) if q_image_path else None,
                top_k=args.top_k,
            )
            gme_retrieval_latency_ms = round((time.time() - retrieval_started) * 1000, 1)
            include_answer_images = should_attach_answer_images(
                question=question,
                has_question_image=bool(q_image_path),
                result=result,
                mode=args.answer_image_mode,
            )
            memory_dicts = build_murag_memory_dicts(
                result,
                args,
                include_images=include_answer_images,
            )
            answer_images = memory_dict_answer_images(memory_dicts)
            answer_context = render_memory_context_preview(memory_dicts)
            prediction, answer_fields = llm_answer(
                question=question,
                point=qa.get("point"),
                memory_dicts=memory_dicts,
                question_image=str(q_image_path) if q_image_path else None,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                args=args,
            )
            answer = str(qa.get("answer") or "")
            judge = llm_judge_answer(
                question=question,
                answer=answer,
                prediction=prediction,
                point=qa.get("point"),
                args=args,
            )
            row = {
                "index": index,
                "scenario": scenario,
                "point": qa.get("point"),
                "question": qa.get("question"),
                "question_image": str(q_image_path) if q_image_path else None,
                "answer": answer,
                "prediction": prediction,
                "answer_context_preview": answer_context[: args.debug_context_chars],
                "memory_context": memory_dicts,
                "em": exact_match(prediction, answer),
                "f1": token_f1(prediction, answer),
                "contains_gt": contains_answer(prediction, answer),
                **answer_fields,
                **judge,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "gme_retrieval_latency_ms": gme_retrieval_latency_ms,
                "retrieval": result.to_dict(),
                "retrieved_image_ids": [
                    image_id
                    for entry in result.entries
                    for image_id in entry.retrieved_image_ids()
                ],
                "retrieved_ids": retrieved_dialogue_ids(result),
                "image_retrieval_audit": build_image_retrieval_audit(result, answer, answer_images),
                "clue": qa.get("clue"),
                "session_id": qa.get("session_id"),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            qa_progress.update(
                index,
                message=f"point={qa.get('point') or ''} judge={judge.get('judge_correct')}",
            )
    qa_progress.close()

    metrics = summarize(rows, ingest_stats, store.stats(), time.time() - started)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = vars(args) | {"scenario": scenario}
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    store.close()
    print(f"[INFO] Saved Mem-Gallery GME run: {run_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return run_dir


def summarize(
    rows: List[Dict[str, Any]],
    ingest_stats: Dict[str, Any],
    store_stats: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    judged_rows = [row for row in rows if row.get("judge_correct") is not None]
    judge_errors = [row for row in judged_rows if row.get("judge_error")]
    audits = [row.get("image_retrieval_audit") or {} for row in rows]
    gold_audits = [audit for audit in audits if audit.get("gold_image_ids")]

    def avg(items: Iterable[float]) -> float:
        values = list(items)
        return sum(values) / len(values) if values else 0.0

    def avg_bool(items: Iterable[Optional[bool]]) -> float:
        values = [1.0 if item else 0.0 for item in items if item is not None]
        return avg(values)

    by_point: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        point = str(row.get("point") or "UNKNOWN")
        rec = by_point.setdefault(
            point,
            {"count": 0, "judge_correct_sum": 0.0, "judge_score_sum": 0.0},
        )
        rec["count"] += 1
        rec["judge_correct_sum"] += float(row.get("judge_correct") or 0)
        rec["judge_score_sum"] += float(row.get("judge_score") or 0.0)
    by_point_metrics = {
        point: {
            "count": rec["count"],
            "judge_accuracy": rec["judge_correct_sum"] / rec["count"] if rec["count"] else 0.0,
            "judge_score": rec["judge_score_sum"] / rec["count"] if rec["count"] else 0.0,
        }
        for point, rec in sorted(by_point.items())
    }

    return {
        "num_results": len(rows),
        "primary_metric": "judge_accuracy",
        "judge_accuracy": avg(row["judge_correct"] for row in judged_rows),
        "judge_score": avg(row["judge_score"] for row in judged_rows),
        "judge_error_rate": len(judge_errors) / len(judged_rows) if judged_rows else 0.0,
        "em": avg(row["em"] for row in rows),
        "f1": avg(row["f1"] for row in rows),
        "contains_gt": avg(row["contains_gt"] for row in rows),
        "avg_latency_ms": avg(row["latency_ms"] for row in rows),
        "avg_gme_retrieval_latency_ms": avg(row["gme_retrieval_latency_ms"] for row in rows),
        "avg_qa_vlm_calls": avg(row.get("qa_vlm_calls", 0) for row in rows),
        "avg_answer_image_count": avg(row.get("answer_image_count", 0) for row in rows),
        "elapsed_seconds": elapsed,
        "image_retrieval": {
            "gold_image_rows": len(gold_audits),
            "embedding_image_gold_any": avg_bool(audit.get("embedding_image_gold_any") for audit in gold_audits),
            "retrieved_entry_gold_any": avg_bool(audit.get("retrieved_entry_gold_any") for audit in gold_audits),
            "answer_image_gold_any": avg_bool(audit.get("answer_image_gold_any") for audit in gold_audits),
        },
        "by_point": by_point_metrics,
        **ingest_stats,
        **store_stats,
    }


def iter_scenarios(data_dir: Path, scenario: Optional[str]) -> Iterable[Path]:
    dialog_dir = data_dir / "data" / "dialog"
    if scenario:
        yield dialog_dir / f"{scenario}.json"
        return
    yield from sorted(dialog_dir.glob("*.json"))


def parse_scenario_names(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()]


def resolve_scenario_paths(args: argparse.Namespace) -> List[Path]:
    if args.all_scenarios:
        paths = list(iter_scenarios(args.data_dir, None))
    else:
        names = parse_scenario_names(args.scenarios)
        if not names:
            names = [args.scenario or "Academic_Animal_Pet_Research_Life"]
        paths = [path for name in names for path in iter_scenarios(args.data_dir, name)]
    if args.max_scenarios is not None:
        paths = paths[: max(0, args.max_scenarios)]
    return paths


def run_scenarios(args: argparse.Namespace) -> List[Path]:
    paths = resolve_scenario_paths(args)
    encoder = GmeQwen2VLEncoder(
        model_path=args.gme_model,
        device=args.gme_device,
        min_image_tokens=args.gme_min_image_tokens,
        max_image_tokens=args.gme_max_image_tokens,
        max_length=args.gme_max_length,
        query_instruction=args.gme_query_instruction or None,
        trust_remote_code=args.gme_trust_remote_code,
    )
    run_dirs = []
    print(f"[INFO] Running {len(paths)} scenario(s)")
    scenario_progress = ProgressBar(
        total=len(paths),
        label="Scenarios",
        enabled=not args.no_progress,
    )
    scenario_progress.update(0)
    for index, path in enumerate(paths, start=1):
        print(f"[INFO] Scenario {index}/{len(paths)}: {path.stem}")
        run_dirs.append(run_scenario(path, args, encoder=encoder))
        scenario_progress.update(index, message=path.stem)
    scenario_progress.close()
    return run_dirs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GME-Qwen2-VL unified-entry memory on Mem-Gallery.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--gme-model", type=str, default=default_gme_model())
    parser.add_argument("--gme-device", type=str, default="cuda:1")
    parser.add_argument("--gme-batch-size", type=int, default=4)
    parser.add_argument("--gme-min-image-tokens", type=int, default=256)
    parser.add_argument("--gme-max-image-tokens", type=int, default=1280)
    parser.add_argument("--gme-max-length", type=int, default=1800)
    parser.add_argument("--gme-query-instruction", type=str, default="")
    parser.add_argument("--gme-trust-remote-code", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--answer-image-top-n", type=int, default=0)
    parser.add_argument("--answer-image-mode", choices=["auto", "always", "off"], default="always")
    parser.add_argument("--answer-context-limit", type=int, default=0)
    parser.add_argument("--evidence-text-chars", type=int, default=0)
    parser.add_argument("--evidence-caption-chars", type=int, default=0)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl-8b-instruct-ctx8k:latest")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:11435/v1")
    parser.add_argument("--api-key", type=str, default="ollama")
    parser.add_argument("--judge-mode", choices=["llm", "off"], default="llm")
    parser.add_argument("--judge-base-url", type=str, default="http://127.0.0.1:11436/v1")
    parser.add_argument("--judge-model", type=str, default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--judge-api-key", type=str, default="ollama")
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--debug-context-chars", type=int, default=6000)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = require_memgallery_dir(args.data_dir)
    args.output_dir = args.output_dir.resolve()
    run_scenarios(args)


if __name__ == "__main__":
    main()
