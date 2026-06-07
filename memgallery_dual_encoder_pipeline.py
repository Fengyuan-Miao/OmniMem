"""Mem-Gallery runner for SigLIP + MiniLM unified dual-encoder memory."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dual_encoder_memory import (  # noqa: E402
    DualEncoderMemoryStore,
    DualEncoderRetriever,
    EvidenceOrganizer,
    ImagePointer,
    MiniLMTextEncoder,
    SigLIPVisionEncoder,
    UnifiedMemoryRecord,
)
from omnimem.config import (  # noqa: E402
    PROJECT_ROOT,
    default_memgallery_dir,
    default_minilm_model,
    default_siglip_model,
    require_memgallery_dir,
)


class SharedEncoders:
    def __init__(self, args: argparse.Namespace):
        self.text_encoder = MiniLMTextEncoder(args.text_model, device=args.text_device)
        self.vision_encoder = SigLIPVisionEncoder(args.vision_model, device=args.vision_device)

DEFAULT_DATA_DIR = default_memgallery_dir()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "memgallery_dual_encoder"
IMAGE_ID_PATTERN = re.compile(r"\bD\d+:IMG_\d+\b")
IMAGE_QUERY_PATTERN = re.compile(
    r"\b(image|images|picture|pictures|photo|photos|visual|figure|figures|"
    r"shown|attached|screenshot|diagram|chart)\b",
    re.IGNORECASE,
)


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

    def advance(self, message: str = "") -> None:
        self.update(self.current + 1, message=message)

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


def make_manual_observed_at(
    date_value: str,
    session_index: int,
    turn_index: int,
    global_turn_index: int,
) -> str:
    """Create a deterministic turn-level timestamp from dataset order."""
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


def image_to_data_url(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


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


def build_answer_context(evidence_set: Any, args: argparse.Namespace) -> str:
    return evidence_set.to_prompt_context(
        group_limit=args.answer_context_limit,
        text_chars=args.evidence_text_chars,
        caption_chars=args.evidence_caption_chars,
    )


def collect_answer_images(ranked_memories: List[Any], top_n: int) -> List[ImagePointer]:
    if top_n <= 0:
        return []
    selected: List[ImagePointer] = []
    seen = set()

    def add_image(image: ImagePointer) -> None:
        key = image.image_id or image.path
        if not key or key in seen or not image.path or not Path(image.path).exists():
            return
        seen.add(key)
        selected.append(image)

    for ranked in ranked_memories:
        image_hits = sorted(
            [hit for hit in ranked.route_hits if hit.route.startswith("image")],
            key=lambda hit: (hit.rank, -hit.score),
        )
        for hit in image_hits:
            for image in ranked.memory.images:
                if (
                    (hit.image_row_id is not None and image.image_row_id == hit.image_row_id)
                    or (hit.image_id and image.image_id == hit.image_id)
                ):
                    add_image(image)
                    break
            if len(selected) >= top_n:
                return selected

    for ranked in ranked_memories:
        for image in ranked.memory.images:
            add_image(image)
            if len(selected) >= top_n:
                return selected
    return selected


def should_attach_answer_images(
    question: str,
    has_question_image: bool,
    retrieval: Any,
    mode: str,
) -> bool:
    if mode == "off":
        return False
    if mode == "always":
        return True
    if has_question_image or IMAGE_QUERY_PATTERN.search(question or ""):
        return True
    if not retrieval.ranked_memories:
        return False
    top = retrieval.ranked_memories[0]
    return top.image_score >= 0.62 and top.image_score > top.text_score + 0.10


def answer_format_instruction(question: str) -> str:
    q = normalize_answer(question)
    if re.search(r"\bwhich\s+(image|picture|photo)s?\b", q) or "image id" in q:
        return "Return only the matching public image id values if the evidence supports image ids."
    starters = (
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
    if q.startswith(starters):
        return "Return only Yes, No, or Not mentioned."
    return "Return a concise answer."


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
    question_image_caption: str,
    context: str,
    answer_images: List[ImagePointer],
    args: argparse.Namespace,
) -> tuple[str, Dict[str, Any]]:
    image_note = ""
    if answer_images:
        image_note = (
            "\nRetrieved evidence images are attached after this text. Each image is "
            "preceded by its public image id and caption. Use public image ids only "
            "when the question asks for image ids.\n"
        )
    prompt = f"""Answer the question using only the retrieved memory evidence.
{answer_format_instruction(question)}

Do not copy memory labels, file paths, or internal ids unless the question asks
for public image ids like D1:IMG_001. If evidence is insufficient, answer
"Not mentioned."
{image_note}

Question:
{question}

Current question image caption:
{question_image_caption}

Memory evidence:
{context}
"""
    messages: List[Dict[str, Any]]
    if answer_images:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in answer_images:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Retrieved evidence image public_image_id={image.image_id}; "
                        f"caption={image.caption}"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image.path)},
                }
            )
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]
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
        }
    except Exception as exc:
        return "", {
            "answer_latency_ms": round((time.time() - started) * 1000, 1),
            "answer_error": str(exc),
            "qa_vlm_calls": 1,
            "answer_image_count": len(answer_images),
            "answer_image_ids": [image.image_id for image in answer_images],
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


def ingest_scenario(
    data: Dict[str, Any],
    data_dir: Path,
    store: DualEncoderMemoryStore,
    text_encoder: MiniLMTextEncoder,
    vision_encoder: SigLIPVisionEncoder,
    args: argparse.Namespace,
) -> Dict[str, int]:
    stats = {
        "turns_seen": 0,
        "memories_stored": 0,
        "images_seen": 0,
        "images_indexed": 0,
    }
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
            manual_observed_at = make_manual_observed_at(
                date_value=date,
                session_index=session_index,
                turn_index=turn_index,
                global_turn_index=global_turn_index,
            )
            text = make_turn_text(turn)
            images: List[ImagePointer] = []
            image_ids = turn.get("image_id") or []
            input_images = turn.get("input_image") or []
            captions = turn.get("image_caption") or []
            for index, rel_image in enumerate(input_images):
                img_id = str(image_ids[index] if index < len(image_ids) else "")
                caption = str(captions[index] if index < len(captions) else "")
                path = image_path(data_dir, rel_image)
                if not path.exists():
                    continue
                images.append(
                    ImagePointer(
                        image_id=img_id,
                        path=str(path),
                        caption=caption,
                        metadata={"relative_path": str(rel_image)},
                    )
                )
                stats["images_seen"] += 1
            if not text and not images:
                continue
            memory = UnifiedMemoryRecord(
                memory_id=f"{session_id}:{turn_id}",
                text=text or " ".join(image.caption for image in images),
                session_id=session_id,
                turn_id=turn_id,
                date=date,
                images=images,
                metadata={
                    "source": "memgallery_dialogue_turn",
                    "manual_observed_at": manual_observed_at,
                    "session_index": session_index,
                    "turn_index": turn_index,
                    "global_turn_index": global_turn_index,
                },
            )
            text_embedding = text_encoder.encode(memory.text)
            stored = store.add_memory(memory, text_embedding=text_embedding)
            image_embeddings = []
            for image in stored.images:
                embedding = vision_encoder.encode_image(image.path)
                image_embeddings.append((image, embedding))
            if image_embeddings:
                store.add_memory(stored, image_embeddings=image_embeddings)
                stats["images_indexed"] += len(image_embeddings)
            stats["memories_stored"] += 1
    store.set_meta("text_model", text_encoder.model_name)
    store.set_meta("vision_model", vision_encoder.model_name)
    store.save_indexes()
    return stats


def build_image_retrieval_audit(retrieval: Any, answer: str) -> Dict[str, Any]:
    gold_ids = extract_public_image_ids(answer)
    text_ids: List[str] = []
    image_ids: List[str] = []
    reranked_ids: List[str] = []
    for hit in retrieval.text_hits:
        if hit.image_id and hit.image_id not in text_ids:
            text_ids.append(hit.image_id)
    for hit in retrieval.image_hits:
        if hit.image_id and hit.image_id not in image_ids:
            image_ids.append(hit.image_id)
    for ranked in retrieval.ranked_memories:
        for image in ranked.memory.images:
            if image.image_id and image.image_id not in reranked_ids:
                reranked_ids.append(image.image_id)

    def any_hit(found: List[str]) -> Optional[bool]:
        return bool(set(gold_ids) & set(found)) if gold_ids else None

    return {
        "gold_image_ids": gold_ids,
        "text_route_image_ids": text_ids,
        "image_route_image_ids": image_ids,
        "reranked_image_ids": reranked_ids,
        "text_route_gold_any": any_hit(text_ids),
        "image_route_gold_any": any_hit(image_ids),
        "reranked_gold_any": any_hit(reranked_ids),
    }


def run_scenario(
    path: Path,
    args: argparse.Namespace,
    encoders: Optional[SharedEncoders] = None,
) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario = path.stem
    run_dir = args.output_dir / scenario / f"{now_stamp()}_siglip_minilm"
    run_dir.mkdir(parents=True, exist_ok=True)

    if encoders is None:
        encoders = SharedEncoders(args)
    text_encoder = encoders.text_encoder
    vision_encoder = encoders.vision_encoder
    store = DualEncoderMemoryStore(run_dir)
    started = time.time()
    ingest_stats = ingest_scenario(data, args.data_dir, store, text_encoder, vision_encoder, args)
    retriever = DualEncoderRetriever(store, text_encoder, vision_encoder)
    evidence_organizer = EvidenceOrganizer(
        neighbor_turn_window=args.evidence_neighbor_window,
    )

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
            question_image_caption = str(qa.get("image_caption") or "")
            t0 = time.time()
            retrieval = retriever.retrieve(
                question,
                question_image=str(q_image_path) if q_image_path else None,
                top_k_text=args.top_k_text,
                top_k_image=args.top_k_image,
                top_k_bm25=args.top_k_bm25,
                rerank_top_k=args.rerank_top_k,
                rrf_k=args.rrf_k,
            )
            evidence_set = evidence_organizer.organize(
                retrieval,
                max_atoms=args.answer_context_limit,
            )
            answer_image_top_n = args.verify_top_n if args.verify_top_n > 0 else args.answer_image_top_n
            answer_image_mode = args.answer_image_mode
            if args.verify_top_n > 0 and answer_image_mode == "off":
                answer_image_mode = "auto"
            answer_images = (
                collect_answer_images(
                    retrieval.ranked_memories,
                    top_n=answer_image_top_n,
                )
                if should_attach_answer_images(
                    question,
                    has_question_image=bool(q_image_path),
                    retrieval=retrieval,
                    mode=answer_image_mode,
                )
                else []
            )
            answer_context = build_answer_context(
                evidence_set,
                args,
            )
            prediction, answer_fields = llm_answer(
                question=question,
                question_image_caption=question_image_caption,
                context=answer_context,
                answer_images=answer_images,
                args=args,
            )
            answer_fields["raw_image_verify_top_n"] = args.verify_top_n
            answer_fields["effective_answer_image_mode"] = answer_image_mode
            answer_fields["effective_answer_image_top_n"] = answer_image_top_n
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
                "em": exact_match(prediction, answer),
                "f1": token_f1(prediction, answer),
                "contains_gt": contains_answer(prediction, answer),
                **answer_fields,
                **judge,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "retrieval": retrieval.to_dict(),
                "evidence": evidence_set.to_dict(),
                "image_retrieval_audit": build_image_retrieval_audit(retrieval, answer),
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
    print(f"[INFO] Saved Mem-Gallery dual-encoder run: {run_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return run_dir


def summarize(
    rows: List[Dict[str, Any]],
    ingest_stats: Dict[str, int],
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
        "avg_qa_vlm_calls": avg(row.get("qa_vlm_calls", 0) for row in rows),
        "avg_answer_image_count": avg(row.get("answer_image_count", 0) for row in rows),
        "elapsed_seconds": elapsed,
        "image_retrieval": {
            "gold_image_rows": len(gold_audits),
            "text_route_gold_any": avg_bool(audit.get("text_route_gold_any") for audit in gold_audits),
            "image_route_gold_any": avg_bool(audit.get("image_route_gold_any") for audit in gold_audits),
            "reranked_gold_any": avg_bool(audit.get("reranked_gold_any") for audit in gold_audits),
        },
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
    encoders = SharedEncoders(args)
    run_dirs = []
    print(f"[INFO] Running {len(paths)} scenario(s)")
    scenario_progress = ProgressBar(
        total=len(paths),
        label="Scenarios",
        enabled=not args.no_progress,
    )
    scenario_progress.update(0)
    for idx, path in enumerate(paths, start=1):
        print(f"[INFO] Scenario {idx}/{len(paths)}: {path.stem}")
        run_dirs.append(run_scenario(path, args, encoders=encoders))
        scenario_progress.update(idx, message=path.stem)
    scenario_progress.close()
    return run_dirs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SigLIP+MiniLM dual-encoder memory on Mem-Gallery.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--text-model", type=str, default=default_minilm_model())
    parser.add_argument("--vision-model", type=str, default=default_siglip_model())
    parser.add_argument("--text-device", type=str, default="cpu")
    parser.add_argument("--vision-device", type=str, default="cpu")
    parser.add_argument("--top-k-text", type=int, default=20)
    parser.add_argument("--top-k-image", type=int, default=20)
    parser.add_argument("--top-k-bm25", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--answer-image-top-n", type=int, default=3)
    parser.add_argument("--answer-image-mode", choices=["auto", "always", "off"], default="off")
    parser.add_argument(
        "--verify-top-n",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Attach up to N retrieved raw images to the answer VLM; no extra verifier call.",
    )
    parser.add_argument("--verify-max-tokens", type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument("--answer-context-limit", type=int, default=10)
    parser.add_argument("--evidence-neighbor-window", type=int, default=1)
    parser.add_argument("--evidence-text-chars", type=int, default=1400)
    parser.add_argument("--evidence-caption-chars", type=int, default=300)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl-8b-instruct-ctx4k:latest")
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
