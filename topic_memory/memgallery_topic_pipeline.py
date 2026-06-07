"""Mem-Gallery runner for topic-gated multimodal memory."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dual_encoder_memory import ImagePointer, MiniLMTextEncoder, SigLIPVisionEncoder, UnifiedMemoryRecord  # noqa: E402
from memgallery_dual_encoder_pipeline import (  # noqa: E402
    ProgressBar,
    answer_format_instruction,
    chat_completion_http,
    collect_answer_images,
    contains_answer,
    exact_match,
    extract_json_object,
    extract_public_image_ids,
    image_path,
    image_to_data_url,
    llm_answer,
    llm_judge_answer,
    make_turn_text,
    optional_image_path,
    token_f1,
)
from topic_memory import (  # noqa: E402
    TopicBuilder,
    TopicMemoryStore,
    TopicRecord,
    TopicRouteDecision,
    TopicScopedRetriever,
    build_ordered_topic_evidence_context,
    build_topic_index_context,
)
from topic_memory.topic_builder import TopicLLMClient  # noqa: E402
from omnimem.config import (  # noqa: E402
    PROJECT_ROOT,
    default_memgallery_dir,
    default_minilm_model,
    default_siglip_model,
    require_memgallery_dir,
)


DEFAULT_DATA_DIR = default_memgallery_dir()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "memgallery_topic_memory"
ALLOWED_MODALITIES = {"text", "image"}


class SharedEncoders:
    def __init__(self, args: argparse.Namespace):
        self.text_encoder = MiniLMTextEncoder(args.text_model, device=args.text_device)
        self.vision_encoder = SigLIPVisionEncoder(args.vision_model, device=args.vision_device)


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_turn_number(turn_id: str) -> int:
    values = re.findall(r"\d+", str(turn_id or ""))
    return int(values[-1]) if values else 0


def scenario_image_pointers(data_dir: Path, turn: Dict[str, Any]) -> List[ImagePointer]:
    images: List[ImagePointer] = []
    image_ids = turn.get("image_id") or []
    input_images = turn.get("input_image") or []
    captions = turn.get("image_caption") or []
    for index, rel_image in enumerate(input_images):
        path = image_path(data_dir, rel_image)
        if not path.exists():
            continue
        images.append(
            ImagePointer(
                image_id=str(image_ids[index] if index < len(image_ids) else ""),
                path=str(path),
                caption=str(captions[index] if index < len(captions) else ""),
                metadata={"relative_path": str(rel_image)},
            )
        )
    return images


def prefilter_router_topics(
    store: TopicMemoryStore,
    text_encoder: MiniLMTextEncoder,
    question: str,
    top_k: int,
) -> List[TopicRecord]:
    topics = store.list_topics()
    if top_k <= 0 or len(topics) <= top_k:
        return topics
    embedding = text_encoder.encode(question)
    return [topic for topic, _score in store.candidate_topics(embedding, top_k)]


def route_topics_with_vlm(
    question: str,
    question_image_caption: str,
    question_image_path: Optional[Path],
    topic_index: Sequence[TopicRecord],
    args: argparse.Namespace,
) -> TopicRouteDecision:
    valid_topic_ids = {topic.topic_id for topic in topic_index}
    index_context = build_topic_index_context(topic_index)
    prompt = f"""You are the first-stage router for a long-term memory QA system.

Given the user's question and the available memory topic index, decide whether
memory retrieval is needed.

If the question is unrelated to all available topics, answer directly and set
use_memory=false. If it is related, set use_memory=true, select relevant topic
ids, and select retrieval modalities.

Modality choices:
- "text": use dialogue text memories.
- "image": use image memories and visual retrieval.
- Use ["text", "image"] when both are useful.

Return only valid JSON:
{{
  "use_memory": true,
  "direct_answer": "",
  "topics": ["T001"],
  "modalities": ["text", "image"],
  "reason": "short"
}}

Question:
{question}

Current question image caption:
{question_image_caption}

Topic index:
{index_context}
"""
    messages: List[Dict[str, Any]]
    if question_image_path is not None:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(question_image_path)},
            }
        )
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    started = time.time()
    raw: Dict[str, Any] = {}
    error = ""
    try:
        content = chat_completion_http(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.vlm_model,
            messages=messages,
            temperature=0.0,
            max_tokens=args.router_max_tokens,
        )
        raw = extract_json_object(content)
        use_memory = bool(raw.get("use_memory"))
        direct_answer = str(raw.get("direct_answer") or raw.get("answer") or "").strip()
        topics = [
            str(item)
            for item in (raw.get("topics") or [])
            if str(item) in valid_topic_ids
        ]
        modalities = [
            str(item).lower()
            for item in (raw.get("modalities") or [])
            if str(item).lower() in ALLOWED_MODALITIES
        ]
        if use_memory and not topics:
            use_memory = False
            error = "router_selected_no_valid_topics"
        if use_memory and not modalities:
            modalities = ["text"]
        if not use_memory and not direct_answer:
            direct_answer = "Not mentioned."
        return TopicRouteDecision(
            use_memory=use_memory,
            direct_answer=direct_answer,
            topics=topics,
            modalities=modalities,
            reason=str(raw.get("reason") or ""),
            raw_response=raw,
            error=error,
            latency_ms=round((time.time() - started) * 1000, 1),
        )
    except Exception as exc:
        return TopicRouteDecision(
            use_memory=False,
            direct_answer="Not mentioned.",
            topics=[],
            modalities=[],
            reason="",
            raw_response=raw,
            error=str(exc),
            latency_ms=round((time.time() - started) * 1000, 1),
        )


def ingest_scenario(
    data: Dict[str, Any],
    data_dir: Path,
    store: TopicMemoryStore,
    text_encoder: MiniLMTextEncoder,
    vision_encoder: SigLIPVisionEncoder,
    topic_builder: TopicBuilder,
    args: argparse.Namespace,
) -> Dict[str, int]:
    stats = {
        "turns_seen": 0,
        "memories_stored": 0,
        "images_seen": 0,
        "images_indexed": 0,
        "topic_assignment_errors": 0,
    }
    sessions = data.get("multi_session_dialogues") or []
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]
    total_turns = 0
    for session in sessions:
        turns = session.get("dialogues") or []
        if args.max_turns is not None:
            turns = turns[: args.max_turns]
        total_turns += len(turns)
    ingest_progress = ProgressBar(
        total_turns,
        "Topic ingest",
        enabled=not args.no_progress,
    )
    ingest_progress.update(0)

    for session in sessions:
        session_id = str(session.get("session_id") or "")
        date = str(session.get("date") or "")
        turns = session.get("dialogues") or []
        if args.max_turns is not None:
            turns = turns[: args.max_turns]
        for turn in turns:
            stats["turns_seen"] += 1
            turn_id = str(turn.get("round") or "")
            user_query = str(turn.get("user") or "")
            images = scenario_image_pointers(data_dir, turn)
            stats["images_seen"] += len(images)
            text = make_turn_text(turn)
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
                    "turn_number": parse_turn_number(turn_id),
                },
            )
            text_embedding = text_encoder.encode(memory.text)
            stored = store.add_memory(memory, text_embedding=text_embedding)
            image_embeddings = []
            for image in stored.images:
                image_embeddings.append((image, vision_encoder.encode_image(image.path)))
            if image_embeddings:
                store.add_memory(stored, image_embeddings=image_embeddings)
                stats["images_indexed"] += len(image_embeddings)
            assignment = topic_builder.assign_turn(user_query=user_query, memory=stored)
            if assignment.error:
                stats["topic_assignment_errors"] += 1
            stats["memories_stored"] += 1
            ingest_progress.update(
                stats["turns_seen"],
                message=f"topics={len(store.list_topics())}",
            )
    ingest_progress.close()
    store.base.set_meta("text_model", text_encoder.model_name)
    store.base.set_meta("vision_model", vision_encoder.model_name)
    store.save_indexes()
    return stats


def build_image_retrieval_audit(retrieval: Any, answer: str) -> Dict[str, Any]:
    gold_ids = extract_public_image_ids(answer)
    text_ids: List[str] = []
    image_ids: List[str] = []
    reranked_ids: List[str] = []
    if retrieval is None:
        return {
            "gold_image_ids": gold_ids,
            "text_route_image_ids": [],
            "image_route_image_ids": [],
            "reranked_image_ids": [],
            "text_route_gold_any": None if gold_ids else None,
            "image_route_gold_any": None if gold_ids else None,
            "reranked_gold_any": None if gold_ids else None,
        }
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
    run_dir = args.output_dir / scenario / f"{now_stamp()}_topic_memory"
    run_dir.mkdir(parents=True, exist_ok=True)

    if encoders is None:
        encoders = SharedEncoders(args)
    text_encoder = encoders.text_encoder
    vision_encoder = encoders.vision_encoder
    store = TopicMemoryStore(run_dir)
    topic_llm = TopicLLMClient(
        base_url=args.topic_llm_base_url,
        model=args.topic_llm_model,
        api_key=args.topic_llm_api_key,
        max_tokens=args.topic_llm_max_tokens,
    )
    topic_builder = TopicBuilder(
        store=store,
        text_encoder=text_encoder,
        llm_client=topic_llm,
        match_top_k=args.topic_match_top_k,
    )
    started = time.time()
    ingest_stats = ingest_scenario(
        data,
        args.data_dir,
        store,
        text_encoder,
        vision_encoder,
        topic_builder,
        args,
    )
    retriever = TopicScopedRetriever(store, text_encoder, vision_encoder)

    qas = data.get("human-annotated QAs") or []
    if args.max_questions is not None:
        qas = qas[: args.max_questions]

    rows: List[Dict[str, Any]] = []
    progress = ProgressBar(len(qas), f"{scenario} QA", enabled=not args.no_progress)
    progress.update(0)
    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for index, qa in enumerate(qas, start=1):
            question = str(qa.get("question") or "")
            answer = str(qa.get("answer") or "")
            q_image_path = optional_image_path(args.data_dir, qa.get("question_image"))
            question_image_caption = str(qa.get("image_caption") or "")
            t0 = time.time()
            router_topics = prefilter_router_topics(
                store,
                text_encoder,
                question,
                args.router_topic_top_k,
            )
            router = route_topics_with_vlm(
                question=question,
                question_image_caption=question_image_caption,
                question_image_path=q_image_path,
                topic_index=router_topics,
                args=args,
            )
            retrieval = None
            answer_context = ""
            answer_fields: Dict[str, Any] = {
                "answer_latency_ms": 0.0,
                "answer_error": "",
                "answer_image_count": 0,
                "answer_image_ids": [],
            }
            if router.use_memory:
                selected_topics = store.get_topics(router.topics)
                retrieval = retriever.retrieve(
                    query=question,
                    topic_ids=router.topics,
                    modalities=router.modalities,
                    question_image=str(q_image_path) if q_image_path else None,
                    top_k_text=args.top_k_text,
                    top_k_image=args.top_k_image,
                    rerank_top_k=args.rerank_top_k,
                )
                answer_context = build_ordered_topic_evidence_context(
                    store=store,
                    selected_topics=selected_topics,
                    ranked_memories=retrieval.ranked_memories,
                    memory_limit=args.answer_context_limit,
                    text_chars=args.evidence_text_chars,
                    caption_chars=args.evidence_caption_chars,
                )
                answer_images = (
                    collect_answer_images(
                        retrieval.ranked_memories,
                        top_n=args.answer_image_top_n,
                    )
                    if "image" in set(router.modalities)
                    else []
                )
                prediction, answer_fields = llm_answer(
                    question=question,
                    question_image_caption=question_image_caption,
                    context=answer_context,
                    answer_images=answer_images,
                    args=args,
                )
                answer_vlm_calls = 1
            else:
                prediction = router.direct_answer
                answer_vlm_calls = 0

            router_vlm_calls = 1
            total_vlm_calls = router_vlm_calls + answer_vlm_calls
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
                "router": router.to_dict(),
                "direct_answer": not router.use_memory,
                "selected_topics": router.topics,
                "modalities": router.modalities,
                "answer_context_preview": answer_context[: args.debug_context_chars],
                "em": exact_match(prediction, answer),
                "f1": token_f1(prediction, answer),
                "contains_gt": contains_answer(prediction, answer),
                **answer_fields,
                **judge,
                "router_vlm_calls": router_vlm_calls,
                "answer_vlm_calls": answer_vlm_calls,
                "qa_vlm_calls": total_vlm_calls,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "topic_retrieval": retrieval.to_dict() if retrieval is not None else None,
                "image_retrieval_audit": build_image_retrieval_audit(retrieval, answer),
                "clue": qa.get("clue"),
                "session_id": qa.get("session_id"),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            progress.update(index, message=f"point={qa.get('point') or ''} judge={judge.get('judge_correct')}")
    progress.close()

    store.dump_topics_jsonl(run_dir / "topics.jsonl")
    store.dump_topic_assignments_jsonl(run_dir / "topic_assignments.jsonl")
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
    print(f"[INFO] Saved Mem-Gallery topic-memory run: {run_dir}")
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
    by_point: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_point[str(row.get("point") or "")].append(row)

    def avg(items: Iterable[float]) -> float:
        values = [item for item in items if item is not None]
        return sum(values) / len(values) if values else 0.0

    def avg_bool(items: Iterable[Optional[bool]]) -> float:
        values = [1.0 if item else 0.0 for item in items if item is not None]
        return avg(values)

    def row_avg(items: List[Dict[str, Any]], key: str) -> float:
        return avg(row.get(key, 0.0) for row in items)

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
        "avg_total_vlm_calls": avg(row.get("qa_vlm_calls", 0) for row in rows),
        "direct_answer_rate": avg(1.0 if row.get("direct_answer") else 0.0 for row in rows),
        "avg_selected_topics": avg(len(row.get("selected_topics") or []) for row in rows),
        "avg_answer_image_count": avg(row.get("answer_image_count", 0) for row in rows),
        "elapsed_seconds": elapsed,
        "image_retrieval": {
            "gold_image_rows": len(gold_audits),
            "text_route_gold_any": avg_bool(audit.get("text_route_gold_any") for audit in gold_audits),
            "image_route_gold_any": avg_bool(audit.get("image_route_gold_any") for audit in gold_audits),
            "reranked_gold_any": avg_bool(audit.get("reranked_gold_any") for audit in gold_audits),
        },
        "by_point": {
            point: {
                "count": len(items),
                "judge_accuracy": row_avg(items, "judge_correct"),
                "judge_score": row_avg(items, "judge_score"),
                "direct_answer_rate": avg(1.0 if row.get("direct_answer") else 0.0 for row in items),
            }
            for point, items in sorted(by_point.items())
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
    progress = ProgressBar(len(paths), "Scenarios", enabled=not args.no_progress)
    progress.update(0)
    for idx, path in enumerate(paths, start=1):
        print(f"[INFO] Scenario {idx}/{len(paths)}: {path.stem}")
        run_dirs.append(run_scenario(path, args, encoders=encoders))
        progress.update(idx, message=path.stem)
    progress.close()
    return run_dirs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run topic-gated memory on Mem-Gallery.")
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
    parser.add_argument("--topic-llm-base-url", type=str, default="http://127.0.0.1:11436/v1")
    parser.add_argument("--topic-llm-model", type=str, default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--topic-llm-api-key", type=str, default="ollama")
    parser.add_argument("--topic-llm-max-tokens", type=int, default=256)
    parser.add_argument("--topic-match-top-k", type=int, default=12)
    parser.add_argument("--router-topic-top-k", type=int, default=30)
    parser.add_argument("--router-max-tokens", type=int, default=256)
    parser.add_argument("--top-k-text", type=int, default=20)
    parser.add_argument("--top-k-image", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--answer-context-limit", type=int, default=10)
    parser.add_argument("--evidence-text-chars", type=int, default=1400)
    parser.add_argument("--evidence-caption-chars", type=int, default=300)
    parser.add_argument("--answer-image-top-n", type=int, default=3)
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
