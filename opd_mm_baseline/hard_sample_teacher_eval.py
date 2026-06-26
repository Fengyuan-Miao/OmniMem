"""Evaluate teacher planning on a mixed hard-sample manifest."""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dual_encoder_memory import MiniLMTextEncoder, SigLIPVisionEncoder
from omnimem.config import (
    PROJECT_ROOT,
    default_minilm_model,
    default_siglip_model,
)

from .memeye import (
    build_scenario_store as build_memeye_store,
    normalize_memeye_data_dir,
    scenario_samples as memeye_scenario_samples,
)
from .memgallery import (
    IMAGE_ID_PATTERN,
    build_scenario_store as build_memgallery_store,
    scenario_samples as memgallery_scenario_samples,
)
from .memgallery_interactive_pipeline import (
    _decision_rows,
    _evidence_redundancy,
    _safe_judge,
    make_components,
)
from .memgallery_pipeline import ProgressBar, now_stamp
from .models import EvidenceItem, MemoryRecord, OPDSample
from .retrieval import HiddenMemoryStore


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "hard_memory_samples"
    / "hard_memory_samples_500_memgallery_main.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "hard_sample_teacher_eval"
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT.parent / "benchmark"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def locomo_records(sample: Dict[str, Any]) -> List[MemoryRecord]:
    records: List[MemoryRecord] = []
    conversation = sample.get("conversation") or {}
    session_numbers = sorted(
        int(key.split("_", 1)[1])
        for key in conversation
        if key.startswith("session_")
        and key.split("_", 1)[1].isdigit()
    )
    global_turn = 0
    for session_index, session_num in enumerate(session_numbers, start=1):
        session_key = f"session_{session_num}"
        session_id = f"D{session_num}"
        date_time = str(conversation.get(f"{session_key}_date_time") or "")
        turns = conversation.get(session_key) or []
        for turn_index, turn in enumerate(turns, start=1):
            global_turn += 1
            dia_id = str(turn.get("dia_id") or f"{session_id}:{turn_index}")
            text = str(turn.get("text") or "").strip()
            caption = str(turn.get("blip_caption") or "").strip()
            content_parts = []
            speaker = str(turn.get("speaker") or "").strip()
            if text:
                content_parts.append(f"{speaker}: {text}" if speaker else text)
            if caption:
                content_parts.append(f"Image caption: {caption}")
            content = "\n".join(content_parts)
            if not content:
                continue
            records.append(
                MemoryRecord(
                    memory_id=f"{dia_id}:turn",
                    turn_id=dia_id,
                    timestamp=date_time or f"order:{global_turn:06d}",
                    author=speaker,
                    modality="text",
                    source_type="conversation",
                    summary=text or caption,
                    content=content,
                    raw_pointer=None,
                    metadata={
                        "session_id": session_id,
                        "session_date": date_time,
                        "session_index": session_index,
                        "turn_index": turn_index,
                        "global_turn_index": global_turn,
                        "locomo_sample_id": sample.get("sample_id"),
                        "has_image_metadata": bool(
                            turn.get("img_url") or caption
                        ),
                    },
                )
            )
    return records


def locomo_sample_from_row(
    row: Dict[str, Any],
    store: HiddenMemoryStore,
) -> OPDSample:
    return OPDSample(
        sample_id=str(row["sample_id"]),
        query=str(row["question"]),
        gold_answer=str(row["answer"]),
        memory_store=store,
        metadata={
            "dataset": "LoCoMo",
            "scenario": row.get("domain"),
            "category": row.get("category"),
            "point": row.get("point"),
            "gold_clue_turn_ids": list(row.get("evidence") or []),
            "gold_image_ids": [],
            "question_image": None,
            "hard_sample": row,
        },
    )


def _sample_map(samples: List[OPDSample]) -> Dict[str, OPDSample]:
    return {sample.sample_id: sample for sample in samples}


class HardSampleResolver:
    def __init__(
        self,
        *,
        memgallery_dir: Path,
        memeye_dir: Path,
        locomo_path: Path,
        dense_encoder: Optional[Any],
        vision_encoder: Optional[Any],
    ):
        self.memgallery_dir = memgallery_dir
        self.memeye_dir = normalize_memeye_data_dir(memeye_dir)
        self.locomo_path = locomo_path
        self.dense_encoder = dense_encoder
        self.vision_encoder = vision_encoder
        self._stores: Dict[tuple[str, str], HiddenMemoryStore] = {}
        self._sample_maps: Dict[tuple[str, str], Dict[str, OPDSample]] = {}
        self._locomo_samples = {
            str(item.get("sample_id")): item
            for item in read_json(locomo_path)
        }

    def resolve(self, row: Dict[str, Any]) -> OPDSample:
        dataset = str(row.get("dataset") or "")
        domain = str(row.get("domain") or "")
        key = (dataset, domain)
        if dataset == "Mem-Gallery":
            if key not in self._stores:
                path = Path(row["source_file"])
                data = read_json(path)
                store, _records = build_memgallery_store(
                    data,
                    data_dir=self.memgallery_dir,
                    dense_encoder=self.dense_encoder,
                    vision_encoder=self.vision_encoder,
                )
                self._stores[key] = store
                self._sample_maps[key] = _sample_map(
                    memgallery_scenario_samples(
                        data,
                        store=store,
                        data_dir=self.memgallery_dir,
                        scenario=domain,
                        include_oracle_profile=False,
                    )
                )
            sample = self._sample_maps[key].get(str(row["sample_id"]))
            if sample is None:
                raise KeyError(f"missing Mem-Gallery sample {row['sample_id']}")
            sample.metadata["hard_sample"] = row
            return sample

        if dataset == "MemEye":
            if key not in self._stores:
                path = Path(row["source_file"])
                data = read_json(path)
                store, _records = build_memeye_store(
                    data,
                    data_dir=self.memeye_dir,
                    dense_encoder=self.dense_encoder,
                    vision_encoder=self.vision_encoder,
                )
                self._stores[key] = store
                self._sample_maps[key] = _sample_map(
                    memeye_scenario_samples(
                        data,
                        store=store,
                        data_dir=self.memeye_dir,
                        scenario=domain,
                    )
                )
            sample = self._sample_maps[key].get(str(row["sample_id"]))
            if sample is None:
                raise KeyError(f"missing MemEye sample {row['sample_id']}")
            sample.metadata["dataset"] = "MemEye"
            sample.metadata["hard_sample"] = row
            return sample

        if dataset == "LoCoMo":
            if key not in self._stores:
                source = self._locomo_samples.get(domain)
                if source is None:
                    raise KeyError(f"missing LoCoMo conversation {domain}")
                self._stores[key] = HiddenMemoryStore(
                    locomo_records(source),
                    dense_encoder=self.dense_encoder,
                    vision_encoder=None,
                )
            return locomo_sample_from_row(row, self._stores[key])

        raise ValueError(f"unsupported dataset: {dataset}")


def support_metrics(
    evidence: List[EvidenceItem],
    clue_turn_ids: List[str],
    gold_image_ids: List[str],
) -> Dict[str, Any]:
    evidence_memory_ids = [item.memory_id for item in evidence]
    hit_turns = [
        clue
        for clue in clue_turn_ids
        if any(
            memory_id == clue or memory_id.startswith(clue + ":")
            for memory_id in evidence_memory_ids
        )
    ]
    serialized = json.dumps(
        [item.to_dict() for item in evidence],
        ensure_ascii=False,
    )
    found_images = list(dict.fromkeys(IMAGE_ID_PATTERN.findall(serialized)))
    return {
        "evidence_image_ids": found_images,
        "gold_image_recall_any": (
            bool(set(found_images) & set(gold_image_ids))
            if gold_image_ids
            else None
        ),
        "gold_image_recall_all": (
            set(gold_image_ids).issubset(found_images)
            if gold_image_ids
            else None
        ),
        "evidence_clue_recall_any": (
            bool(hit_turns) if clue_turn_ids else None
        ),
        "evidence_clue_recall_all": (
            len(hit_turns) == len(clue_turn_ids)
            if clue_turn_ids
            else None
        ),
        "support_turn_recall": (
            len(hit_turns) / len(clue_turn_ids)
            if clue_turn_ids
            else None
        ),
    }


def average(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _bool_average(rows: List[Dict[str, Any]], key: str) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return average(float(bool(value)) for value in values)


def summarize(rows: List[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    by_point: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    by_category: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    action_usage: collections.Counter[str] = collections.Counter()
    action_sources: collections.Counter[str] = collections.Counter()
    shapes: collections.Counter[str] = collections.Counter()
    for row in rows:
        by_dataset[str(row.get("dataset") or "unknown")].append(row)
        if row.get("point") is not None:
            by_point[str(row.get("point"))].append(row)
        if row.get("category") is not None:
            by_category[str(row.get("category"))].append(row)
        actions = row.get("actions") or []
        action_usage.update(action.get("tool") for action in actions)
        shapes["->".join(action.get("tool", "") for action in actions)] += 1
        action_sources.update(
            decision.get("action_source", "unknown")
            for decision in row.get("teacher_decisions") or []
        )

    def group_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "count": len(items),
            "teacher_answer_accuracy": average(
                float(row.get("correct", False)) for row in items
            ),
            "teacher_strict_validation_rate": average(
                float(
                    bool(
                        (row.get("strict_answer_validation") or {}).get(
                            "correct"
                        )
                    )
                )
                for row in items
            ),
            "evidence_recall_any": _bool_average(
                items,
                "evidence_clue_recall_any",
            ),
            "evidence_recall_all": _bool_average(
                items,
                "evidence_clue_recall_all",
            ),
            "avg_actions": average(
                float(len(row.get("actions") or [])) for row in items
            ),
            "avg_planner_calls": average(
                float(row.get("planner_calls") or 0) for row in items
            ),
            "avg_answer_validator_calls": average(
                float(row.get("answer_validator_calls") or 0) for row in items
            ),
            "avg_validation_cache_hits": average(
                float(row.get("validation_cache_hits") or 0) for row in items
            ),
            "avg_planner_seconds": average(
                float(row.get("planner_seconds") or 0.0) for row in items
            ),
            "avg_answer_validator_seconds": average(
                float(row.get("answer_validator_seconds") or 0.0)
                for row in items
            ),
            "avg_sample_latency_seconds": average(
                float(row.get("sample_latency_seconds") or 0.0)
                for row in items
            ),
        }

    return {
        "num_results": len(rows),
        "primary_metric": "teacher_answer_accuracy",
        "teacher_answer_accuracy": average(
            float(row.get("correct", False)) for row in rows
        ),
        "teacher_answer_score": average(
            float(row.get("score") or 0.0) for row in rows
        ),
        "teacher_strict_validation_rate": average(
            float(
                bool((row.get("strict_answer_validation") or {}).get("correct"))
            )
            for row in rows
        ),
        "evidence_recall_any": _bool_average(rows, "evidence_clue_recall_any"),
        "evidence_recall_all": _bool_average(rows, "evidence_clue_recall_all"),
        "gold_image_recall_any": _bool_average(rows, "gold_image_recall_any"),
        "avg_actions": average(float(len(row.get("actions") or [])) for row in rows),
        "avg_evidence_items": average(
            float(len((row.get("execution") or {}).get("evidence") or []))
            for row in rows
        ),
        "avg_planner_calls": average(
            float(row.get("planner_calls") or 0) for row in rows
        ),
        "avg_answer_validator_calls": average(
            float(row.get("answer_validator_calls") or 0) for row in rows
        ),
        "avg_candidates_evaluated": average(
            float(row.get("candidates_evaluated") or 0) for row in rows
        ),
        "avg_validation_cache_hits": average(
            float(row.get("validation_cache_hits") or 0) for row in rows
        ),
        "total_validation_cache_hits": sum(
            int(row.get("validation_cache_hits") or 0) for row in rows
        ),
        "avg_planner_seconds": average(
            float(row.get("planner_seconds") or 0.0) for row in rows
        ),
        "avg_answer_validator_seconds": average(
            float(row.get("answer_validator_seconds") or 0.0) for row in rows
        ),
        "avg_sample_latency_seconds": average(
            float(row.get("sample_latency_seconds") or 0.0) for row in rows
        ),
        "judge_error_rate": average(
            float(bool(row.get("judge_error"))) for row in rows
        ),
        "row_error_rate": average(float(bool(row.get("error"))) for row in rows),
        "action_usage": dict(sorted(action_usage.items())),
        "teacher_action_sources": dict(sorted(action_sources.items())),
        "trajectory_shapes": dict(
            sorted(shapes.items(), key=lambda item: (-item[1], item[0]))
        ),
        "by_dataset": {
            key: group_summary(items)
            for key, items in sorted(by_dataset.items())
        },
        "by_point": {
            key: group_summary(items)
            for key, items in sorted(by_point.items())
        },
        "by_category": {
            key: group_summary(items)
            for key, items in sorted(by_category.items())
        },
        "elapsed_seconds": elapsed,
    }


def select_rows(
    rows: List[Dict[str, Any]],
    max_samples: Optional[int],
    datasets: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if datasets:
        allowed = set(datasets)
        rows = [row for row in rows if row.get("dataset") in allowed]
    if max_samples is None or len(rows) <= max_samples:
        return rows
    grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset") or "")].append(row)
    selected: List[Dict[str, Any]] = []
    keys = sorted(grouped)
    while len(selected) < max_samples and keys:
        next_keys = []
        for key in keys:
            if len(selected) >= max_samples:
                break
            bucket = grouped[key]
            if bucket:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def sanitized_config(args: argparse.Namespace) -> Dict[str, Any]:
    data = vars(args).copy()
    for key in list(data):
        if key.endswith("_api_key"):
            value = str(data[key] or "")
            data[key] = value if value.startswith("env:") else "<redacted>"
    return data


def component_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        max_chunk_actions=args.max_chunk_actions,
        max_top_k=args.max_top_k,
        raw_inspection=args.raw_inspection,
        planner_base_url=args.planner_base_url,
        planner_model=args.planner_model,
        planner_api_key=args.planner_api_key,
        planner_service=args.planner_service,
        planner_max_tokens=args.planner_max_tokens,
        planner_thinking_token_budget=args.planner_thinking_token_budget,
        planner_prompt_mode=args.planner_prompt_mode,
        planner_enable_thinking=args.planner_enable_thinking,
        teacher_recall_only=False,
        teacher_validation="answer",
        teacher_min_answer_score=args.teacher_min_answer_score,
        answer_base_url=args.api_base_url,
        answer_model=args.api_model,
        answer_api_key=args.api_key,
        answer_service="api",
        answer_max_tokens=args.answer_max_tokens,
        answer_max_images=args.answer_max_images,
        inspect_max_tokens=args.inspect_max_tokens,
        judge_mode="llm",
        judge_base_url=args.api_base_url,
        judge_model=args.api_model,
        judge_api_key=args.api_key,
        judge_service="api",
        judge_max_tokens=args.judge_max_tokens,
        hybrid_alpha=args.hybrid_alpha,
        max_rounds=args.max_rounds,
        beam_size=args.beam_size,
        candidates_per_node=args.candidates_per_node,
        max_actions=args.max_actions,
        max_evidence=args.max_evidence,
        max_raw_inspections=args.max_raw_inspections,
        trajectory_action_cost=args.trajectory_action_cost,
        trajectory_evidence_cost=args.trajectory_evidence_cost,
    )


def run_eval(args: argparse.Namespace) -> Path:
    started = time.time()
    run_dir = args.output_dir.expanduser().resolve() / f"{now_stamp()}_teacher_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.manifest = args.manifest.expanduser().resolve()

    dense_encoder = (
        MiniLMTextEncoder(args.dense_model, device=args.dense_device)
        if args.dense_mode == "minilm"
        else None
    )
    vision_encoder = (
        SigLIPVisionEncoder(args.vision_model, device=args.vision_device)
        if args.vision_mode == "siglip"
        else None
    )
    resolver = HardSampleResolver(
        memgallery_dir=args.memgallery_dir.expanduser().resolve(),
        memeye_dir=args.memeye_dir.expanduser().resolve(),
        locomo_path=args.locomo_path.expanduser().resolve(),
        dense_encoder=dense_encoder,
        vision_encoder=vision_encoder,
    )
    components = make_components(component_args(args))

    rows = select_rows(
        list(iter_jsonl(args.manifest)),
        args.max_samples,
        args.datasets,
    )
    config = sanitized_config(args)
    config["resolved_sample_count"] = len(rows)
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    predictions_path = run_dir / "predictions.jsonl"
    partial_path = run_dir / "partial_metrics.json"
    output_rows: List[Dict[str, Any]] = []
    progress = ProgressBar(
        len(rows),
        "hard teacher eval",
        not args.no_progress,
    )
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, manifest_row in enumerate(rows, start=1):
            row: Dict[str, Any]
            sample_started = time.time()
            try:
                sample = resolver.resolve(manifest_row)
                question_image = sample.metadata.get("question_image")
                search = components["teacher_search"].search(
                    query=sample.query,
                    gold_answer=sample.gold_answer,
                    memory_store=sample.memory_store,
                    question_image=question_image,
                )
                strict_validation = search.answer_validation
                if strict_validation is not None:
                    prediction = strict_validation.prediction
                    correct = strict_validation.correct
                    score = strict_validation.score
                    reason = strict_validation.reason
                    judge_error = strict_validation.error
                else:
                    prediction = components["answer_model"].answer(
                        sample.query,
                        search.execution.evidence,
                        question_image=question_image,
                    )
                    correct, score, reason, judge_error = _safe_judge(
                        components["judge"],
                        sample.query,
                        prediction,
                        sample.gold_answer,
                    )
                support = support_metrics(
                    search.execution.evidence,
                    list(sample.metadata.get("gold_clue_turn_ids") or []),
                    list(sample.metadata.get("gold_image_ids") or []),
                )
                row = {
                    "uid": manifest_row.get("uid"),
                    "dataset": manifest_row.get("dataset"),
                    "domain": manifest_row.get("domain"),
                    "sample_id": sample.sample_id,
                    "query": sample.query,
                    "gold_answer": sample.gold_answer,
                    "category": manifest_row.get("category"),
                    "point": manifest_row.get("point"),
                    "difficulty_score": manifest_row.get("difficulty_score"),
                    "difficulty_signals": manifest_row.get("difficulty_signals"),
                    "prior_eval": manifest_row.get("prior_eval"),
                    "question_image": question_image,
                    "actions": [action.to_dict() for action in search.actions],
                    "execution": search.execution.to_dict(),
                    "verification": search.verification.to_dict(),
                    "teacher_decisions": _decision_rows(search),
                    "prediction": prediction,
                    "correct": bool(correct),
                    "score": float(score),
                    "judge_reason": reason,
                    "judge_error": judge_error,
                    "strict_answer_validation": (
                        strict_validation.to_dict()
                        if strict_validation is not None
                        else None
                    ),
                    "planner_calls": search.planner_calls,
                    "verifier_calls": search.verifier_calls,
                    "answer_validator_calls": search.answer_validator_calls,
                    "validation_cache_hits": search.validation_cache_hits,
                    "planner_seconds": search.planner_seconds,
                    "answer_validator_seconds": search.answer_validator_seconds,
                    "candidates_evaluated": search.candidates_evaluated,
                    "chunk_count": len(search.decisions),
                    "sample_latency_seconds": time.time() - sample_started,
                    "error": "",
                    **support,
                }
            except Exception as exc:
                row = {
                    "uid": manifest_row.get("uid"),
                    "dataset": manifest_row.get("dataset"),
                    "domain": manifest_row.get("domain"),
                    "sample_id": manifest_row.get("sample_id"),
                    "query": manifest_row.get("question"),
                    "gold_answer": manifest_row.get("answer"),
                    "category": manifest_row.get("category"),
                    "point": manifest_row.get("point"),
                    "difficulty_score": manifest_row.get("difficulty_score"),
                    "correct": False,
                    "score": 0.0,
                    "actions": [],
                    "execution": {"evidence": [], "steps": []},
                    "teacher_decisions": [],
                    "planner_calls": 0,
                    "answer_validator_calls": 0,
                    "validation_cache_hits": 0,
                    "planner_seconds": 0.0,
                    "answer_validator_seconds": 0.0,
                    "candidates_evaluated": 0,
                    "sample_latency_seconds": time.time() - sample_started,
                    "error": str(exc),
                }
            output_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            metrics = summarize(output_rows, time.time() - started)
            metrics["partial"] = index < len(rows)
            metrics["completed_samples"] = index
            metrics["total_samples"] = len(rows)
            partial_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            progress.update(
                index,
                message=(
                    f"{row.get('dataset')} correct={int(bool(row.get('correct')))}"
                ),
            )
    progress.close()
    metrics = summarize(output_rows, time.time() - started)
    metrics["partial"] = False
    metrics["completed_samples"] = len(output_rows)
    metrics["total_samples"] = len(rows)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    partial_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Saved hard-sample teacher eval: {run_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate teacher planning on hard memory samples."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--memgallery-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR / "Mem-Gallery",
    )
    parser.add_argument(
        "--memeye-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR / "MemEye" / "data",
    )
    parser.add_argument(
        "--locomo-path",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR / "LoCoMo" / "data" / "locomo10.json",
    )
    parser.add_argument("--dense-mode", choices=["minilm", "off"], default="minilm")
    parser.add_argument("--dense-model", default=default_minilm_model())
    parser.add_argument("--dense-device", default="cpu")
    parser.add_argument("--vision-mode", choices=["siglip", "off"], default="siglip")
    parser.add_argument("--vision-model", default=default_siglip_model())
    parser.add_argument("--vision-device", default="cpu")
    parser.add_argument("--planner-base-url", default="http://127.0.0.1:11438/v1")
    parser.add_argument("--planner-model", default="qwen3-vl-4b-thinking-vllm")
    parser.add_argument("--planner-api-key", default="ollama")
    parser.add_argument("--planner-service", choices=["auto", "local", "api"], default="local")
    parser.add_argument("--planner-max-tokens", type=int, default=768)
    parser.add_argument("--planner-thinking-token-budget", type=int, default=256)
    parser.add_argument("--planner-prompt-mode", choices=["teacher_compact", "student_simple"], default="teacher_compact")
    parser.add_argument(
        "--planner-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--api-base-url", default="https://api.llm.ustc.edu.cn/v1")
    parser.add_argument("--api-model", default="qwen-chat")
    parser.add_argument("--api-key", default="env:USTC_API_KEY")
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--answer-max-images", type=int, default=3)
    parser.add_argument("--judge-max-tokens", type=int, default=192)
    parser.add_argument("--teacher-min-answer-score", type=float, default=0.9)
    parser.add_argument("--raw-inspection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-raw-inspections", type=int, default=3)
    parser.add_argument("--inspect-max-tokens", type=int, default=160)
    parser.add_argument("--max-rounds", type=int, default=9)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--candidates-per-node", type=int, default=3)
    parser.add_argument("--max-chunk-actions", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=9)
    parser.add_argument("--max-top-k", type=int, default=50)
    parser.add_argument("--max-evidence", type=int, default=40)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--trajectory-action-cost", type=float, default=0.08)
    parser.add_argument("--trajectory-evidence-cost", type=float, default=0.01)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_eval(parse_args())


if __name__ == "__main__":
    main()
