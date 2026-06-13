"""Mem-Gallery runner for interactive next-action OPD-MM policies."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dual_encoder_memory import MiniLMTextEncoder, SigLIPVisionEncoder
from omnimem.config import (
    PROJECT_ROOT,
    default_memgallery_dir,
    default_minilm_model,
    default_siglip_model,
    require_memgallery_dir,
)

from .clients import (
    ChatAnswerJudge,
    ChatAnswerModel,
    ChatRawInspector,
    HeuristicAnswerJudge,
    OpenAICompatibleClient,
)
from .interactive import (
    ChatGoldEvidenceVerifier,
    ChatInteractivePlanner,
    InteractiveActionValidator,
    InteractivePolicyRunner,
    InteractiveSearchResult,
    InteractiveTeacherSearch,
    VerificationResult,
)
from .memgallery import (
    IMAGE_ID_PATTERN,
    build_scenario_store,
    scenario_samples,
)
from .memgallery_pipeline import (
    AlwaysIncorrectJudge,
    EmptyAnswerModel,
    ProgressBar,
    now_stamp,
    resolve_scenarios,
    write_memory_manifest,
)
from .models import EvidenceItem, ExecutionResult, MemoryRecord, SFTExample
from .retrieval import TurnAwareHybridRetriever


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "memgallery_opd_interactive"


def _average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evidence_image_ids(evidence: List[EvidenceItem]) -> List[str]:
    serialized = json.dumps(
        [item.to_dict() for item in evidence],
        ensure_ascii=False,
    )
    return list(dict.fromkeys(IMAGE_ID_PATTERN.findall(serialized)))


def _support_metrics(
    execution: ExecutionResult,
    clue_turn_ids: List[str],
    gold_image_ids: List[str],
) -> Dict[str, Any]:
    evidence_memory_ids = [item.memory_id for item in execution.evidence]
    hit_turns = [
        clue
        for clue in clue_turn_ids
        if any(
            memory_id == clue or memory_id.startswith(clue + ":")
            for memory_id in evidence_memory_ids
        )
    ]
    found_images = _evidence_image_ids(execution.evidence)
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


def _select_sft_trajectory(
    quality_filter: str,
    verification: VerificationResult,
    support: Dict[str, Any],
    answer_correct: bool = False,
) -> bool:
    if quality_filter == "all":
        return True
    if not verification.answerable:
        return False
    if quality_filter == "answerable":
        return True
    required = [
        value
        for value in (
            support.get("evidence_clue_recall_any"),
            support.get("gold_image_recall_all"),
        )
        if value is not None
    ]
    support_grounded = (
        all(bool(value) for value in required) if required else True
    )
    if quality_filter == "answer-correct":
        return support_grounded and answer_correct
    return support_grounded


def _decision_rows(result: InteractiveSearchResult) -> List[Dict[str, Any]]:
    return [
        {
            "step": index,
            "observation": decision.observation.to_dict(),
            "actions": [action.to_dict() for action in decision.actions],
            "observation_after": decision.observation_after.to_dict(),
            "verifier_feedback_used_by_teacher": decision.privileged_feedback,
            "verification_after": decision.verification_after.to_dict(),
            "planner_raw_response": decision.planner_raw_response,
            "action_source": decision.action_source,
        }
        for index, decision in enumerate(result.decisions)
    ]


def _safe_judge(
    judge: Any,
    query: str,
    prediction: str,
    gold_answer: str,
) -> tuple[bool, float, str, str]:
    try:
        correct, score, reason = judge.evaluate(
            query,
            prediction,
            gold_answer,
        )
        return bool(correct), float(score), str(reason), ""
    except Exception as exc:
        return False, 0.0, "", str(exc)


def make_components(args: argparse.Namespace) -> Dict[str, Any]:
    validator = InteractiveActionValidator(
        max_chunk_actions=args.max_chunk_actions,
        max_top_k=args.max_top_k,
        allow_inspect_raw=args.raw_inspection,
    )
    planner = ChatInteractivePlanner(
        OpenAICompatibleClient(
            args.planner_base_url,
            args.planner_model,
            args.planner_api_key,
        ),
        validator=validator,
        max_tokens=args.planner_max_tokens,
    )
    verifier = ChatGoldEvidenceVerifier(
        OpenAICompatibleClient(
            args.verifier_base_url,
            args.verifier_model,
            args.verifier_api_key,
        ),
        max_tokens=args.verifier_max_tokens,
    )
    if args.teacher_recall_only:
        answer_model = EmptyAnswerModel()
        judge = AlwaysIncorrectJudge()
        raw_inspector = None
    else:
        answer_client = OpenAICompatibleClient(
            args.answer_base_url,
            args.answer_model,
            args.answer_api_key,
        )
        answer_model = ChatAnswerModel(
            answer_client,
            max_tokens=args.answer_max_tokens,
            max_images=args.answer_max_images,
        )
        raw_inspector = (
            ChatRawInspector(answer_client, max_tokens=args.inspect_max_tokens)
            if args.raw_inspection
            else None
        )
        judge = (
            ChatAnswerJudge(
                OpenAICompatibleClient(
                    args.judge_base_url,
                    args.judge_model,
                    args.judge_api_key,
                ),
                max_tokens=args.judge_max_tokens,
            )
            if args.judge_mode == "llm"
            else HeuristicAnswerJudge()
        )
    retriever = TurnAwareHybridRetriever(args.hybrid_alpha)
    return {
        "validator": validator,
        "planner": planner,
        "verifier": verifier,
        "answer_model": answer_model,
        "judge": judge,
        "teacher_search": InteractiveTeacherSearch(
            planner=planner,
            verifier=verifier,
            validator=validator,
            retriever=retriever,
            raw_inspector=raw_inspector,
            max_rounds=args.max_rounds,
            beam_size=args.beam_size,
            candidates_per_node=args.candidates_per_node,
            max_actions=args.max_actions,
            max_evidence=args.max_evidence,
            max_raw_inspections=args.max_raw_inspections,
        ),
        "policy_runner": InteractivePolicyRunner(
            planner=planner,
            validator=validator,
            retriever=retriever,
            raw_inspector=raw_inspector,
            max_rounds=args.max_rounds,
            max_actions=args.max_actions,
            max_raw_inspections=args.max_raw_inspections,
        ),
    }


def summarize(
    rows: List[Dict[str, Any]],
    memory_count: int,
    elapsed: float,
    mode: str,
    teacher_recall_only: bool,
) -> Dict[str, Any]:
    clue_rows = [
        row for row in rows if row.get("evidence_clue_recall_any") is not None
    ]
    image_rows = [
        row for row in rows if row.get("gold_image_recall_any") is not None
    ]
    by_point: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_point[str(row.get("point") or "unknown")].append(row)
    action_counts = Counter(
        action["tool"]
        for row in rows
        for action in row.get("actions") or []
    )
    shapes = Counter(
        "->".join(action["tool"] for action in row.get("actions") or [])
        or "EMPTY"
        for row in rows
    )
    action_sources = Counter(
        decision.get("action_source", "unknown")
        for row in rows
        for decision in row.get("teacher_decisions") or []
    )
    primary_metric = (
        "verifier_answerable_rate"
        if teacher_recall_only
        else "answer_accuracy"
    )
    return {
        "num_results": len(rows),
        "memory_count": memory_count,
        "mode": mode,
        "primary_metric": primary_metric,
        "answer_accuracy": _average(
            [float(row["correct"]) for row in rows]
        ),
        "answer_score": _average([float(row["score"]) for row in rows]),
        "verifier_answerable_rate": _average(
            [float(row["verification"]["answerable"]) for row in rows]
        ),
        "verifier_relevance": _average(
            [float(row["verification"]["relevance"]) for row in rows]
        ),
        "verifier_completeness": _average(
            [float(row["verification"]["completeness"]) for row in rows]
        ),
        "evidence_recall_any": _average(
            [float(row["evidence_clue_recall_any"]) for row in clue_rows]
        ),
        "evidence_recall_all": _average(
            [float(row["evidence_clue_recall_all"]) for row in clue_rows]
        ),
        "support_turn_recall": _average(
            [float(row["support_turn_recall"]) for row in clue_rows]
        ),
        "gold_image_recall_any": _average(
            [float(row["gold_image_recall_any"]) for row in image_rows]
        ),
        "gold_image_recall_all": _average(
            [float(row["gold_image_recall_all"]) for row in image_rows]
        ),
        "clue_labeled_questions": len(clue_rows),
        "gold_image_questions": len(image_rows),
        "avg_planner_calls": _average(
            [float(row["planner_calls"]) for row in rows]
        ),
        "avg_verifier_calls": _average(
            [float(row["verifier_calls"]) for row in rows]
        ),
        "avg_candidates_evaluated": _average(
            [float(row["candidates_evaluated"]) for row in rows]
        ),
        "avg_chunks": _average([float(row["chunk_count"]) for row in rows]),
        "avg_actions": _average(
            [float(len(row.get("actions") or [])) for row in rows]
        ),
        "avg_evidence_items": _average(
            [
                float(len((row.get("execution") or {}).get("evidence") or []))
                for row in rows
            ]
        ),
        "query_rewrite_rate": _average(
            [
                float(
                    any(
                        action.get("tool") == "RETRIEVE"
                        and bool(action.get("query"))
                        for action in row.get("actions") or []
                    )
                )
                for row in rows
            ]
        ),
        "action_usage": dict(sorted(action_counts.items())),
        "teacher_action_sources": dict(sorted(action_sources.items())),
        "trajectory_shapes": dict(
            sorted(shapes.items(), key=lambda item: (-item[1], item[0]))
        ),
        "judge_error_rate": _average(
            [float(bool(row.get("judge_error"))) for row in rows]
        ),
        "sft_selected_trajectories": sum(
            bool(row.get("sft_selected")) for row in rows
        ),
        "sft_selection_rate": _average(
            [float(bool(row.get("sft_selected"))) for row in rows]
        ),
        "elapsed_seconds": elapsed,
        "by_point": {
            point: {
                "count": len(items),
                "answer_accuracy": _average(
                    [float(item["correct"]) for item in items]
                ),
                "verifier_answerable_rate": _average(
                    [
                        float(item["verification"]["answerable"])
                        for item in items
                    ]
                ),
                "evidence_recall_any": _average(
                    [
                        float(item["evidence_clue_recall_any"])
                        for item in items
                        if item.get("evidence_clue_recall_any") is not None
                    ]
                ),
            }
            for point, items in sorted(by_point.items())
        },
    }


def run_scenario(
    path: Path,
    args: argparse.Namespace,
    components: Dict[str, Any],
    dense_encoder: Optional[Any],
    vision_encoder: Optional[Any],
) -> Path:
    started = time.time()
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario = path.stem
    run_dir = (
        args.output_dir
        / scenario
        / f"{now_stamp()}_opd_interactive"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    store, records = build_scenario_store(
        data,
        data_dir=args.data_dir,
        dense_encoder=dense_encoder,
        vision_encoder=vision_encoder,
        max_sessions=args.max_sessions,
        max_turns=args.max_turns,
    )
    samples = scenario_samples(
        data,
        store=store,
        data_dir=args.data_dir,
        scenario=scenario,
        max_questions=args.max_questions,
        include_oracle_profile=False,
    )
    for sample in samples:
        sample.metadata.pop("teacher_privileged_context", None)

    rows: List[Dict[str, Any]] = []
    sft_examples: List[SFTExample] = []
    progress = ProgressBar(
        len(samples),
        f"{scenario} interactive",
        not args.no_progress,
    )
    predictions_path = run_dir / "predictions.jsonl"
    sft_path = run_dir / "sft_steps.jsonl"
    with predictions_path.open("w", encoding="utf-8") as prediction_handle:
        for index, sample in enumerate(samples, start=1):
            question_image = sample.metadata.get("question_image")
            if args.mode == "collect-sft":
                search = components["teacher_search"].search(
                    query=sample.query,
                    gold_answer=sample.gold_answer,
                    memory_store=sample.memory_store,
                    question_image=question_image,
                )
                execution = search.execution
                verification = search.verification
                actions = [action.to_dict() for action in search.actions]
                planner_calls = search.planner_calls
                verifier_calls = search.verifier_calls
                candidates_evaluated = search.candidates_evaluated
                decisions = _decision_rows(search)
            else:
                policy = components["policy_runner"].run(
                    query=sample.query,
                    memory_store=sample.memory_store,
                    question_image=question_image,
                )
                execution = policy.execution
                actions = [action.to_dict() for action in policy.actions]
                planner_calls = policy.planner_calls
                verifier_calls_before = components["verifier"].calls
                verification = components["verifier"].evaluate(
                    sample.query,
                    sample.gold_answer,
                    execution.evidence,
                )
                verifier_calls = (
                    components["verifier"].calls - verifier_calls_before
                )
                candidates_evaluated = 0
                decisions = [
                    {"planner_raw_response": raw}
                    for raw in policy.planner_raw_responses
                ]

            support = _support_metrics(
                execution,
                list(sample.metadata.get("gold_clue_turn_ids") or []),
                list(sample.metadata.get("gold_image_ids") or []),
            )
            prediction = components["answer_model"].answer(
                sample.query,
                execution.evidence,
                question_image=question_image,
            )
            correct, score, reason, judge_error = _safe_judge(
                components["judge"],
                sample.query,
                prediction,
                sample.gold_answer,
            )
            sft_selected = (
                args.mode == "collect-sft"
                and _select_sft_trajectory(
                    args.sft_quality_filter,
                    verification,
                    support,
                    answer_correct=correct,
                )
            )
            if sft_selected:
                sft_examples.extend(
                    search.sft_examples(
                        sample.sample_id,
                        sample.query,
                        components["validator"].schema_text(),
                    )
                )
            row = {
                "sample_id": sample.sample_id,
                "query": sample.query,
                "gold_answer": sample.gold_answer,
                "point": sample.metadata.get("point"),
                "question_image": question_image,
                "actions": actions,
                "execution": execution.to_dict(),
                "verification": verification.to_dict(),
                "teacher_decisions": decisions,
                "prediction": prediction,
                "correct": correct,
                "score": score,
                "judge_reason": reason,
                "judge_error": judge_error,
                "planner_calls": planner_calls,
                "verifier_calls": verifier_calls,
                "candidates_evaluated": candidates_evaluated,
                "chunk_count": len(decisions),
                "sft_selected": sft_selected,
                **support,
            }
            rows.append(row)
            prediction_handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
            prediction_handle.flush()
            progress.update(
                index,
                message=(
                    f"point={sample.metadata.get('point')} "
                    f"answerable={int(verification.answerable)}"
                ),
            )
    progress.close()

    with sft_path.open("w", encoding="utf-8") as handle:
        for example in sft_examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    write_memory_manifest(run_dir / "hidden_memory_manifest.jsonl", records)
    metrics = summarize(
        rows,
        memory_count=len(store),
        elapsed=time.time() - started,
        mode=args.mode,
        teacher_recall_only=args.teacher_recall_only,
    )
    metrics["sft_step_examples"] = len(sft_examples)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = vars(args) | {
        "scenario": scenario,
        "teacher_privilege": (
            "gold answer visible only to post-action evidence verifier; "
            "planner sees query, action history, executor observations, and "
            "coarse verifier feedback during teacher search"
        ),
        "student_sft_input_privilege": "none",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[INFO] Saved interactive OPD-MM run: {run_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return run_dir


def run_scenarios(args: argparse.Namespace) -> List[Path]:
    paths = resolve_scenarios(args)
    if not paths:
        raise FileNotFoundError("no Mem-Gallery scenario files matched")
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
    components = make_components(args)
    outputs = []
    for index, path in enumerate(paths, start=1):
        print(f"[INFO] Scenario {index}/{len(paths)}: {path.stem}")
        outputs.append(
            run_scenario(
                path,
                args,
                components,
                dense_encoder,
                vision_encoder,
            )
        )
    return outputs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interactive next-action OPD-MM on Mem-Gallery."
    )
    parser.add_argument("--data-dir", type=Path, default=default_memgallery_dir())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["collect-sft", "evaluate"],
        default="collect-sft",
    )
    parser.add_argument("--teacher-recall-only", action="store_true")
    parser.add_argument(
        "--sft-quality-filter",
        choices=[
            "all",
            "answerable",
            "support-grounded",
            "answer-correct",
        ],
        default="support-grounded",
        help=(
            "answer-correct additionally requires answer generation and judging, "
            "so it should not be combined with --teacher-recall-only."
        ),
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--candidates-per-node", type=int, default=3)
    parser.add_argument("--max-chunk-actions", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=9)
    parser.add_argument("--max-top-k", type=int, default=50)
    parser.add_argument("--max-evidence", type=int, default=40)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--dense-mode", choices=["minilm", "off"], default="minilm")
    parser.add_argument("--dense-model", default=default_minilm_model())
    parser.add_argument("--dense-device", default="cpu")
    parser.add_argument(
        "--vision-mode",
        choices=["siglip", "off"],
        default="siglip",
    )
    parser.add_argument("--vision-model", default=default_siglip_model())
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--planner-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--planner-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--planner-api-key", default="ollama")
    parser.add_argument("--planner-max-tokens", type=int, default=768)
    parser.add_argument("--verifier-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--verifier-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--verifier-api-key", default="ollama")
    parser.add_argument("--verifier-max-tokens", type=int, default=192)
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--answer-model", default="qwen3-vl-8b-instruct-ctx8k:latest")
    parser.add_argument("--answer-api-key", default="ollama")
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--answer-max-images", type=int, default=3)
    parser.add_argument("--raw-inspection", action="store_true")
    parser.add_argument("--max-raw-inspections", type=int, default=3)
    parser.add_argument("--inspect-max-tokens", type=int, default=160)
    parser.add_argument("--judge-mode", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--judge-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--judge-api-key", default="ollama")
    parser.add_argument("--judge-max-tokens", type=int, default=192)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if (
        args.teacher_recall_only
        and args.sft_quality_filter == "answer-correct"
    ):
        raise ValueError(
            "--sft-quality-filter answer-correct requires answer generation; "
            "remove --teacher-recall-only"
        )
    args.data_dir = require_memgallery_dir(args.data_dir)
    args.output_dir = args.output_dir.expanduser().resolve()
    run_scenarios(args)


if __name__ == "__main__":
    main()
