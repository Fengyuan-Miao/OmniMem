"""Build a balanced hard-sample set across Mem-Gallery, LoCoMo, and MemEye.

The output is a lightweight JSONL manifest of QA samples with provenance and
interpretable difficulty signals. It does not call model services or build
memory indexes; downstream pipelines can use the source fields to reconstruct
the proper benchmark memory store.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from omnimem.config import PROJECT_ROOT, default_memgallery_dir

from .memeye import DEFAULT_OPEN_TASKS, normalize_memeye_data_dir


REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "benchmark"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "hard_memory_samples"

TURN_ID_PATTERN = re.compile(r"\b[A-Za-z0-9_-]+:\d+\b")
MEMGALLERY_IMAGE_ID_PATTERN = re.compile(r"\bD\d+:IMG_\d+\b")
MEMEYE_IMAGE_ID_PATTERN = re.compile(
    r"\b(?:[A-Za-z0-9_-]+:)?IMG_\d+\b|\b[A-Z]{1,4}\d*:\d+\b"
)

QUERY_COMPLEXITY_PATTERNS = {
    "temporal": re.compile(
        r"\b(when|date|after|before|first|last|latest|previous|next|yesterday|today|chronolog|recent)\b",
        re.I,
    ),
    "comparison": re.compile(
        r"\b(compare|same|different|difference|similar|than|versus|vs\.?|match)\b",
        re.I,
    ),
    "aggregation": re.compile(
        r"\b(all|both|list|which.*(ones|items|people)|how many|count|total)\b",
        re.I,
    ),
    "reasoning": re.compile(r"\b(why|how|likely|would|should|infer)\b", re.I),
    "visual_detail": re.compile(
        r"\b(image|picture|photo|visual|color|background|text|logo|shape|wearing|shown|seen|look)\b",
        re.I,
    ),
}


@dataclass
class PriorEvalSignal:
    eval_runs: int = 0
    wrong_count: int = 0
    scores: List[float] = field(default_factory=list)
    evidence_miss_count: int = 0
    gold_image_miss_count: int = 0
    action_counts: List[int] = field(default_factory=list)
    planner_calls: List[int] = field(default_factory=list)
    source_runs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        avg_score = sum(self.scores) / len(self.scores) if self.scores else None
        return {
            "eval_runs": self.eval_runs,
            "wrong_count": self.wrong_count,
            "wrong_rate": self.wrong_count / self.eval_runs
            if self.eval_runs
            else None,
            "avg_score": avg_score,
            "evidence_miss_count": self.evidence_miss_count,
            "gold_image_miss_count": self.gold_image_miss_count,
            "max_action_count": max(self.action_counts) if self.action_counts else 0,
            "avg_action_count": (
                sum(self.action_counts) / len(self.action_counts)
                if self.action_counts
                else None
            ),
            "max_planner_calls": max(self.planner_calls)
            if self.planner_calls
            else 0,
            "source_runs": self.source_runs[:8],
        }


@dataclass
class HardSample:
    uid: str
    dataset: str
    domain: str
    source_file: str
    sample_id: str
    question: str
    answer: str
    category: Any = None
    point: Any = None
    evidence: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)
    question_image: Optional[str] = None
    gold_image_ids: List[str] = field(default_factory=list)
    difficulty_score: float = 0.0
    difficulty_signals: Dict[str, Any] = field(default_factory=dict)
    prior_eval: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "dataset": self.dataset,
            "domain": self.domain,
            "source_file": self.source_file,
            "sample_id": self.sample_id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "point": self.point,
            "evidence": self.evidence,
            "session_ids": self.session_ids,
            "question_image": self.question_image,
            "gold_image_ids": self.gold_image_ids,
            "difficulty_score": round(self.difficulty_score, 4),
            "difficulty_signals": self.difficulty_signals,
            "prior_eval": self.prior_eval,
            "metadata": self.metadata,
        }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_float(value: str, seed: int = 17) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def flatten_point(value: Any) -> List[str]:
    out: List[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif item is not None:
            out.append(str(item))

    visit(value)
    return out


def unique_strs(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def clue_session_id(turn_id: str) -> str:
    value = str(turn_id)
    return value.split(":", 1)[0] if ":" in value else value


def evidence_span(evidence: List[str]) -> int:
    session_numbers = []
    for item in evidence:
        match = re.match(r"[A-Za-z_]*?(\d+)", str(item).split(":", 1)[0])
        if match:
            session_numbers.append(int(match.group(1)))
    if not session_numbers:
        return 0
    return max(session_numbers) - min(session_numbers)


def query_signals(question: str) -> Dict[str, bool]:
    return {
        name: bool(pattern.search(question or ""))
        for name, pattern in QUERY_COMPLEXITY_PATTERNS.items()
    }


def bool_score(value: bool, weight: float) -> float:
    return weight if value else 0.0


def count_execution_actions(row: Dict[str, Any]) -> int:
    actions = row.get("actions")
    if isinstance(actions, list):
        return len(actions)
    execution = row.get("execution") or {}
    steps = execution.get("steps") if isinstance(execution, dict) else None
    if isinstance(steps, list):
        return len(steps)
    return int(row.get("planner_calls") or 0)


def _as_bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def load_prior_eval_signals(runs_dir: Path) -> Dict[tuple[str, str], PriorEvalSignal]:
    signals: Dict[tuple[str, str], PriorEvalSignal] = {}
    if not runs_dir.is_dir():
        return signals
    for path in runs_dir.rglob("predictions.jsonl"):
        path_text = str(path)
        if "memgallery" in path_text:
            dataset = "Mem-Gallery"
        elif "memeye" in path_text:
            dataset = "MemEye"
        else:
            continue
        try:
            handle = path.open(encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = str(row.get("sample_id") or "")
                if not sample_id:
                    continue
                key = (dataset, sample_id)
                signal = signals.setdefault(key, PriorEvalSignal())
                signal.eval_runs += 1
                correct = _as_bool_or_none(row.get("correct"))
                if correct is False:
                    signal.wrong_count += 1
                try:
                    signal.scores.append(float(row.get("score")))
                except (TypeError, ValueError):
                    pass
                evidence_recall = _as_bool_or_none(
                    row.get("evidence_clue_recall_any")
                    if "evidence_clue_recall_any" in row
                    else row.get("support_turn_recall")
                )
                if evidence_recall is False:
                    signal.evidence_miss_count += 1
                gold_image_recall = _as_bool_or_none(row.get("gold_image_recall_any"))
                if gold_image_recall is False:
                    signal.gold_image_miss_count += 1
                actions = count_execution_actions(row)
                if actions:
                    signal.action_counts.append(actions)
                try:
                    signal.planner_calls.append(int(row.get("planner_calls") or 0))
                except (TypeError, ValueError):
                    pass
                run_name = str(path.parent)
                if run_name not in signal.source_runs:
                    signal.source_runs.append(run_name)
    return signals


def prior_score(signal: PriorEvalSignal) -> float:
    if signal.eval_runs <= 0:
        return 0.0
    score = 0.0
    score += 4.0 * signal.wrong_count
    score += 2.0 * signal.evidence_miss_count
    score += 1.5 * signal.gold_image_miss_count
    if signal.scores:
        score += max(0.0, 1.0 - sum(signal.scores) / len(signal.scores)) * 2.5
    if signal.action_counts:
        score += min(2.0, max(signal.action_counts) / 5.0)
    return score


def score_common(
    question: str,
    answer: str,
    evidence: List[str],
    session_ids: List[str],
    question_image: Optional[str],
    gold_image_ids: List[str],
) -> tuple[float, Dict[str, Any]]:
    q = query_signals(question)
    multi_support = len(evidence) > 1
    multi_session = len(set(session_ids)) > 1
    span = evidence_span(evidence)
    answer_has_list = bool(re.search(r",| and |;|\n", answer or "", re.I))
    signals: Dict[str, Any] = {
        "query_signals": q,
        "evidence_count": len(evidence),
        "multi_support": multi_support,
        "multi_session_support": multi_session,
        "evidence_session_span": span,
        "answer_has_list": answer_has_list,
        "question_has_image": bool(question_image),
        "gold_image_id_count": len(gold_image_ids),
        "question_length": len((question or "").split()),
    }
    score = 0.0
    score += min(2.0, len(evidence) * 0.45)
    score += bool_score(multi_support, 1.0)
    score += bool_score(multi_session, 1.25)
    score += min(1.5, span * 0.25)
    score += bool_score(answer_has_list, 0.8)
    score += bool_score(bool(question_image), 1.5)
    score += min(1.2, len(gold_image_ids) * 0.6)
    score += min(1.0, len((question or "").split()) / 30.0)
    for name, present in q.items():
        weight = {
            "temporal": 1.0,
            "comparison": 1.2,
            "aggregation": 1.2,
            "reasoning": 1.0,
            "visual_detail": 1.1,
        }[name]
        score += bool_score(present, weight)
    return score, signals


def memgallery_samples(
    data_dir: Path,
    prior: Dict[tuple[str, str], PriorEvalSignal],
) -> List[HardSample]:
    rows: List[HardSample] = []
    for path in sorted((data_dir / "data" / "dialog").glob("*.json")):
        data = read_json(path)
        scenario = path.stem
        for index, qa in enumerate(data.get("human-annotated QAs") or [], start=1):
            question = str(qa.get("question") or "")
            answer = str(qa.get("answer") or "")
            sample_id = f"{scenario}:{index}"
            evidence = unique_strs(TURN_ID_PATTERN.findall(str(qa.get("clue") or "")))
            session_ids = unique_strs(qa.get("session_id") or map(clue_session_id, evidence))
            question_image = qa.get("question_image") or None
            gold_image_ids = MEMGALLERY_IMAGE_ID_PATTERN.findall(answer)
            common_score, signals = score_common(
                question,
                answer,
                evidence,
                session_ids,
                str(question_image) if question_image else None,
                gold_image_ids,
            )
            point = qa.get("point")
            point_weight = {
                "VR": 2.2,
                "CD": 2.0,
                "TTL": 1.9,
                "TR": 1.5,
                "MR": 1.4,
                "VS": 1.2,
                "AR": 1.0,
                "FR": 0.7,
                "KR": 0.6,
            }.get(str(point), 0.8)
            prior_signal = prior.get(("Mem-Gallery", sample_id), PriorEvalSignal())
            signals.update({"point_weight": point_weight})
            sample = HardSample(
                uid=f"memgallery::{sample_id}",
                dataset="Mem-Gallery",
                domain=scenario,
                source_file=str(path),
                sample_id=sample_id,
                question=question,
                answer=answer,
                point=point,
                evidence=evidence,
                session_ids=session_ids,
                question_image=str(question_image) if question_image else None,
                gold_image_ids=gold_image_ids,
                difficulty_score=common_score + point_weight + prior_score(prior_signal),
                difficulty_signals=signals,
                prior_eval=prior_signal.to_dict(),
                metadata={"benchmark_kind": "multimodal_dialog_memory"},
            )
            rows.append(sample)
    return rows


def memeye_samples(
    data_dir: Path,
    prior: Dict[tuple[str, str], PriorEvalSignal],
    task_names: Iterable[str],
) -> List[HardSample]:
    rows: List[HardSample] = []
    dialog_dir = data_dir / "data" / "dialog"
    for name in task_names:
        path = dialog_dir / f"{name}.json"
        if not path.is_file():
            continue
        data = read_json(path)
        task = path.stem
        for index, qa in enumerate(data.get("human-annotated QAs") or [], start=1):
            question_id = str(qa.get("question_id") or index)
            sample_id = f"{task}:{question_id}"
            question = str(qa.get("question") or "")
            answer = str(qa.get("answer") or "")
            evidence = unique_strs(qa.get("clue") or [])
            session_ids = unique_strs(qa.get("session_id") or map(clue_session_id, evidence))
            question_image = qa.get("question_image") or None
            gold_image_ids = MEMEYE_IMAGE_ID_PATTERN.findall(answer)
            common_score, signals = score_common(
                question,
                answer,
                evidence,
                session_ids,
                str(question_image) if question_image else None,
                gold_image_ids,
            )
            point_values = flatten_point(qa.get("point"))
            visual_task = bool(
                question_image
                or gold_image_ids
                or QUERY_COMPLEXITY_PATTERNS["visual_detail"].search(question)
            )
            raw_point_complexity = len(point_values)
            task_weight = 1.5 if visual_task else 0.7
            task_weight += min(1.5, raw_point_complexity * 0.25)
            prior_signal = prior.get(("MemEye", sample_id), PriorEvalSignal())
            signals.update(
                {
                    "raw_point_count": raw_point_complexity,
                    "visual_memory_task": visual_task,
                    "task_weight": task_weight,
                }
            )
            rows.append(
                HardSample(
                    uid=f"memeye::{sample_id}",
                    dataset="MemEye",
                    domain=task,
                    source_file=str(path),
                    sample_id=sample_id,
                    question=question,
                    answer=answer,
                    point=qa.get("point"),
                    evidence=evidence,
                    session_ids=session_ids,
                    question_image=str(question_image) if question_image else None,
                    gold_image_ids=gold_image_ids,
                    difficulty_score=common_score
                    + task_weight
                    + prior_score(prior_signal),
                    difficulty_signals=signals,
                    prior_eval=prior_signal.to_dict(),
                    metadata={
                        "benchmark_kind": "multimodal_dialog_memory",
                        "question_id": question_id,
                    },
                )
            )
    return rows


def locomo_dialog_index(sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    conv = sample.get("conversation") or {}
    index: Dict[str, Dict[str, Any]] = {}
    session_nums = sorted(
        int(match.group(1))
        for key in conv
        if (match := re.fullmatch(r"session_(\d+)", key))
    )
    for session_num in session_nums:
        session_key = f"session_{session_num}"
        date_time = str(conv.get(f"{session_key}_date_time") or "")
        for turn_index, turn in enumerate(conv.get(session_key) or [], start=1):
            dia_id = str(turn.get("dia_id") or f"D{session_num}:{turn_index}")
            index[dia_id] = {
                "session_id": f"D{session_num}",
                "session_num": session_num,
                "date_time": date_time,
                "turn_index": turn_index,
                "has_image": bool(turn.get("img_url") or turn.get("blip_caption")),
            }
    return index


def locomo_samples(data_path: Path) -> List[HardSample]:
    rows: List[HardSample] = []
    data = read_json(data_path)
    category_weight = {
        5: 2.4,  # adversarial/no-information option in the official evaluator
        3: 2.2,  # multi-hop / inference-heavy in practice
        2: 1.8,  # temporal date questions
        4: 1.3,
        1: 0.8,
    }
    for sample in data:
        conv_id = str(sample.get("sample_id") or "conversation")
        dialog_index = locomo_dialog_index(sample)
        for index, qa in enumerate(sample.get("qa") or [], start=1):
            question = str(qa.get("question") or "")
            answer = str(
                qa.get("answer")
                or qa.get("adversarial_answer")
                or qa.get("gold_answer")
                or ""
            )
            if not question or not answer:
                continue
            evidence = unique_strs(qa.get("evidence") or [])
            session_ids = unique_strs(
                dialog_index.get(eid, {}).get("session_id") or clue_session_id(eid)
                for eid in evidence
            )
            common_score, signals = score_common(
                question,
                answer,
                evidence,
                session_ids,
                None,
                [],
            )
            category = qa.get("category")
            cat_weight = category_weight.get(category, 1.0)
            evidence_has_images = any(
                bool(dialog_index.get(eid, {}).get("has_image")) for eid in evidence
            )
            max_session_num = max(
                [
                    int(dialog_index.get(eid, {}).get("session_num") or 0)
                    for eid in evidence
                ]
                or [0]
            )
            long_horizon_bonus = min(2.0, max_session_num / 15.0)
            signals.update(
                {
                    "locomo_category_weight": cat_weight,
                    "adversarial_answer_field": "adversarial_answer" in qa
                    and "answer" not in qa,
                    "evidence_has_image_metadata": evidence_has_images,
                    "max_evidence_session_num": max_session_num,
                    "long_horizon_bonus": long_horizon_bonus,
                }
            )
            rows.append(
                HardSample(
                    uid=f"locomo::{conv_id}:{index}",
                    dataset="LoCoMo",
                    domain=conv_id,
                    source_file=str(data_path),
                    sample_id=f"{conv_id}:{index}",
                    question=question,
                    answer=answer,
                    category=category,
                    evidence=evidence,
                    session_ids=session_ids,
                    difficulty_score=common_score
                    + cat_weight
                    + long_horizon_bonus
                    + bool_score(evidence_has_images, 0.6),
                    difficulty_signals=signals,
                    prior_eval={},
                    metadata={
                        "benchmark_kind": "long_text_conversation_memory",
                        "selection_group": f"{conv_id}:cat{category}",
                        "speaker_a": (sample.get("conversation") or {}).get("speaker_a"),
                        "speaker_b": (sample.get("conversation") or {}).get("speaker_b"),
                    },
                )
            )
    return rows


def round_robin_select(
    candidates: List[HardSample],
    quota: int,
    seed: int,
) -> List[HardSample]:
    groups: Dict[str, List[HardSample]] = collections.defaultdict(list)
    for item in candidates:
        group_key = str(item.metadata.get("selection_group") or item.domain)
        groups[group_key].append(item)
    for values in groups.values():
        values.sort(
            key=lambda item: (
                item.difficulty_score,
                stable_float(item.uid, seed),
            ),
            reverse=True,
        )
    domains = sorted(
        groups,
        key=lambda domain: (
            max(item.difficulty_score for item in groups[domain]),
            stable_float(domain, seed),
        ),
        reverse=True,
    )
    selected: List[HardSample] = []
    while len(selected) < quota and domains:
        next_domains: List[str] = []
        for domain in domains:
            if len(selected) >= quota:
                break
            bucket = groups[domain]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if bucket:
                next_domains.append(domain)
        domains = next_domains
    return selected


def fill_shortfall(
    selected: List[HardSample],
    all_candidates: List[HardSample],
    target_size: int,
    seed: int,
) -> List[HardSample]:
    seen = {item.uid for item in selected}
    leftovers = [item for item in all_candidates if item.uid not in seen]
    leftovers.sort(
        key=lambda item: (
            item.difficulty_score,
            stable_float(item.uid, seed),
        ),
        reverse=True,
    )
    for item in leftovers:
        if len(selected) >= target_size:
            break
        selected.append(item)
    return selected


def summarize(rows: List[HardSample]) -> Dict[str, Any]:
    by_dataset = collections.Counter(item.dataset for item in rows)
    by_domain = collections.Counter((item.dataset, item.domain) for item in rows)
    by_point = collections.Counter(
        str(item.point) for item in rows if item.point is not None
    )
    by_category = collections.Counter(
        str(item.category) for item in rows if item.category is not None
    )
    signal_counts: collections.Counter[str] = collections.Counter()
    prior_wrong = 0
    for item in rows:
        q_signals = item.difficulty_signals.get("query_signals") or {}
        for name, present in q_signals.items():
            if present:
                signal_counts[name] += 1
        if (item.prior_eval or {}).get("wrong_count", 0):
            prior_wrong += 1
    scores = [item.difficulty_score for item in rows]
    return {
        "total": len(rows),
        "by_dataset": dict(by_dataset),
        "by_domain": {
            f"{dataset}/{domain}": count
            for (dataset, domain), count in sorted(by_domain.items())
        },
        "by_point": dict(by_point),
        "by_category": dict(by_category),
        "query_signal_counts": dict(signal_counts),
        "samples_with_prior_wrong": prior_wrong,
        "difficulty_score": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "avg": sum(scores) / len(scores) if scores else None,
        },
    }


def parse_quota(value: str, target_size: int) -> Dict[str, int]:
    if not value:
        memgallery = math.ceil(target_size * 0.60)
        memeye = math.ceil(target_size * 0.24)
        base = {
            "Mem-Gallery": memgallery,
            "MemEye": memeye,
            "LoCoMo": target_size - memgallery - memeye,
        }
        return base
    quota: Dict[str, int] = {}
    aliases = {
        "memgallery": "Mem-Gallery",
        "mem-gallery": "Mem-Gallery",
        "gallery": "Mem-Gallery",
        "locomo": "LoCoMo",
        "memeye": "MemEye",
    }
    for part in value.split(","):
        if not part.strip():
            continue
        name, raw_count = part.split("=", 1)
        key = aliases.get(name.strip().lower(), name.strip())
        quota[key] = int(raw_count)
    return quota


def build_hard_samples(args: argparse.Namespace) -> Path:
    memgallery_dir = args.memgallery_dir.expanduser().resolve()
    memeye_dir = normalize_memeye_data_dir(args.memeye_dir)
    locomo_path = args.locomo_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prior = load_prior_eval_signals(args.runs_dir.expanduser().resolve())

    pools = {
        "Mem-Gallery": memgallery_samples(memgallery_dir, prior),
        "LoCoMo": locomo_samples(locomo_path),
        "MemEye": memeye_samples(memeye_dir, prior, args.memeye_tasks),
    }
    quotas = parse_quota(args.quota, args.target_size)

    selected: List[HardSample] = []
    for dataset, candidates in pools.items():
        quota = max(0, int(quotas.get(dataset, 0)))
        selected.extend(round_robin_select(candidates, quota, args.seed))

    all_candidates = [item for candidates in pools.values() for item in candidates]
    selected = fill_shortfall(selected, all_candidates, args.target_size, args.seed)
    selected = selected[: args.target_size]
    selected.sort(
        key=lambda item: (
            item.dataset,
            item.domain,
            -item.difficulty_score,
            item.sample_id,
        )
    )

    stem = args.output_name
    jsonl_path = output_dir / f"{stem}.jsonl"
    manifest_path = output_dir / f"{stem}.manifest.json"
    write_jsonl(jsonl_path, (item.to_dict() for item in selected))
    manifest = {
        "output": str(jsonl_path),
        "target_size": args.target_size,
        "requested_quota": quotas,
        "candidate_counts": {
            dataset: len(candidates) for dataset, candidates in pools.items()
        },
        "selection_summary": summarize(selected),
        "difficulty_policy": {
            "primary_sources": [
                "Mem-Gallery is the main distribution",
                "MemEye is used as hard multimodal/visual-memory supplement",
                "LoCoMo is used as pure text long-term memory supplement",
                "historical prediction failures and evidence misses when available",
                "multi-support and cross-session evidence",
                "temporal/comparison/aggregation/reasoning/visual-detail query signals",
                "benchmark category or point weights",
                "long-horizon LoCoMo evidence sessions",
            ],
            "no_model_calls": True,
        },
        "source_paths": {
            "memgallery_dir": str(memgallery_dir),
            "memeye_dir": str(memeye_dir),
            "locomo_path": str(locomo_path),
            "runs_dir": str(args.runs_dir.expanduser().resolve()),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return jsonl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a balanced hard-sample QA manifest from Mem-Gallery, "
            "LoCoMo, and MemEye."
        )
    )
    parser.add_argument("--target-size", type=int, default=500)
    parser.add_argument(
        "--quota",
        default="",
        help=(
            "Optional comma list, e.g. Mem-Gallery=170,LoCoMo=170,MemEye=160. "
            "Defaults to approximately 34/33/33 percent."
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="hard_memory_samples_500")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--memgallery-dir", type=Path, default=default_memgallery_dir())
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
    parser.add_argument(
        "--memeye-tasks",
        nargs="*",
        default=DEFAULT_OPEN_TASKS,
        help="MemEye task names without .json. Defaults to Open tasks.",
    )
    return parser.parse_args()


def main() -> None:
    build_hard_samples(parse_args())


if __name__ == "__main__":
    main()
