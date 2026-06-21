"""Streaming on-policy distillation for the interactive memory policy.

Unlike the legacy round-batch runner, this worker never builds a static train
split. Each update is produced by the current student:

1. The student samples tool actions from its simple prompt.
2. The interactive teacher finds a trajectory that the answer model validates.
3. Each teacher decision becomes a next-action target for its own state.
4. Student and privileged-teacher logits are aligned on the teacher action.
5. A hard-label action loss anchors the student to the validated trajectory.
6. The optimizer updates the student before the next rollout batch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, DistributedDataParallelKwargs
from accelerate.utils import gather_object, set_seed
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from dual_encoder_memory import MiniLMTextEncoder, SigLIPVisionEncoder
from omnimem.config import require_memgallery_dir

from .build_opd_dataset import split_name
from .clients import extract_json_array
from .interactive import (
    ChatInteractivePlanner,
    InteractiveActionValidator,
    build_simple_student_policy_prompt,
)
from .memgallery import build_scenario_store, scenario_samples
from .memgallery_online_pipeline import make_components
from .memgallery_pipeline import now_stamp, resolve_scenarios
from .models import OPDSample, ToolAction
from .online import OnlineCorrection, OnlineSampleResult, OnlineSelfDistiller


STUDENT_ADAPTER = "student"
TEACHER_ADAPTER = "teacher"


def state_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def minimal_wandb_settings() -> Any:
    """Disable automatic telemetry while keeping explicit training metrics."""
    os.environ.update(
        {
            "WANDB__DISABLE_STATS": "true",
            "WANDB__DISABLE_MACHINE_INFO": "true",
            "WANDB__DISABLE_META": "true",
            "WANDB__SAVE_REQUIREMENTS": "false",
            "WANDB_DISABLE_CODE": "true",
            "WANDB_DISABLE_GIT": "true",
            "WANDB_CONSOLE": "off",
        }
    )
    import wandb

    return wandb.Settings(
        x_disable_stats=True,
        x_disable_machine_info=True,
        x_disable_meta=True,
        x_save_requirements=False,
        disable_code=True,
        disable_git=True,
        save_code=False,
        console="off",
    )


def build_deepspeed_plugin(
    args: argparse.Namespace,
) -> Optional[DeepSpeedPlugin]:
    zero_stage = int(getattr(args, "zero_stage", 0))
    if zero_stage == 0:
        return None
    if zero_stage != 2:
        raise ValueError(
            "Streaming OPD currently supports --zero-stage 0 or 2."
        )
    plugin = DeepSpeedPlugin(
        zero_stage=zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
        offload_optimizer_device=args.zero_offload_optimizer,
    )
    plugin.deepspeed_config.update(
        {
            "train_micro_batch_size_per_gpu": args.train_batch_size,
            "train_batch_size": (
                args.train_batch_size
                * args.gradient_accumulation_steps
                * args.accelerate_num_processes
            ),
            "zero_allow_untested_optimizer": True,
        }
    )
    return plugin


@dataclass
class StreamingExample:
    sample_id: str
    state_key: str
    state_index: int
    teacher_decision_index: int
    policy_version: int
    prompt: str
    teacher_prompt: str
    completion: str
    privileged_context: str
    teacher_action_source: str
    teacher_answer_correct: bool
    teacher_answer_score: float
    teacher_verification: Dict[str, Any]
    student_actions: List[Dict[str, Any]]
    teacher_actions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "state_key": self.state_key,
            "state_index": self.state_index,
            "teacher_decision_index": self.teacher_decision_index,
            "policy_version": self.policy_version,
            "prompt": self.prompt,
            "teacher_prompt": self.teacher_prompt,
            "completion": self.completion,
            "privileged_context": self.privileged_context,
            "teacher_action_source": self.teacher_action_source,
            "teacher_answer_correct": self.teacher_answer_correct,
            "teacher_answer_score": self.teacher_answer_score,
            "teacher_verification": self.teacher_verification,
            "student_actions": self.student_actions,
            "teacher_actions": self.teacher_actions,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "StreamingExample":
        return cls(
            sample_id=str(value["sample_id"]),
            state_key=str(
                value.get("state_key") or state_key(str(value["prompt"]))
            ),
            state_index=int(value["state_index"]),
            teacher_decision_index=int(
                value.get("teacher_decision_index", 0)
            ),
            policy_version=int(value.get("policy_version", 0)),
            prompt=str(value["prompt"]),
            teacher_prompt=str(
                value.get("teacher_prompt") or value["prompt"]
            ),
            completion=str(value["completion"]),
            privileged_context=str(value["privileged_context"]),
            teacher_action_source=str(value.get("teacher_action_source") or ""),
            teacher_answer_correct=bool(
                value.get("teacher_answer_correct", True)
            ),
            teacher_answer_score=float(value.get("teacher_answer_score", 1.0)),
            teacher_verification=dict(value.get("teacher_verification") or {}),
            student_actions=list(value.get("student_actions") or []),
            teacher_actions=list(value.get("teacher_actions") or []),
        )


class LocalStudentPlanner(ChatInteractivePlanner):
    """ChatInteractivePlanner backed by the live trainable model."""

    def __init__(
        self,
        generator: "LocalPolicyGenerator",
        validator: InteractiveActionValidator,
    ):
        self.generator = generator
        self.validator = validator
        self.calls = 0
        self.last_raw_response = ""
        self.last_candidate_sources: Dict[str, str] = {}
        self.last_candidate_rationales: Dict[str, Dict[str, str]] = {}

    def propose(
        self,
        query: str,
        history: List[ToolAction],
        observation: Any,
        candidate_count: int = 1,
        privileged_feedback: Optional[Dict[str, Any]] = None,
    ) -> List[List[ToolAction]]:
        del candidate_count, privileged_feedback
        self.calls += 1
        prompt = build_simple_student_policy_prompt(
            query=query,
            history=history,
            observation=observation,
            schema=self.validator.schema_text(),
        )
        raw = self.generator.generate(prompt)
        self.last_raw_response = raw
        values = extract_json_array(raw)
        normalized = []
        for value in values:
            if not isinstance(value, dict):
                normalized.append(value)
                continue
            if "tool" in value:
                normalized.append(value)
                continue
            action_name = value.get("action") or value.get("name")
            arguments = value.get("arguments")
            if action_name and isinstance(arguments, dict):
                normalized.append({"tool": action_name, **arguments})
            elif action_name:
                normalized.append({"tool": action_name})
            else:
                normalized.append(value)
        actions = self.validator.repair(normalized)
        return [actions]


class LocalPolicyGenerator:
    def __init__(
        self,
        accelerator: Accelerator,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))
        self.top_p = max(0.0, min(1.0, float(top_p)))
        self.enable_thinking = bool(enable_thinking)

    def generate(self, prompt: str) -> str:
        model = self.accelerator.unwrap_model(self.model)
        set_policy_role(model, STUDENT_ADAPTER)
        was_training = model.training
        model.eval()
        prompt_ids = render_policy_prompt(
            self.tokenizer,
            prompt,
            enable_thinking=self.enable_thinking,
        )
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.accelerator.device,
        )
        attention_mask = torch.ones_like(input_ids)
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.temperature > 0.0:
            generation_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        completion_ids = generated[0, input_ids.size(1) :]
        raw = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if was_training:
            model.train()
        return raw


def render_policy_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    privileged_context: Optional[str] = None,
    enable_thinking: bool = False,
) -> List[int]:
    if privileged_context:
        user_content = (
            f"{prompt.rstrip()}\n\n"
            "A successful expert demonstration for this exact state is shown "
            "below. It is privileged training context and will not be present "
            "at deployment.\n\n"
            f"{privileged_context.strip()}\n\n"
            "Now produce your own next action for the original state. "
            "Return only the final JSON array containing exactly one "
            "executable tool action."
        )
    else:
        user_content = prompt
    messages = [
        {
            "role": "system",
            "content": (
                "Think privately if enabled. Supervise only the final answer: "
                "a JSON array of executable tool actions."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not enable_thinking and rendered.endswith("<think>\n"):
        rendered += "</think>\n\n"
    return tokenizer(
        rendered,
        add_special_tokens=False,
    )["input_ids"]


def _teacher_completion(correction: OnlineCorrection) -> Optional[str]:
    actions = [action.to_dict() for action in correction.teacher_actions]
    if len(actions) != 1:
        return None
    return json.dumps(
        actions,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _teacher_prompt(correction: OnlineCorrection) -> str:
    metadata = correction.example.metadata or {}
    opd_metadata = metadata.get("opd") or {}
    teacher_prompt = str(opd_metadata.get("teacher_input") or "").strip()
    if teacher_prompt:
        return teacher_prompt
    return correction.example.input


def _privileged_context(
    correction: OnlineCorrection,
    attempt: Dict[str, Any],
) -> str:
    decision_index = int(
        correction.example.metadata.get("teacher_decision_index", 0)
    )
    payload: Dict[str, Any] = {
        "validated_outcome": (
            "This decision belongs to a trajectory that produced a correct "
            "final answer."
        ),
        "trajectory_step": decision_index,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _actions_are_stop(actions: List[Dict[str, Any]]) -> bool:
    return bool(actions) and all(action.get("tool") == "STOP" for action in actions)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _streaming_example_rank(example: StreamingExample) -> tuple:
    verification = example.teacher_verification or {}
    source_priority = {
        "planner": 3,
        "planner_repaired": 2,
        "answer_stop": 1,
        "verifier_stop": 1,
    }.get(example.teacher_action_source, 0)
    return (
        int(example.teacher_answer_correct),
        float(example.teacher_answer_score),
        _safe_float(verification.get("completeness")),
        _safe_float(verification.get("relevance")),
        int(_actions_are_stop(example.teacher_actions)),
        source_priority,
        -len(example.teacher_actions),
        -len(example.completion),
        -example.teacher_decision_index,
    )


def _select_best_state_targets(
    examples: List[StreamingExample],
) -> List[StreamingExample]:
    best_by_state: Dict[str, StreamingExample] = {}
    for example in examples:
        current = best_by_state.get(example.state_key)
        if current is None or (
            _streaming_example_rank(example)
            > _streaming_example_rank(current)
        ):
            best_by_state[example.state_key] = example
    return sorted(
        best_by_state.values(),
        key=lambda example: (
            example.state_index,
            example.teacher_decision_index,
            example.sample_id,
            example.state_key,
        ),
    )


def _deterministic_keep_state0(
    example: StreamingExample,
    keep_ratio: float,
) -> bool:
    if keep_ratio >= 1.0:
        return True
    if keep_ratio <= 0.0:
        return False
    digest = hashlib.sha256(
        f"{example.sample_id}:{example.policy_version}:{example.state_key}".encode(
            "utf-8"
        )
    ).hexdigest()
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return bucket < keep_ratio


def _rebalance_state_examples(
    examples: List[StreamingExample],
    *,
    state0_keep_ratio: float,
    positive_state_repeat: int,
) -> List[StreamingExample]:
    state0 = [example for example in examples if example.state_index <= 0]
    positive = [example for example in examples if example.state_index > 0]
    selected: List[StreamingExample] = []
    if positive:
        selected.extend(
            example
            for example in state0
            if _deterministic_keep_state0(example, state0_keep_ratio)
        )
    elif state0:
        if state0_keep_ratio >= 1.0:
            selected.extend(state0)
        else:
            selected.append(state0[0])
    repeat = max(1, int(positive_state_repeat))
    for example in positive:
        selected.extend([example] * repeat)
    return selected


def streaming_examples_from_result(
    result: OnlineSampleResult,
    *,
    quality_filter: str,
    policy_version: int = 0,
    state0_keep_ratio: float = 1.0,
    positive_state_repeat: int = 1,
) -> List[StreamingExample]:
    if quality_filter == "student-answer-failure":
        if result.student_answer_validation.correct:
            return []
    elif quality_filter == "student-sufficiency-failure":
        if result.student_evidence_sufficiency.correct:
            return []
    elif quality_filter != "teacher-correct":
        raise ValueError(f"invalid quality filter: {quality_filter}")

    attempts = {
        int(attempt.get("state_index", -1)): attempt
        for attempt in result.teacher_attempts
    }
    examples: List[StreamingExample] = []
    corrections = sorted(
        result.corrections,
        key=lambda correction: (
            correction.state_index,
            int(
                correction.example.metadata.get(
                    "teacher_decision_index",
                    0,
                )
            ),
        ),
    )
    for correction in corrections:
        completion = _teacher_completion(correction)
        if completion is None:
            continue
        state_index = correction.state_index
        decision_index = int(
            correction.example.metadata.get("teacher_decision_index", 0)
        )
        attempt = attempts.get(state_index, {})
        prompt = correction.example.input
        verification = dict(correction.teacher_verification or {})
        examples.append(
            StreamingExample(
                sample_id=result.sample_id,
                state_key=state_key(prompt),
                state_index=state_index,
                teacher_decision_index=decision_index,
                policy_version=policy_version,
                prompt=prompt,
                teacher_prompt=_teacher_prompt(correction),
                completion=completion,
                privileged_context=_privileged_context(correction, attempt),
                teacher_action_source=correction.teacher_action_source,
                teacher_answer_correct=correction.teacher_answer_validation.correct,
                teacher_answer_score=float(
                    correction.teacher_answer_validation.score
                ),
                teacher_verification=verification,
                student_actions=[
                    action.to_dict() for action in correction.student_actions
                ],
                teacher_actions=[
                    action.to_dict() for action in correction.teacher_actions
                ],
            )
        )
    deduped = _select_best_state_targets(examples)
    return _rebalance_state_examples(
        deduped,
        state0_keep_ratio=state0_keep_ratio,
        positive_state_repeat=positive_state_repeat,
    )


def _copy_student_to_teacher(model: Any) -> None:
    student_state = get_peft_model_state_dict(
        model,
        adapter_name=STUDENT_ADAPTER,
    )
    set_peft_model_state_dict(
        model,
        student_state,
        adapter_name=TEACHER_ADAPTER,
    )


def set_policy_role(model: Any, role: str) -> None:
    if getattr(model, "peft_config", None) and hasattr(model, "set_adapter"):
        model.set_adapter(role)


@torch.no_grad()
def update_ema_teacher(model: Any, decay: float) -> None:
    if not getattr(model, "peft_config", None):
        return
    decay = max(0.0, min(1.0, float(decay)))
    student_state = get_peft_model_state_dict(
        model,
        adapter_name=STUDENT_ADAPTER,
    )
    teacher_state = get_peft_model_state_dict(
        model,
        adapter_name=TEACHER_ADAPTER,
    )
    updated = {
        key: teacher_state[key].mul(decay).add(
            student_state[key],
            alpha=1.0 - decay,
        )
        for key in student_state
    }
    set_peft_model_state_dict(
        model,
        updated,
        adapter_name=TEACHER_ADAPTER,
    )


def load_policy_model(
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> tuple[Any, Any]:
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.train_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.train_model,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.training_mode == "lora":
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(base, lora, adapter_name=STUDENT_ADAPTER)
        model.add_adapter(TEACHER_ADAPTER, copy.deepcopy(lora))
        _copy_student_to_teacher(model)
        model.set_adapter(STUDENT_ADAPTER)
        for name, parameter in model.named_parameters():
            if f".{TEACHER_ADAPTER}." in name:
                parameter.requires_grad_(False)
    else:
        model = base
        if args.freeze_vision_tower:
            for parameter in model.model.visual.parameters():
                parameter.requires_grad_(False)
        if args.freeze_token_embeddings:
            for parameter in model.get_input_embeddings().parameters():
                parameter.requires_grad_(False)
            for parameter in model.lm_head.parameters():
                parameter.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    if accelerator.is_main_process and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    elif accelerator.is_main_process:
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"trainable params: {trainable:,} || all params: {total:,} || "
            f"trainable%: {100.0 * trainable / total:.4f}"
        )
    return model, tokenizer


def _pad(
    values: List[List[int]],
    *,
    pad_id: int,
    side: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(value) for value in values)
    ids = []
    masks = []
    for value in values:
        padding = [pad_id] * (width - len(value))
        mask_padding = [0] * len(padding)
        if side == "left":
            ids.append(padding + value)
            masks.append(mask_padding + [1] * len(value))
        else:
            ids.append(value + padding)
            masks.append([1] * len(value) + mask_padding)
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def _head_tail_truncate(
    ids: List[int],
    *,
    max_length: int,
    head_tokens: int,
) -> tuple[List[int], bool]:
    if max_length <= 0 or len(ids) <= max_length:
        return ids, False
    head = max(0, min(int(head_tokens), max_length))
    tail = max_length - head
    if tail <= 0:
        return ids[:max_length], True
    return [*ids[:head], *ids[-tail:]], True


def _percentile(values: List[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def _prompt_batch_stats(
    lengths: List[int],
    truncated: List[bool],
    *,
    prefix: str,
) -> Dict[str, float]:
    if not lengths:
        return {
            f"{prefix}_prompt_truncation_rate": 0.0,
            f"{prefix}_prompt_tokens_p50": 0.0,
            f"{prefix}_prompt_tokens_p90": 0.0,
            f"{prefix}_prompt_tokens_max": 0.0,
        }
    return {
        f"{prefix}_prompt_truncation_rate": (
            sum(1 for value in truncated if value) / len(truncated)
        ),
        f"{prefix}_prompt_tokens_p50": _percentile(lengths, 0.50),
        f"{prefix}_prompt_tokens_p90": _percentile(lengths, 0.90),
        f"{prefix}_prompt_tokens_max": float(max(lengths)),
    }


def build_distillation_batch(
    examples: List[StreamingExample],
    tokenizer: Any,
    *,
    device: torch.device,
    max_prompt_length: int,
    max_completion_length: int,
    prompt_head_tokens: int,
    enable_thinking: bool,
) -> Dict[str, Any]:
    student_prompts = []
    teacher_prompts = []
    completions = []
    student_lengths: List[int] = []
    teacher_lengths: List[int] = []
    student_truncated: List[bool] = []
    teacher_truncated: List[bool] = []
    for example in examples:
        student_ids_full = render_policy_prompt(
            tokenizer,
            example.prompt,
            enable_thinking=enable_thinking,
        )
        teacher_ids_full = render_policy_prompt(
            tokenizer,
            example.teacher_prompt,
            enable_thinking=enable_thinking,
        )
        student_ids, was_student_truncated = _head_tail_truncate(
            student_ids_full,
            max_length=max_prompt_length,
            head_tokens=prompt_head_tokens,
        )
        teacher_ids, was_teacher_truncated = _head_tail_truncate(
            teacher_ids_full,
            max_length=max_prompt_length,
            head_tokens=prompt_head_tokens,
        )
        student_lengths.append(len(student_ids_full))
        teacher_lengths.append(len(teacher_ids_full))
        student_truncated.append(was_student_truncated)
        teacher_truncated.append(was_teacher_truncated)
        completion_ids = tokenizer(
            example.completion,
            add_special_tokens=False,
        )["input_ids"]
        completion_ids = completion_ids[: max(1, max_completion_length - 1)]
        completion_ids.append(tokenizer.eos_token_id)
        student_prompts.append(student_ids)
        teacher_prompts.append(teacher_ids)
        completions.append(completion_ids)

    student_ids, student_mask = _pad(
        student_prompts,
        pad_id=tokenizer.pad_token_id,
        side="left",
        device=device,
    )
    teacher_ids, teacher_mask = _pad(
        teacher_prompts,
        pad_id=tokenizer.pad_token_id,
        side="left",
        device=device,
    )
    completion_ids, completion_mask = _pad(
        completions,
        pad_id=tokenizer.pad_token_id,
        side="right",
        device=device,
    )
    return {
        "student_input_ids": torch.cat([student_ids, completion_ids], dim=1),
        "student_attention_mask": torch.cat(
            [student_mask, completion_mask],
            dim=1,
        ),
        "teacher_input_ids": torch.cat([teacher_ids, completion_ids], dim=1),
        "teacher_attention_mask": torch.cat(
            [teacher_mask, completion_mask],
            dim=1,
        ),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "prompt_stats": {
            **_prompt_batch_stats(
                student_lengths,
                student_truncated,
                prefix="student",
            ),
            **_prompt_batch_stats(
                teacher_lengths,
                teacher_truncated,
                prefix="teacher",
            ),
        },
    }


def _forward_completion_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_length: int,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=completion_length + 1,
    )
    logits = outputs.logits[:, :-1, :]
    return logits[:, -completion_length:, :]


def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    captured = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    captured = torch.clamp(captured, max=-1e-7)
    tail = torch.log(-torch.expm1(captured))
    return torch.cat([log_probs, tail], dim=-1)


def reverse_kl_topk_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    top_k: int,
    add_tail: bool,
) -> torch.Tensor:
    top_k = min(max(1, int(top_k)), student_logits.size(-1))
    student_log_z = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    top_student, indices = torch.topk(student_logits, k=top_k, dim=-1)
    student_log_probs = top_student - student_log_z
    teacher_log_z = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
    teacher_log_probs = torch.gather(
        teacher_logits,
        dim=-1,
        index=indices,
    ) - teacher_log_z
    if add_tail:
        student_log_probs = _add_tail_bucket(student_log_probs)
        teacher_log_probs = _add_tail_bucket(teacher_log_probs)
    else:
        student_log_probs = student_log_probs - torch.logsumexp(
            student_log_probs,
            dim=-1,
            keepdim=True,
        )
        teacher_log_probs = teacher_log_probs - torch.logsumexp(
            teacher_log_probs,
            dim=-1,
            keepdim=True,
        )
    per_token = F.kl_div(
        teacher_log_probs,
        student_log_probs,
        reduction="none",
        log_target=True,
    ).sum(-1)
    mask = completion_mask.to(per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def reverse_kl_topk_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    add_tail: bool,
) -> torch.Tensor:
    if add_tail:
        student_log_probs = _add_tail_bucket(student_log_probs)
        teacher_log_probs = _add_tail_bucket(teacher_log_probs)
    else:
        student_log_probs = student_log_probs - torch.logsumexp(
            student_log_probs,
            dim=-1,
            keepdim=True,
        )
        teacher_log_probs = teacher_log_probs - torch.logsumexp(
            teacher_log_probs,
            dim=-1,
            keepdim=True,
        )
    per_token = F.kl_div(
        teacher_log_probs,
        student_log_probs,
        reduction="none",
        log_target=True,
    ).sum(-1)
    mask = completion_mask.to(per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def _scale_logits(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature == 1.0:
        return logits
    return logits / temperature


def _student_distillation_stats(
    logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    top_k = min(max(1, int(top_k)), logits.size(-1))
    scaled = _scale_logits(logits, temperature)
    log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
    top_values, indices = torch.topk(scaled, k=top_k, dim=-1)
    return top_values - log_z, indices


@torch.no_grad()
def _offload_teacher_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = _scale_logits(logits, temperature)
    log_z = torch.logsumexp(scaled, dim=-1, keepdim=True)
    if scaled is logits:
        cpu_logits = logits.to("cpu")
    else:
        cpu_logits = scaled.to("cpu")
    return cpu_logits, log_z.to("cpu")


def _teacher_log_probs_from_cpu(
    teacher_logits: torch.Tensor,
    teacher_log_z: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    selected = torch.gather(
        teacher_logits,
        dim=-1,
        index=indices.detach().to("cpu"),
    )
    return selected.to(device) - teacher_log_z.to(device)


def activation_offload_context(
    model: Any,
    *,
    enabled: bool,
    device: torch.device,
) -> Any:
    if not enabled or device.type != "cuda":
        return nullcontext()

    parameter_storages = {
        parameter.untyped_storage().data_ptr()
        for parameter in model.parameters()
    }

    def pack(tensor: torch.Tensor) -> tuple[str, Any, Any]:
        if (
            tensor.device.type != "cuda"
            or tensor.layout != torch.strided
            or tensor.untyped_storage().data_ptr() in parameter_storages
        ):
            return ("keep", tensor, None)
        cpu_tensor = torch.empty(
            tensor.size(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device="cpu",
            pin_memory=True,
        )
        cpu_tensor.copy_(tensor, non_blocking=False)
        return ("cpu", cpu_tensor, tensor.device)

    def unpack(packed: tuple[str, Any, Any]) -> torch.Tensor:
        location, tensor, original_device = packed
        if location == "keep":
            return tensor
        return tensor.to(original_device, non_blocking=True)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def masked_completion_nll(
    student_logits: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    per_token = F.cross_entropy(
        student_logits.transpose(1, 2),
        completion_ids,
        reduction="none",
    )
    mask = completion_mask.to(per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def train_distillation_batch(
    *,
    accelerator: Accelerator,
    model: Any,
    optimizer: Any,
    tokenizer: Any,
    examples: List[StreamingExample],
    args: argparse.Namespace,
) -> Dict[str, float]:
    batch = build_distillation_batch(
        examples,
        tokenizer,
        device=accelerator.device,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        prompt_head_tokens=args.prompt_head_tokens,
        enable_thinking=bool(getattr(args, "supervise_thinking", False)),
    )
    unwrapped = accelerator.unwrap_model(model)
    with accelerator.accumulate(model):
        temperature = max(float(args.distill_temperature), 1e-6)
        with torch.no_grad():
            set_policy_role(unwrapped, TEACHER_ADAPTER)
            teacher_logits = _forward_completion_logits(
                unwrapped,
                batch["teacher_input_ids"],
                batch["teacher_attention_mask"],
                batch["completion_ids"].size(1),
            )
            teacher_logits_cpu, teacher_log_z_cpu = (
                _offload_teacher_logits(
                    teacher_logits,
                    temperature=temperature,
                )
            )
            del teacher_logits
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()

        set_policy_role(unwrapped, STUDENT_ADAPTER)
        model.train()
        activation_context = activation_offload_context(
            unwrapped,
            enabled=args.activation_offload,
            device=accelerator.device,
        )
        with activation_context:
            student_logits = _forward_completion_logits(
                model,
                batch["student_input_ids"],
                batch["student_attention_mask"],
                batch["completion_ids"].size(1),
            )
            nll_loss = masked_completion_nll(
                student_logits,
                batch["completion_ids"],
                batch["completion_mask"],
            )
            student_log_probs, top_indices = _student_distillation_stats(
                student_logits,
                top_k=args.distill_top_k,
                temperature=temperature,
            )
            teacher_log_probs = _teacher_log_probs_from_cpu(
                teacher_logits_cpu,
                teacher_log_z_cpu,
                top_indices,
                device=student_logits.device,
            )
            del teacher_logits_cpu, teacher_log_z_cpu
            kl_loss = reverse_kl_topk_log_probs(
                student_log_probs,
                teacher_log_probs,
                batch["completion_mask"],
                add_tail=args.distill_add_tail,
            ) * (temperature**2)
            loss = (
                float(args.distill_kl_weight) * kl_loss
                + float(args.distill_nll_weight) * nll_loss
            )
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if accelerator.sync_gradients and args.training_mode == "lora":
            update_ema_teacher(
                unwrapped,
                args.teacher_ema_decay,
            )
    metrics = {
        "loss": float(loss.detach().item()),
        "reverse_kl": float(kl_loss.detach().item()),
        "teacher_action_nll": float(nll_loss.detach().item()),
    }
    metrics.update(
        {
            f"prompt_{key}": float(value)
            for key, value in batch["prompt_stats"].items()
        }
    )
    return metrics


def _iter_local_samples(
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> Iterable[OPDSample]:
    data_dir = require_memgallery_dir(args.data_dir)
    paths = resolve_scenarios(args)
    local_paths = paths[
        accelerator.process_index :: accelerator.num_processes
    ]
    dense_encoder = (
        MiniLMTextEncoder(args.dense_model, device=args.dense_device)
        if args.dense_mode == "minilm"
        else None
    )
    vision_device = args.vision_device
    if str(vision_device).startswith("cuda"):
        vision_device = str(accelerator.device)
    vision_encoder = (
        SigLIPVisionEncoder(args.vision_model, device=vision_device)
        if args.vision_mode == "siglip"
        else None
    )
    for path in local_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        store, _records = build_scenario_store(
            data,
            data_dir=data_dir,
            dense_encoder=dense_encoder,
            vision_encoder=vision_encoder,
            max_sessions=args.max_sessions,
            max_turns=args.max_turns,
        )
        samples = scenario_samples(
            data,
            store=store,
            data_dir=data_dir,
            scenario=path.stem,
            max_questions=args.max_questions,
            include_oracle_profile=False,
        )
        for sample in samples:
            sample.metadata.pop("teacher_privileged_context", None)
            if (
                args.val_ratio > 0
                and split_name(
                    sample.sample_id,
                    args.val_ratio,
                    args.split_seed,
                )
                == "val"
            ):
                continue
            yield sample


def _batch_local_samples(
    iterator: Iterable[OPDSample],
    batch_size: int,
) -> Iterable[List[OPDSample]]:
    batch = []
    for sample in iterator:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _prepare_rank_batch(
    queue: Deque[StreamingExample],
    *,
    rank: int,
    world_size: int,
    per_device_batch_size: int,
    flush: bool,
) -> Optional[List[StreamingExample]]:
    global_batch = world_size * per_device_batch_size
    if len(queue) < global_batch and not flush:
        return None
    if not queue:
        return None
    selected = []
    while queue and len(selected) < global_batch:
        selected.append(queue.popleft())
    while len(selected) < global_batch:
        selected.append(selected[len(selected) % len(selected)])
    start = rank * per_device_batch_size
    return selected[start : start + per_device_batch_size]


def _update_supervision_counters(
    examples: Iterable[StreamingExample],
    *,
    state_index_counts: Counter,
    teacher_decision_index_counts: Counter,
    first_tool_counts: Counter,
    trajectory_shape_counts: Counter,
    action_source_counts: Counter,
) -> None:
    for example in examples:
        state_index_counts[str(example.state_index)] += 1
        teacher_decision_index_counts[str(example.teacher_decision_index)] += 1
        first_tool = (
            example.teacher_actions[0].get("tool")
            if example.teacher_actions
            else "EMPTY"
        )
        first_tool_counts[str(first_tool)] += 1
        trajectory_shape_counts[
            "->".join(
                str(action.get("tool", "?"))
                for action in example.teacher_actions
            )
        ] += 1
        action_source_counts[str(example.teacher_action_source or "unknown")] += 1


def save_checkpoint(
    accelerator: Accelerator,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    step: int,
    args: argparse.Namespace,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    set_policy_role(unwrapped, STUDENT_ADAPTER)
    state_dict = accelerator.get_state_dict(model)
    save_kwargs: Dict[str, Any] = {
        "state_dict": state_dict,
        "safe_serialization": True,
    }
    if args.training_mode == "lora":
        save_kwargs["selected_adapters"] = [STUDENT_ADAPTER]
    unwrapped.save_pretrained(checkpoint_dir, **save_kwargs)
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "online_state.json").write_text(
        json.dumps(
            {
                "optimizer_step": step,
                "base_model": args.train_model,
                "training_mode": args.training_mode,
                "adapter": (
                    STUDENT_ADAPTER if args.training_mode == "lora" else None
                ),
                "teacher": (
                    "ema_lora"
                    if args.training_mode == "lora"
                    else "live_privileged_context"
                ),
                "teacher_ema_decay": (
                    args.teacher_ema_decay
                    if args.training_mode == "lora"
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_streaming_opd(args: argparse.Namespace) -> Path:
    proxy = str(args.proxy_url or "").strip()
    if proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[key] = proxy
    mixed_precision = "bf16" if args.bf16 else "fp16"
    deepspeed_plugin = build_deepspeed_plugin(args)
    accelerator_kwargs: Dict[str, Any] = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "log_with": "wandb" if args.report_to == "wandb" else None,
        "deepspeed_plugin": deepspeed_plugin,
    }
    if deepspeed_plugin is None:
        accelerator_kwargs["kwargs_handlers"] = [
            DistributedDataParallelKwargs(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        ]
    accelerator = Accelerator(
        **accelerator_kwargs,
    )
    set_seed(args.seed + accelerator.process_index)

    run_name = args.wandb_run_name or f"opd-stream-{now_stamp()}"
    run_dir_value = [""]
    if accelerator.is_main_process:
        run_dir = (
            args.output_dir.expanduser().resolve()
            / f"{now_stamp()}_opd_stream"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir_value[0] = str(run_dir)
    gathered_run_dir = gather_object(run_dir_value)
    run_dir = Path(next(value for value in gathered_run_dir if value))
    if accelerator.is_main_process:
        (run_dir / "config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()

    if args.report_to == "wandb" and args.wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = args.wandb_mode
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config={key: str(value) for key, value in vars(args).items()},
            init_kwargs={
                "wandb": {
                    "name": run_name,
                    "group": args.wandb_group or "opd-streaming",
                    "settings": minimal_wandb_settings(),
                    **(
                        {"entity": args.wandb_entity}
                        if args.wandb_entity
                        else {}
                    ),
                }
            },
        )

    model, tokenizer = load_policy_model(args, accelerator)
    optimizer_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if args.optimizer in {"adamw8bit", "paged_adamw8bit"}:
        from bitsandbytes.optim import AdamW8bit, PagedAdamW8bit

        optimizer_class = (
            PagedAdamW8bit
            if args.optimizer == "paged_adamw8bit"
            else AdamW8bit
        )
        optimizer = optimizer_class(
            optimizer_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            optimizer_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    model, optimizer = accelerator.prepare(model, optimizer)

    components_args = copy.copy(args)
    components = make_components(components_args)
    generator = LocalPolicyGenerator(
        accelerator,
        model,
        tokenizer,
        max_new_tokens=args.student_rollout_max_tokens,
        temperature=args.student_rollout_temperature,
        top_p=args.student_rollout_top_p,
        enable_thinking=args.student_enable_thinking,
    )
    local_planner = LocalStudentPlanner(generator, components["validator"])
    distiller = OnlineSelfDistiller(
        student_planner=local_planner,
        teacher_search=components["teacher_search"],
        answer_validator=components["answer_validator"],
        answer_model=components["answer_model"],
        answer_judge=components["judge"],
        validator=components["validator"],
        retriever=components["retriever"],
        max_student_rounds=args.student_max_rounds,
        max_student_actions=args.max_actions,
        raw_inspector=components["raw_inspector"],
        max_raw_inspections=args.max_raw_inspections,
    )

    local_iterator = iter(
        _batch_local_samples(
            _iter_local_samples(args, accelerator),
            args.online_samples_per_rank,
        )
    )
    pending: Deque[StreamingExample] = deque()
    samples_seen = 0
    examples_seen = 0
    optimizer_step = 0
    losses: List[float] = []
    reverse_kl_losses: List[float] = []
    teacher_action_nll_losses: List[float] = []
    prompt_metric_history: Dict[str, List[float]] = {}
    supervision_state_index_counts: Counter = Counter()
    supervision_teacher_decision_index_counts: Counter = Counter()
    supervision_first_tool_counts: Counter = Counter()
    supervision_trajectory_shape_counts: Counter = Counter()
    supervision_action_source_counts: Counter = Counter()
    atomic_skipped_multi_action_targets = 0
    rollout_batch_index = 0
    started = time.time()
    rollouts_path = run_dir / "online_rollouts.jsonl"
    examples_path = run_dir / "online_examples.jsonl"
    metrics_path = run_dir / "metrics.json"

    while True:
        local_batch = next(local_iterator, None)
        active = torch.tensor(
            int(local_batch is not None),
            device=accelerator.device,
            dtype=torch.long,
        )
        active_total = accelerator.reduce(active, reduction="sum").item()
        if active_total <= 0:
            break

        local_rollouts = []
        local_examples = []
        local_skipped_multi_action_targets = 0
        if local_batch is not None:
            for sample in local_batch:
                result = distiller.collect_sample(
                    sample,
                    round_index=optimizer_step,
                )
                rollout_row = result.to_dict()
                rollout_row["policy_version"] = optimizer_step
                local_rollouts.append(rollout_row)
                local_skipped_multi_action_targets += sum(
                    1
                    for correction in result.corrections
                    if len(correction.teacher_actions) != 1
                )
                local_examples.extend(
                    example.to_dict()
                    for example in streaming_examples_from_result(
                        result,
                        quality_filter=args.quality_filter,
                        policy_version=optimizer_step,
                        state0_keep_ratio=args.state0_keep_ratio,
                        positive_state_repeat=args.positive_state_repeat,
                    )
                )

        gathered_rollouts = gather_object(local_rollouts)
        gathered_examples = gather_object(local_examples)
        gathered_skipped_multi_action_targets = gather_object(
            [local_skipped_multi_action_targets]
        )
        new_examples = [
            StreamingExample.from_dict(value)
            for value in gathered_examples
        ]
        pending.extend(new_examples)
        samples_seen += len(gathered_rollouts)
        examples_seen += len(new_examples)
        atomic_skipped_multi_action_targets += sum(
            int(value) for value in gathered_skipped_multi_action_targets
        )
        _update_supervision_counters(
            new_examples,
            state_index_counts=supervision_state_index_counts,
            teacher_decision_index_counts=(
                supervision_teacher_decision_index_counts
            ),
            first_tool_counts=supervision_first_tool_counts,
            trajectory_shape_counts=supervision_trajectory_shape_counts,
            action_source_counts=supervision_action_source_counts,
        )
        rollout_batch_index += 1

        if accelerator.is_main_process:
            _write_jsonl(rollouts_path, gathered_rollouts)
            _write_jsonl(examples_path, gathered_examples)

        while True:
            rank_batch = _prepare_rank_batch(
                pending,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
                per_device_batch_size=args.train_batch_size,
                flush=False,
            )
            if rank_batch is None:
                break
            train_metrics = train_distillation_batch(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                examples=rank_batch,
                args=args,
            )
            if accelerator.sync_gradients:
                optimizer_step += 1
                losses.append(train_metrics["loss"])
                reverse_kl_losses.append(train_metrics["reverse_kl"])
                teacher_action_nll_losses.append(
                    train_metrics["teacher_action_nll"]
                )
                for key, value in train_metrics.items():
                    if key.startswith("prompt_"):
                        prompt_metric_history.setdefault(key, []).append(value)
                log_payload = {
                        "train/loss": train_metrics["loss"],
                        "train/reverse_kl": train_metrics["reverse_kl"],
                        "train/teacher_action_nll": (
                            train_metrics["teacher_action_nll"]
                        ),
                        "online/samples_seen": samples_seen,
                        "online/examples_seen": examples_seen,
                        "online/atomic_skipped_multi_action_targets": (
                            atomic_skipped_multi_action_targets
                        ),
                    }
                log_payload.update(
                    {
                        f"prompt/{key.removeprefix('prompt_')}": value
                        for key, value in train_metrics.items()
                        if key.startswith("prompt_")
                    }
                )
                accelerator.log(log_payload, step=optimizer_step)
                if (
                    args.save_steps > 0
                    and optimizer_step % args.save_steps == 0
                ):
                    save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        run_dir,
                        step=optimizer_step,
                        args=args,
                    )

    while pending:
        rank_batch = _prepare_rank_batch(
            pending,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            per_device_batch_size=args.train_batch_size,
            flush=True,
        )
        if rank_batch is None:
            break
        train_metrics = train_distillation_batch(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            examples=rank_batch,
            args=args,
        )
        if accelerator.sync_gradients:
            optimizer_step += 1
            losses.append(train_metrics["loss"])
            reverse_kl_losses.append(train_metrics["reverse_kl"])
            teacher_action_nll_losses.append(
                train_metrics["teacher_action_nll"]
            )
            for key, value in train_metrics.items():
                if key.startswith("prompt_"):
                    prompt_metric_history.setdefault(key, []).append(value)
            log_payload = {
                    "train/loss": train_metrics["loss"],
                    "train/reverse_kl": train_metrics["reverse_kl"],
                    "train/teacher_action_nll": (
                        train_metrics["teacher_action_nll"]
                    ),
                    "online/samples_seen": samples_seen,
                    "online/examples_seen": examples_seen,
                    "online/atomic_skipped_multi_action_targets": (
                        atomic_skipped_multi_action_targets
                    ),
                }
            log_payload.update(
                {
                    f"prompt/{key.removeprefix('prompt_')}": value
                    for key, value in train_metrics.items()
                    if key.startswith("prompt_")
                }
            )
            accelerator.log(log_payload, step=optimizer_step)

    if args.save_final:
        save_checkpoint(
            accelerator,
            model,
            tokenizer,
            run_dir,
            step=optimizer_step,
            args=args,
        )
    elapsed = time.time() - started
    if accelerator.is_main_process:
        metrics = {
            "mode": "streaming_on_policy_distillation",
            "samples_seen": samples_seen,
            "on_policy_examples": examples_seen,
            "optimizer_steps": optimizer_step,
            "atomic_skipped_multi_action_targets": (
                atomic_skipped_multi_action_targets
            ),
            "elapsed_seconds": elapsed,
            "samples_per_second": samples_seen / max(elapsed, 1e-6),
            "mean_loss": (
                sum(losses) / len(losses) if losses else None
            ),
            "last_loss": losses[-1] if losses else None,
            "mean_reverse_kl": (
                sum(reverse_kl_losses) / len(reverse_kl_losses)
                if reverse_kl_losses
                else None
            ),
            "last_reverse_kl": (
                reverse_kl_losses[-1] if reverse_kl_losses else None
            ),
            "mean_teacher_action_nll": (
                sum(teacher_action_nll_losses)
                / len(teacher_action_nll_losses)
                if teacher_action_nll_losses
                else None
            ),
            "last_teacher_action_nll": (
                teacher_action_nll_losses[-1]
                if teacher_action_nll_losses
                else None
            ),
            "loss": (
                "teacher_action_nll_plus_reverse_kl_to_ema_teacher"
                if args.training_mode == "lora"
                else (
                    "teacher_action_nll_plus_reverse_kl_to_"
                    "live_privileged_teacher"
                )
            ),
            "distill_kl_weight": args.distill_kl_weight,
            "distill_nll_weight": args.distill_nll_weight,
            "training_mode": args.training_mode,
            "teacher_context": (
                "validated teacher trajectory scored under the original "
                "privileged teacher prompt with "
                + (
                    "EMA adapter"
                    if args.training_mode == "lora"
                    else "live full-parameter teacher"
                )
            ),
            "static_train_dataset": False,
            "prompt_stats": {
                key: {
                    "mean": (
                        sum(values) / len(values) if values else None
                    ),
                    "last": values[-1] if values else None,
                }
                for key, values in sorted(prompt_metric_history.items())
            },
            "supervision_distribution": {
                "state_index": dict(
                    sorted(supervision_state_index_counts.items())
                ),
                "teacher_decision_index": dict(
                    sorted(supervision_teacher_decision_index_counts.items())
                ),
                "first_tool": dict(
                    supervision_first_tool_counts.most_common()
                ),
                "trajectory_shape": dict(
                    supervision_trajectory_shape_counts.most_common()
                ),
                "action_source": dict(
                    supervision_action_source_counts.most_common()
                ),
            },
        }
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    accelerator.end_training()
    return run_dir
