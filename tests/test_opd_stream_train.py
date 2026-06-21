from __future__ import annotations

import argparse
import json

import torch

from opd_mm_baseline.interactive import AnswerValidationResult
from opd_mm_baseline.interactive import (
    ExecutorSession,
    InteractiveActionValidator,
)
from opd_mm_baseline.models import MemoryRecord
from opd_mm_baseline.retrieval import HiddenMemoryStore
from opd_mm_baseline.models import SFTExample, ToolAction
from opd_mm_baseline.online import (
    OnlineCorrection,
    OnlineSampleResult,
)
from opd_mm_baseline.opd_stream_train import (
    _head_tail_truncate,
    _offload_teacher_logits,
    _student_distillation_stats,
    _teacher_log_probs_from_cpu,
    activation_offload_context,
    build_deepspeed_plugin,
    LocalStudentPlanner,
    masked_completion_nll,
    reverse_kl_topk_log_probs,
    reverse_kl_topk_loss,
    streaming_examples_from_result,
)


def _completion_actions(completion: str):
    value = json.loads(completion)
    assert isinstance(value, list)
    return value


def test_deepspeed_plugin_is_optional_and_configures_zero2() -> None:
    args = argparse.Namespace(
        zero_stage=0,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        zero_offload_optimizer="none",
        train_batch_size=2,
        accelerate_num_processes=4,
    )
    assert build_deepspeed_plugin(args) is None

    args.zero_stage = 2
    plugin = build_deepspeed_plugin(args)
    assert plugin is not None
    config = plugin.deepspeed_config
    assert config["zero_optimization"]["stage"] == 2
    assert config["zero_optimization"]["offload_optimizer"]["device"] == "none"
    assert config["train_micro_batch_size_per_gpu"] == 2
    assert config["train_batch_size"] == 64
    assert config["zero_allow_untested_optimizer"] is True


def _result(student_raw_response: str) -> OnlineSampleResult:
    validation = AnswerValidationResult(
        correct=True,
        score=1.0,
        prediction="ok",
    )
    correction = OnlineCorrection(
        sample_id="sample",
        state_index=0,
        student_actions=[ToolAction("RETRIEVE", {"method": "bm25"})],
        teacher_actions=[
            ToolAction("RETRIEVE", {"method": "hybrid", "top_k": 5})
        ],
        teacher_answer_validation=validation,
        example=SFTExample(
            sample_id="sample:step:0",
            input="student prompt",
            target='[{"tool":"RETRIEVE","method":"hybrid","top_k":5}]',
            metadata={
                "teacher_decision_index": 0,
                "opd": {
                    "teacher_input": "privileged teacher prompt",
                },
            },
        ),
        student_raw_response=student_raw_response,
    )
    return OnlineSampleResult(
        sample_id="sample",
        student_actions=list(correction.student_actions),
        student_evidence_sufficiency=validation,
        student_answer_validation=AnswerValidationResult(
            correct=False,
            score=0.0,
            prediction="",
        ),
        corrections=[correction],
        student_planner_calls=1,
        teacher_attempts=[{"state_index": 0}],
    )


def test_streaming_example_uses_validated_teacher_action_as_completion() -> None:
    student_completion = '[{"tool":"RETRIEVE","method":"bm25"}]'
    examples = streaming_examples_from_result(
        _result(student_completion),
        quality_filter="teacher-correct",
    )
    assert len(examples) == 1
    completion = _completion_actions(examples[0].completion)
    assert completion == [
        {"tool": "RETRIEVE", "method": "hybrid", "top_k": 5}
    ]
    assert examples[0].teacher_prompt == "privileged teacher prompt"
    assert examples[0].teacher_actions[0]["method"] == "hybrid"
    privileged = json.loads(examples[0].privileged_context)
    assert "successful_next_action" not in privileged
    assert privileged["trajectory_step"] == 0
    assert "gold" not in examples[0].privileged_context.lower()


def test_streaming_example_keeps_teacher_target_after_bad_student_output() -> None:
    examples = streaming_examples_from_result(
        _result("I could not decide what to do."),
        quality_filter="teacher-correct",
    )
    assert len(examples) == 1
    assert _completion_actions(examples[0].completion)[0]["method"] == "hybrid"


def test_streaming_examples_keep_every_validated_teacher_decision() -> None:
    result = _result('[{"tool":"RETRIEVE","method":"bm25"}]')
    first = result.corrections[0]
    result.corrections.append(
        OnlineCorrection(
            sample_id="sample",
            state_index=0,
            student_actions=list(first.student_actions),
            teacher_actions=[ToolAction("READ", {"fields": ["summary"]})],
            teacher_answer_validation=first.teacher_answer_validation,
            example=SFTExample(
                sample_id="sample:step:1",
                input="student prompt after retrieval",
                target='[{"tool":"READ","fields":["summary"]}]',
                metadata={
                    "teacher_decision_index": 1,
                    "opd": {
                        "teacher_input": (
                            "privileged teacher prompt after retrieval"
                        ),
                    },
                },
            ),
            student_raw_response=first.student_raw_response,
        )
    )
    examples = streaming_examples_from_result(
        result,
        quality_filter="teacher-correct",
    )

    assert len(examples) == 2
    assert [example.teacher_decision_index for example in examples] == [0, 1]
    assert _completion_actions(examples[1].completion) == [
        {"tool": "READ", "fields": ["summary"]}
    ]
    privileged = json.loads(examples[1].privileged_context)
    assert "teacher_diagnosis" not in privileged


def test_streaming_examples_skip_multi_action_teacher_targets() -> None:
    result = _result('[{"tool":"RETRIEVE","method":"bm25"}]')
    first = result.corrections[0]
    first.teacher_actions = [
        ToolAction("RETRIEVE", {"method": "hybrid", "top_k": 5}),
        ToolAction("READ", {"fields": ["summary"]}),
    ]
    result.corrections.append(
        OnlineCorrection(
            sample_id="sample",
            state_index=1,
            student_actions=[ToolAction("READ", {"fields": ["summary"]})],
            teacher_actions=[ToolAction("READ", {"fields": ["summary"]})],
            teacher_answer_validation=first.teacher_answer_validation,
            example=SFTExample(
                sample_id="sample:state1",
                input="student prompt after retrieval",
                target='[{"tool":"READ","fields":["summary"]}]',
                metadata={
                    "teacher_decision_index": 0,
                    "opd": {
                        "teacher_input": (
                            "privileged teacher prompt after retrieval"
                        ),
                    },
                },
            ),
        )
    )

    examples = streaming_examples_from_result(
        result,
        quality_filter="teacher-correct",
    )

    assert len(examples) == 1
    assert _completion_actions(examples[0].completion) == [
        {"tool": "READ", "fields": ["summary"]}
    ]
    assert all(len(_completion_actions(example.completion)) == 1 for example in examples)


def test_streaming_examples_dedupe_same_state_to_best_target() -> None:
    result = _result('[{"tool":"RETRIEVE","method":"bm25"}]')
    first = result.corrections[0]
    result.corrections.append(
        OnlineCorrection(
            sample_id="sample",
            state_index=0,
            student_actions=list(first.student_actions),
            teacher_actions=[ToolAction("STOP")],
            teacher_answer_validation=first.teacher_answer_validation,
            example=SFTExample(
                sample_id="sample:step:stop",
                input=first.example.input,
                target='[{"tool":"STOP"}]',
                metadata={
                    "teacher_decision_index": 1,
                    "opd": {
                        "teacher_input": "privileged teacher prompt",
                    },
                },
            ),
            student_raw_response=first.student_raw_response,
            teacher_action_source="answer_stop",
            teacher_verification={
                "answerable": True,
                "relevance": 1.0,
                "completeness": 1.0,
                "redundancy": 0.0,
            },
        )
    )
    examples = streaming_examples_from_result(
        result,
        quality_filter="teacher-correct",
    )
    assert len(examples) == 1
    assert _completion_actions(examples[0].completion) == [{"tool": "STOP"}]
    assert examples[0].state_key
    assert examples[0].teacher_action_source == "answer_stop"


def test_streaming_examples_rebalance_toward_positive_states() -> None:
    result = _result('[{"tool":"RETRIEVE","method":"bm25"}]')
    first = result.corrections[0]
    result.corrections.append(
        OnlineCorrection(
            sample_id="sample",
            state_index=1,
            student_actions=[ToolAction("READ", {"fields": ["summary"]})],
            teacher_actions=[ToolAction("READ", {"fields": ["summary"]})],
            teacher_answer_validation=first.teacher_answer_validation,
            example=SFTExample(
                sample_id="sample:state1",
                input="student prompt after evidence",
                target='[{"tool":"READ","fields":["summary"]}]',
                metadata={
                    "teacher_decision_index": 0,
                    "opd": {
                        "teacher_input": (
                            "privileged teacher prompt after evidence"
                        ),
                    },
                },
            ),
        )
    )
    examples = streaming_examples_from_result(
        result,
        quality_filter="teacher-correct",
        state0_keep_ratio=0.0,
        positive_state_repeat=2,
    )
    assert [example.state_index for example in examples] == [1, 1]


def test_head_tail_truncation_preserves_prompt_front_and_back() -> None:
    truncated, was_truncated = _head_tail_truncate(
        list(range(10)),
        max_length=6,
        head_tokens=2,
    )
    assert was_truncated
    assert truncated == [0, 1, 6, 7, 8, 9]


def test_local_student_planner_normalizes_action_arguments_shape() -> None:
    class Generator:
        @staticmethod
        def generate(_prompt: str) -> str:
            return json.dumps(
                [
                    {
                        "action": "RETRIEVE",
                        "arguments": {"method": "bm25", "top_k": 5},
                    }
                ]
            )

    validator = InteractiveActionValidator()
    planner = LocalStudentPlanner(Generator(), validator)
    session = ExecutorSession(
        query="find Lena",
        memory_store=HiddenMemoryStore(
            [
                MemoryRecord(
                    memory_id="m1",
                    turn_id="D1:1",
                    timestamp="2026-01-01",
                    author="user",
                    modality="text",
                    source_type="conversation",
                    summary="Lena studies biology.",
                )
            ]
        ),
        validator=validator,
    )
    actions = planner.propose(
        query="find Lena",
        history=[],
        observation=session.observation(),
    )[0]
    assert actions[0].tool == "RETRIEVE"
    assert actions[0].arguments["method"] == "bm25"


def test_local_student_planner_accepts_json_action_array() -> None:
    class Generator:
        @staticmethod
        def generate(_prompt: str) -> str:
            return json.dumps(
                [
                    {
                        "tool": "READ",
                        "fields": ["summary"],
                    }
                ]
            )

    validator = InteractiveActionValidator()
    planner = LocalStudentPlanner(Generator(), validator)
    session = ExecutorSession(
        query="find Lena",
        memory_store=HiddenMemoryStore(
            [
                MemoryRecord(
                    memory_id="m1",
                    turn_id="D1:1",
                    timestamp="2026-01-01",
                    author="user",
                    modality="text",
                    source_type="conversation",
                    summary="Lena studies biology.",
                )
            ]
        ),
        validator=validator,
    )
    session.execute_chunk([ToolAction("RETRIEVE", {"method": "bm25"})])
    actions = planner.propose(
        query="find Lena",
        history=session.history,
        observation=session.observation(),
    )[0]
    assert actions == [ToolAction("READ", {"fields": ["summary"]})]


def test_reverse_kl_topk_is_zero_for_matching_logits() -> None:
    logits = torch.tensor(
        [[[2.0, 1.0, -1.0], [0.5, 0.25, -0.5]]],
        requires_grad=True,
    )
    mask = torch.ones((1, 2), dtype=torch.long)
    loss = reverse_kl_topk_loss(
        logits,
        logits.detach().clone(),
        mask,
        top_k=2,
        add_tail=True,
    )
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_offloaded_topk_kl_matches_full_logits() -> None:
    student = torch.tensor(
        [[[2.0, 1.0, -1.0], [0.5, 0.25, -0.5]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[1.5, 1.25, -0.5], [0.25, 0.75, -0.25]]],
    )
    mask = torch.ones((1, 2), dtype=torch.long)
    expected = reverse_kl_topk_loss(
        student,
        teacher,
        mask,
        top_k=2,
        add_tail=True,
    )
    student_log_probs, indices = _student_distillation_stats(
        student,
        top_k=2,
        temperature=1.0,
    )
    teacher_cpu, teacher_log_z_cpu = _offload_teacher_logits(
        teacher,
        temperature=1.0,
    )
    teacher_log_probs = _teacher_log_probs_from_cpu(
        teacher_cpu,
        teacher_log_z_cpu,
        indices,
        device=student.device,
    )
    actual = reverse_kl_topk_log_probs(
        student_log_probs,
        teacher_log_probs,
        mask,
        add_tail=True,
    )
    assert torch.allclose(actual, expected)


def test_activation_offload_context_is_noop_on_cpu() -> None:
    model = torch.nn.Linear(3, 2)
    context = activation_offload_context(
        model,
        enabled=True,
        device=torch.device("cpu"),
    )
    with context:
        output = model(torch.ones(1, 3)).sum()
    output.backward()
    assert model.weight.grad is not None


def test_masked_completion_nll_ignores_padding() -> None:
    logits = torch.tensor(
        [
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        ],
        requires_grad=True,
    )
    completion_ids = torch.tensor([[0, 1, 0]])
    completion_mask = torch.tensor([[1, 1, 0]])
    loss = masked_completion_nll(
        logits,
        completion_ids,
        completion_mask,
    )
    expected = torch.nn.functional.cross_entropy(
        logits[:, :2, :].transpose(1, 2),
        completion_ids[:, :2],
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
