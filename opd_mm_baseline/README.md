# OPD-MM Baseline

This directory implements the lightweight query-only on-policy distillation
scheme described in `../opd_mm_baseline.md`.

## Isolation guarantees

- The student receives only the user query and executable tool schema.
- The teacher receives only the query, gold answer, student trajectory,
  student answer, and correctness.
- Neither policy receives the memory store, candidate list, evidence package,
  executor trace, or memory IDs.
- Only `ToolExecutor` receives `HiddenMemoryStore`.
- Every generated action is validated before execution.
- `RETRIEVE` always uses the original query and rejects custom query fields.

The teacher supports four training-time privilege levels:

- `minimal`: query, gold answer, student trajectory, answer, and correctness.
- `diagnostic` (default): additionally receives per-action pool counts and only
  the evidence already observed by the student, with IDs and paths removed.
- `oracle-feedback`: receives only the abstract support modality/source
  profile. Hidden replay can return aggregate coverage and evidence-count
  feedback for iterative revisions, but never retrieval ranks, a recommended
  method, an exact `top_k`, support timestamps, IDs, or content.
- `oracle-profile`: additionally receives an abstract gold-support profile
  containing modality, author, source type, time span, support count, and
  content-free retrieval ranks, but no support content or identifiers. It also
  provides verified action advice: the minimum `top_k` needed to cover every
  support record, followed by `READ` without a smaller trailing `TOPK`. This
  mode is intended strictly as an upper-bound ablation. It should not be the
  default source of SFT targets because the teacher can copy the recommended
  method and exact retrieval depth.

## Modules

```text
models.py               memory, action, evidence, rollout, and SFT records
schema.py               strict action-space validation
retrieval.py            current-pool BM25, dense, and hybrid ranking
executor.py             FILTER/SORT/TOPK/RETRIEVE/READ/INSPECT_RAW/STOP
clients.py              student, teacher, answer, VLM inspector, and judge
training.py             on-policy rollout and hindsight SFT generation
sft.py                  generic Hugging Face causal-LM fine-tuning
memgallery.py           Mem-Gallery to hidden-memory conversion
memgallery_pipeline.py  benchmark and SFT-data runner
```

## Retrieval semantics

`RETRIEVE(top_k=k)` ranks unified dialogue turns rather than independent text
and image rows. Once a turn is selected, its conversation text and all image
records are returned together. `TOPK` uses the same turn-level semantics.

The default hybrid retriever combines:

- BM25 over memory text and image captions
- MiniLM dense similarity over full-query and clause-level query views
- SigLIP similarity between the question image (or query text) and memory images
- decayed score propagation to adjacent turns in the same session

This keeps modality pointers attached to their textual context and supports
queries whose visual meaning is carried by an attached image rather than by
words such as "the content in the picture".

## Mem-Gallery smoke run

```bash
omnimem benchmark memgallery opd \
  --scenario Academic_Animal_Pet_Research_Life \
  --max-sessions 2 \
  --max-questions 5 \
  --mode collect-sft \
  --teacher-privilege oracle-profile \
  --sft-quality-filter support-verified
```

For datasets without support annotations, use `diagnostic` privilege and the
`valid` filter. Use `oracle-profile` plus `support-verified` only to measure the
retrieval upper bound. Use `oracle-feedback` to study less leaky supervision;
its candidate trajectories are replayed and selected by support coverage first
and evidence count second. The original LLM result remains in
`teacher_model_policy`, and all candidate actions and replay diagnostics are
stored in `teacher_candidate_diagnostics`.

Use retrieval-only evaluation to skip unrelated answer and judge calls:

```bash
omnimem benchmark memgallery opd \
  --all-scenarios \
  --max-scenarios 5 \
  --max-questions 20 \
  --teacher-privilege oracle-profile \
  --teacher-recall-only
```

`predictions.jsonl` contains student trajectories, execution traces, answers,
teacher corrections, and metrics. `sft_data.jsonl` contains the distilled
`input` and `target` pairs.

Raw inspection is opt-in:

```bash
omnimem benchmark memgallery opd --raw-inspection
```

## SFT

```bash
omnimem-opd-sft \
  --model /path/to/student-model \
  --data /path/to/sft_data.jsonl \
  --output-dir /path/to/output
```

Prompt tokens are masked from the loss; only the corrected JSON trajectory is
trained with standard causal cross-entropy.
