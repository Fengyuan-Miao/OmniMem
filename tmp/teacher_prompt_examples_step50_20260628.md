# Teacher Prompt Examples from step50 training data

Source: `/home/miaofy/SimpleMem/OmniMem/runs/opd_online_train/20260627_061046_opd_stream/online_examples.jsonl`

## Example 1

### Metadata

```json
{
  "sample_id": "conv-26:70",
  "state_index": 2,
  "teacher_decision_index": 1,
  "teacher_action_source": "planner",
  "teacher_answer_correct": true,
  "teacher_answer_score": 1.0,
  "trajectory_action_count": 9,
  "trajectory_evidence_count": 26,
  "sample_weight": 0.05555555555555555
}
```

### Privileged Context

```json
{
  "validated_outcome": "This decision belongs to a trajectory that produced a correct final answer.",
  "trajectory_step": 1
}
```

### Completion Target

```json
[
  {
    "tool": "READ",
    "fields": [
      "summary",
      "content",
      "timestamp",
      "session_date",
      "turn_id",
      "author",
      "modality",
      "source_type",
      "raw_pointer"
    ]
  }
]
```

### Teacher Prompt

```text
Return a memory-tool policy JSON.

State:
{"q": "What personality traits might Melanie say Caroline has?", "history": [{"tool": "RETRIEVE", "method": "hybrid", "top_k": 5, "scope": "all"}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}], "obs": {"pool": {"records": 5, "turns": 5, "score_min": 0.6578, "score_max": 0.9022}, "candidates": [{"rank": 1, "score": 0.9022, "time": "8:56 pm on 20 July, 2023", "modalities": ["text"], "summary": "Hey Melanie! Just wanted to say hi!"}, {"rank": 2, "score": 0.812, "time": "8:56 pm on 20 July, 2023", "modalities": ["text"], "summary": "Hey Caroline! Good to talk to you again. What's up? Anything new since last t..."}], "evidence": {"count": 5, "new": 0, "items": [{"source": "READ", "fields": "{'content': \"Melanie: Wow, Caro, that painting is amazing! You've made so much progress. I'm super proud of you for b..."}, {"source": "READ", "fields": "{'content': \"Melanie: Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?\", 'summary': \"Wow..."}]}, "last_retrieval": {"method": "hybrid", "top_k": 5, "query": "What personality traits might Melanie say Caroline has?", "scope": "all"}, "stopped": false, "last_error": "", "has_question_image": false}, "fb": {"answerable": false, "relevance": "low", "completeness": "low", "continue_required": true}}

Tools:
- RETRIEVE(method=bm25|dense|vision|hybrid, top_k, query?, scope?): search memory and replace/extend the candidate pool; it does NOT add answer evidence
- READ(fields): read text and metadata from the current candidate pool into answer evidence
- EXPAND_NEIGHBORS(window): add nearby turns around current candidates to recover surrounding context; it does NOT read them
- FILTER(field, op, value): narrow the current candidate pool by metadata; it does NOT add evidence
- SORT(field, order): reorder the current candidate pool; it does NOT add evidence
- TOPK(k): keep only the first k current candidates; it does NOT add evidence
- INSPECT_RAW(current_pool): inspect raw images in the current pool and add query-relevant visual observations as evidence
- STOP(): finish only when the accumulated evidence is sufficient to answer; STOP does not retrieve or read anything

Constraints:
- 1 candidates; 1-3 actions each.
- Pool-changing actions before READ/INSPECT_RAW.
- Analyze the query before choosing RETRIEVE.method.
- Use bm25 for exact names, IDs, dates, quoted phrases, or distinctive words.
- Use dense for semantic text memory when wording may differ.
- Use vision for SigLIP visual search over memory images. Prefer it when
  obs.has_question_image is true, or when the query asks what is visible,
  which image matches, visual similarity, object identity, color, layout, or
  fine-grained visual attributes.
- Use hybrid when both text/caption clues and visual evidence are useful, or
  when unsure which route should dominate.
- RETRIEVE alone cannot support an answer. Usually follow a useful retrieval with READ in the same chunk. If the missing evidence is visual, use INSPECT_RAW; if the missing evidence may be in nearby dialogue turns, use EXPAND_NEIGHBORS then READ.
- If candidates exist but evidence is empty, read or inspect them instead of
  repeating the same retrieval.
- If the candidate pool changed but evidence.new is 0, the new candidates have
  not been used as evidence yet. Prefer READ; if surrounding dialogue is needed,
  use EXPAND_NEIGHBORS then READ.
- Do not STOP while relevant candidates remain unread or uninspected.
- No memory IDs or file paths.
- If fb says visual/image gap, use INSPECT_RAW.
- If fb says text/context/date/person relation is missing after a READ,
  consider EXPAND_NEIGHBORS, FILTER, SORT, or a different retrieval route
  instead of another similar RETRIEVE.

Answer-model feedback:
- fb is a privileged assessment of whether the current evidence can support an
  answer. Use it as the main diagnosis, but do not copy hidden answer content.
- If fb.continue_required is true, do not STOP. Choose the next action that
  most directly addresses fb.failure_diagnostic.evidence_gap.
- If fb.failure_diagnostic has recommended_tool, make at least one candidate
  follow that tool unless the observation makes it impossible.
- If recommended_tool is READ, read the current candidate pool instead of
  retrieving again. If it is INSPECT_RAW, inspect current visual candidates. If
  it is RETRIEVE, change the retrieval method, query focus, scope, or top_k so
  the attempt is not a repeat of last_retrieval. If it is EXPAND_NEIGHBORS,
  recover surrounding turns before reading. If it names STOP, stop only when
  fb says the evidence is answerable.
- If fb.failure_diagnostic.failure_type is unread_candidate_pool, use READ
  now; do not retrieve again before reading the current pool.
- Respect avoid_action as a warning about a repeated or unhelpful move.
- Treat repeated failure as a signal to switch tools. For example, after a
  failed semantic search try bm25/vision/hybrid with a focused query; after
  reading a plausible memory but missing chronology, expand neighbors or sort
  by time; after seeing image candidates but lacking visual confirmation,
  inspect raw images.

Planning guidance:
- evidence.new is the evidence added by the most recent action chunk.
  last_retrieval is the exact retrieval that actually ran.
- When evidence.new is 0, the last attempt made no progress. Repeating the
  same retrieval is likely to return the same pool, so reconsider the search
  rather than merely trying it again.
- Use the observed evidence gap to make the next attempt meaningfully
  different. Depending on the state, this may mean expressing the missing
  concept in a focused query, changing retrieval breadth or method, exploring
  neighboring turns, or inspecting relevant raw images.
- In chunk mode, prefer a compact repair plan over a single habitual action:
  retrieve+read for a new pool, expand+read for surrounding context,
  filter/sort+read for narrowing or temporal questions, inspect_raw for
  visual verification. Keep the chunk short and purposeful.
- Use FILTER when the current pool already contains plausible candidates but
  must be narrowed by author/person, modality, source_type, date/session, or a
  distinctive text value; usually follow FILTER with READ.
- Use SORT for before/after, latest/earliest, timeline, first/last, or other
  chronology questions; sort by timestamp/turn_id, optionally TOPK, then READ.
- Use TOPK after FILTER or SORT when the pool is broad and only the strongest
  or temporally relevant candidates should be read.
- Candidate metadata and executable actions must agree. If next_tool says a
  more targeted search or broader context is needed, the action parameters
  should visibly implement that change.
- Each candidate should include a short diagnosis explaining which feedback
  gap it addresses. Prefer candidates with different tools or retrieval
  methods when multiple plausible repairs exist.
- Prefer a deliberate step whose effect can be judged from the next
  observation.

Final JSON shape (format example only; choose the method from query analysis):
{"candidates":[{"diagnosis":"why this chunk addresses the current feedback","next_tool":"RETRIEVE","expected_gain":"find and read a focused candidate pool","actions":[{"tool":"RETRIEVE","method":"vision","top_k":5,"scope":"all"},{"tool":"READ","fields":["summary","content","timestamp","session_date","turn_id","author","modality","source_type","raw_pointer"]}]}]}
```

## Example 2

### Metadata

```json
{
  "sample_id": "conv-26:70",
  "state_index": 1,
  "teacher_decision_index": 1,
  "teacher_action_source": "answer_stop",
  "teacher_answer_correct": true,
  "teacher_answer_score": 1.0,
  "trajectory_action_count": 5,
  "trajectory_evidence_count": 26,
  "sample_weight": 0.05555555555555555
}
```

### Privileged Context

```json
{
  "validated_outcome": "This decision belongs to a trajectory that produced a correct final answer.",
  "trajectory_step": 1
}
```

### Completion Target

```json
[
  {
    "tool": "STOP"
  }
]
```

### Teacher Prompt

```text
Return a memory-tool policy JSON.

State:
{"q": "What personality traits might Melanie say Caroline has?", "history": [{"tool": "RETRIEVE", "method": "hybrid", "top_k": 5, "scope": "all"}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}, {"tool": "EXPAND_NEIGHBORS", "window": 3}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}], "obs": {"pool": {"records": 26, "turns": 26, "score_min": 0.0, "score_max": 0.9022}, "candidates": [{"rank": 1, "score": 0.0, "time": "1:56 pm on 8 May, 2023", "modalities": ["text"], "summary": "The support group has made me feel accepted and given me courage to embrace m..."}, {"rank": 2, "score": 0.0, "time": "1:56 pm on 8 May, 2023", "modalities": ["text"], "summary": "That's really cool. You've got guts. What now?"}], "evidence": {"count": 26, "new": 21, "items": [{"source": "READ", "fields": "{'content': 'Melanie: That sounds awesome! What did you take away from it to use in your life?', 'summary': 'That sou..."}, {"source": "READ", "fields": "{'content': \"Caroline: It taught me self-acceptance and how to find support. It also showed me that tough times don't..."}]}, "last_retrieval": {"method": "hybrid", "top_k": 5, "query": "What personality traits might Melanie say Caroline has?", "scope": "all"}, "stopped": false, "last_error": "", "has_question_image": false}, "fb": {"answerable": true, "relevance": "high", "completeness": "sufficient", "continue_required": false}}

Tools:
- RETRIEVE(method=bm25|dense|vision|hybrid, top_k, query?, scope?): search memory and replace/extend the candidate pool; it does NOT add answer evidence
- READ(fields): read text and metadata from the current candidate pool into answer evidence
- EXPAND_NEIGHBORS(window): add nearby turns around current candidates to recover surrounding context; it does NOT read them
- FILTER(field, op, value): narrow the current candidate pool by metadata; it does NOT add evidence
- SORT(field, order): reorder the current candidate pool; it does NOT add evidence
- TOPK(k): keep only the first k current candidates; it does NOT add evidence
- INSPECT_RAW(current_pool): inspect raw images in the current pool and add query-relevant visual observations as evidence
- STOP(): finish only when the accumulated evidence is sufficient to answer; STOP does not retrieve or read anything

Constraints:
- 1 candidates; 1-3 actions each.
- Pool-changing actions before READ/INSPECT_RAW.
- Analyze the query before choosing RETRIEVE.method.
- Use bm25 for exact names, IDs, dates, quoted phrases, or distinctive words.
- Use dense for semantic text memory when wording may differ.
- Use vision for SigLIP visual search over memory images. Prefer it when
  obs.has_question_image is true, or when the query asks what is visible,
  which image matches, visual similarity, object identity, color, layout, or
  fine-grained visual attributes.
- Use hybrid when both text/caption clues and visual evidence are useful, or
  when unsure which route should dominate.
- RETRIEVE alone cannot support an answer. Usually follow a useful retrieval with READ in the same chunk. If the missing evidence is visual, use INSPECT_RAW; if the missing evidence may be in nearby dialogue turns, use EXPAND_NEIGHBORS then READ.
- If candidates exist but evidence is empty, read or inspect them instead of
  repeating the same retrieval.
- If the candidate pool changed but evidence.new is 0, the new candidates have
  not been used as evidence yet. Prefer READ; if surrounding dialogue is needed,
  use EXPAND_NEIGHBORS then READ.
- Do not STOP while relevant candidates remain unread or uninspected.
- No memory IDs or file paths.
- If fb says visual/image gap, use INSPECT_RAW.
- If fb says text/context/date/person relation is missing after a READ,
  consider EXPAND_NEIGHBORS, FILTER, SORT, or a different retrieval route
  instead of another similar RETRIEVE.

Answer-model feedback:
- fb is a privileged assessment of whether the current evidence can support an
  answer. Use it as the main diagnosis, but do not copy hidden answer content.
- If fb.continue_required is true, do not STOP. Choose the next action that
  most directly addresses fb.failure_diagnostic.evidence_gap.
- If fb.failure_diagnostic has recommended_tool, make at least one candidate
  follow that tool unless the observation makes it impossible.
- If recommended_tool is READ, read the current candidate pool instead of
  retrieving again. If it is INSPECT_RAW, inspect current visual candidates. If
  it is RETRIEVE, change the retrieval method, query focus, scope, or top_k so
  the attempt is not a repeat of last_retrieval. If it is EXPAND_NEIGHBORS,
  recover surrounding turns before reading. If it names STOP, stop only when
  fb says the evidence is answerable.
- If fb.failure_diagnostic.failure_type is unread_candidate_pool, use READ
  now; do not retrieve again before reading the current pool.
- Respect avoid_action as a warning about a repeated or unhelpful move.
- Treat repeated failure as a signal to switch tools. For example, after a
  failed semantic search try bm25/vision/hybrid with a focused query; after
  reading a plausible memory but missing chronology, expand neighbors or sort
  by time; after seeing image candidates but lacking visual confirmation,
  inspect raw images.

Planning guidance:
- evidence.new is the evidence added by the most recent action chunk.
  last_retrieval is the exact retrieval that actually ran.
- When evidence.new is 0, the last attempt made no progress. Repeating the
  same retrieval is likely to return the same pool, so reconsider the search
  rather than merely trying it again.
- Use the observed evidence gap to make the next attempt meaningfully
  different. Depending on the state, this may mean expressing the missing
  concept in a focused query, changing retrieval breadth or method, exploring
  neighboring turns, or inspecting relevant raw images.
- In chunk mode, prefer a compact repair plan over a single habitual action:
  retrieve+read for a new pool, expand+read for surrounding context,
  filter/sort+read for narrowing or temporal questions, inspect_raw for
  visual verification. Keep the chunk short and purposeful.
- Use FILTER when the current pool already contains plausible candidates but
  must be narrowed by author/person, modality, source_type, date/session, or a
  distinctive text value; usually follow FILTER with READ.
- Use SORT for before/after, latest/earliest, timeline, first/last, or other
  chronology questions; sort by timestamp/turn_id, optionally TOPK, then READ.
- Use TOPK after FILTER or SORT when the pool is broad and only the strongest
  or temporally relevant candidates should be read.
- Candidate metadata and executable actions must agree. If next_tool says a
  more targeted search or broader context is needed, the action parameters
  should visibly implement that change.
- Each candidate should include a short diagnosis explaining which feedback
  gap it addresses. Prefer candidates with different tools or retrieval
  methods when multiple plausible repairs exist.
- Prefer a deliberate step whose effect can be judged from the next
  observation.

Final JSON shape (format example only; choose the method from query analysis):
{"candidates":[{"diagnosis":"why this chunk addresses the current feedback","next_tool":"RETRIEVE","expected_gain":"find and read a focused candidate pool","actions":[{"tool":"RETRIEVE","method":"vision","top_k":5,"scope":"all"},{"tool":"READ","fields":["summary","content","timestamp","session_date","turn_id","author","modality","source_type","raw_pointer"]}]}]}
```

## Example 3

### Metadata

```json
{
  "sample_id": "conv-26:70",
  "state_index": 2,
  "teacher_decision_index": 3,
  "teacher_action_source": "planner",
  "teacher_answer_correct": true,
  "teacher_answer_score": 1.0,
  "trajectory_action_count": 9,
  "trajectory_evidence_count": 26,
  "sample_weight": 0.05555555555555555
}
```

### Privileged Context

```json
{
  "validated_outcome": "This decision belongs to a trajectory that produced a correct final answer.",
  "trajectory_step": 3
}
```

### Completion Target

```json
[
  {
    "tool": "EXPAND_NEIGHBORS",
    "window": 3
  },
  {
    "tool": "READ",
    "fields": [
      "summary",
      "content",
      "timestamp",
      "session_date",
      "turn_id",
      "author",
      "modality",
      "source_type",
      "raw_pointer"
    ]
  }
]
```

### Teacher Prompt

```text
Return a memory-tool policy JSON.

State:
{"q": "What personality traits might Melanie say Caroline has?", "history": [{"tool": "RETRIEVE", "method": "hybrid", "top_k": 5, "scope": "all"}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}, {"tool": "READ", "fields": ["summary", "content", "timestamp", "session_date", "turn_id", "author", "modality", "source_type", "raw_pointer"]}], "obs": {"pool": {"records": 5, "turns": 5, "score_min": 0.6578, "score_max": 0.9022}, "candidates": [{"rank": 1, "score": 0.9022, "time": "8:56 pm on 20 July, 2023", "modalities": ["text"], "summary": "Hey Melanie! Just wanted to say hi!"}, {"rank": 2, "score": 0.812, "time": "8:56 pm on 20 July, 2023", "modalities": ["text"], "summary": "Hey Caroline! Good to talk to you again. What's up? Anything new since last t..."}], "evidence": {"count": 5, "new": 0, "items": [{"source": "READ", "fields": "{'content': \"Melanie: Wow, Caro, that painting is amazing! You've made so much progress. I'm super proud of you for b..."}, {"source": "READ", "fields": "{'content': \"Melanie: Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?\", 'summary': \"Wow..."}]}, "last_retrieval": {"method": "hybrid", "top_k": 5, "query": "What personality traits might Melanie say Caroline has?", "scope": "all"}, "stopped": false, "last_error": "", "has_question_image": false}, "fb": {"answerable": false, "relevance": "low", "completeness": "low", "continue_required": true}}

Tools:
- RETRIEVE(method=bm25|dense|vision|hybrid, top_k, query?, scope?): search memory and replace/extend the candidate pool; it does NOT add answer evidence
- READ(fields): read text and metadata from the current candidate pool into answer evidence
- EXPAND_NEIGHBORS(window): add nearby turns around current candidates to recover surrounding context; it does NOT read them
- FILTER(field, op, value): narrow the current candidate pool by metadata; it does NOT add evidence
- SORT(field, order): reorder the current candidate pool; it does NOT add evidence
- TOPK(k): keep only the first k current candidates; it does NOT add evidence
- INSPECT_RAW(current_pool): inspect raw images in the current pool and add query-relevant visual observations as evidence
- STOP(): finish only when the accumulated evidence is sufficient to answer; STOP does not retrieve or read anything

Constraints:
- 1 candidates; 1-3 actions each.
- Pool-changing actions before READ/INSPECT_RAW.
- Analyze the query before choosing RETRIEVE.method.
- Use bm25 for exact names, IDs, dates, quoted phrases, or distinctive words.
- Use dense for semantic text memory when wording may differ.
- Use vision for SigLIP visual search over memory images. Prefer it when
  obs.has_question_image is true, or when the query asks what is visible,
  which image matches, visual similarity, object identity, color, layout, or
  fine-grained visual attributes.
- Use hybrid when both text/caption clues and visual evidence are useful, or
  when unsure which route should dominate.
- RETRIEVE alone cannot support an answer. Usually follow a useful retrieval with READ in the same chunk. If the missing evidence is visual, use INSPECT_RAW; if the missing evidence may be in nearby dialogue turns, use EXPAND_NEIGHBORS then READ.
- If candidates exist but evidence is empty, read or inspect them instead of
  repeating the same retrieval.
- If the candidate pool changed but evidence.new is 0, the new candidates have
  not been used as evidence yet. Prefer READ; if surrounding dialogue is needed,
  use EXPAND_NEIGHBORS then READ.
- Do not STOP while relevant candidates remain unread or uninspected.
- No memory IDs or file paths.
- If fb says visual/image gap, use INSPECT_RAW.
- If fb says text/context/date/person relation is missing after a READ,
  consider EXPAND_NEIGHBORS, FILTER, SORT, or a different retrieval route
  instead of another similar RETRIEVE.

Answer-model feedback:
- fb is a privileged assessment of whether the current evidence can support an
  answer. Use it as the main diagnosis, but do not copy hidden answer content.
- If fb.continue_required is true, do not STOP. Choose the next action that
  most directly addresses fb.failure_diagnostic.evidence_gap.
- If fb.failure_diagnostic has recommended_tool, make at least one candidate
  follow that tool unless the observation makes it impossible.
- If recommended_tool is READ, read the current candidate pool instead of
  retrieving again. If it is INSPECT_RAW, inspect current visual candidates. If
  it is RETRIEVE, change the retrieval method, query focus, scope, or top_k so
  the attempt is not a repeat of last_retrieval. If it is EXPAND_NEIGHBORS,
  recover surrounding turns before reading. If it names STOP, stop only when
  fb says the evidence is answerable.
- If fb.failure_diagnostic.failure_type is unread_candidate_pool, use READ
  now; do not retrieve again before reading the current pool.
- Respect avoid_action as a warning about a repeated or unhelpful move.
- Treat repeated failure as a signal to switch tools. For example, after a
  failed semantic search try bm25/vision/hybrid with a focused query; after
  reading a plausible memory but missing chronology, expand neighbors or sort
  by time; after seeing image candidates but lacking visual confirmation,
  inspect raw images.

Planning guidance:
- evidence.new is the evidence added by the most recent action chunk.
  last_retrieval is the exact retrieval that actually ran.
- When evidence.new is 0, the last attempt made no progress. Repeating the
  same retrieval is likely to return the same pool, so reconsider the search
  rather than merely trying it again.
- Use the observed evidence gap to make the next attempt meaningfully
  different. Depending on the state, this may mean expressing the missing
  concept in a focused query, changing retrieval breadth or method, exploring
  neighboring turns, or inspecting relevant raw images.
- In chunk mode, prefer a compact repair plan over a single habitual action:
  retrieve+read for a new pool, expand+read for surrounding context,
  filter/sort+read for narrowing or temporal questions, inspect_raw for
  visual verification. Keep the chunk short and purposeful.
- Use FILTER when the current pool already contains plausible candidates but
  must be narrowed by author/person, modality, source_type, date/session, or a
  distinctive text value; usually follow FILTER with READ.
- Use SORT for before/after, latest/earliest, timeline, first/last, or other
  chronology questions; sort by timestamp/turn_id, optionally TOPK, then READ.
- Use TOPK after FILTER or SORT when the pool is broad and only the strongest
  or temporally relevant candidates should be read.
- Candidate metadata and executable actions must agree. If next_tool says a
  more targeted search or broader context is needed, the action parameters
  should visibly implement that change.
- Each candidate should include a short diagnosis explaining which feedback
  gap it addresses. Prefer candidates with different tools or retrieval
  methods when multiple plausible repairs exist.
- Prefer a deliberate step whose effect can be judged from the next
  observation.

Final JSON shape (format example only; choose the method from query analysis):
{"candidates":[{"diagnosis":"why this chunk addresses the current feedback","next_tool":"RETRIEVE","expected_gain":"find and read a focused candidate pool","actions":[{"tool":"RETRIEVE","method":"vision","top_k":5,"scope":"all"},{"tool":"READ","fields":["summary","content","timestamp","session_date","turn_id","author","modality","source_type","raw_pointer"]}]}]}
```

## Example 4

### Metadata

```json
{
  "sample_id": "AI_Robotics_Automation_Future_Tech:50",
  "state_index": 1,
  "teacher_decision_index": 0,
  "teacher_action_source": "planner_repaired",
  "teacher_answer_correct": true,
  "teacher_answer_score": 1.0,
  "trajectory_action_count": 6,
  "trajectory_evidence_count": 9,
  "sample_weight": 0.08333333333333333
}
```

### Privileged Context

```json
{
  "validated_outcome": "This decision belongs to a trajectory that produced a correct final answer.",
  "trajectory_step": 0
}
```

### Completion Target

```json
[
  {
    "tool": "INSPECT_RAW"
  }
]
```

### Teacher Prompt

```text
Return a memory-tool policy JSON.

State:
{"q": "Which chatbot mentioned in the conversation and the product in the picture are from the same company?", "history": [{"tool": "RETRIEVE", "method": "bm25", "top_k": 5, "scope": "all"}, {"tool": "RETRIEVE", "method": "dense", "top_k": 5, "scope": "all"}], "obs": {"pool": {"records": 6, "turns": 5, "score_min": 0.3963, "score_max": 0.4403}, "candidates": [{"rank": 1, "score": 0.4403, "time": "2024-11-24T00:23:10Z", "modalities": ["text"], "summary": "What about for customer service? Which AI would be a better fit?"}, {"rank": 2, "score": 0.4099, "time": "2024-08-21T00:12:40Z", "modalities": ["image", "text"], "summary": "Some AI systems are even capable of providing personalized therapy sessions,..."}], "evidence": {"count": 0, "new": 0, "items": []}, "last_retrieval": {"method": "dense", "top_k": 5, "query": "Which chatbot mentioned in the conversation and the product in the picture are from the same company?", "scope": "all"}, "stopped": false, "last_error": "", "has_question_image": true}}

Tools:
- RETRIEVE(method=bm25|dense|vision|hybrid, top_k, query?, scope?): search memory and replace/extend the candidate pool; it does NOT add answer evidence
- READ(fields): read text and metadata from the current candidate pool into answer evidence
- EXPAND_NEIGHBORS(window): add nearby turns around current candidates to recover surrounding context; it does NOT read them
- FILTER(field, op, value): narrow the current candidate pool by metadata; it does NOT add evidence
- SORT(field, order): reorder the current candidate pool; it does NOT add evidence
- TOPK(k): keep only the first k current candidates; it does NOT add evidence
- INSPECT_RAW(current_pool): inspect raw images in the current pool and add query-relevant visual observations as evidence
- STOP(): finish only when the accumulated evidence is sufficient to answer; STOP does not retrieve or read anything

Constraints:
- 1 candidates; 1-3 actions each.
- Pool-changing actions before READ/INSPECT_RAW.
- Analyze the query before choosing RETRIEVE.method.
- Use bm25 for exact names, IDs, dates, quoted phrases, or distinctive words.
- Use dense for semantic text memory when wording may differ.
- Use vision for SigLIP visual search over memory images. Prefer it when
  obs.has_question_image is true, or when the query asks what is visible,
  which image matches, visual similarity, object identity, color, layout, or
  fine-grained visual attributes.
- Use hybrid when both text/caption clues and visual evidence are useful, or
  when unsure which route should dominate.
- RETRIEVE alone cannot support an answer. Usually follow a useful retrieval with READ in the same chunk. If the missing evidence is visual, use INSPECT_RAW; if the missing evidence may be in nearby dialogue turns, use EXPAND_NEIGHBORS then READ.
- If candidates exist but evidence is empty, read or inspect them instead of
  repeating the same retrieval.
- If the candidate pool changed but evidence.new is 0, the new candidates have
  not been used as evidence yet. Prefer READ; if surrounding dialogue is needed,
  use EXPAND_NEIGHBORS then READ.
- Do not STOP while relevant candidates remain unread or uninspected.
- No memory IDs or file paths.
- If fb says visual/image gap, use INSPECT_RAW.
- If fb says text/context/date/person relation is missing after a READ,
  consider EXPAND_NEIGHBORS, FILTER, SORT, or a different retrieval route
  instead of another similar RETRIEVE.

Answer-model feedback:
- fb is a privileged assessment of whether the current evidence can support an
  answer. Use it as the main diagnosis, but do not copy hidden answer content.
- If fb.continue_required is true, do not STOP. Choose the next action that
  most directly addresses fb.failure_diagnostic.evidence_gap.
- If fb.failure_diagnostic has recommended_tool, make at least one candidate
  follow that tool unless the observation makes it impossible.
- If recommended_tool is READ, read the current candidate pool instead of
  retrieving again. If it is INSPECT_RAW, inspect current visual candidates. If
  it is RETRIEVE, change the retrieval method, query focus, scope, or top_k so
  the attempt is not a repeat of last_retrieval. If it is EXPAND_NEIGHBORS,
  recover surrounding turns before reading. If it names STOP, stop only when
  fb says the evidence is answerable.
- If fb.failure_diagnostic.failure_type is unread_candidate_pool, use READ
  now; do not retrieve again before reading the current pool.
- Respect avoid_action as a warning about a repeated or unhelpful move.
- Treat repeated failure as a signal to switch tools. For example, after a
  failed semantic search try bm25/vision/hybrid with a focused query; after
  reading a plausible memory but missing chronology, expand neighbors or sort
  by time; after seeing image candidates but lacking visual confirmation,
  inspect raw images.

Planning guidance:
- evidence.new is the evidence added by the most recent action chunk.
  last_retrieval is the exact retrieval that actually ran.
- When evidence.new is 0, the last attempt made no progress. Repeating the
  same retrieval is likely to return the same pool, so reconsider the search
  rather than merely trying it again.
- Use the observed evidence gap to make the next attempt meaningfully
  different. Depending on the state, this may mean expressing the missing
  concept in a focused query, changing retrieval breadth or method, exploring
  neighboring turns, or inspecting relevant raw images.
- In chunk mode, prefer a compact repair plan over a single habitual action:
  retrieve+read for a new pool, expand+read for surrounding context,
  filter/sort+read for narrowing or temporal questions, inspect_raw for
  visual verification. Keep the chunk short and purposeful.
- Use FILTER when the current pool already contains plausible candidates but
  must be narrowed by author/person, modality, source_type, date/session, or a
  distinctive text value; usually follow FILTER with READ.
- Use SORT for before/after, latest/earliest, timeline, first/last, or other
  chronology questions; sort by timestamp/turn_id, optionally TOPK, then READ.
- Use TOPK after FILTER or SORT when the pool is broad and only the strongest
  or temporally relevant candidates should be read.
- Candidate metadata and executable actions must agree. If next_tool says a
  more targeted search or broader context is needed, the action parameters
  should visibly implement that change.
- Each candidate should include a short diagnosis explaining which feedback
  gap it addresses. Prefer candidates with different tools or retrieval
  methods when multiple plausible repairs exist.
- Prefer a deliberate step whose effect can be judged from the next
  observation.

Final JSON shape (format example only; choose the method from query analysis):
{"candidates":[{"diagnosis":"why this chunk addresses the current feedback","next_tool":"RETRIEVE","expected_gain":"find and read a focused candidate pool","actions":[{"tool":"RETRIEVE","method":"vision","top_k":5,"scope":"all"},{"tool":"READ","fields":["summary","content","timestamp","session_date","turn_id","author","modality","source_type","raw_pointer"]}]}]}
```

