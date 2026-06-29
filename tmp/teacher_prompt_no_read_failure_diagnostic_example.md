# Current Teacher Prompt Example

```text
Return a memory-tool policy JSON.

State:
{"q": "Which project did Bob select?", "history": [], "obs": {"pool": {"records": 1, "turns": 1, "score_min": 0.0, "score_max": 0.0}, "candidates": [], "evidence": {"count": 0, "new": 0, "items": []}, "last_retrieval": {}, "stopped": false, "last_error": "", "has_question_image": false, "has_visual_candidates": false}, "fb": {"answerable": false, "relevance": "low", "completeness": "low", "continue_required": true, "failure_diagnostic": {"failure_type": "no_evidence", "evidence_gap": "No evidence pool has been retrieved for this query yet.", "recommended_change": "Choose an initial retrieval method based on the query: bm25 for names/dates/exact terms, dense for paraphrased text memory, vision for visual matching, or hybrid when mixed.", "recommended_tool": "RETRIEVE", "recommended_retrieval_method": "hybrid", "needs_text_evidence": true, "needs_visual_evidence": false, "needs_neighbor_context": false, "avoid_action": "STOP"}}}

Tools:
- RETRIEVE(method=bm25|dense|vision|hybrid, top_k, query?, scope?): search memory and replace/extend the current evidence pool
- EXPAND_NEIGHBORS(window): add nearby turns around current evidence to recover surrounding context
- FILTER(field, op, value): narrow the current evidence pool by metadata
- SORT(field, order): reorder the current evidence pool
- TOPK(k): keep only the first k current evidence turns
- INSPECT_RAW(current_pool): inspect raw images in the current pool and add query-relevant visual observations as evidence
- STOP(): finish only when the accumulated evidence is sufficient to answer; STOP does not retrieve anything

Constraints:
- 1 candidates; 1-3 actions each.
- Pool-changing actions before INSPECT_RAW.
- Analyze the query before choosing RETRIEVE.method.
- Use bm25 for exact names, IDs, dates, quoted phrases, or distinctive words.
- Use dense for semantic text memory when wording may differ.
- Use vision for SigLIP visual search over memory images. Prefer it when
  obs.has_question_image is true, or when the query asks what is visible,
  which image matches, visual similarity, object identity, color, layout, or
  fine-grained visual attributes.
- Use hybrid when both text/caption clues and visual evidence are useful, or
  when unsure which route should dominate.
- RETRIEVE directly replaces the current evidence pool. If the new pool is broad, pair retrieval with FILTER/SORT/TOPK. If surrounding dialogue may contain missing context, use EXPAND_NEIGHBORS. If visual details are missing, use INSPECT_RAW.
- If evidence exists but is off-topic, change retrieval method/query/scope
  rather than stopping.
- If surrounding dialogue is needed, use EXPAND_NEIGHBORS.
- Do not STOP while the current evidence is insufficient or only visual/textual
  half of the problem is covered.
- No memory IDs or file paths.
- If fb says visual/image gap, use INSPECT_RAW.
- If fb says text/context/date/person relation is missing,
  consider EXPAND_NEIGHBORS, FILTER, SORT, or a different retrieval route
  instead of another similar RETRIEVE.

Answer-model feedback:
- fb is a privileged assessment of whether the current evidence can support an
  answer. Use it as the main diagnosis, but do not copy hidden answer content.
- If fb.continue_required is true, do not STOP. Choose the next action that
  most directly addresses fb.failure_diagnostic.evidence_gap.
- If fb.failure_diagnostic has recommended_tool, make at least one candidate
  follow that tool unless the observation makes it impossible.
- If recommended_tool is INSPECT_RAW, inspect current visual candidates. If it
  is RETRIEVE, change the retrieval method, query focus, scope, or top_k so the
  attempt is not a repeat of last_retrieval. If it is EXPAND_NEIGHBORS, recover
  surrounding turns. If it names FILTER/SORT/TOPK, narrow or reorder the current
  evidence pool. If it names STOP, stop only when fb says the evidence is
  answerable.
- If old feedback names READ, treat it as a request to use the current evidence:
  if it is insufficient, retrieve/filter/sort/expand/inspect instead.
- Respect avoid_action as a warning about a repeated or unhelpful move.
- Treat repeated failure as a signal to switch tools. For example, after a
  failed semantic search try bm25/vision/hybrid with a focused query; after
  finding a plausible memory but missing chronology, expand neighbors or sort
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
  retrieve for a new pool, expand for surrounding context, filter/sort/topk for
  narrowing or temporal questions, inspect_raw for visual verification. Keep
  the chunk short and purposeful.
- Use FILTER when the current pool already contains plausible candidates but
  must be narrowed by author/person, modality, source_type, date/session, or a
  distinctive text value.
- Use SORT for before/after, latest/earliest, timeline, first/last, or other
  chronology questions; sort by timestamp/turn_id, optionally TOPK.
- Use TOPK after FILTER or SORT when the pool is broad and only the strongest
  or temporally relevant candidates should remain as evidence.
- Candidate metadata and executable actions must agree. If next_tool says a
  more targeted search or broader context is needed, the action parameters
  should visibly implement that change.
- Each candidate should include a short diagnosis explaining which feedback
  gap it addresses. Prefer candidates with different tools or retrieval
  methods when multiple plausible repairs exist.
- Prefer a deliberate step whose effect can be judged from the next
  observation.

Final JSON shape (format example only; choose the method from query analysis):
{"candidates":[{"diagnosis":"why this chunk addresses the current feedback","next_tool":"RETRIEVE","expected_gain":"find a focused evidence pool","actions":[{"tool":"RETRIEVE","method":"vision","top_k":5,"scope":"all"},{"tool":"INSPECT_RAW","target":"current_pool","instruction":"answer_query_related_visual_details"}]}]}

```
