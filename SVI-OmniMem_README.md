# SVI-OmniMem: Legacy Compatibility Experiment

> This document describes the earlier SVI design. The standalone OmniMem
> project now uses GME unified-entry memory and SigLIP/MiniLM dual-encoder
> memory as its primary implementations. SVI remains available as an optional
> compatibility experiment and requires the external `omni_memory` package.

> **设计提案 / Implementation-Oriented Research README**  
> **目标代码库：** `OmniSimpleMem`  
> **核心方法：** 结构化视觉索引（Structured Visual Index, SVI）+ 原图证据核验（Raw-Evidence Verification）+ 查询驱动事实写回（Verified Fact Promotion）  
> **设计目标：** 在不引入密集 patch / region 索引开销的前提下，提高长期多模态记忆对实体、属性、OCR、空间关系及简单视觉状态变化问题的检索与回答能力。

---

## 1. Motivation

Omni-SimpleMem 已提供一套高效的多模态长期记忆骨架：

- 通过视觉信息价值触发机制选择性接收图像；
- 使用 MAU（Multimodal Atomic Unit）统一表示多模态记忆；
- 在 hot storage 中保存可检索摘要与元数据；
- 在 cold storage 中保存原始图像；
- 使用 dense retrieval、BM25 与图关系完成候选召回；
- 仅在需要时读取原始证据。

这一设计对 scene-level 的图像事件记忆较有效，但其图像路径仍容易受 **caption-centric indexing** 限制：如果写入阶段只生成一句全局 caption，未来查询所依赖的局部视觉事实可能没有进入可检索表示。

例如，原图包含：

```text
桌上有一台银色笔记本电脑，左侧有一个带白色条纹的红色陶瓷杯，
显示器边缘贴着三张黄色便签，其中一张写着 “meeting 3pm”。
```

而图像摘要仅为：

```text
A desk with a laptop and office items.
```

则未来查询可能无法定位该图像：

```text
杯子是什么颜色？
便签上写了什么？
电脑左侧是什么物体？
最近一次桌面照片里杯子的颜色是什么？
```

虽然原始图像仍保存在 cold storage 中，但如果检索入口没有覆盖相关实体或属性，原图不会被读取，视觉证据等同于不可访问。

### 1.1 Why Not Dense Patch / Region Indexing?

一种直接增强方式是将每张图像拆成多个 patch 或 region，并为每个局部区域生成向量、caption、关系边或独立 MAU。该方案理论上增强局部证据召回能力，但会显著增加：

- 写入阶段视觉编码或 VLM 调用；
- 向量索引规模；
- crop / region 存储量；
- 多路 reranking 和图结构维护复杂度；
- 对大量从未被查询图片的无效预处理。

对于强调轻量化的 Omni-SimpleMem，这种 eager region expansion 会削弱原系统的效率优势。

### 1.2 Research Question

SVI-OmniMem 研究的问题是：

> **在不构建密集 patch / region memory 的前提下，结构化视觉索引与原图按需核验能否以更优的质量—成本权衡，支持长期多模态 agent 对细粒度视觉事实与演化状态的记忆？**

---

## 2. Design Goals and Non-Goals

### 2.1 Goals

SVI-OmniMem 旨在实现：

1. **可搜索的图像细节入口**  
   让对象类别、显著属性、OCR、空间关系和视觉观察状态进入检索索引，而不只依赖全局 caption。

2. **原图作为最终视觉证据**  
   将写入阶段结构化输出视为检索线索；在回答颜色、文字、数量、关系或状态问题前，根据原图重新核验。

3. **较低写入成本**  
   每张图像仅进行一次主要结构化 VLM 抽取，不为所有图像生成大量 patch embedding 或 region graph。

4. **真实使用驱动的高置信记忆积累**  
   只有真实 query 使用并通过原图核验的视觉事实，才写回为 `query_verified` 记忆。

5. **可解释的视觉状态观察记录**  
   维护带时间与证据指针的 observation ledger，支持“之前”“最近一次”“是否观察到变化”等查询。

### 2.2 Non-Goals

首版不计划解决：

- 完整 object-level bounding box 检索；
- 密集局部视觉 recall 的理论上界；
- 复杂对象跨时间 identity tracking；
- 通用视频记忆；
- 为所有图像建立完整空间关系图；
- 依靠结构化抽取结果直接替代原始视觉证据。

---

## 3. Core Idea: Index Is Not Evidence

SVI-OmniMem 的核心原则是：

# **Index is not Evidence**

SVI 的 `StructuredVisualCard` 应被理解为一个 **coarse visual inverted index**，而不是事实库。它的作用接近“视觉倒排索引”：用较小的写入成本暴露实体、属性、OCR、关系和状态入口，让系统能找到可能相关的原图。

结构化视觉卡片承担的是：

- 让未来查询找到可能相关的历史图像；
- 对候选图像进行快速过滤与排序；
- 组织时间观察与检索锚点；
- 为 answer-time verification 提供候选声明。

它**不承担**：

- 证明颜色、OCR、数量、局部关系一定正确；
- 证明单次观察在之后仍然有效；
- 取代原图作为视觉证据。

因此，检索结果必须强制区分三类信息：

| Field | Source | Allowed Use |
|---|---|---|
| `retrieval_claims` | `StructuredVisualCard` / text mirror / inverted fields | 只能解释为什么召回该图像，不能直接作为最终答案证据 |
| `verified_evidence` | 当前 query 中重新读取 raw image 的 verifier 输出 | 可进入 answer prompt，作为本轮回答依据 |
| `promoted_facts` | 历史 query 已验证并写回的事实 | 在无冲突、无更新观察时可复用；仍保留 source provenance |

回答生成阶段不得把 `unverified_extraction` 字段当作已证实事实使用。若没有 `verified_evidence` 或可复用的 `promoted_facts`，系统应返回不确定性或说明证据不足。

整体流程如下：

```text
Ingestion
─────────
Image
  → VisualEntropyTrigger
  → Save raw image in cold storage
  → One-pass VLM structured extraction
  → Image MAU + StructuredVisualCard + searchable mirror

Query
─────
User Query
  → Query requirement parsing
  → Caption dense retrieval
     + Entity / relation / OCR / state matching
     + Optional global visual fallback
  → Top candidate source images
  → Bounded raw-image verification
  → Answer with provenance
  → Promote verified facts for future reuse
```

---

## 4. Positioning Against Related Work

### 4.1 Omni-SimpleMem

Omni-SimpleMem provides the base MAU abstraction, selective ingestion, hot/cold storage, hybrid retrieval and progressive evidence access. SVI-OmniMem is an incremental visual indexing extension.

| Omni-SimpleMem Image Path | SVI-OmniMem Extension |
|---|---|
| Image → concise caption → global retrieval representation | Image → structured searchable card + original image |
| Fine detail mainly remains in cold raw image | Salient entity / OCR / relation / observation anchors enter hot index |
| Raw image is useful only after the image is retrieved | Structured index increases the chance that detail-relevant images are retrieved |
| Caption can become the bottleneck | Structured index routes queries; raw image verifies claims |

### 4.2 Mem-Gallery

Mem-Gallery evaluates multimodal long-term conversational memory and is the primary benchmark for the current SVI implementation. It provides multi-session dialogues with image captions, image IDs, visual search questions, visual reasoning questions, and text-memory questions, making it a good fit for testing whether SVI improves source-image retrieval without benchmark-specific query rules.

The current implementation uses Mem-Gallery point labels only for reporting, not for retrieval routing.

### 4.3 MemEye

MemEye evaluates multimodal memory using visual evidence granularity and evidence-use complexity, including cases where captions cannot substitute for decisive visual evidence and cases involving changing visual states. SVI-OmniMem is directly motivated by this setting:

- raw image preservation supports fine-grained verification;
- structured retrieval anchors support evidence routing;
- observation ledger supports limited temporal state reasoning.

MemEye is now treated as an optional stress test rather than the primary benchmark, because it encourages aggregate/table-style visual reasoning that belongs to a later SVI executor layer.

### 4.4 MM-Mem

MM-Mem preserves fine-grained video evidence in a sensory layer and retrieves downward from semantic memory when needed. SVI-OmniMem shares the principle of retaining original perceptual evidence but differs in target and mechanism:

| MM-Mem | SVI-OmniMem |
|---|---|
| Long-horizon video agents | Multi-session image-grounded memory |
| Keyframe / sub-clip sensory evidence | Original image as evidence |
| Pyramidal video memory | Structured visual index card |
| Progressive temporal down-drill | Candidate image retrieval followed by raw-image verification |
| Video-centric compression | Low-cost static-image visual routing |

---

## 5. Architecture Overview

SVI-OmniMem introduces four components:

```text
┌───────────────────────────────────────────────┐
│ 1. StructuredVisualExtractor                  │
│    One-pass VLM extraction at image ingestion  │
└─────────────────────┬─────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│ 2. StructuredVisualStore                      │
│    Cards, inverted fields, state observations  │
└─────────────────────┬─────────────────────────┘
                      │
             Query + Retrieval
                      │
                      ▼
┌───────────────────────────────────────────────┐
│ 3. RawEvidenceVerifier                        │
│    Bounded loading of original images          │
└─────────────────────┬─────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│ 4. VerifiedFactPromoter + VisualStateLedger   │
│    Reusable facts and temporal observations    │
└───────────────────────────────────────────────┘
```

### 5.1 Memory Views Per Image

Each accepted image yields:

- `ImageGlobalMAU`: existing Omni-SimpleMem-compatible visual MAU;
- `StructuredVisualCard`: structured searchable index record;
- `TextMirrorMAU` or searchable mirror text: optional adapter for existing BM25/dense stores;
- `raw_pointer`: pointer to original image in cold storage;
- later, zero or more `VerifiedVisualFact` items created only after query-time verification.

---

## 6. Data Model

### 6.1 Verification Status Vocabulary

Every visual assertion stored at ingestion should carry a verification state:

| Status | Meaning |
|---|---|
| `unverified_extraction` | VLM extracted this at ingestion; usable for retrieval, not final proof |
| `query_verified` | Confirmed by re-reading raw image for a real query |
| `contradicted` | Later raw-image verification did not support the extraction |
| `superseded` | A later relevant verified observation replaces it for latest-state queries |
| `uncertain` | Evidence insufficient or ambiguous |

### 6.2 Python Dataclasses

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IndexedAttribute:
    value: str
    confidence: float
    verification_status: str = "unverified_extraction"


@dataclass
class RetrievalAnchor:
    anchor_id: str
    category: str
    aliases: list[str] = field(default_factory=list)
    coarse_position: Optional[str] = None
    salient_attributes: dict[str, IndexedAttribute] = field(default_factory=dict)
    distinguishing_features: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class VisualRelation:
    subject_anchor_id: str
    predicate: str
    object_anchor_id: Optional[str] = None
    object_text: Optional[str] = None
    confidence: float = 0.0
    verification_status: str = "unverified_extraction"


@dataclass
class OCRObservation:
    text: str
    context: Optional[str] = None
    confidence: float = 0.0
    verification_status: str = "unverified_extraction"


@dataclass
class StateObservation:
    slot: list[str]                 # e.g. ["desk", "cup", "color"]
    value: str
    observed_at: str
    source_anchor_id: Optional[str] = None
    confidence: float = 0.0
    verification_status: str = "unverified_extraction"
    validity: str = "observation"   # observation / latest_observed / historical / uncertain


@dataclass
class StructuredVisualCard:
    card_id: str
    image_mau_id: str
    session_id: Optional[str]
    turn_id: Optional[str]
    observed_at: str
    raw_pointer: str
    global_caption: str
    retrieval_anchors: list[RetrievalAnchor] = field(default_factory=list)
    relations: list[VisualRelation] = field(default_factory=list)
    ocr_observations: list[OCRObservation] = field(default_factory=list)
    state_observations: list[StateObservation] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    extraction_scope: str = "salient_entities_only"
    schema_version: str = "svi_v1"


@dataclass
class VerifiedVisualFact:
    fact_id: str
    source_image_mau_id: str
    source_card_id: str
    subject: str
    predicate: str
    value: str
    evidence_description: str
    observation_time: str
    verified_at: str
    confidence: float
    query_type: str
    evidence_scope: str       # full_image / localized_object / ocr_area
    raw_pointer: str
```

### 6.3 Structured Visual Card JSON Example

```json
{
  "card_id": "visual_card_013",
  "image_mau_id": "mau_image_013",
  "session_id": "session_04",
  "turn_id": "turn_019",
  "observed_at": "2026-03-01T10:23:00",
  "raw_pointer": "cold://visual/session_04/image_013.jpg",
  "global_caption": "A desk with a laptop, a cup, and several paper notes.",
  "retrieval_anchors": [
    {
      "anchor_id": "cup_1",
      "category": "cup",
      "aliases": ["mug", "coffee cup"],
      "coarse_position": "left side of desk",
      "confidence": 0.88,
      "salient_attributes": {
        "color": {
          "value": "red",
          "confidence": 0.82,
          "verification_status": "unverified_extraction"
        },
        "material": {
          "value": "ceramic",
          "confidence": 0.68,
          "verification_status": "unverified_extraction"
        },
        "markings": {
          "value": "white stripe",
          "confidence": 0.61,
          "verification_status": "unverified_extraction"
        }
      },
      "distinguishing_features": ["possible small crack near rim"]
    },
    {
      "anchor_id": "laptop_1",
      "category": "laptop",
      "aliases": ["computer", "notebook computer"],
      "coarse_position": "center of desk",
      "confidence": 0.94,
      "salient_attributes": {}
    }
  ],
  "relations": [
    {
      "subject_anchor_id": "cup_1",
      "predicate": "left_of",
      "object_anchor_id": "laptop_1",
      "confidence": 0.79,
      "verification_status": "unverified_extraction"
    }
  ],
  "ocr_observations": [
    {
      "text": "meeting 3pm",
      "context": "yellow paper note near monitor",
      "confidence": 0.86,
      "verification_status": "unverified_extraction"
    }
  ],
  "state_observations": [
    {
      "slot": ["desk", "cup", "color"],
      "value": "red",
      "observed_at": "2026-03-01T10:23:00",
      "source_anchor_id": "cup_1",
      "confidence": 0.82,
      "verification_status": "unverified_extraction",
      "validity": "observation"
    }
  ],
  "tags": ["desk", "office", "image_memory"],
  "extraction_scope": "salient_entities_only",
  "schema_version": "svi_v1"
}
```

### 6.4 Why Use `StateObservation` Instead of `state_events.observed_current`

A single image supports only the claim:

```text
At observation time T, the image visibly contained value V.
```

It does not establish:

```text
V remains the current true state at all later times.
```

Therefore:

```json
{
  "slot": ["desk", "cup", "color"],
  "value": "red",
  "observed_at": "2026-03-01",
  "validity": "observation"
}
```

is preferred over:

```json
{
  "subject": "cup_on_desk",
  "predicate": "color",
  "value": "red",
  "temporal_status": "observed_current"
}
```

---

## 7. Storage and MAU Mapping

### 7.1 ImageGlobalMAU

Reuse the current visual MAU for source images:

```python
image_mau = MultimodalAtomicUnit(
    summary=card.global_caption,
    modality_type="visual",
    raw_pointer=card.raw_pointer,
    metadata={
        "session_id": card.session_id,
        "turn_id": card.turn_id,
        "timestamp": card.observed_at,
        "structured_visual_card_id": card.card_id,
        "schema_version": card.schema_version
    }
)
```

### 7.2 Structured Visual Store

Recommended storage files:

```text
hot_storage/
  structured_visual_cards.jsonl
  verified_visual_facts.jsonl
  visual_state_ledger.jsonl
  structured_index_metadata.json
```

During first implementation, JSONL plus in-memory inverted indexes is sufficient. SQLite can be introduced only if query or update costs become problematic.

### 7.3 Text Mirror for Existing Retrieval Infrastructure

To reuse existing dense/BM25 routes, serialize every card into a compact text mirror:

```text
[Structured visual index; raw image verification required for fine-grained claims]
Scene: A desk with a laptop, a cup, and several paper notes.
Entities: cup/mug (red, ceramic, white stripe); laptop/computer.
Relations: cup left_of laptop.
OCR: "meeting 3pm" on a paper note near monitor.
Observation: desk cup color observed as red at 2026-03-01.
Source image: mau_image_013.
Verification status: extracted, not yet query-verified.
```

The text mirror is a **retrieval adapter**, not a trusted final answer source.

### 7.4 Indexes

| Index | Indexed Content | Query Types |
|---|---|---|
| Caption Dense Index | global caption / card mirror | scene and semantic search |
| Entity Alias Index | `cup`, `mug`, `laptop`, etc. | object-centered retrieval |
| Attribute Index | `red`, `ceramic`, `white stripe` | known-attribute filtering |
| Relation Index | `cup left_of laptop` | spatial relation queries |
| OCR Index | recognized strings / text contexts | text-in-image questions |
| State Slot Index | `["desk","cup","color"] → observations` | temporal state questions |
| Optional Global Visual Index | shared text-image global representation | fallback when extraction omitted anchors |
| Verified Fact Index | query-verified facts only | repeated-query acceleration |

---

## 8. Ingestion Algorithm

### 8.1 API

```python
def add_image_structured(
    self,
    image,
    text_context: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    timestamp: str | None = None,
    tags: list[str] | None = None,
    force: bool = False,
) -> ProcessingResult:
    ...
```

### 8.2 Pipeline

```text
Input image + dialogue context + timestamp
        │
        ▼
1. Visual entropy filtering
        │ accepted
        ▼
2. Save original image to cold storage
        │
        ▼
3. One-pass structured VLM extraction
        │
        ├── global_caption
        ├── retrieval_anchors
        ├── relations
        ├── OCR observations
        └── state observations
        ▼
4. Create ImageGlobalMAU
        │
        ▼
5. Save StructuredVisualCard
        │
        ▼
6. Build compact text mirror and field indexes
        │
        ▼
7. Optional global visual fallback embedding
```

### 8.3 Step 1: Visual Entropy Filtering

Reuse the existing visual acceptance trigger:

- reject low-information or duplicate images unless `force=True`;
- accepted images enter structured ingestion.

### 8.4 Step 2: Raw Image Preservation

The original image must remain in cold storage. No fine-grained visual claim is considered fully grounded unless it can be traced back to this source evidence.

```python
raw_pointer = cold_storage.save(image, session_id=session_id)
```

### 8.5 Step 3: One-Pass Structured VLM Extraction

Replace or extend the original one-sentence caption call with a single structured JSON call. The VLM is instructed to extract **salient, searchable and observable** details rather than exhaustively enumerate pixels.

Recommended extraction instruction:

```text
You are building a searchable visual memory index for a long-term agent.

The original image will be retained and reopened for later verification.
Extract concise visual retrieval anchors:
- a short scene-level caption;
- clearly visible salient entities and useful aliases;
- salient attributes only when observable;
- meaningful object relations;
- readable OCR text;
- state observations tied to the supplied timestamp.

Rules:
- Do not infer hidden facts.
- Do not claim permanence or "current truth"; record only observations at this time.
- Return concise searchable information.
- Fine-grained claims remain unverified until checked against the original image during a real query.
```

### 8.6 Step 4: Hard Extraction Budget

The structured extractor must stay compact. Otherwise SVI quietly degenerates into a long dense caption or eager region description system.

Recommended first-version budget:

```yaml
extraction_budget:
  max_card_tokens: 220
  max_retrieval_anchors: 8
  max_attributes_per_anchor: 4
  max_relations_per_image: 6
  max_ocr_observations: 5
  max_state_observations: 6
  max_aliases_per_anchor: 4
  max_distinguishing_features_per_anchor: 3
```

When the VLM returns more content than the budget allows, keep fields in this order:

1. entities explicitly mentioned by the dialogue context or query-like user wording;
2. OCR observations;
3. distinctive object attributes such as color, brand, material, damage or markings;
4. relations involving important entities;
5. state observations useful for temporal memory;
6. generic background objects.

This budget should be reported in experiments so the method is compared fairly against dense caption baselines.

### 8.7 Step 5: Confidence Filtering

Before indexing:

```yaml
min_anchor_confidence: 0.45
min_attribute_confidence: 0.45
min_relation_confidence: 0.55
min_ocr_confidence: 0.55
min_state_observation_confidence: 0.50
```

Low-confidence fields may be retained in raw card logs for analysis, but should not enter primary retrieval indexes.

### 8.8 Step 6: Omission-Aware Fallback Indexing

The expected weakness of SVI is extraction omission. A small or unusual object may be visible in the raw image but absent from the structured card. To reduce hard failures without building dense patch memory, keep two low-cost fallback routes:

```yaml
omission_fallback:
  retain_global_caption: true
  retain_optional_global_visual_embedding: true
  fallback_recent_images: 3
  fallback_same_session_images: 3
  enable_when_structured_score_below: 0.35
```

If a query is visually necessary and structured routes return no confident candidates, the retriever may inspect a small number of recent or same-session images under the same verification budget. This is not a replacement for dense region recall; it is a bounded safety valve for omitted anchors.

### 8.9 Step 7: Optional Global Visual Fallback

If a shared text-image model is already available or cheap to add, retain one global image embedding per accepted image. This is not meant to solve small-object retrieval; it is a low-cost fallback when structured extraction misses a relevant object but the global image representation still carries some signal.

---

## 9. Query Understanding

### 9.1 Query Interface

```python
def query_structured_visual(
    self,
    query: str,
    top_k: int = 10,
    verify: bool = True,
    verification_budget: int = 3,
    writeback_verified_fact: bool = True,
    tags_filter: list[str] | None = None,
    time_range: tuple[str, str] | None = None,
) -> RetrievalResult:
    ...
```

### 9.2 Query Requirement Schema

```python
@dataclass
class VisualQueryRequirement:
    requires_visual_evidence: bool
    entities: list[str]
    requested_attributes: list[str]
    relation_constraints: list[str]
    ocr_terms: list[str]
    state_slots: list[list[str]]
    temporal_scope: str      # any / historical / latest_observed / at_time / change_over_time
    target_time: str | None
    requires_raw_verification: bool
```

### 9.3 Examples

#### Attribute Query

```text
Query: 办公桌照片中的杯子是什么颜色？
```

```json
{
  "requires_visual_evidence": true,
  "entities": ["cup", "desk"],
  "requested_attributes": ["color"],
  "relation_constraints": [],
  "ocr_terms": [],
  "state_slots": [["desk", "cup", "color"]],
  "temporal_scope": "any",
  "target_time": null,
  "requires_raw_verification": true
}
```

#### OCR Query

```text
Query: 我之前照片里的黄色便签写了什么？
```

```json
{
  "requires_visual_evidence": true,
  "entities": ["sticky note"],
  "requested_attributes": ["text"],
  "relation_constraints": [],
  "ocr_terms": [],
  "state_slots": [],
  "temporal_scope": "historical",
  "target_time": null,
  "requires_raw_verification": true
}
```

#### Latest State Query

```text
Query: 最近一次桌面照片里杯子的颜色是什么？
```

```json
{
  "requires_visual_evidence": true,
  "entities": ["cup", "desk"],
  "requested_attributes": ["color"],
  "relation_constraints": [],
  "ocr_terms": [],
  "state_slots": [["desk", "cup", "color"]],
  "temporal_scope": "latest_observed",
  "target_time": null,
  "requires_raw_verification": true
}
```

---

## 10. Retrieval Algorithm

### 10.1 Retrieval Routes

For every visual-memory query, SVI-OmniMem executes only the routes relevant to the parsed requirement.

#### Route A: Caption Dense Retrieval

Use the full query or a scene-level rewrite against:

- existing visual MAU summaries;
- structured visual card text mirrors;
- query-verified fact mirrors.

Good for broad context:

```text
我之前是否发过办公桌相关的照片？
```

#### Route B: Entity / Alias Retrieval

Match normalized entity names and aliases:

```text
cup OR mug OR coffee cup
laptop OR computer
sticky note OR note
```

Good for retrieving a source image even when its global caption does not mention the entity.

#### Route C: Attribute Retrieval

Use only when the attribute value is already present in the query:

```text
哪张照片里有红色杯子？
```

Search:

```text
entity=cup AND attribute.color=red
```

Do **not** use an unknown answer attribute as a filter:

```text
杯子是什么颜色？    # retrieve cup, not "color=unknown"
```

#### Route D: Relation Retrieval

Retrieve images indexed with matching structural relations:

```text
电脑左边的杯子是什么颜色？
→ cup left_of laptop
```

#### Route E: OCR Retrieval

Use OCR exact/BM25 matching for queries containing target text:

```text
哪张图片里出现了 "SALE 50%"？
```

For open OCR questions such as “便签写了什么？”, use entity/context anchors to identify candidate images, then verify raw images.

#### Route F: State Slot Retrieval

Retrieve observations keyed by slot:

```text
["desk", "cup", "color"]
```

This route is required for temporal questions.

#### Route G: Optional Global Visual Fallback

Use shared text-image global embeddings as a low-cost fallback for candidate expansion when structured routes return weak results.

### 10.2 Candidate Fusion

Candidates are merged by `image_mau_id`, preserving the routes and fields that caused retrieval:

```json
{
  "image_mau_id": "mau_image_013",
  "card_id": "visual_card_013",
  "routes": ["entity_alias", "relation", "caption_dense"],
  "matched_fields": [
    "entity=cup",
    "relation=cup left_of laptop"
  ],
  "observation_time": "2026-03-01T10:23:00"
}
```

### 10.3 Pre-Verification Score

The reranking score only determines which original images to inspect:

\[
S_{\text{retrieve}}(I,q)=
w_c S_{\text{caption}}
+w_e S_{\text{entity}}
+w_a S_{\text{attribute}}
+w_r S_{\text{relation}}
+w_o S_{\text{ocr}}
+w_t S_{\text{temporal}}
+w_v S_{\text{global-visual}}
+w_f S_{\text{verified-fact}}
\]

Recommended defaults:

```yaml
retrieval_weights:
  caption_dense: 0.22
  entity_alias: 0.24
  attribute: 0.10
  relation: 0.10
  ocr: 0.14
  temporal_state: 0.10
  global_visual_fallback: 0.04
  verified_fact: 0.06
```

The system should dynamically reweight routes:

| Query Type | Upweighted Routes |
|---|---|
| OCR query | OCR, entity/context |
| Known attribute search | entity, attribute |
| Spatial relation | entity, relation |
| Latest/historical state | state slot, timestamp, verified fact |
| Broad scene query | caption dense, global visual |

---

## 11. Raw-Evidence Verification

### 11.1 When Verification Is Mandatory

Raw-image verification is required for:

- colors, material, pattern, damage or appearance;
- object count;
- OCR content;
- object-to-object spatial relations;
- state selection for “current” / “latest observed” queries;
- any answer relying on `unverified_extraction`;
- any case where multiple candidates disagree.

It may be skipped, by configuration, for:

- pure source retrieval (“我是否发过办公桌照片？”);
- a repeated query answered by an unconflicted `query_verified` fact.

### 11.2 Verification Budget

Only top candidates are opened:

```yaml
verification:
  enabled: true
  max_images_per_query: 3
  allow_abstain: true
  require_provenance: true
```

### 11.3 Query-Aware Candidate Packing

The verifier should not always inspect the top-3 candidates by the same score. Candidate packing should depend on the query requirement:

| Query Type | Candidate Packing Rule |
|---|---|
| Attribute question with unknown value, e.g. “杯子是什么颜色？” | Rank by entity/context match first; do not filter by a guessed attribute value |
| Known-attribute search, e.g. “哪张照片有红色杯子？” | Rank by entity + attribute match, then verify the color in raw image |
| OCR question with target text | Prioritize OCR index matches, then same-session/image-caption candidates |
| Open OCR question, e.g. “便签写了什么？” | Prioritize entity/context anchors such as note, sign, label, screen; verification reads the text |
| Spatial relation question | Prioritize relation matches plus candidates containing both entities |
| Count question | Prefer candidate diversity across relevant images or sessions; avoid spending the whole budget on near-duplicates |
| Latest observed state | Sort relevant candidates by observation time descending, then verify newest first |
| Historical / at-time state | Restrict or rank by target time before content score |
| Conflicting candidates | Include the strongest candidates from each conflicting value group |

If all structured candidates are weak and the query requires visual evidence, use the omission-aware fallback from Section 8.8 within the same budget. The system should log whether a verified answer came from structured retrieval or fallback inspection.

### 11.4 Verification Prompt

```text
You are verifying visual evidence from an original stored image.

Question:
{query}

Candidate index information:
{candidate_index_claims}

Observation time:
{observed_at}

Instructions:
- Candidate index information is only a retrieval hint and may be wrong.
- Use only facts visibly supported by this original image.
- Verify only the information needed to answer the question.
- If the image does not clearly support the requested answer, return supports=false.
- Do not infer that an observed state remains current across later sessions.

Return JSON:
{
  "supports": true,
  "answer_fragment": "...",
  "visible_evidence": "...",
  "verified_facts": [
    {
      "subject": "...",
      "predicate": "...",
      "value": "...",
      "confidence": 0.0
    }
  ],
  "confidence": 0.0
}
```

### 11.5 Evidence Bundle for Answer Generation

The answer model receives compact, provenance-preserving evidence:

```text
Question: 最近一次桌面照片里杯子的颜色是什么？

Verified Evidence:
- Source image: mau_image_027
- Observation time: 2026-04-10
- Verified fact: desk.cup.color = blue
- Visible evidence: A blue cup is visible next to the laptop.
- Verification confidence: 0.95

Relevant historical observation:
- Source image: mau_image_013
- Observation time: 2026-03-01
- Verified fact: desk.cup.color = red
```

---

## 12. Query-Driven Verified Fact Promotion

### 12.1 Motivation

When a real user query has already required raw-image verification, the verified result is more valuable than the original unverified extraction. Saving only the required verified fact enables later reuse without turning every image into a dense region memory record.

### 12.2 Promotion Policy

```yaml
writeback:
  promote_verified_facts: true
  min_verification_confidence: 0.80
  store_only_requested_facts: true
  deduplicate_same_fact: true
  preserve_raw_pointer: true
  update_original_card_status: true
```

### 12.3 Anti-Contamination Rules

Verified-fact promotion must be conservative. A single mistaken verifier output should not become a permanent high-priority memory.

```yaml
promote_only_if:
  verifier_confidence_at_least: 0.80
  answer_fragment_not_empty: true
  source_image_available: true
  evidence_description_not_empty: true
  fact_is_requested_by_query: true
  no_later_conflicting_verified_fact: true

do_not_promote_if:
  verifier_supports_false: true
  verifier_abstained: true
  raw_pointer_missing: true
  answer_requires_identity_tracking_but_identity_uncertain: true
  query_is_speculative_or_hypothetical: true

demote_or_mark_if:
  later_verification_contradicts: "contradicted"
  later_observation_replaces_latest_state: "superseded"
  raw_pointer_becomes_unavailable: "uncertain"
```

For latest-state queries, older facts should not be marked false. They should become `historical` or `superseded` only with respect to a newer verified observation for the same state slot.

### 12.4 Example

After verifying:

```text
Q: 办公桌照片中的杯子是什么颜色？
A: red
```

write:

```json
{
  "fact_id": "verified_fact_002",
  "source_image_mau_id": "mau_image_013",
  "source_card_id": "visual_card_013",
  "subject": "desk.cup",
  "predicate": "color",
  "value": "red",
  "evidence_description": "A red cup is visible to the left of the laptop.",
  "observation_time": "2026-03-01T10:23:00",
  "verified_at": "2026-05-28T09:11:00",
  "confidence": 0.96,
  "query_type": "visual_attribute",
  "evidence_scope": "full_image",
  "raw_pointer": "cold://visual/session_04/image_013.jpg"
}
```

The corresponding card field is updated from:

```text
unverified_extraction → query_verified
```

### 12.5 Reuse Policy

A verified fact may answer a later query without re-opening the image only when:

- the query asks the same or logically weaker fact;
- the fact is not contradicted by a later verified observation;
- the query does not explicitly request source inspection;
- the query is not asking for current state when newer observations exist.

---

## 13. Visual State Observation Ledger

### 13.1 Scope

The ledger supports:

- observations at specific times;
- latest verified observation;
- historical comparison;
- evidence provenance;
- abstention under identity ambiguity.

It does not solve full object tracking.

### 13.2 Ledger Representation

```json
{
  "slot": ["desk", "cup", "color"],
  "observations": [
    {
      "value": "red",
      "observed_at": "2026-03-01",
      "source_image_mau_id": "mau_image_013",
      "verification_status": "query_verified",
      "validity": "historical"
    },
    {
      "value": "blue",
      "observed_at": "2026-04-10",
      "source_image_mau_id": "mau_image_027",
      "verification_status": "query_verified",
      "validity": "latest_observed"
    }
  ],
  "latest_verified_value": "blue",
  "latest_verified_source": "mau_image_027"
}
```

### 13.3 Query Resolution Rules

| Query Scope | Resolution |
|---|---|
| Historical / specific date | Verify and use the observation nearest the requested time |
| Latest observed | Verify the temporally latest relevant candidate |
| Change over time | Return an ordered sequence of verified observations |
| Conflicting unverified observations | Inspect source images before selecting |
| Entity identity uncertain | State observed differences without asserting the same object changed |

Example conservative answer:

```text
3 月 1 日的照片中观察到红色杯子；4 月 10 日的照片中观察到蓝色杯子。
现有证据不足以确认它们是否为同一个杯子。
```

---

## 14. Failure Handling

| Failure Case | Required Behaviour |
|---|---|
| Structured JSON parse failure | Fall back to original global-caption image MAU |
| Low-confidence anchor | Do not index it in primary retrieval; retain for debug only |
| Low-confidence relation | Do not use as relation match candidate |
| OCR failure | Keep scene/entity index and raw image |
| Missing raw image at verification | Never treat unverified structured claim as grounded answer |
| Verification contradicts extraction | Mark field `contradicted`; optionally store correction |
| Verification insufficient | Abstain or present uncertainty |
| Verification budget exhausted | Return evidence-insufficient result; record budget miss |
| Multiple state candidates unresolved | Return timestamped observations without forcing a current-state conclusion |
| Global visual fallback unavailable | Continue with caption and structured field routes |

---

## 15. Proposed Code Structure

```text
OmniSimpleMem/omni_memory/
  core/
    structured_visual_card.py
    verified_visual_fact.py
    visual_state_ledger.py

  processors/
    structured_visual_extractor.py
    visual_query_parser.py

  storage/
    structured_visual_store.py
    verified_fact_store.py
    visual_state_store.py

  retrieval/
    structured_visual_retriever.py
    raw_evidence_verifier.py
    verified_fact_promoter.py

  orchestrator.py
```

### 15.1 `core/structured_visual_card.py`

Contains:

- `IndexedAttribute`
- `RetrievalAnchor`
- `VisualRelation`
- `OCRObservation`
- `StateObservation`
- `StructuredVisualCard`

Requirements:

- `to_dict()` / `from_dict()`;
- JSON schema validation;
- versioned schema;
- robust defaults for missing optional fields.

### 15.2 `processors/structured_visual_extractor.py`

```python
class StructuredVisualExtractor:
    def extract(
        self,
        image,
        text_context: str | None,
        timestamp: str,
        session_id: str | None = None,
        turn_id: str | None = None,
    ) -> StructuredVisualCard:
        ...
```

Implementation notes:

- use the existing VLM path;
- output caption and index fields in one call;
- validate JSON and filter low-confidence records;
- never label ingestion-time outputs as verified.

### 15.3 `processors/visual_query_parser.py`

```python
class VisualQueryParser:
    def parse(self, query: str) -> VisualQueryRequirement:
        ...
```

First version may use:

- deterministic keyword templates for OCR/state/attribute triggers;
- one compact LLM call only when parsing fails or the query is complex;
- cached parsing for repeated queries.

### 15.4 `storage/structured_visual_store.py`

Supports:

```python
append(card)
get_by_card_id(card_id)
get_by_image_mau_id(image_mau_id)
search_entity_alias(entity_terms)
search_attribute(entity, attribute, value)
search_relation(subject, predicate, object)
search_ocr(text)
search_state_slot(slot, time_range=None)
```

### 15.5 `retrieval/structured_visual_retriever.py`

Responsible for:

- running relevant retrieval routes;
- preserving retrieval provenance;
- merging candidates by source image;
- temporal filtering;
- pre-verification reranking;
- returning images to inspect under the verification budget.

### 15.6 `retrieval/raw_evidence_verifier.py`

Responsible for:

- loading source image from cold storage;
- calling VLM on bounded candidate images;
- parsing verified facts and evidence descriptions;
- handling abstention and contradictions.

### 15.7 `retrieval/verified_fact_promoter.py`

Responsible for:

- persisting verified facts;
- updating original card field statuses;
- updating state ledger;
- deduplicating repeated verified claims;
- preserving source provenance.

### 15.8 `orchestrator.py` API Additions

```python
def add_image_structured(
    self,
    image,
    text_context: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    timestamp: str | None = None,
    tags: list[str] | None = None,
    force: bool = False,
) -> ProcessingResult:
    ...
```

```python
def query_structured_visual(
    self,
    query: str,
    top_k: int = 10,
    verify: bool = True,
    verification_budget: int = 3,
    writeback_verified_fact: bool = True,
    tags_filter: list[str] | None = None,
    time_range: tuple[str, str] | None = None,
) -> RetrievalResult:
    ...
```

Existing APIs should remain unchanged for backward compatibility and baseline evaluation.

---

## 16. Integration with Existing Omni-SimpleMem Components

### 16.1 `VisualEntropyTrigger`

Reuse unchanged:

```text
low-value visual input → skip
accepted visual input → SVI structured extraction
```

### 16.2 `MultimodalAtomicUnit`

No mandatory schema replacement is required. Attach a pointer in MAU metadata/details:

```json
{
  "structured_visual_card_id": "visual_card_013",
  "verified_fact_ids": ["verified_fact_002"],
  "schema_version": "svi_v1"
}
```

### 16.3 `ColdStorageManager`

Continue storing original images only; do not generate persistent crops by default.

### 16.4 `BM25Store`

Index:

- caption;
- serialized entity aliases;
- visible attributes;
- OCR strings;
- relation strings;
- verified fact strings.

### 16.5 `HybridVectorStore`

Index:

- compact card text mirrors;
- verified fact mirrors;
- optional global visual embeddings.

### 16.6 `KnowledgeGraph`

First version should add only stable/high-value edges:

```text
IMAGE_HAS_ENTITY
IMAGE_HAS_OCR
IMAGE_OBSERVED_STATE
VERIFIED_FACT_SUPPORTED_BY_IMAGE
OBSERVATION_PRECEDES_OBSERVATION
```

Do not initially create a dense region relation graph.

---

## 17. Experimental Design

### 17.1 Research Questions

| ID | Research Question |
|---|---|
| RQ1 | Can structured visual cards retrieve fine-grained relevant source images better than global captions? |
| RQ2 | Under equal ingestion-token budgets, do structured fields outperform dense captions? |
| RQ3 | Does raw-image verification reduce unsupported visual claims and extraction errors? |
| RQ4 | Does verified-fact writeback reduce repeated visual verification cost without hurting accuracy? |
| RQ5 | Can a lightweight observation ledger improve temporal visual-state queries? |
| RQ6 | Does SVI offer a better accuracy–cost tradeoff than eager region/patch indexing? |

### 17.2 Datasets

#### Primary: Mem-Gallery

Use Mem-Gallery as the primary benchmark for the current low-prior SVI implementation. The evaluation should ingest dialogues chronologically, store image cards, retrieve by generic card-text similarity, optionally verify original images, and report scores by Mem-Gallery point label only after prediction.

#### Optional Stress Test: MemEye

Use MemEye only after the Mem-Gallery pipeline is stable. MemEye remains valuable for fine-grained visual evidence and evolving-state stress tests, but aggregate/table-style questions should be reported separately until SVI has a dedicated evidence-table executor.

#### Optional Generalization: MemLens or Image-Linked Long Dialogue Subset

Only add if integration effort is low and the primary experiments are stable.

### 17.3 Baselines

| Method | Description | Why Required |
|---|---|---|
| Omni-SimpleMem | Original caption-centric image memory path | Direct baseline |
| Dense Caption OmniMem | Longer textual description with matched token budget | Rules out “just write more text” |
| Structured Index Only | SVI retrieval without raw verification | Measures extraction-only limitations |
| SVI + Raw Verification | Structured retrieval plus source image verification | Core method |
| SVI + Raw Verification + Writeback | Full method with verified fact cache | Measures longitudinal reuse |
| Optional Eager Region / Patch Baseline | Pre-expanded local visual memory | Accuracy/cost comparison upper bound |

### 17.4 Equal-Budget Ingestion Comparison

To avoid an unfair advantage from longer VLM output, compare:

| Method | Maximum Ingestion Output |
|---|---:|
| Original Caption | original budget |
| Dense Caption | 200 tokens/image |
| Structured Visual Card | 200 tokens/image |
| SVI + Verification | 200 tokens/image plus bounded query-time verification |

### 17.5 Required Diagnostic Split: Caption-Blind Visual Evidence

Construct or filter a diagnostic subset satisfying:

```text
1. The original image visibly supports the answer.
2. The original Omni-SimpleMem caption does not contain the required answer
   or sufficient identifying clue.
3. The query requires entity, attribute, OCR, relation, count, or temporal
   visual evidence.
```

This subset directly tests the central hypothesis:

> SVI improves access to visual evidence that a concise caption omits, without requiring dense local indexing.

This split should be reported as a primary diagnostic, not only as an auxiliary check.

Recommended reporting:

| Report | Purpose |
|---|---|
| Overall Mem-Gallery score | General multimodal memory quality |
| Caption-Blind Visual Evidence score | Core SVI hypothesis |
| Caption-Covered score | Ensures SVI does not harm easy caption-answerable cases |
| Evidence granularity breakdown | Scene / entity / attribute / OCR / relation / state |
| Verification-required breakdown | Measures benefit of raw evidence verification |

### 17.6 Cost Model and Equal-Budget Protocol

SVI's main claim is a quality-cost tradeoff, so experiments should report cost explicitly.

Let:

```text
N_img      = number of accepted images
N_q_vis    = number of visual-evidence queries
K_region   = average generated regions or patches per image
B_verify   = verification images opened per query
C_cap      = cost of one caption call
C_struct   = cost of one structured extraction call
C_region   = cost of one region caption / crop embedding unit
C_verify   = cost of one raw-image verification call
```

Approximate method costs:

```text
Cost_original_caption =
  N_img * C_cap

Cost_dense_caption =
  N_img * C_struct   # matched output-token budget, but no structured fields

Cost_eager_region =
  N_img * (C_cap + K_region * C_region)

Cost_svi =
  N_img * C_struct + N_q_vis * B_verify * C_verify
```

Storage growth should also be reported:

```text
Storage_original = O(N_img)
Storage_region   = O(N_img * K_region)
Storage_svi      = O(N_img + N_verified_facts)
```

Equal-budget ingestion comparison:

| Method | Ingestion Constraint |
|---|---|
| Original Caption | Existing OmniSimpleMem caption budget |
| Dense Caption | Same max output tokens as SVI card |
| Structured Visual Card | Same max output tokens as dense caption |
| Eager Region / Patch | Report separately as a higher-cost upper-bound baseline |

This separates the value of structure from simply allowing the model to write a longer description.

### 17.7 Metrics

#### Answer Quality

| Metric | Description |
|---|---|
| QA Accuracy / F1 | Overall answer correctness |
| Category Accuracy | Scene / entity / attribute / OCR / relation / state |
| Abstention Accuracy | Correctly refusing unsupported answers |

#### Retrieval Quality

| Metric | Description |
|---|---|
| Image Evidence Recall@K | Whether the supporting original image is retrieved |
| Anchor Hit@K | Whether relevant structured anchor participates in retrieval |
| Caption-Blind Recall@K | Recall when original caption omits needed clue |
| MRR | Ranking of correct source image |

#### Verification and Grounding

| Metric | Description |
|---|---|
| Verified Evidence Precision | Fraction of accepted evidence that truly supports answer |
| Unsupported Claim Rate | Answers relying on non-supported visual claims |
| Extraction Error Correction Rate | Errors corrected by raw-image verification |
| State Resolution Accuracy | Correct latest/history observation selection |

#### Efficiency

| Metric | Description |
|---|---|
| Write-Time VLM Calls per Image | Main ingestion call count |
| Ingestion Output Tokens per Image | Written index size |
| Storage Records per Image | Memory growth |
| Verification Calls per Query | Query-time visual cost |
| Retrieved Tokens per Query | Context sent to answering model |
| End-to-End Latency | Total runtime |
| Verified Fact Cache Hit Rate | Reuse rate |
| Accuracy per Token / Second | Quality–cost tradeoff |

### 17.8 Ablations

| Ablation | Purpose |
|---|---|
| Caption only | Baseline image memory |
| + Entity Anchors | Object-level retrieval benefit |
| + Attributes | Attribute-filtered retrieval benefit |
| + Relations | Spatial relation benefit |
| + OCR | Text-in-image benefit |
| + State Observations | Temporal state benefit |
| + Global Visual Fallback | Mitigation for extraction omission |
| + Raw Verification | Necessity of evidence checking |
| + Verified Writeback | Reuse and cost reduction |
| Without Confidence Filtering | Sensitivity to noisy extractions |

---

## 18. Failure Analysis Taxonomy

SVI-OmniMem should explicitly report failure sources:

| Failure Type | Description |
|---|---|
| Extraction Omission | Target entity/detail exists in raw image but is absent from card |
| Extraction Hallucination | Structured card includes an incorrect visible fact |
| Retrieval Miss | Required indexed clue exists but source image is not retrieved |
| Verification Failure | Correct source image is retrieved but verifier gives wrong result |
| Temporal Resolution Error | System selects wrong historical/latest observation |
| Entity Identity Ambiguity | System cannot establish whether objects across images are the same |
| Verification Budget Miss | Correct source image falls outside verification budget |
| Over-Trust of Index | Answer uses unverified extraction as final evidence |

This analysis is critical because the expected primary limitation of SVI is **extraction omission**, while the expected primary limitation of eager patch/region methods is **compute and storage cost**.

---

## 19. Implementation Milestones

### Milestone 1: Schema and Baseline

- Run original Omni-SimpleMem visual-memory path.
- Define `StructuredVisualCard` and JSON schema.
- Persist structured cards without changing retrieval.
- Confirm raw pointer recovery works.

### Milestone 2: One-Pass Structured Ingestion

- Modify/extend visual extraction call to emit structured JSON.
- Build compact text mirror.
- Build entity, relation, OCR and state indexes.
- Log extraction confidence and card size.

### Milestone 3: Structured Retrieval

- Implement query requirement parsing.
- Implement route-specific recall.
- Implement candidate fusion/reranking and route provenance logging.
- Compare image recall against caption-only retrieval.

### Milestone 4: Raw-Evidence Verification

- Load top candidate images from cold storage.
- Implement bounded VLM verification.
- Return evidence bundle with confidence and provenance.
- Compare `Structured Index Only` vs. `SVI + Raw Verification`.

### Milestone 5: Verified Fact Promotion and State Ledger

- Persist verified facts.
- Update verification status on visual cards.
- Maintain timestamped state observations.
- Measure cache reuse and repeat-query cost.

### Milestone 6: Evaluation and Paper-Ready Analysis

- Run Mem-Gallery primary evaluation.
- Optionally run MemEye stress evaluation after evidence-table support is available.
- Complete equal-budget dense-caption comparison.
- Produce accuracy/cost curves, ablations and failure analysis.
- Package configs and reproducibility scripts.

---

## 20. Recommended Configuration

```yaml
svi_omnimem:
  enabled: true
  schema_version: "svi_v1"

  ingestion:
    reuse_visual_entropy_trigger: true
    save_raw_image: true
    one_pass_structured_vlm_extraction: true
    extraction_scope: "salient_entities_only"
    max_card_tokens: 220
    max_retrieval_anchors: 8
    max_attributes_per_anchor: 4
    max_relations: 6
    max_ocr_observations: 5
    max_state_observations: 6
    max_aliases_per_anchor: 4
    max_distinguishing_features_per_anchor: 3
    min_anchor_confidence: 0.45
    min_attribute_confidence: 0.45
    min_relation_confidence: 0.55
    min_ocr_confidence: 0.55
    create_text_mirror: true
    generate_patch_embeddings: false
    generate_region_captions: false
    build_dense_region_graph: false

  indexing:
    caption_dense_index: true
    entity_alias_index: true
    attribute_index: true
    relation_index: true
    ocr_bm25_index: true
    state_slot_index: true
    global_visual_fallback_index: true
    verified_fact_index: true

  omission_fallback:
    enabled: true
    fallback_recent_images: 3
    fallback_same_session_images: 3
    enable_when_structured_score_below: 0.35

  retrieval:
    top_k_candidates: 10
    deduplicate_by_image: true
    dynamic_route_weights: true
    weights:
      caption_dense: 0.22
      entity_alias: 0.24
      attribute: 0.10
      relation: 0.10
      ocr: 0.14
      temporal_state: 0.10
      global_visual_fallback: 0.04
      verified_fact: 0.06

  verification:
    enabled: true
    verification_budget: 3
    query_aware_candidate_packing: true
    verify_visual_attributes: true
    verify_counts: true
    verify_ocr: true
    verify_relations: true
    verify_state_answers: true
    allow_abstain: true
    require_provenance: true

  writeback:
    promote_verified_facts: true
    min_verification_confidence: 0.80
    store_only_requested_facts: true
    deduplicate_same_fact: true
    preserve_raw_pointer: true
    maintain_state_ledger: true
    require_nonempty_answer_fragment: true
    require_nonempty_evidence_description: true
    block_identity_uncertain_promotions: true
    demote_on_later_contradiction: true
```

---

## 21. Acceptance Criteria

SVI-OmniMem is successful if it demonstrates:

1. **Better fine-grained visual memory**
   - Improved Mem-Gallery visual-search / visual-reasoning performance relative to Omni-SimpleMem.

2. **Caption-blind evidence recovery**
   - Clear improvements on the Caption-Blind Visual Evidence split.

3. **Independent value over longer captions**
   - At equal ingestion token budgets, structured visual indexing outperforms dense caption storage.

4. **Necessity of source verification**
   - Raw verification reduces unsupported claim rate and corrects ingestion-time extraction errors.

5. **Preserved lightweight character**
   - One main structured extraction call per accepted image;
   - no per-image dense patch/region vector growth;
   - query verification strictly bounded by budget.

6. **Useful long-term writeback**
   - Verified-fact promotion reduces repeated verification cost without accuracy loss.

7. **Transparent limitations**
   - Explicitly report failures involving omitted tiny objects, ambiguous entity identity and insufficient visual coverage.

---

## 22. Expected Benefits

SVI-OmniMem is expected to improve queries such as:

```text
照片中的杯子是什么颜色？
便签上写了什么？
哪个物体位于电脑左侧？
哪张图片中出现了带白色条纹的红杯子？
最近一次桌面照片里杯子的颜色是什么？
我以前是否拍过写着 “SALE 50%” 的标牌？
```

The expected gain is not merely answer accuracy. The method should produce more trustworthy memory behaviour:

- retrieved claims have source-image provenance;
- fine-grained answers are checked against raw images;
- repeated validated facts become cheaper to answer;
- temporal observations remain traceable rather than silently overwritten.

---

## 23. Limitations

1. **Structured extraction remains lossy.**  
   If a tiny or unexpected object is absent from `retrieval_anchors` and not recoverable through global visual fallback, the source image may remain undiscovered.

2. **SVI does not replace dense local visual memory.**  
   Patch- or region-level methods may provide better recall for dense scenes or very small objects. SVI targets a better cost–quality balance.

3. **First-query verification still has cost.**  
   The method saves write-time and storage costs but may pay bounded query-time VLM cost for new fine-grained questions.

4. **Temporal observations do not guarantee object identity.**  
   Seeing a red cup in one image and a blue cup in another does not prove a single object changed color.

5. **Schema quality depends on the VLM.**  
   Extraction coverage, OCR accuracy and structured-output reliability must be measured and reported.

---

## 24. Paper Claim

Recommended claim:

> Existing multimodal agent memory systems may retain raw images while accessing them through lossy caption-centric indexes. We introduce **SVI-OmniMem**, a lightweight structured visual indexing framework that separates searchable visual claims from raw visual evidence. SVI-OmniMem performs one-pass structured extraction for low-cost image routing, executes bounded raw-image verification only when a query requires fine-grained evidence, and promotes verified facts for later reuse. This improves caption-blind visual memory without the ingestion and storage cost of dense patch or region indexing.

中文表述：

> 现有多模态 agent memory 即便保留了原始图像，其检索入口仍可能受有损 caption 限制。SVI-OmniMem 使用一次性结构化视觉索引完成低成本候选图片定位，将原图核验限定在真正需要细粒度证据的问题上，并将验证后的事实写回为可复用记忆，从而在避免密集 patch / region 索引成本的同时，提高 caption 无法覆盖的视觉记忆能力。

---

## 25. References

- Omni-SimpleMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory. arXiv:2604.01007.  
  <https://arxiv.org/abs/2604.01007>

- MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory. arXiv:2605.15128.  
  <https://arxiv.org/abs/2605.15128>

- From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck for Long-Horizon Video Agents (MM-Mem). arXiv:2603.01455.  
  <https://arxiv.org/abs/2603.01455>

- Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents. arXiv:2601.03515.  
  <https://arxiv.org/abs/2601.03515>

---

## 26. Final Implementation Recommendation

First implementation should include only:

- one-pass structured VLM ingestion;
- `StructuredVisualCard`;
- caption/entity/relation/OCR/state retrieval;
- bounded original-image verification;
- verified fact writeback;
- a lightweight observation ledger;
- Mem-Gallery evaluation.

First implementation should **not** include:

- dense patch embeddings;
- pre-generated region captions;
- complete region graph;
- persistent crops for every image;
- general-purpose object identity tracking;
- heavy detector/OCR/VLM cascades.

The intended system boundary is:

```text
SVI-OmniMem does not attempt to store every possible visual detail in advance.
It creates a cheap structured route to likely source images,
opens raw evidence only when a real query demands detail,
and gradually turns verified, useful observations into reusable long-term memory.
```
