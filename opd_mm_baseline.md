# OPD-MM Baseline：面向多模态 Agent Memory 检索的轻量 On-Policy Distillation 方案

## 0. 设计目标

本文档定义一个轻量版 **OPD-MM baseline**，用于实现多模态 agent memory 检索策略学习。

核心目标是：

> 不依赖复杂手写规则，也不让 policy model 直接读取完整 memory index，而是让学生模型仅根据用户 query 生成一串可执行 tool action。训练时，教师模型基于 query、gold answer 和学生 rollout 进行 hindsight correction，并将修正后的 action trajectory 蒸馏给学生。

这个 baseline 只实现核心闭环：

```text
Student query-only rollout
        ↓
Tool executor runs on hidden memory store
        ↓
Answer model answers with retrieved evidence
        ↓
Teacher hindsight correction
        ↓
SFT distillation to student
```

不做复杂优化：

```text
不做 Best-of-N teacher traces
不做 counterfactual perturbation
不做复杂 verifier
不做 PPO / GRPO
不做 memory update / delete
不做 graph memory
不做 patch-level visual indexing
不让 student / teacher 读取完整 memory index
```

---

## 1. 核心约束

### 1.1 动作不能过度预定义

不使用类似 `SELECT_LAST_IMAGE` 这种强规则动作。  
所有动作都抽象成通用 tool，例如：

```text
FILTER(field=modality, value=image)
FILTER(field=author, value=user)
SORT(field=timestamp, order=desc)
TOPK(k=1)
READ(fields=[summary, ocr])
```

也就是说，“选择上次用户发的图片”不是一个单独动作，而是由通用 tool 组合完成：

```json
[
  {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
  {"tool": "FILTER", "field": "author", "op": "eq", "value": "user"},
  {"tool": "SORT", "field": "timestamp", "order": "desc"},
  {"tool": "TOPK", "k": 1},
  {"tool": "READ", "fields": ["summary", "ocr"]}
]
```

这样可以避免将具体 query pattern 写死在 action space 里。

---

### 1.2 Student 和 teacher 不看完整 memory index

学生模型测试时只看：

```text
user query
allowed tool schema
```

学生模型不看：

```text
完整 memory bank
完整 memory index
候选 memory 列表
memory_id
检索结果
```

训练时教师模型也不看完整 memory index。教师可以看到：

```text
user query
gold answer
student trajectory
student answer 是否正确
allowed tool schema
```

教师不看：

```text
完整 memory bank
完整 memory index
候选 memory 列表
具体 memory_id
```

因此，教师输出的是**抽象检索计划**，而不是直接选择某条 memory。

---

### 1.3 每个 action 都必须是可执行 tool

模型不能输出自由形式的 reasoning step，例如：

```json
{"step": "understand the query intent"}
```

只能输出系统支持的 tool call：

```json
{"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"}
```

所有 tool 都由 executor 在隐藏 memory store 上执行。

---

## 2. 系统总览

### 2.1 推理阶段

```text
User Query
   ↓
Student Policy Model
   - input: query only
   - output: tool trajectory
   ↓
Tool Executor
   - executes tools over hidden memory store
   - retrieves evidence package
   ↓
Answer Model
   - input: query + evidence package
   - output: final answer
```

### 2.2 训练阶段

```text
Training Sample: query + gold answer + hidden memory store
   ↓
Student generates trajectory from query
   ↓
Executor executes trajectory on hidden memory store
   ↓
Answer model produces answer
   ↓
Teacher sees:
   - query
   - gold answer
   - student trajectory
   - student answer correctness
   - allowed tool schema
   ↓
Teacher outputs corrected abstract trajectory
   ↓
SFT student on corrected trajectory
```

这里的关键是：

> teacher 的 hindsight correction 只修正“应该用什么工具组合”，而不直接告诉学生某条 memory_id 是答案证据。

---

## 3. Memory Store 设计

虽然 student / teacher 不看完整 memory index，但 executor 需要在隐藏 memory store 上执行工具。

每条 memory 使用统一 schema。

### 3.1 文本 memory

```json
{
  "memory_id": "m_0012",
  "turn_id": 12,
  "timestamp": "2026-06-12T10:20:00",
  "author": "user",
  "modality": "text",
  "source_type": "conversation",
  "summary": "用户表示自己主要研究结直肠癌免疫治疗响应预测。",
  "content": "我现在主要研究结直肠癌免疫治疗响应预测。",
  "ocr": null,
  "raw_pointer": null
}
```

### 3.2 图像 memory

```json
{
  "memory_id": "m_0017",
  "turn_id": 17,
  "timestamp": "2026-06-12T10:35:00",
  "author": "user",
  "modality": "image",
  "source_type": "uploaded_image",
  "summary": "A screenshot of an ICLR 2026 registration receipt.",
  "content": null,
  "ocr": "ICLR 2026 Registration. Amount: $450.",
  "raw_pointer": "images/img_0017.png"
}
```

### 3.3 文档 / 截图 memory，可选

```json
{
  "memory_id": "m_0021",
  "turn_id": 21,
  "timestamp": "2026-06-12T10:40:00",
  "author": "user",
  "modality": "image",
  "source_type": "screenshot",
  "summary": "A screenshot of a conference reimbursement form.",
  "content": null,
  "ocr": "International conference registration fee reimbursement ...",
  "raw_pointer": "images/screenshot_0021.png"
}
```

---

## 4. Tool Action Space

Baseline 使用少量通用工具。

### 4.1 `FILTER`

根据 metadata 字段过滤当前候选池。

#### Schema

```json
{
  "tool": "FILTER",
  "field": "modality | author | source_type | timestamp | status",
  "op": "eq | neq | before | after | contains",
  "value": "..."
}
```

#### 例子

```json
{"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"}
```

```json
{"tool": "FILTER", "field": "author", "op": "eq", "value": "user"}
```

```json
{"tool": "FILTER", "field": "source_type", "op": "eq", "value": "screenshot"}
```

#### 实现

```python
def tool_filter(pool, field, op, value):
    if op == "eq":
        return [m for m in pool if m.get(field) == value]
    if op == "neq":
        return [m for m in pool if m.get(field) != value]
    if op == "contains":
        return [m for m in pool if value in str(m.get(field, ""))]
    if op == "before":
        return [m for m in pool if m.get(field) < value]
    if op == "after":
        return [m for m in pool if m.get(field) > value]
    return pool
```

---

### 4.2 `SORT`

按照某个字段排序。

#### Schema

```json
{
  "tool": "SORT",
  "field": "timestamp | turn_id | score",
  "order": "asc | desc"
}
```

#### 例子

```json
{"tool": "SORT", "field": "timestamp", "order": "desc"}
```

#### 实现

```python
def tool_sort(pool, field, order):
    reverse = order == "desc"
    return sorted(pool, key=lambda x: x.get(field), reverse=reverse)
```

---

### 4.3 `TOPK`

保留当前候选池前 k 条。

#### Schema

```json
{
  "tool": "TOPK",
  "k": 1
}
```

#### 实现

```python
def tool_topk(pool, k):
    return pool[:k]
```

---

### 4.4 `RETRIEVE`

在当前候选池上执行检索。检索器内部可以使用 BM25、embedding 或 hybrid search。模型不需要指定新的 search query，默认使用原始用户 query。

#### Schema

```json
{
  "tool": "RETRIEVE",
  "method": "bm25 | dense | hybrid",
  "top_k": 5
}
```

#### 例子

```json
{"tool": "RETRIEVE", "method": "hybrid", "top_k": 5}
```

#### 实现

```python
def tool_retrieve(pool, query, method="hybrid", top_k=5):
    if method == "bm25":
        scored = bm25_score(query, pool)
    elif method == "dense":
        scored = dense_score(query, pool)
    else:
        bm25 = bm25_score(query, pool)
        dense = dense_score(query, pool)
        scored = combine_scores(bm25, dense, alpha=0.5)

    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
    return [x["memory"] for x in ranked[:top_k]]
```

---

### 4.5 `READ`

读取当前候选 memory 的指定字段，构造 evidence package。

#### Schema

```json
{
  "tool": "READ",
  "fields": ["summary", "content", "ocr", "timestamp", "raw_pointer"]
}
```

#### 例子

```json
{"tool": "READ", "fields": ["summary", "ocr", "timestamp"]}
```

#### 实现

```python
def tool_read(pool, fields):
    evidence = []
    for m in pool:
        item = {"memory_id": m["memory_id"]}
        for f in fields:
            item[f] = m.get(f)
        evidence.append(item)
    return evidence
```

---

### 4.6 `INSPECT_RAW`

将当前候选中的原始图像 / 文档交给 VLM 或 OCR 模块进一步检查。

Baseline 里不做复杂视觉定位，只调用 VLM 得到与 query 相关的简短 observation。

#### Schema

```json
{
  "tool": "INSPECT_RAW",
  "target": "current_pool",
  "instruction": "answer_query_related_visual_details"
}
```

#### 例子

```json
{
  "tool": "INSPECT_RAW",
  "target": "current_pool",
  "instruction": "answer_query_related_visual_details"
}
```

#### 实现

```python
def tool_inspect_raw(pool, query, vlm):
    observations = []
    for m in pool:
        if m.get("raw_pointer") is None:
            continue
        obs = vlm.inspect(
            image_path=m["raw_pointer"],
            instruction=f"Answer visual details relevant to this query: {query}"
        )
        observations.append({
            "memory_id": m["memory_id"],
            "visual_observation": obs
        })
    return observations
```

---

### 4.7 `STOP`

结束检索，将当前 evidence package 交给 answer model。

#### Schema

```json
{
  "tool": "STOP"
}
```

#### 实现

```python
def tool_stop(evidence):
    return evidence
```

---

## 5. Tool Trajectory 示例

### 5.1 Query：我上次发的图片是关于什么的？

模型不使用 `SELECT_LAST_IMAGE`，而是输出通用 tool 组合：

```json
[
  {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
  {"tool": "FILTER", "field": "author", "op": "eq", "value": "user"},
  {"tool": "SORT", "field": "timestamp", "order": "desc"},
  {"tool": "TOPK", "k": 1},
  {"tool": "READ", "fields": ["summary", "ocr", "timestamp"]},
  {"tool": "STOP"}
]
```

---

### 5.2 Query：我之前说我的研究方向是什么？

```json
[
  {"tool": "FILTER", "field": "modality", "op": "eq", "value": "text"},
  {"tool": "RETRIEVE", "method": "hybrid", "top_k": 5},
  {"tool": "READ", "fields": ["summary", "content", "timestamp"]},
  {"tool": "STOP"}
]
```

---

### 5.3 Query：我之前发的收据金额是多少？

```json
[
  {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
  {"tool": "RETRIEVE", "method": "hybrid", "top_k": 5},
  {"tool": "READ", "fields": ["summary", "ocr", "timestamp", "raw_pointer"]},
  {"tool": "STOP"}
]
```

如果 OCR 不足，学生也可以输出：

```json
[
  {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
  {"tool": "RETRIEVE", "method": "hybrid", "top_k": 3},
  {"tool": "INSPECT_RAW", "target": "current_pool", "instruction": "answer_query_related_visual_details"},
  {"tool": "STOP"}
]
```

---

### 5.4 Query：刚才那个截图左上角是什么按钮？

```json
[
  {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
  {"tool": "FILTER", "field": "source_type", "op": "eq", "value": "screenshot"},
  {"tool": "FILTER", "field": "author", "op": "eq", "value": "user"},
  {"tool": "SORT", "field": "timestamp", "order": "desc"},
  {"tool": "TOPK", "k": 1},
  {"tool": "INSPECT_RAW", "target": "current_pool", "instruction": "answer_query_related_visual_details"},
  {"tool": "STOP"}
]
```

---

## 6. Student Policy Model

### 6.1 输入

学生只看 query 和 tool schema。

```text
You are a multimodal memory retrieval planner.
Given a user query, output a sequence of executable tool calls.
You cannot see the memory index.
You must not output memory IDs.
You must not create a new search query using unknown answer words.
Use only the allowed tools.

User query:
{query}

Allowed tools:
FILTER, SORT, TOPK, RETRIEVE, READ, INSPECT_RAW, STOP
```

### 6.2 输出

学生一次性输出完整 tool trajectory。

```json
[
  {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
  {"tool": "FILTER", "field": "author", "op": "eq", "value": "user"},
  {"tool": "SORT", "field": "timestamp", "order": "desc"},
  {"tool": "TOPK", "k": 1},
  {"tool": "READ", "fields": ["summary", "ocr", "timestamp"]},
  {"tool": "STOP"}
]
```

### 6.3 模型规模建议

baseline 可以使用：

```text
1.5B manager：轻量对照
3B / 4B manager：主 baseline
7B / 8B manager：强 baseline
```

---

## 7. Teacher Hindsight Correction

### 7.1 教师输入

教师不看完整 memory index。教师看到：

```text
user query
gold answer
student trajectory
student answer
whether student answer is correct
allowed tool schema
```

示例 prompt：

```text
You are a hindsight teacher for multimodal memory retrieval.

You are given:
1. User query.
2. Gold answer.
3. Student tool trajectory.
4. Student answer and correctness.

You cannot see the memory index.
You must not output memory IDs.
You must not use answer-specific words as a new search query.
Your task is to output a corrected abstract tool trajectory.
Use only the allowed tools.
Prefer simple and general trajectories.

Allowed tools:
FILTER, SORT, TOPK, RETRIEVE, READ, INSPECT_RAW, STOP
```

### 7.2 教师输出

教师输出 corrected trajectory：

```json
[
  {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
  {"tool": "FILTER", "field": "author", "op": "eq", "value": "user"},
  {"tool": "SORT", "field": "timestamp", "order": "desc"},
  {"tool": "TOPK", "k": 1},
  {"tool": "READ", "fields": ["summary", "ocr", "timestamp"]},
  {"tool": "STOP"}
]
```

---

## 8. OPD 训练流程

### 8.1 数据样本

每个样本包含：

```json
{
  "sample_id": "s_001",
  "query": "我上次发的图片是关于什么的？",
  "gold_answer": "一张 ICLR 2026 注册收据。",
  "memory_store": "hidden_memory_store_for_this_session"
}
```

训练时 `memory_store` 只给 executor 使用，不给 student / teacher。

---

### 8.2 单轮 OPD

```python
def opd_round(dataset, student, teacher, executor, answer_model):
    sft_data = []

    for sample in dataset:
        query = sample["query"]
        gold_answer = sample["gold_answer"]
        memory_store = sample["memory_store"]

        # 1. Student outputs query-only trajectory
        student_trace = student.generate_trace(query)

        # 2. Executor runs trace over hidden memory store
        evidence = executor.run(
            trace=student_trace,
            query=query,
            memory_store=memory_store
        )

        # 3. Answer model answers with retrieved evidence
        student_answer = answer_model.answer(
            query=query,
            evidence=evidence
        )

        correct = evaluate(student_answer, gold_answer)

        # 4. Teacher outputs corrected abstract trajectory
        teacher_trace = teacher.correct(
            query=query,
            gold_answer=gold_answer,
            student_trace=student_trace,
            student_answer=student_answer,
            correct=correct
        )

        # 5. Add SFT pair: query -> corrected trace
        sft_data.append({
            "input": build_student_prompt(query),
            "target": teacher_trace
        })

    return sft_data
```

---

### 8.3 多轮 OPD

```python
def train_opd(dataset, student, teacher, executor, answer_model, num_rounds=3):
    all_sft_data = []

    for r in range(num_rounds):
        round_data = opd_round(
            dataset=dataset,
            student=student,
            teacher=teacher,
            executor=executor,
            answer_model=answer_model
        )

        all_sft_data.extend(round_data)
        student = sft_train(student, all_sft_data)

    return student
```

Baseline 中建议：

```text
num_rounds = 2 或 3
max_trace_length = 6
loss = standard cross entropy
no RL reward
no preference optimization
```

---

## 9. Executor 实现

Executor 是唯一能访问 memory store 的模块。

```python
class ToolExecutor:
    def __init__(self, bm25, dense_encoder, vlm=None):
        self.bm25 = bm25
        self.dense_encoder = dense_encoder
        self.vlm = vlm

    def run(self, trace, query, memory_store):
        pool = list(memory_store)
        evidence = []

        for action in trace:
            tool = action["tool"]

            if tool == "FILTER":
                pool = tool_filter(
                    pool,
                    field=action["field"],
                    op=action["op"],
                    value=action["value"]
                )

            elif tool == "SORT":
                pool = tool_sort(
                    pool,
                    field=action["field"],
                    order=action["order"]
                )

            elif tool == "TOPK":
                pool = tool_topk(pool, k=action["k"])

            elif tool == "RETRIEVE":
                pool = tool_retrieve(
                    pool,
                    query=query,
                    method=action.get("method", "hybrid"),
                    top_k=action.get("top_k", 5)
                )

            elif tool == "READ":
                evidence.extend(tool_read(pool, fields=action["fields"]))

            elif tool == "INSPECT_RAW":
                evidence.extend(tool_inspect_raw(pool, query=query, vlm=self.vlm))

            elif tool == "STOP":
                break

        return evidence
```

---

## 10. 数据构造

### 10.1 可用数据来源

可使用以下类型数据：

```text
Mem-Gallery：通用长期图文对话 memory
MemEye：视觉细节、图像依赖、状态变化
SMMBench：多来源 memory、截图、文档、表格、action prediction
Synthetic temporal-reference data：专门构造“上次/刚才/第一张/最后一张图”类问题
```

---

### 10.2 重点合成数据

因为 baseline 关注 query-only trajectory planning，建议重点构造时间指代和模态指代样本。

#### 模板

```text
我上次发的图片是关于什么的？
我刚才发的截图里有什么？
我最后一次上传的图显示了什么？
我之前发的收据金额是多少？
我第一张图和最后一张图分别是什么？
我刚才那个界面左上角是什么按钮？
```

#### Gold answer

由隐藏 memory store 中对应 memory 的 caption / OCR / raw image inspection 生成。

#### Teacher trajectory

不依赖具体 memory_id，只生成抽象 tool plan。

---

## 11. Baseline 对照方法

建议比较：

| 方法 | 说明 |
|---|---|
| Dense RAG | 所有 memory 文本化后直接向量检索 |
| Rule-Routed RAG | 简单关键词规则选择 text / image / hybrid |
| Always Hybrid | 每次同时查 text 和 image |
| Offline Distillation | teacher 直接基于 query + gold answer 生成轨迹，不使用 student rollout |
| OPD-MM Baseline | 学生 rollout 后，teacher hindsight correction，再 SFT |
| OPD-MM + Optimization | 未来完整方法，可加入 verifier / Best-of-N / cost-aware reward |

---

## 12. 评价指标

建议报告：

```text
Answer Accuracy
Evidence Recall@K
Modality Routing Accuracy
Temporal Reference Accuracy
Raw Image Invocation Rate
Average Tool Steps
Average Retrieval Cost
Wrong Modality Rate
```

其中最重要的是：

| 指标 | 含义 |
|---|---|
| Evidence Recall@K | executor 根据学生轨迹是否能找到正确 memory |
| Temporal Reference Accuracy | “上次/刚才/最后一次”等时间指代是否解析正确 |
| Modality Routing Accuracy | 是否选择了正确模态路径 |
| Raw Image Invocation Rate | 是否过度调用原图检查 |
| Average Tool Steps | 检索计划是否过长 |

---

## 13. 论文中可以这样描述

英文版：

```text
We implement a lightweight OPD baseline for multimodal memory retrieval. The student policy model observes only the user query and outputs an abstract trajectory of executable tool calls. The policy does not access the full memory index or candidate memories. The executor applies the predicted tools to a hidden memory store to retrieve an evidence package, which is then passed to a fixed answer model. During training, a teacher model receives the query, gold answer, student trajectory, and student answer correctness, but not the full memory index. The teacher provides a corrected abstract tool trajectory under the same constrained tool schema. The corrected trajectories are used as supervision to fine-tune the student with standard cross-entropy loss.
```

中文版：

```text
我们实现了一个轻量 OPD 多模态 memory 检索 baseline。学生 policy model 只观察用户 query，并输出一串抽象的可执行 tool call。学生不能访问完整 memory index 或候选 memory。Executor 在隐藏 memory store 上执行这些工具，得到 evidence package，并将其交给固定 answer model。训练时，教师模型可以看到 query、gold answer、学生轨迹以及学生答案是否正确，但不能看到完整 memory index。教师在相同的受限工具空间中输出 corrected abstract trajectory，并用这些轨迹通过标准交叉熵损失微调学生。
```

---

## 14. Baseline 的能力边界

### 能实现

```text
query-only 检索计划生成
通用 FILTER / SORT / TOPK / RETRIEVE / READ tool 组合
on-policy student rollout
teacher hindsight correction
SFT 蒸馏
基本图文 memory 检索
基本时间指代处理
```

### 暂不实现

```text
不让 policy 看 memory index
不让 teacher 选择 memory_id
不设计 SELECT_LAST_IMAGE 这类强规则动作
不做复杂 trajectory verification
不做反事实扰动
不做 RL reward
不做 memory update/delete
不做视觉 patch 检索
```

---

## 15. 最小版本总结

最终 baseline 可以简化为一句话：

> Student 和 teacher 都不访问完整 memory index，只根据 query 生成由通用 tool 组成的抽象检索轨迹；executor 在隐藏 memory store 上执行轨迹并返回 evidence；训练时 teacher 基于 gold answer 对学生轨迹进行 hindsight correction，学生通过 SFT 学习 corrected trajectory。

最小 tool set：

```text
FILTER
SORT
TOPK
RETRIEVE
READ
INSPECT_RAW
STOP
```

最小训练流程：

```text
student rollout
→ execute tools
→ answer
→ teacher correction
→ SFT
→ repeat 2~3 rounds
```

