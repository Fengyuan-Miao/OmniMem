# OmniMem

OmniMem is a standalone multimodal long-term memory research project. It
stores complete dialogue memories with their original text and optional image
pointers, builds persistent SQLite and FAISS indexes, and evaluates retrieval
and answer generation on Mem-Gallery.

The project no longer requires the surrounding SimpleMem repository layout.
Mem-Gallery data and model weights remain external assets and can be supplied
through command-line arguments or environment variables.

## Implemented Methods

### GME unified-entry memory

Each dialogue turn is one memory entry. GME-Qwen2-VL encodes the entry's text
and first valid image into one normalized multimodal embedding. Retrieval uses
a single FAISS index and returns the full original entry, including all text
and image pointers.

This is the primary MuRAG-style implementation:

```text
text + optional image -> one entry embedding -> FAISS top-k
                                            -> original text + images
```

### SigLIP + MiniLM dual-encoder memory

Each turn is still one unified memory record, while the index contains:

- MiniLM text embeddings
- SigLIP image embeddings
- BM25 text statistics

The three ranked lists are fused with reciprocal-rank fusion. Retrieved
memories are organized into chronological evidence groups before answer
generation.

### Topic memory

An experimental topic-gated extension of the dual-encoder method. It uses an
LLM-maintained topic index to narrow retrieval before multimodal search.

### SVI compatibility experiment

The earlier Structured Visual Index implementation remains available for
ablation. Its Mem-Gallery runner depends on the external `omni_memory` package
from OmniSimpleMem; the GME, dual-encoder, and topic methods do not.

## Installation

Create an environment with the PyTorch build appropriate for the machine, then
install OmniMem:

```bash
cd OmniMem
pip install -e ".[all]"
```

For a smaller installation:

```bash
pip install -e ".[gme]"
pip install -e ".[dual]"
```

The core package requires Python 3.10+, NumPy, Pillow, SQLite from the Python
standard library, and FAISS.

## Mem-Gallery Data

The benchmark dataset is intentionally not copied into this repository. Point
OmniMem at an existing Mem-Gallery checkout in one of three ways:

```bash
export OMNIMEM_MEMGALLERY_DIR=/path/to/Mem-Gallery
```

```bash
omnimem benchmark memgallery gme \
  --data-dir /path/to/Mem-Gallery \
  --scenario Academic_Animal_Pet_Research_Life
```

Or place a checkout or symlink at:

```text
OmniMem/benchmarks/Mem-Gallery/
```

The resolved directory must contain `data/dialog/*.json` and the referenced
images.

## Model Configuration

Model defaults are portable. OmniMem first checks the environment variables
below, then known local paths on the current research machine, and finally the
public Hugging Face model identifiers.

| Variable | Default public model |
| --- | --- |
| `OMNIMEM_GME_MODEL` | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` |
| `OMNIMEM_MINILM_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `OMNIMEM_SIGLIP_MODEL` | `google/siglip-base-patch16-384` |

Every runner also accepts the corresponding `--gme-model`, `--text-model`, or
`--vision-model` argument.

## Running Mem-Gallery

Run a small GME smoke test:

```bash
omnimem benchmark memgallery gme \
  --scenario Academic_Animal_Pet_Research_Life \
  --max-sessions 2 \
  --max-questions 5 \
  --gme-device cuda:1
```

Run the three-route dual-encoder method:

```bash
omnimem benchmark memgallery dual \
  --scenario Academic_Animal_Pet_Research_Life \
  --max-sessions 2 \
  --max-questions 5 \
  --text-device cpu \
  --vision-device cpu
```

Run all scenarios:

```bash
omnimem benchmark memgallery gme --all-scenarios --gme-device cuda:1
```

Equivalent direct entry points are installed:

```text
omnimem-memgallery-gme
omnimem-memgallery-dual
omnimem-memgallery-topic
omnimem-memgallery-svi
```

Use `--help` on a runner to see all retrieval, context, answer-model, and judge
options.

## Answer And Judge Services

The runners use OpenAI-compatible HTTP endpoints. Current defaults are:

```text
answer VLM: http://127.0.0.1:11435/v1
LLM judge:  http://127.0.0.1:11436/v1
```

They can point to Ollama or any compatible service:

```bash
omnimem benchmark memgallery gme \
  --base-url http://host:port/v1 \
  --vlm-model your-vlm \
  --judge-base-url http://host:port/v1 \
  --judge-model your-judge
```

Use `--judge-mode off` when only retrieval and debug metrics are needed.
OmniMem never starts or stops model services itself.

## Outputs

Runs are written under `runs/` by default:

```text
runs/
  memgallery_gme/<scenario>/<timestamp>_gme_qwen2vl_unified/
  memgallery_dual_encoder/<scenario>/<timestamp>_siglip_minilm/
```

Each run records:

- `config.json`
- `predictions.jsonl`
- `metrics.json`
- SQLite memory records and embedding blobs
- persisted FAISS indexes

`judge_accuracy` is the primary answer metric when the LLM judge is enabled.
Predictions also retain EM, token F1, image recall, retrieved entries, answer
images, latency, and judge diagnostics.

## Repository Layout

```text
omnimem/                    project configuration and CLI
gme_memory/                 unified multimodal entry memory
dual_encoder_memory/        MiniLM, SigLIP, BM25, RRF, evidence organization
topic_memory/               topic-gated experimental retrieval
svi_omnimem/                legacy SVI compatibility experiment
memgallery_*_pipeline.py    Mem-Gallery runners
ollama/                     local Ollama Modelfile examples
tests/                      storage, retrieval, evidence, and packaging tests
```

## Development

```bash
pip install -e ".[all,dev]"
python -m pytest
python -m compileall omnimem gme_memory dual_encoder_memory topic_memory
```

Generated runs, indexes, SQLite files, local datasets, and model weights are
excluded by the project-level `.gitignore`.
