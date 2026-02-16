# RAPTOR Benchmarks

Evaluation on real RAG datasets — including the three from the [RAPTOR ICLR 2024 paper](https://arxiv.org/abs/2401.18059).

## Datasets

| Dataset | What it Tests | Metric | RAPTOR Paper Result | Script |
|---|---|---|---|---|
| **QuALITY** | Long-document MC QA (~5K token articles) | Accuracy | 82.6% (GPT-4) | `run_quality.py` |
| **QASPER** | QA on scientific NLP papers | Token F1 | 55.7% (GPT-4) | `run_qasper.py` |
| **NarrativeQA** | Book/script comprehension | ROUGE-L, METEOR | 30.87% ROUGE-L | `run_narrativeqa.py` |
| **FRAMES** | Multi-hop QA (Wikipedia articles) | Accuracy | N/A (newer benchmark) | `run_frames.py` |

## Prerequisites

```bash
# Install with benchmark dependencies
uv sync --extra benchmarks

# Set API key
export OPENAI_API_KEY=sk-...
```

## Quick Start

```bash
# QuALITY — small run (2 docs, 5 questions)
python -m benchmarks.run_quality --max_docs 2 --max_questions 5

# QASPER — full validation set
python -m benchmarks.run_qasper

# NarrativeQA — using summaries (cheaper)
python -m benchmarks.run_narrativeqa --use_summaries

# FRAMES — 10 questions
python -m benchmarks.run_frames --max_questions 10

# Collapsed tree + cross-encoder reranking
python -m benchmarks.run_quality --reranker cross_encoder --max_docs 2 --max_questions 5

# Tree traversal mode (no collapsed tree)
python -m benchmarks.run_quality --collapse_tree false --max_docs 2 --max_questions 5

# Collapsed tree + LiteLLM reranking (Cohere)
python -m benchmarks.run_quality --reranker litellm --max_docs 2 --max_questions 5
```

## Cost Estimates (gpt-4.1-nano)

gpt-4.1-nano pricing: $0.10/1M input, $0.40/1M output.

| Dataset | Questions | Est. Cost |
|---|---|---|
| QuALITY (validation) | ~230 | ~$3–5 |
| QASPER (test) | ~1,500 | ~$5–10 |
| NarrativeQA (summaries) | ~10K | ~$3–8 |
| FRAMES (test) | ~824 | ~$5–15 |

Costs scale linearly with `--max_questions`. Use `--max_docs 2 --max_questions 5` for smoke tests.

## CLI Reference

All benchmarks share these arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `gpt-4.1-nano` | OpenAI model for summarization and QA |
| `--max_docs` | all | Limit number of documents |
| `--max_questions` | all | Limit number of questions |
| `--top_k` | 5 | Top-k nodes for retrieval |
| `--max_tokens` | 100 | Max tokens per chunk (tree building) |
| `--num_layers` | 5 | Number of tree layers |
| `--cache_dir` | `benchmarks/.cache/` | Tree cache directory |
| `--output` | per-benchmark | Output JSON path |
| `--no_resume` | false | Don't resume from partial results |
| `--reranker` | none | Reranker after retrieval: `cross_encoder` or `litellm` |
| `--reranker_model` | auto | Model for reranker (see defaults below) |
| `--collapse_tree` | `true` | `true` for collapsed tree, `false` for tree traversal |

### Benchmark-Specific Arguments

**QuALITY:**
- `--split` (default: `validation`) — Dataset split
- `--hard_only` — Only evaluate QuALITY-HARD questions

**NarrativeQA:**
- `--use_summaries` (default: true) — Use document summaries
- `--use_full_text` — Use full document text (much more expensive)

**Reranker defaults:**
- `cross_encoder`: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, requires `raptor-rag[reranker]`)
- `litellm`: `cohere/rerank-english-v3.0` (API-based, requires Cohere key)

## Caching

Trees are cached in `benchmarks/.cache/<benchmark>/` (gitignored). Cache key includes doc_id, model, num_layers, and max_tokens. Re-running the same config skips tree building.

To clear cache: `rm -rf benchmarks/.cache/`

## Resume

Interrupted runs save progress to `*.partial.json` (gitignored). Restarting the same command skips completed questions. Use `--no_resume` to start fresh.

## Compression Utility

`run_compression.py` measures tree compression ratio on any pre-built RAPTOR tree:

```bash
python benchmarks/run_compression.py --tree path/to/tree.pkl
```

The paper reports ~72% compression ratio (summaries are ~28% of child text length).
