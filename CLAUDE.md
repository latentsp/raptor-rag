# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) — a Python library that builds hierarchical tree structures from documents for context-aware RAG retrieval. Based on the [ICLR 2024 paper](https://arxiv.org/abs/2401.18059). Maintained by Latent.

## Setup & Installation

```bash
# Install with uv (required — never use pip directly)
devbox run uv sync --extra dev
```

**Important:** Always manage dependencies via `pyproject.toml` and `uv sync`. Never use `pip install` or `uv pip install` directly. To add a new dependency, add it to the appropriate group in `pyproject.toml` then run `devbox run uv sync --extra dev`.

Requires Python 3.10+. Core package deps: numpy, tiktoken, scikit-learn, umap-learn, scipy, tenacity, litellm. Heavy deps are optional:
- `[reranker]` — sentence-transformers cross-encoder
- `[huggingface]` — sentence-transformers, torch, transformers
- `[faiss]` — faiss-cpu
- `[all]` — everything
- `[dev]` — all + pytest, pytest-cov, ruff

## Development Commands

```bash
# Run tests (no API keys needed)
uv run pytest -m "not integration"

# Run integration tests (needs OPENAI_API_KEY)
uv run pytest -m integration

# Lint
uv run ruff check raptor/ tests/ integrations/

# Format
uv run ruff format raptor/ tests/ integrations/

# Build
uv build

# Test integration sub-packages
cd integrations/langchain-raptor-rag && uv run pytest
cd integrations/llama-index-raptor-rag && uv run pytest
```

## Architecture

The library lives in `raptor/`. The main entry point is `RetrievalAugmentation` which composes a `TreeBuilder` and `TreeRetriever`.

### Core Pipeline

1. **Text Splitting** (`utils.py`) — Pluggable via `BaseTextSplitter`. Default splits into ~100-token chunks.
2. **Embedding** (`embedding_models.py`) — Pluggable. LiteLLM (default), SBert, HuggingFace, or custom.
3. **Tree Construction** (`cluster_tree_builder.py`) — UMAP + GMM clustering, LLM summarization, recursive.
4. **Retrieval** (`tree_retriever.py`) — Collapsed tree or tree traversal. Optional reranking.
5. **Answer Generation** — Pluggable QA model.

### Key Abstractions (all in `raptor/`)

| Base Class | Purpose | Implementations |
|---|---|---|
| `BaseEmbeddingModel` | Text → vector | `LiteLLMEmbeddingModel`, `SBertEmbeddingModel`, `HuggingFaceEmbeddingModel` |
| `BaseSummarizationModel` | Cluster text → summary | `LiteLLMSummarizationModel` |
| `BaseQAModel` | Context + question → answer | `LiteLLMQAModel`, `UnifiedQAModel` |
| `BaseReRanker` | Rerank retrieved nodes | `CrossEncoderReRanker`, `LiteLLMReRanker` |
| `BaseTextSplitter` | Text → chunks | `DefaultTextSplitter` |
| `BaseRetriever` | Tree → relevant nodes | `TreeRetriever`, `FaissRetriever` |

### Config Classes

- `RetrievalAugmentationConfig` — top-level config, threads params to builder/retriever
- `TreeBuilderConfig` / `ClusterTreeConfig` — tree construction params
- `TreeRetrieverConfig` — retrieval params (including `reranker=`)

### Project Structure

```
raptor/                        # Core library
├── __init__.py                # All public exports
├── embedding_models.py        # LiteLLM, SBert, HuggingFace embeddings
├── summarization_models.py    # LiteLLM summarization
├── qa_models.py               # LiteLLM, UnifiedQA
├── reranker.py                # CrossEncoder, LiteLLM reranking
├── tree_builder.py            # Base TreeBuilder + TreeBuilderConfig
├── cluster_tree_builder.py    # Cluster-based tree construction
├── cluster_utils.py           # UMAP + GMM clustering
├── tree_retriever.py          # Collapsed tree + tree traversal retrieval
├── tree_structures.py         # Node, Tree data classes
├── utils.py                   # Text splitting, distance functions
├── retrievers.py              # BaseRetriever ABC
├── faiss_retriever.py         # FAISS-based flat retriever
└── retrieval_augmentation.py  # Main entry point

tests/                      # Test suite (pytest)
integrations/               # Framework sub-packages
├── langchain-raptor-rag/   # LangChain integration (separate PyPI package)
└── llama-index-raptor-rag/ # LlamaIndex integration (separate PyPI package)
notebooks/                  # Jupyter notebooks
benchmarks/                 # MTEB/BEIR benchmarks
docs/                       # Blog post draft
data/                       # Sample data (cinderella pickle is gitignored)
.github/workflows/          # CI (ruff + pytest) and publish (PyPI via OIDC)
```

### Default Parameters

| Parameter | Default | Source |
|---|---|---|
| `max_tokens` (chunk size) | 100 tokens | `TreeBuilderConfig` |
| `num_layers` | 5 | `TreeBuilderConfig` |
| `threshold` (retrieval) | 0.5 | `TreeBuilderConfig` |
| `top_k` | 5 | `TreeBuilderConfig` |
| `selection_mode` | `"top_k"` | `TreeBuilderConfig` |
| `summarization_length` | 100 tokens | `TreeBuilderConfig` |
| Default summarization model | `gpt-4o-mini` | `LiteLLMSummarizationModel` |
| Default QA model | `gpt-4o-mini` | `LiteLLMQAModel` |
| Default embedding model | `text-embedding-ada-002` | `LiteLLMEmbeddingModel` |

### Lazy Imports for Heavy Dependencies

LiteLLM is a core dependency imported at module level. Heavy optional deps (sentence-transformers, torch, faiss) use lazy imports via `_compat._lazy_import` so the core package works without them installed.

## Paper Context (ICLR 2024)

RAPTOR addresses a key limitation of traditional RAG: retrieving only short, contiguous chunks loses large-scale discourse structure. By recursively summarizing and clustering, RAPTOR builds a tree where leaf nodes are original text chunks and higher layers are progressively more abstract summaries.

**Key results:** QuALITY 82.6% accuracy (+20% over SOTA), NarrativeQA 19.1 METEOR, ~72% compression ratio, ~4% hallucination rate.
