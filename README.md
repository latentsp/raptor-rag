<picture>
  <source media="(prefers-color-scheme: dark)" srcset="raptor_dark.png">
  <img alt="RAPTOR" src="raptor.jpg">
</picture>

# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

[![CI](https://github.com/latentsp/raptor-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/latentsp/raptor-rag/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/raptor-rag.svg)](https://pypi.org/project/raptor-rag/)
[![Python versions](https://img.shields.io/pypi/pyversions/raptor-rag.svg)](https://pypi.org/project/raptor-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2401.18059)

**RAPTOR** builds hierarchical tree structures from documents for context-aware RAG retrieval. Instead of retrieving flat chunks, RAPTOR recursively clusters and summarizes text, enabling retrieval at multiple granularity levels — from specific details to high-level themes.

Based on the [ICLR 2024 paper](https://arxiv.org/abs/2401.18059) by Sarthi et al., maintained by [Latent](https://latent.space).

### What's New in v0.3

- **LiteLLM as default provider layer** — use 100+ LLM providers (OpenAI, Anthropic, Cohere, local models, etc.)
- **Cross-encoder and LiteLLM reranking** — improve retrieval precision with a reranking stage
- **FAISS retriever** — flat index retrieval for large-scale deployments
- **Pluggable text splitting** — bring your own chunking strategy
- **LangChain + LlamaIndex integrations** — separate PyPI packages (`langchain-raptor-rag`, `llama-index-raptor-rag`)
- **Benchmarks suite** — QuALITY, NarrativeQA, QASPER, FRAMES evaluation harness
- **Modern Python packaging** — `pyproject.toml`, optional dependency groups, PEP 8 module naming

## Installation

```bash
pip install raptor-rag

# With cross-encoder reranking
pip install raptor-rag[reranker]

# With HuggingFace embeddings
pip install raptor-rag[huggingface]

# Everything
pip install raptor-rag[all]

# Development
pip install raptor-rag[dev]
```

Requires Python 3.10+.

## Quick Start

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"  # LiteLLM reads this natively for OpenAI models

from raptor import RetrievalAugmentation

ra = RetrievalAugmentation()
ra.add_documents(open("document.txt").read())
answer = ra.answer_question("What is the main argument?")
```

## Features

### Provider-Agnostic with LiteLLM

Use any LLM provider — OpenAI, Anthropic, Cohere, local models, and more:

```python
from raptor import RetrievalAugmentationConfig, LiteLLMSummarizationModel, LiteLLMQAModel

config = RetrievalAugmentationConfig(
    summarization_model=LiteLLMSummarizationModel(model="anthropic/claude-3-haiku-20240307"),
    qa_model=LiteLLMQAModel(model="anthropic/claude-3-5-sonnet-20241022"),
)
ra = RetrievalAugmentation(config=config)
```

### Cross-Encoder Reranking

Improve retrieval precision with a reranking stage:

```python
from raptor import RetrievalAugmentationConfig, CrossEncoderReRanker

config = RetrievalAugmentationConfig(
    tr_reranker=CrossEncoderReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
)
```

### HuggingFace Embeddings

Use any HuggingFace embedding model:

```python
from raptor import RetrievalAugmentationConfig, HuggingFaceEmbeddingModel

config = RetrievalAugmentationConfig(
    embedding_model=HuggingFaceEmbeddingModel(model_name="BAAI/bge-small-en-v1.5"),
)
```

### Pluggable Text Splitting

Bring your own text splitting strategy:

```python
from raptor import BaseTextSplitter, RetrievalAugmentationConfig

class MyTextSplitter(BaseTextSplitter):
    def split_text(self, text, tokenizer, max_tokens):
        # Your custom splitting logic
        return chunks

config = RetrievalAugmentationConfig(tb_text_splitter=MyTextSplitter())
```

### Configurable Prompts

Customize summarization and QA prompts:

```python
from raptor import LiteLLMSummarizationModel, LiteLLMQAModel

summarizer = LiteLLMSummarizationModel(
    model="gpt-4o-mini",
    system_prompt="You are an expert summarizer.",
    user_prompt_template="Summarize the key points: {context}",
)

qa = LiteLLMQAModel(
    model="gpt-4o",
    system_prompt="Answer questions precisely.",
    user_prompt_template="Context: {context}\n\nQuestion: {question}\nAnswer:",
)
```

## Framework Integrations

### LangChain

```bash
pip install langchain-raptor-rag
```

```python
from raptor import RetrievalAugmentation
from langchain_raptor_rag import RaptorRetriever

ra = RetrievalAugmentation()
ra.add_documents(text)
retriever = RaptorRetriever(ra=ra)
docs = retriever.invoke("What happened to Cinderella?")
```

### LlamaIndex

```bash
pip install llama-index-raptor-rag
```

```python
from raptor import RetrievalAugmentation
from llama_index_raptor_rag import RaptorRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

ra = RetrievalAugmentation()
ra.add_documents(text)
retriever = RaptorRetriever(ra=ra)
query_engine = RetrieverQueryEngine.from_args(retriever)
response = query_engine.query("What happened to Cinderella?")
```

## Extending RAPTOR

RAPTOR is designed to be modular. Extend any component by subclassing:

```python
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, BaseReRanker

class MySummarizer(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return "your summary"

class MyQA(BaseQAModel):
    def answer_question(self, context, question):
        return "your answer"

class MyEmbedding(BaseEmbeddingModel):
    def create_embedding(self, text):
        return [0.0] * 768

class MyReRanker(BaseReRanker):
    def rerank(self, query, nodes, top_k=5):
        return sorted(nodes, key=lambda n: relevance(query, n))[:top_k]
```

## Saving and Loading Trees

```python
# Save
ra.save("my_tree.pkl")

# Load
ra = RetrievalAugmentation(tree="my_tree.pkl")
answer = ra.answer_question("Your question?")
```

## Benchmarks

The `benchmarks/` directory contains evaluation harnesses for QuALITY, NarrativeQA, QASPER, and FRAMES datasets. See `benchmarks/README.md` for setup and usage.

## Development

```bash
git clone https://github.com/latentsp/raptor-rag.git
cd raptor-rag
devbox shell
uv sync --extra dev

# Run tests (no API keys needed)
uv run pytest -m "not integration"

# Run integration tests (needs OPENAI_API_KEY)
uv run pytest -m integration

# Lint
uv run ruff check raptor/ tests/ integrations/

# Format
uv run ruff format raptor/ tests/ integrations/

# Test framework integrations
cd integrations/langchain-raptor-rag && uv run pytest
cd integrations/llama-index-raptor-rag && uv run pytest
```

## Contributing

Contributions welcome! Areas of interest:
- New model integrations
- Alternative clustering algorithms
- Benchmark results
- Documentation improvements

## License

MIT License. See [LICENSE.txt](LICENSE.txt).

## Citation

```bibtex
@inproceedings{sarthi2024raptor,
    title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
    author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```

## Credits

Original paper and implementation by Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning.

Modernized and maintained by [Latent](https://www.reachlatent.com).
