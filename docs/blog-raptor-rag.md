# Beyond Flat Chunks: How RAPTOR's Tree Structure Transforms RAG

## The Problem with Flat-Chunk RAG

Traditional RAG systems split documents into fixed-size chunks and retrieve the most similar ones to a query. This works well for factoid questions, but falls apart when answers require understanding across multiple sections of a document — synthesizing themes, tracking character arcs, or answering questions that span chapters.

The fundamental issue: flat chunking discards the hierarchical structure of text. A 100-token chunk can't capture the thesis of a 50-page paper.

## How RAPTOR Solves This (ICLR 2024)

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a tree from your documents:

1. **Chunk** the text into leaf nodes (~100 tokens each)
2. **Embed** each chunk using a sentence embedding model
3. **Cluster** similar chunks using UMAP dimensionality reduction + Gaussian Mixture Models
4. **Summarize** each cluster with an LLM, creating parent nodes
5. **Repeat** — cluster and summarize the summaries, building progressively more abstract layers
6. **Retrieve** across all layers simultaneously (collapsed tree) or top-down (tree traversal)

The key insight: retrieval operates at multiple granularity levels. A query about a specific detail matches leaf nodes. A query about themes matches higher-level summaries. In practice, 18-57% of retrieved nodes come from summary layers.

**Results from the paper:**
- QuALITY benchmark: 82.6% accuracy (+20% absolute over prior SOTA)
- NarrativeQA: 19.1 METEOR (new SOTA)
- ~72% compression ratio (summaries are ~28% of input)

## What Latent Improved

We forked the original Stanford implementation and modernized it into a production-ready Python package:

### Provider-Agnostic via LiteLLM

The original code was locked to OpenAI. We added LiteLLM support, giving you access to 100+ providers with a single model string:

```python
from raptor import RetrievalAugmentationConfig, LiteLLMSummarizationModel, LiteLLMQAModel

config = RetrievalAugmentationConfig(
    summarization_model=LiteLLMSummarizationModel(model="anthropic/claude-3-haiku-20240307"),
    qa_model=LiteLLMQAModel(model="anthropic/claude-3-5-sonnet-20241022"),
)
```

### Cross-Encoder Reranking

Bi-encoder retrieval (cosine similarity) is fast but imprecise. We added a reranking stage that re-scores retrieved nodes with a cross-encoder for higher precision:

```python
from raptor import RetrievalAugmentationConfig, CrossEncoderReRanker

config = RetrievalAugmentationConfig(
    tr_reranker=CrossEncoderReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
)
```

### Pluggable Architecture

Every component is now swappable via clean base classes:
- `BaseEmbeddingModel` — bring your own embeddings (HuggingFace, OpenAI, custom)
- `BaseSummarizationModel` — any LLM for summarization
- `BaseQAModel` — any LLM for question answering
- `BaseReRanker` — cross-encoder, LiteLLM rerank API, or custom
- `BaseTextSplitter` — custom chunking strategies

### Framework Integrations

Drop RAPTOR into existing LangChain or LlamaIndex pipelines:

```python
# LangChain
from langchain_raptor_rag import RaptorRetriever
retriever = RaptorRetriever(ra=ra)
docs = retriever.invoke("What happened?")

# LlamaIndex
from llama_index_raptor_rag import RaptorRetriever
query_engine = RetrieverQueryEngine.from_args(RaptorRetriever(ra=ra))
```

### Modern Packaging

- Slim core dependencies (litellm included; no torch required by default)
- Optional dependency groups: `pip install raptor-rag[reranker]`
- Python 3.10+ with hatchling build backend
- Full test suite with pytest
- CI/CD with GitHub Actions
- Ruff for linting and formatting

## The Graph Connection

RAPTOR's soft clustering (GMM with threshold-based assignment) means nodes can belong to multiple clusters simultaneously. This creates a DAG (directed acyclic graph), not a strict tree — child nodes can have multiple parents.

This overlapping membership is a feature: it means summaries capture richer context by including nodes that are relevant to multiple topics. It's a natural stepping stone toward full graph-augmented retrieval, where relationships between concepts are explicitly modeled.

## Token Compression

RAPTOR achieves significant token compression. Instead of retrieving 10 raw chunks (potentially redundant), you can retrieve 3 leaf nodes + 2 summary nodes that cover the same ground with less redundancy. The summary layers act as a learned index over the document's semantic structure.

## Getting Started

```bash
pip install raptor-rag
```

```python
from raptor import RetrievalAugmentation

ra = RetrievalAugmentation()
ra.add_documents(open("my_document.txt").read())
answer = ra.answer_question("What is the main argument?")
```

## Contributing

RAPTOR-RAG is open source under the MIT license. We welcome contributions:
- New embedding/summarization model integrations
- Alternative clustering algorithms
- Benchmark results on additional datasets
- Documentation improvements

Repository: github.com/latentsp/raptor-rag

## Credits

Based on the ICLR 2024 paper by Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning.

Maintained by Latent Engineering Firm Ltd.
