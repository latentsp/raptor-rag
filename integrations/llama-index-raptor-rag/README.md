# llama-index-raptor-rag

LlamaIndex retriever integration for [RAPTOR RAG](https://github.com/latentsp/raptor-rag). Wraps RAPTOR's hierarchical tree retrieval as a LlamaIndex `BaseRetriever`, returning `NodeWithScore` objects with node metadata and rank-based scores.

## Installation

```bash
pip install llama-index-raptor-rag
```

This installs `raptor-rag` and `llama-index-core` as dependencies.

## Quick Start

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"

from raptor import RetrievalAugmentation
from llama_index_raptor_rag import RaptorRetriever

# Build a RAPTOR tree
ra = RetrievalAugmentation()
ra.add_documents(open("document.txt").read())

# Wrap as a LlamaIndex retriever
retriever = RaptorRetriever(ra=ra)
results = retriever.retrieve("What is the main argument?")

for r in results:
    print(f"[{r.score:.2f}] Layer {r.node.metadata['layer_number']}: {r.node.text[:80]}...")
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `ra` | *required* | A `RetrievalAugmentation` instance with a built tree |
| `top_k` | `10` | Maximum number of nodes to retrieve |
| `max_tokens` | `3500` | Maximum total tokens across retrieved nodes |
| `collapse_tree` | `True` | Use collapsed tree retrieval (vs. tree traversal) |

## Usage with RetrieverQueryEngine

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

query_engine = RetrieverQueryEngine.from_args(
    retriever,
    llm=OpenAI(model="gpt-4o-mini"),
)

response = query_engine.query("What happened to Cinderella?")
print(response)
```

## Scores

Results are scored by retrieval rank: the first result gets `1.0`, and scores decrease linearly. This provides a useful ordering signal even though RAPTOR's native retrieval doesn't produce calibrated scores.

## Metadata

Each returned `NodeWithScore` includes:

- `node_index` — the node's position in the RAPTOR tree
- `layer_number` — `0` for leaf (original text), higher for summary layers

## Development

```bash
cd integrations/llama-index-raptor-rag
pip install -e ".[dev]"
pytest
```
