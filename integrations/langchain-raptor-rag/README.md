# langchain-raptor-rag

LangChain retriever integration for [RAPTOR RAG](https://github.com/latentsp/raptor-rag). Wraps RAPTOR's hierarchical tree retrieval as a LangChain `BaseRetriever`, returning `Document` objects with node metadata.

## Installation

```bash
pip install langchain-raptor-rag
```

This installs `raptor-rag` and `langchain-core` as dependencies.

## Quick Start

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"

from raptor import RetrievalAugmentation
from langchain_raptor_rag import RaptorRetriever

# Build a RAPTOR tree
ra = RetrievalAugmentation()
ra.add_documents(open("document.txt").read())

# Wrap as a LangChain retriever
retriever = RaptorRetriever(ra=ra)
docs = retriever.invoke("What is the main argument?")

for doc in docs:
    print(f"Layer {doc.metadata['layer_number']}: {doc.page_content[:80]}...")
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `ra` | *required* | A `RetrievalAugmentation` instance with a built tree |
| `top_k` | `10` | Maximum number of nodes to retrieve |
| `max_tokens` | `3500` | Maximum total tokens across retrieved nodes |
| `collapse_tree` | `True` | Use collapsed tree retrieval (vs. tree traversal) |

## Usage with LangChain Chains

```python
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context:\n\n{context}"),
    ("human", "{input}"),
])

combine_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, combine_chain)

response = chain.invoke({"input": "What happened to Cinderella?"})
print(response["answer"])
```

## Metadata

Each returned `Document` includes:

- `node_index` — the node's position in the RAPTOR tree
- `layer_number` — `0` for leaf (original text), higher for summary layers

## Development

```bash
cd integrations/langchain-raptor-rag
pip install -e ".[dev]"
pytest
```
