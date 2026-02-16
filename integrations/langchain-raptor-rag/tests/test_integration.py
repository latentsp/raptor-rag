"""Integration tests for LangChain RAPTOR retriever.

Exercises the full path: build a real RAPTOR tree with stub models (no API keys)
→ wrap in framework retriever → call retrieve → verify framework-native output types.
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from raptor import (
    BaseEmbeddingModel,
    BaseQAModel,
    BaseSummarizationModel,
    ClusterTreeConfig,
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
)

from langchain_raptor_rag import RaptorRetriever

# ---------------------------------------------------------------------------
# Stub models — deterministic, no API keys needed
# ---------------------------------------------------------------------------


class StubEmbeddingModel(BaseEmbeddingModel):
    """Deterministic hash-based 8-dim embedding with values in [-1, 1]."""

    def create_embedding(self, text):
        h = hashlib.sha256(text.encode()).digest()
        return [(b / 127.5) - 1.0 for b in h[:8]]


class StubSummarizationModel(BaseSummarizationModel):
    """Returns a truncated version of the context as a summary."""

    def summarize(self, context, max_tokens=150):
        words = context.split()
        return " ".join(words[: min(len(words), 20)])


class StubQAModel(BaseQAModel):
    """Returns a fixed answer string."""

    def answer_question(self, context, question):
        return "stub answer"


# ---------------------------------------------------------------------------
# Test text — short enough for minimal tree config
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Once upon a time there was a young girl named Cinderella. "
    "She lived with her wicked stepmother and two stepsisters. "
    "One day an invitation arrived for a grand ball at the royal palace. "
    "Cinderella wished she could attend the ball more than anything. "
    "Her fairy godmother appeared and transformed her rags into a gown. "
    "At the ball she danced with the prince and lost her glass slipper. "
    "The prince searched the kingdom to find the owner of the slipper. "
    "When Cinderella tried it on the slipper fit perfectly. "
    "The prince and Cinderella were married and lived happily ever after. "
    "The story teaches that kindness and perseverance are always rewarded."
)


def _build_ra() -> RetrievalAugmentation:
    """Build a RetrievalAugmentation with stub models and minimal config."""
    embedding_model = StubEmbeddingModel()
    config = RetrievalAugmentationConfig(
        embedding_model=embedding_model,
        qa_model=StubQAModel(),
        tree_builder_config=ClusterTreeConfig(
            reduction_dimension=2,
            num_layers=1,
            max_tokens=30,
            summarization_length=20,
            summarization_model=StubSummarizationModel(),
            embedding_models={"EMB": embedding_model},
            cluster_embedding_model="EMB",
        ),
        tr_embedding_model=embedding_model,
        tr_context_embedding_model="EMB",
    )
    ra = RetrievalAugmentation(config=config)
    ra.add_documents(SAMPLE_TEXT)
    return ra


class TestLangChainIntegration:
    def test_invoke_returns_documents(self):
        ra = _build_ra()
        retriever = RaptorRetriever(ra=ra, top_k=5, max_tokens=3500, collapse_tree=True)

        docs = retriever.invoke("What happened to Cinderella?")

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_documents_have_metadata(self):
        ra = _build_ra()
        retriever = RaptorRetriever(ra=ra, top_k=5)

        docs = retriever.invoke("Who is Cinderella?")

        for doc in docs:
            assert "node_index" in doc.metadata
            assert "layer_number" in doc.metadata
            assert isinstance(doc.metadata["node_index"], int)
            assert isinstance(doc.metadata["layer_number"], int)
            assert len(doc.page_content) > 0

    def test_documents_have_text_from_tree(self):
        ra = _build_ra()
        retriever = RaptorRetriever(ra=ra, top_k=3)

        docs = retriever.invoke("fairy godmother")

        # All returned text should exist in the tree's nodes
        all_node_texts = {node.text for node in ra.tree.all_nodes.values()}
        for doc in docs:
            assert doc.page_content in all_node_texts
