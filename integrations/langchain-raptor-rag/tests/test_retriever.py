"""Tests for the LangChain RAPTOR retriever integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document

from langchain_raptor_rag import RaptorRetriever


def _make_mock_ra(nodes: dict[int, str], layer_info: list[dict]) -> MagicMock:
    """Create a mock RetrievalAugmentation with the given nodes and layer info."""
    mock_ra = MagicMock()

    # Build mock node objects accessible via ra.retriever.tree.all_nodes[idx]
    mock_nodes = {}
    for idx, text in nodes.items():
        node = SimpleNamespace(text=text, index=idx)
        mock_nodes[idx] = node

    mock_ra.retriever.tree.all_nodes = mock_nodes
    mock_ra.retrieve.return_value = ("combined context string", layer_info)
    return mock_ra


class TestRaptorRetriever:
    def test_returns_documents_with_correct_metadata(self):
        nodes = {0: "First chunk of text.", 3: "Summary of cluster."}
        layer_info = [
            {"node_index": 0, "layer_number": 0},
            {"node_index": 3, "layer_number": 1},
        ]
        mock_ra = _make_mock_ra(nodes, layer_info)

        retriever = RaptorRetriever(ra=mock_ra, top_k=5, max_tokens=2000, collapse_tree=True)
        docs = retriever.invoke("What is the story about?")

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

        assert docs[0].page_content == "First chunk of text."
        assert docs[0].metadata == {"node_index": 0, "layer_number": 0}

        assert docs[1].page_content == "Summary of cluster."
        assert docs[1].metadata == {"node_index": 3, "layer_number": 1}

    def test_retrieve_called_with_correct_params(self):
        nodes = {1: "Node text."}
        layer_info = [{"node_index": 1, "layer_number": 0}]
        mock_ra = _make_mock_ra(nodes, layer_info)

        retriever = RaptorRetriever(ra=mock_ra, top_k=7, max_tokens=1500, collapse_tree=False)
        retriever.invoke("test query")

        mock_ra.retrieve.assert_called_once_with(
            "test query",
            top_k=7,
            max_tokens=1500,
            collapse_tree=False,
            return_layer_information=True,
        )

    def test_empty_results(self):
        mock_ra = _make_mock_ra({}, [])

        retriever = RaptorRetriever(ra=mock_ra)
        docs = retriever.invoke("query with no results")

        assert docs == []

    def test_default_parameters(self):
        nodes = {0: "text"}
        layer_info = [{"node_index": 0, "layer_number": 0}]
        mock_ra = _make_mock_ra(nodes, layer_info)

        retriever = RaptorRetriever(ra=mock_ra)

        assert retriever.top_k == 10
        assert retriever.max_tokens == 3500
        assert retriever.collapse_tree is True
