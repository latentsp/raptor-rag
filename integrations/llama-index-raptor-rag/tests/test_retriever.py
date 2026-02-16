"""Tests for the LlamaIndex RAPTOR retriever integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore

from llama_index_raptor_rag import RaptorRetriever


def _make_mock_ra(nodes: dict[int, str], layer_info: list[dict]) -> MagicMock:
    """Create a mock RetrievalAugmentation with the given nodes and layer info."""
    mock_ra = MagicMock()

    mock_nodes = {}
    for idx, text in nodes.items():
        node = SimpleNamespace(text=text, index=idx)
        mock_nodes[idx] = node

    mock_ra.retriever.tree.all_nodes = mock_nodes
    mock_ra.retrieve.return_value = ("combined context string", layer_info)
    return mock_ra


class TestRaptorRetriever:
    def test_returns_nodes_with_correct_metadata(self):
        nodes = {0: "First chunk of text.", 3: "Summary of cluster."}
        layer_info = [
            {"node_index": 0, "layer_number": 0},
            {"node_index": 3, "layer_number": 1},
        ]
        mock_ra = _make_mock_ra(nodes, layer_info)

        retriever = RaptorRetriever(ra=mock_ra, top_k=5, max_tokens=2000, collapse_tree=True)
        results = retriever.retrieve("What is the story about?")

        assert len(results) == 2
        assert all(isinstance(r, NodeWithScore) for r in results)

        assert results[0].node.text == "First chunk of text."
        assert results[0].node.metadata["node_index"] == 0
        assert results[0].node.metadata["layer_number"] == 0

        assert results[1].node.text == "Summary of cluster."
        assert results[1].node.metadata["node_index"] == 3
        assert results[1].node.metadata["layer_number"] == 1

    def test_scores_are_rank_based(self):
        nodes = {0: "A", 1: "B", 2: "C"}
        layer_info = [
            {"node_index": 0, "layer_number": 0},
            {"node_index": 1, "layer_number": 0},
            {"node_index": 2, "layer_number": 1},
        ]
        mock_ra = _make_mock_ra(nodes, layer_info)

        retriever = RaptorRetriever(ra=mock_ra)
        results = retriever.retrieve("query")

        # First result should have highest score
        assert results[0].score > results[1].score
        assert results[1].score > results[2].score
        # First should be 1.0, last should be > 0
        assert results[0].score == 1.0
        assert results[2].score > 0

    def test_retrieve_called_with_correct_params(self):
        nodes = {1: "Node text."}
        layer_info = [{"node_index": 1, "layer_number": 0}]
        mock_ra = _make_mock_ra(nodes, layer_info)

        retriever = RaptorRetriever(ra=mock_ra, top_k=7, max_tokens=1500, collapse_tree=False)
        retriever.retrieve("test query")

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
        results = retriever.retrieve("query with no results")

        assert results == []

    def test_single_result_has_score_one(self):
        nodes = {5: "Only node."}
        layer_info = [{"node_index": 5, "layer_number": 2}]
        mock_ra = _make_mock_ra(nodes, layer_info)

        retriever = RaptorRetriever(ra=mock_ra)
        results = retriever.retrieve("query")

        assert len(results) == 1
        assert results[0].score == 1.0

    def test_default_parameters(self):
        mock_ra = MagicMock()
        mock_ra.retrieve.return_value = ("", [])

        retriever = RaptorRetriever(ra=mock_ra)

        assert retriever._top_k == 10
        assert retriever._max_tokens == 3500
        assert retriever._collapse_tree is True
