"""Tests for the reranker module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from raptor.reranker import BaseReRanker, CrossEncoderReRanker, LiteLLMReRanker
from raptor.tree_structures import Node


@pytest.fixture
def nodes():
    """Create a list of test nodes."""
    return [
        Node(text="Machine learning is great", index=0, children=set(), embeddings={"test": [0.1]}),
        Node(text="Deep learning uses neural networks", index=1, children=set(), embeddings={"test": [0.2]}),
        Node(text="NLP processes text data", index=2, children=set(), embeddings={"test": [0.3]}),
        Node(text="Computer vision analyzes images", index=3, children=set(), embeddings={"test": [0.4]}),
        Node(text="Reinforcement learning trains agents", index=4, children=set(), embeddings={"test": [0.5]}),
    ]


class TestBaseReRanker:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseReRanker()

    def test_subclass_must_implement_rerank(self):
        class IncompleteReRanker(BaseReRanker):
            pass

        with pytest.raises(TypeError):
            IncompleteReRanker()

    def test_custom_reranker(self, nodes):
        class ReverseReRanker(BaseReRanker):
            def rerank(self, query, nodes, top_k=5):
                return list(reversed(nodes))[:top_k]

        reranker = ReverseReRanker()
        result = reranker.rerank("query", nodes, top_k=3)
        assert len(result) == 3
        assert result[0].index == 4  # Last node first
        assert result[1].index == 3
        assert result[2].index == 2


class TestCrossEncoderReRanker:
    @patch("sentence_transformers.CrossEncoder")
    def test_instantiation(self, mock_ce_cls):
        CrossEncoderReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        mock_ce_cls.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2", device=None)

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank(self, mock_ce_cls, nodes):
        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model
        # Return scores that rank nodes 2, 4, 0, 1, 3
        mock_model.predict.return_value = np.array([0.5, 0.3, 0.9, 0.1, 0.7])

        reranker = CrossEncoderReRanker()
        result = reranker.rerank("query about NLP", nodes, top_k=3)

        assert len(result) == 3
        assert result[0].index == 2  # Highest score (0.9)
        assert result[1].index == 4  # Second highest (0.7)
        assert result[2].index == 0  # Third (0.5)

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_empty_nodes(self, mock_ce_cls):
        reranker = CrossEncoderReRanker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_with_device(self, mock_ce_cls):
        CrossEncoderReRanker(device="cpu")
        mock_ce_cls.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


class TestLiteLLMReRanker:
    def test_instantiation(self):
        reranker = LiteLLMReRanker(model="cohere/rerank-english-v3.0")
        assert reranker.model == "cohere/rerank-english-v3.0"

    @patch("litellm.rerank")
    def test_rerank(self, mock_rerank, nodes):
        # Mock the rerank response
        mock_result0 = MagicMock()
        mock_result0.index = 2
        mock_result1 = MagicMock()
        mock_result1.index = 0
        mock_result2 = MagicMock()
        mock_result2.index = 4

        mock_response = MagicMock()
        mock_response.results = [mock_result0, mock_result1, mock_result2]
        mock_rerank.return_value = mock_response

        reranker = LiteLLMReRanker()
        result = reranker.rerank("query", nodes, top_k=3)

        assert len(result) == 3
        assert result[0].index == 2
        assert result[1].index == 0
        assert result[2].index == 4

    def test_rerank_empty_nodes(self):
        reranker = LiteLLMReRanker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []
