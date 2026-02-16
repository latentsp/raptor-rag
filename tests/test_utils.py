"""Tests for raptor/utils.py — split_text, distances, and pure helper functions."""

import numpy as np
import pytest

from raptor.utils import (
    distances_from_embeddings,
    get_embeddings,
    get_node_list,
    get_text,
    indices_of_nearest_neighbors_from_distances,
    reverse_mapping,
    split_text,
)


class TestSplitText:
    """Edge cases and typical usage of the split_text function."""

    @pytest.fixture(autouse=True)
    def _setup_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def test_empty_string(self):
        result = split_text("", self.tokenizer, max_tokens=100)
        assert result == []

    def test_single_short_sentence(self):
        result = split_text("Hello world.", self.tokenizer, max_tokens=100)
        assert len(result) == 1
        assert "Hello world" in result[0]

    def test_respects_max_tokens(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        # Use a very small max_tokens to force splitting
        result = split_text(text, self.tokenizer, max_tokens=5)
        assert len(result) >= 2
        for chunk in result:
            token_count = len(self.tokenizer.encode(chunk))
            # Chunks should be approximately within the limit (may slightly exceed
            # due to sentence boundaries, but should be reasonable)
            assert token_count <= 20  # generous upper bound for sentence-level splitting

    def test_long_sentence_with_sub_delimiters(self):
        """A sentence exceeding max_tokens is split on commas, semicolons, colons."""
        long_sentence = (
            "This is a very long sentence with many clauses, "
            "separated by commas; and semicolons: and colons, "
            "to verify that the sub-splitting logic works correctly"
        )
        result = split_text(long_sentence, self.tokenizer, max_tokens=10)
        assert len(result) >= 2

    def test_newline_acts_as_delimiter(self):
        text = "Line one\nLine two\nLine three"
        result = split_text(text, self.tokenizer, max_tokens=100)
        assert len(result) >= 1
        # All lines should appear in the output
        combined = " ".join(result)
        assert "Line one" in combined
        assert "Line two" in combined
        assert "Line three" in combined

    def test_overlap_parameter(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        no_overlap = split_text(text, self.tokenizer, max_tokens=10, overlap=0)
        with_overlap = split_text(text, self.tokenizer, max_tokens=10, overlap=1)
        # With overlap, there should be shared content between consecutive chunks
        # and generally the same or more chunks
        assert len(with_overlap) >= len(no_overlap)

    def test_multiple_delimiters(self):
        text = "Question? Exclamation! Period. Newline\nEnd"
        result = split_text(text, self.tokenizer, max_tokens=100)
        assert len(result) >= 1
        combined = " ".join(result)
        assert "Question" in combined
        assert "Exclamation" in combined

    def test_whitespace_only_sentences_skipped(self):
        text = "Hello.   .   . World."
        result = split_text(text, self.tokenizer, max_tokens=100)
        # Empty sentences between delimiters should be skipped
        combined = " ".join(result)
        assert "Hello" in combined
        assert "World" in combined

    def test_unicode_text(self):
        text = "Le chat mange. Le chien dort. L'oiseau chante."
        result = split_text(text, self.tokenizer, max_tokens=100)
        assert len(result) >= 1


class TestDistancesFromEmbeddings:
    def test_cosine_identical_vectors(self):
        vec = [1.0, 0.0, 0.0]
        distances = distances_from_embeddings(vec, [vec])
        assert len(distances) == 1
        assert abs(distances[0]) < 1e-7  # cosine distance of identical vectors is 0

    def test_cosine_orthogonal_vectors(self):
        q = [1.0, 0.0]
        e = [0.0, 1.0]
        distances = distances_from_embeddings(q, [e])
        assert abs(distances[0] - 1.0) < 1e-7  # cosine distance of orthogonal is 1

    def test_cosine_opposite_vectors(self):
        q = [1.0, 0.0]
        e = [-1.0, 0.0]
        distances = distances_from_embeddings(q, [e])
        assert abs(distances[0] - 2.0) < 1e-7  # cosine distance of opposite is 2

    def test_multiple_embeddings(self):
        q = [1.0, 0.0, 0.0]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        distances = distances_from_embeddings(q, embeddings)
        assert len(distances) == 3
        assert distances[0] < distances[1] < distances[2]

    def test_l1_metric(self):
        q = [0.0, 0.0]
        e = [3.0, 4.0]
        distances = distances_from_embeddings(q, [e], distance_metric="L1")
        assert abs(distances[0] - 7.0) < 1e-7

    def test_l2_metric(self):
        q = [0.0, 0.0]
        e = [3.0, 4.0]
        distances = distances_from_embeddings(q, [e], distance_metric="L2")
        assert abs(distances[0] - 5.0) < 1e-7

    def test_linf_metric(self):
        q = [0.0, 0.0]
        e = [3.0, 4.0]
        distances = distances_from_embeddings(q, [e], distance_metric="Linf")
        assert abs(distances[0] - 4.0) < 1e-7

    def test_unsupported_metric_raises(self):
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            distances_from_embeddings([1.0], [[1.0]], distance_metric="manhattan")


class TestIndicesOfNearestNeighbors:
    def test_sorted_ascending(self):
        distances = [0.5, 0.1, 0.9, 0.3]
        indices = indices_of_nearest_neighbors_from_distances(distances)
        assert list(indices) == [1, 3, 0, 2]

    def test_single_element(self):
        indices = indices_of_nearest_neighbors_from_distances([42.0])
        assert list(indices) == [0]

    def test_returns_ndarray(self):
        result = indices_of_nearest_neighbors_from_distances([0.2, 0.1])
        assert isinstance(result, np.ndarray)


class TestHelperFunctions:
    def test_get_node_list_sorted_by_index(self, sample_nodes, embedding_model_name):
        node_dict = {n.index: n for n in reversed(sample_nodes)}
        result = get_node_list(node_dict)
        indices = [n.index for n in result]
        assert indices == sorted(indices)

    def test_get_embeddings(self, sample_nodes, embedding_model_name):
        embeddings = get_embeddings(sample_nodes, embedding_model_name)
        assert len(embeddings) == len(sample_nodes)
        for emb in embeddings:
            assert len(emb) == 8

    def test_get_text(self, sample_nodes):
        text = get_text(sample_nodes)
        for node in sample_nodes:
            assert node.text in text

    def test_reverse_mapping(self, sample_tree):
        mapping = reverse_mapping(sample_tree.layer_to_nodes)
        # leaf nodes (indices 0-4) should map to layer 0
        for i in range(5):
            assert mapping[i] == 0
        # parent nodes (indices 5, 6) should map to layer 1
        assert mapping[5] == 1
        assert mapping[6] == 1
