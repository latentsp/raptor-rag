"""Tests for all Config classes: defaults, validation, and error paths."""

from unittest.mock import MagicMock

import pytest

from raptor.embedding_models import BaseEmbeddingModel
from raptor.qa_models import BaseQAModel
from raptor.reranker import BaseReRanker
from raptor.summarization_models import BaseSummarizationModel
from raptor.utils import BaseTextSplitter, DefaultTextSplitter


class StubEmbeddingModel(BaseEmbeddingModel):
    def create_embedding(self, text):
        return [0.0] * 8


class StubSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=500):
        return "stub summary"


class StubQAModel(BaseQAModel):
    def answer_question(self, context, question):
        return "stub answer"


class StubReRanker(BaseReRanker):
    def rerank(self, query, nodes, top_k=5):
        return nodes[:top_k]


class StubTextSplitter(BaseTextSplitter):
    def split_text(self, text, tokenizer, max_tokens):
        return [text]


class TestTreeBuilderConfig:
    @pytest.fixture
    def make_config(self):
        """Helper that imports and builds TreeBuilderConfig with stub models."""
        from raptor.tree_builder import TreeBuilderConfig

        def _make(**overrides):
            defaults = {
                "summarization_model": StubSummarizationModel(),
                "embedding_models": {"test": StubEmbeddingModel()},
                "cluster_embedding_model": "test",
            }
            defaults.update(overrides)
            return TreeBuilderConfig(**defaults)

        return _make

    def test_defaults(self, make_config):
        cfg = make_config()
        assert cfg.max_tokens == 100
        assert cfg.num_layers == 5
        assert cfg.threshold == 0.5
        assert cfg.top_k == 5
        assert cfg.selection_mode == "top_k"
        assert cfg.summarization_length == 100
        assert isinstance(cfg.text_splitter, DefaultTextSplitter)

    def test_custom_values(self, make_config):
        cfg = make_config(max_tokens=200, num_layers=3, threshold=0.8, top_k=10, selection_mode="threshold")
        assert cfg.max_tokens == 200
        assert cfg.num_layers == 3
        assert cfg.threshold == 0.8
        assert cfg.top_k == 10
        assert cfg.selection_mode == "threshold"

    def test_custom_text_splitter(self, make_config):
        splitter = StubTextSplitter()
        cfg = make_config(text_splitter=splitter)
        assert cfg.text_splitter is splitter

    def test_invalid_max_tokens_zero(self, make_config):
        with pytest.raises(ValueError, match="max_tokens"):
            make_config(max_tokens=0)

    def test_invalid_max_tokens_negative(self, make_config):
        with pytest.raises(ValueError, match="max_tokens"):
            make_config(max_tokens=-5)

    def test_invalid_max_tokens_float(self, make_config):
        with pytest.raises(ValueError, match="max_tokens"):
            make_config(max_tokens=3.5)

    def test_invalid_num_layers(self, make_config):
        with pytest.raises(ValueError, match="num_layers"):
            make_config(num_layers=0)

    def test_invalid_threshold_above_1(self, make_config):
        with pytest.raises(ValueError, match="threshold"):
            make_config(threshold=1.5)

    def test_invalid_threshold_negative(self, make_config):
        with pytest.raises(ValueError, match="threshold"):
            make_config(threshold=-0.1)

    def test_invalid_top_k(self, make_config):
        with pytest.raises(ValueError, match="top_k"):
            make_config(top_k=0)

    def test_invalid_selection_mode(self, make_config):
        with pytest.raises(ValueError, match="selection_mode"):
            make_config(selection_mode="invalid")

    def test_invalid_summarization_model(self, make_config):
        with pytest.raises(ValueError, match="summarization_model"):
            make_config(summarization_model="not_a_model")

    def test_invalid_embedding_models_not_dict(self, make_config):
        with pytest.raises(ValueError, match="embedding_models"):
            make_config(embedding_models=[StubEmbeddingModel()])

    def test_invalid_embedding_models_wrong_type(self, make_config):
        with pytest.raises(ValueError, match="embedding models"):
            make_config(embedding_models={"bad": "not_an_embedding_model"})

    def test_invalid_cluster_embedding_model_missing_key(self, make_config):
        with pytest.raises(ValueError, match="cluster_embedding_model"):
            make_config(cluster_embedding_model="nonexistent")

    def test_invalid_text_splitter_type(self, make_config):
        with pytest.raises(ValueError, match="text_splitter"):
            make_config(text_splitter="not_a_splitter")

    def test_log_config_returns_string(self, make_config):
        cfg = make_config()
        log = cfg.log_config()
        assert isinstance(log, str)
        assert "Max Tokens" in log


class TestTreeRetrieverConfig:
    @pytest.fixture
    def make_config(self):
        from raptor.tree_retriever import TreeRetrieverConfig

        def _make(**overrides):
            defaults = {
                "embedding_model": StubEmbeddingModel(),
            }
            defaults.update(overrides)
            return TreeRetrieverConfig(**defaults)

        return _make

    def test_defaults(self, make_config):
        cfg = make_config()
        assert cfg.threshold == 0.5
        assert cfg.top_k == 5
        assert cfg.selection_mode == "top_k"
        assert cfg.context_embedding_model == "default"
        assert cfg.num_layers is None
        assert cfg.start_layer is None
        assert cfg.reranker is None

    def test_custom_reranker(self, make_config):
        reranker = StubReRanker()
        cfg = make_config(reranker=reranker)
        assert cfg.reranker is reranker

    def test_invalid_threshold_not_float(self, make_config):
        with pytest.raises(ValueError, match="threshold"):
            make_config(threshold="high")

    def test_invalid_top_k(self, make_config):
        with pytest.raises(ValueError, match="top_k"):
            make_config(top_k=-1)

    def test_invalid_selection_mode(self, make_config):
        with pytest.raises(ValueError, match="selection_mode"):
            make_config(selection_mode="random")

    def test_invalid_context_embedding_model(self, make_config):
        with pytest.raises(ValueError, match="context_embedding_model"):
            make_config(context_embedding_model=123)

    def test_invalid_embedding_model_type(self):
        from raptor.tree_retriever import TreeRetrieverConfig

        with pytest.raises(ValueError, match="embedding_model"):
            TreeRetrieverConfig(embedding_model="not_a_model")

    def test_invalid_num_layers_negative(self, make_config):
        with pytest.raises(ValueError, match="num_layers"):
            make_config(num_layers=-1)

    def test_invalid_start_layer_negative(self, make_config):
        with pytest.raises(ValueError, match="start_layer"):
            make_config(start_layer=-1)

    def test_invalid_reranker_type(self, make_config):
        with pytest.raises(ValueError, match="reranker"):
            make_config(reranker="not_a_reranker")

    def test_log_config(self, make_config):
        cfg = make_config()
        log = cfg.log_config()
        assert isinstance(log, str)
        assert "Top K" in log


class TestClusterTreeConfig:
    @pytest.fixture
    def make_config(self):
        from raptor.cluster_tree_builder import ClusterTreeConfig

        def _make(**overrides):
            defaults = {
                "summarization_model": StubSummarizationModel(),
                "embedding_models": {"test": StubEmbeddingModel()},
                "cluster_embedding_model": "test",
            }
            defaults.update(overrides)
            return ClusterTreeConfig(**defaults)

        return _make

    def test_defaults(self, make_config):
        cfg = make_config()
        assert cfg.reduction_dimension == 10
        # Inherits TreeBuilderConfig defaults
        assert cfg.max_tokens == 100

    def test_custom_reduction_dimension(self, make_config):
        cfg = make_config(reduction_dimension=5)
        assert cfg.reduction_dimension == 5

    def test_custom_clustering_params(self, make_config):
        params = {"threshold": 0.2, "verbose": True}
        cfg = make_config(clustering_params=params)
        assert cfg.clustering_params == params

    def test_log_config_includes_cluster_info(self, make_config):
        cfg = make_config()
        log = cfg.log_config()
        assert "Reduction Dimension" in log
        assert "Clustering Algorithm" in log


class TestRetrievalAugmentationConfig:
    @pytest.fixture
    def make_config(self):
        from raptor.retrieval_augmentation import RetrievalAugmentationConfig

        def _make(**overrides):
            defaults = {
                "tb_summarization_model": StubSummarizationModel(),
                "tb_embedding_models": {"test": StubEmbeddingModel()},
                "tb_cluster_embedding_model": "test",
                "tr_embedding_model": StubEmbeddingModel(),
                "tr_context_embedding_model": "test",
                "qa_model": StubQAModel(),
            }
            defaults.update(overrides)
            return RetrievalAugmentationConfig(**defaults)

        return _make

    def test_defaults(self, make_config):
        cfg = make_config()
        assert cfg.tree_builder_type == "cluster"
        assert cfg.tree_builder_config is not None
        assert cfg.tree_retriever_config is not None
        assert cfg.qa_model is not None

    def test_invalid_tree_builder_type(self, make_config):
        with pytest.raises(ValueError, match="tree_builder_type"):
            make_config(tree_builder_type="invalid")

    def test_invalid_qa_model(self, make_config):
        with pytest.raises(ValueError, match="qa_model"):
            make_config(qa_model="not_a_model")

    def test_invalid_embedding_model(self, make_config):
        with pytest.raises(ValueError, match="embedding_model"):
            make_config(embedding_model="not_a_model")

    def test_invalid_summarization_model(self):
        from raptor.retrieval_augmentation import RetrievalAugmentationConfig

        with pytest.raises(ValueError, match="summarization_model"):
            RetrievalAugmentationConfig(
                summarization_model="not_a_model",
                tb_embedding_models={"test": StubEmbeddingModel()},
                tb_cluster_embedding_model="test",
                tr_embedding_model=StubEmbeddingModel(),
                tr_context_embedding_model="test",
                qa_model=StubQAModel(),
            )

    def test_shortcut_embedding_model(self):
        """Providing embedding_model sets both tb and tr embedding models."""
        from raptor.retrieval_augmentation import RetrievalAugmentationConfig

        emb = StubEmbeddingModel()
        cfg = RetrievalAugmentationConfig(
            embedding_model=emb,
            tb_summarization_model=StubSummarizationModel(),
            qa_model=StubQAModel(),
        )
        assert cfg.tree_builder_config.cluster_embedding_model == "EMB"
        assert cfg.tree_retriever_config.context_embedding_model == "EMB"

    def test_both_embedding_model_and_tb_raises(self):
        from raptor.retrieval_augmentation import RetrievalAugmentationConfig

        with pytest.raises(ValueError, match="Only one of"):
            RetrievalAugmentationConfig(
                embedding_model=StubEmbeddingModel(),
                tb_embedding_models={"test": StubEmbeddingModel()},
                tb_summarization_model=StubSummarizationModel(),
                qa_model=StubQAModel(),
            )

    def test_both_summarization_model_and_tb_raises(self):
        from raptor.retrieval_augmentation import RetrievalAugmentationConfig

        with pytest.raises(ValueError, match="Only one of"):
            RetrievalAugmentationConfig(
                summarization_model=StubSummarizationModel(),
                tb_summarization_model=StubSummarizationModel(),
                tb_embedding_models={"test": StubEmbeddingModel()},
                tb_cluster_embedding_model="test",
                tr_embedding_model=StubEmbeddingModel(),
                tr_context_embedding_model="test",
                qa_model=StubQAModel(),
            )

    def test_custom_reranker(self, make_config):
        reranker = StubReRanker()
        cfg = make_config(tr_reranker=reranker)
        assert cfg.tree_retriever_config.reranker is reranker

    def test_invalid_reranker(self, make_config):
        with pytest.raises(ValueError, match="tr_reranker"):
            make_config(tr_reranker="not_a_reranker")

    def test_custom_text_splitter(self, make_config):
        splitter = StubTextSplitter()
        cfg = make_config(tb_text_splitter=splitter)
        assert cfg.tree_builder_config.text_splitter is splitter

    def test_invalid_text_splitter(self, make_config):
        with pytest.raises(ValueError, match="tb_text_splitter"):
            make_config(tb_text_splitter="not_a_splitter")

    def test_log_config(self, make_config):
        cfg = make_config()
        log = cfg.log_config()
        assert isinstance(log, str)
        assert "RetrievalAugmentationConfig" in log

    def test_custom_tree_builder_config_wrong_type(self, make_config):
        """Passing a non-ClusterTreeConfig as tree_builder_config should raise."""
        from raptor.tree_builder import TreeBuilderConfig

        # TreeBuilderConfig is NOT ClusterTreeConfig
        mock_tb_config = MagicMock(spec=TreeBuilderConfig)
        # It won't be an instance of ClusterTreeConfig
        with pytest.raises(ValueError, match="tree_builder_config must be"):
            make_config(tree_builder_config=mock_tb_config)

    def test_custom_tree_retriever_config_wrong_type(self, make_config):
        with pytest.raises(ValueError, match="tree_retriever_config must be"):
            make_config(tree_retriever_config="not_a_config")
