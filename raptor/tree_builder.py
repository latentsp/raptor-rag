import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken

from .embedding_models import BaseEmbeddingModel, LiteLLMEmbeddingModel
from .summarization_models import BaseSummarizationModel, LiteLLMSummarizationModel
from .tree_structures import Node, Tree
from .utils import (
    BaseTextSplitter,
    DefaultTextSplitter,
)

logger = logging.getLogger(__name__)


class TreeBuilderConfig:
    def __init__(
        self,
        tokenizer=None,
        max_tokens=None,
        num_layers=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        summarization_length=None,
        summarization_model=None,
        embedding_models=None,
        cluster_embedding_model=None,
        text_splitter=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        if max_tokens is None:
            max_tokens = 100
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens

        if num_layers is None:
            num_layers = 5
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")
        self.num_layers = num_layers

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode must be either 'top_k' or 'threshold'")
        self.selection_mode = selection_mode

        if summarization_length is None:
            summarization_length = 100
        self.summarization_length = summarization_length

        if summarization_model is None:
            summarization_model = LiteLLMSummarizationModel()
        if not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError("summarization_model must be an instance of BaseSummarizationModel")
        self.summarization_model = summarization_model

        if embedding_models is None:
            embedding_models = {"default": LiteLLMEmbeddingModel()}
        if not isinstance(embedding_models, dict):
            raise ValueError("embedding_models must be a dictionary of model_name: instance pairs")
        for model in embedding_models.values():
            if not isinstance(model, BaseEmbeddingModel):
                raise ValueError("All embedding models must be an instance of BaseEmbeddingModel")
        self.embedding_models = embedding_models

        if cluster_embedding_model is None:
            cluster_embedding_model = "default"
        if cluster_embedding_model not in self.embedding_models:
            raise ValueError("cluster_embedding_model must be a key in the embedding_models dictionary")
        self.cluster_embedding_model = cluster_embedding_model

        if text_splitter is None:
            text_splitter = DefaultTextSplitter()
        if not isinstance(text_splitter, BaseTextSplitter):
            raise ValueError("text_splitter must be an instance of BaseTextSplitter")
        self.text_splitter = text_splitter

    def log_config(self):
        config_log = f"""
        TreeBuilderConfig:
            Tokenizer: {self.tokenizer}
            Max Tokens: {self.max_tokens}
            Num Layers: {self.num_layers}
            Threshold: {self.threshold}
            Top K: {self.top_k}
            Selection Mode: {self.selection_mode}
            Summarization Length: {self.summarization_length}
            Summarization Model: {self.summarization_model}
            Embedding Models: {self.embedding_models}
            Cluster Embedding Model: {self.cluster_embedding_model}
        """
        return config_log


class TreeBuilder(ABC):
    def __init__(self, config) -> None:
        self.config = config

        logger.info("Successfully initialized TreeBuilder with Config %s", config.log_config())

    def create_node(self, index: int, text: str, children_indices: set[int] | None = None) -> tuple[int, Node]:
        if children_indices is None:
            children_indices = set()

        embeddings = {
            model_name: model.create_embedding(text) for model_name, model in self.config.embedding_models.items()
        }
        return (index, Node(text, index, children_indices, embeddings))

    def create_embedding(self, text) -> list[float]:
        return self.config.embedding_models[self.config.cluster_embedding_model].create_embedding(text)

    def summarize(self, context, max_tokens) -> str:
        return self.config.summarization_model.summarize(context, max_tokens)

    def multithreaded_create_leaf_nodes(self, chunks: list[str]) -> dict[int, Node]:
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text) for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    def build_from_text(self, text: str, use_multithreading: bool = True) -> Tree:
        chunks = self.config.text_splitter.split_text(text, self.config.tokenizer, self.config.max_tokens)

        logger.info("Creating Leaf Nodes")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, chunk in enumerate(chunks):
                _, node = self.create_node(index, chunk)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logger.info("Created %d Leaf Embeddings", len(leaf_nodes))
        logger.info("Building All Nodes")

        all_nodes = dict(leaf_nodes)
        root_nodes, num_layers_built = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)
        tree = Tree(all_nodes, root_nodes, leaf_nodes, num_layers_built, layer_to_nodes)

        return tree

    @abstractmethod
    def construct_tree(
        self,
        current_level_nodes: dict[int, Node],
        all_tree_nodes: dict[int, Node],
        layer_to_nodes: dict[int, list[Node]],
        use_multithreading: bool = True,
    ) -> tuple[dict[int, Node], int]:
        pass
