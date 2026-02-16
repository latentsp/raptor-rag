import logging

import tiktoken

from .embedding_models import BaseEmbeddingModel, LiteLLMEmbeddingModel
from .reranker import BaseReRanker
from .retrievers import BaseRetriever
from .tree_structures import Node, Tree
from .utils import (
    distances_from_embeddings,
    get_embeddings,
    get_node_list,
    get_text,
    indices_of_nearest_neighbors_from_distances,
    reverse_mapping,
)

logger = logging.getLogger(__name__)


class TreeRetrieverConfig:
    def __init__(
        self,
        tokenizer=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        context_embedding_model=None,
        embedding_model=None,
        num_layers=None,
        start_layer=None,
        reranker=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

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
        if not isinstance(selection_mode, str) or selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode must be a string and either 'top_k' or 'threshold'")
        self.selection_mode = selection_mode

        if context_embedding_model is None:
            context_embedding_model = "default"
        if not isinstance(context_embedding_model, str):
            raise ValueError("context_embedding_model must be a string")
        self.context_embedding_model = context_embedding_model

        if embedding_model is None:
            embedding_model = LiteLLMEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError("embedding_model must be an instance of BaseEmbeddingModel")
        self.embedding_model = embedding_model

        if num_layers is not None and (not isinstance(num_layers, int) or num_layers < 0):
            raise ValueError("num_layers must be an integer and at least 0")
        self.num_layers = num_layers

        if start_layer is not None and (not isinstance(start_layer, int) or start_layer < 0):
            raise ValueError("start_layer must be an integer and at least 0")
        self.start_layer = start_layer

        if reranker is not None and not isinstance(reranker, BaseReRanker):
            raise ValueError("reranker must be an instance of BaseReRanker")
        self.reranker = reranker

    def log_config(self):
        config_log = f"""
        TreeRetrieverConfig:
            Tokenizer: {self.tokenizer}
            Threshold: {self.threshold}
            Top K: {self.top_k}
            Selection Mode: {self.selection_mode}
            Context Embedding Model: {self.context_embedding_model}
            Embedding Model: {self.embedding_model}
            Num Layers: {self.num_layers}
            Start Layer: {self.start_layer}
            ReRanker: {self.reranker}
        """
        return config_log


class TreeRetriever(BaseRetriever):
    def __init__(self, config, tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError("num_layers in config must be less than or equal to tree.num_layers + 1")

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError("start_layer in config must be less than or equal to tree.num_layers")

        self.config = config
        self.tree = tree
        self.num_layers = config.num_layers if config.num_layers is not None else tree.num_layers + 1
        self.start_layer = config.start_layer if config.start_layer is not None else tree.num_layers

        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)

        logger.info("Successfully initialized TreeRetriever with Config %s", config.log_config())

    def create_embedding(self, text: str) -> list[float]:
        return self.config.embedding_model.create_embedding(text)

    def _apply_reranker(self, query: str, selected_nodes: list[Node]) -> list[Node]:
        if self.config.reranker is not None and selected_nodes:
            return self.config.reranker.rerank(query, selected_nodes, top_k=len(selected_nodes))
        return selected_nodes

    def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> tuple[list[Node], str]:
        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)
        embeddings = get_embeddings(node_list, self.config.context_embedding_model)
        distances = distances_from_embeddings(query_embedding, embeddings)
        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        for idx in indices[:top_k]:
            node = node_list[idx]
            node_tokens = len(self.config.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        selected_nodes = self._apply_reranker(query, selected_nodes)

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(self, current_nodes: list[Node], query: str, num_layers: int) -> tuple[list[Node], str]:
        query_embedding = self.create_embedding(query)

        selected_nodes = []
        node_list = current_nodes

        for layer in range(num_layers):
            embeddings = get_embeddings(node_list, self.config.context_embedding_model)
            distances = distances_from_embeddings(query_embedding, embeddings)
            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.config.selection_mode == "threshold":
                best_indices = [index for index in indices if distances[index] > self.config.threshold]
            else:
                best_indices = indices[: self.config.top_k]

            nodes_to_add = [node_list[idx] for idx in best_indices]
            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:
                child_nodes = []
                for index in best_indices:
                    child_nodes.extend(node_list[index].children)
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        selected_nodes = self._apply_reranker(query, selected_nodes)

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve(
        self,
        query: str,
        start_layer: int | None = None,
        num_layers: int | None = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> str | tuple[str, list[dict]]:
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        start_layer = self.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        if not isinstance(start_layer, int) or not (0 <= start_layer <= self.tree.num_layers):
            raise ValueError("start_layer must be an integer between 0 and tree.num_layers")

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")

        if num_layers > (start_layer + 1):
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        if collapse_tree:
            logger.info("Using collapsed_tree")
            selected_nodes, context = self.retrieve_information_collapse_tree(query, top_k, max_tokens)
        else:
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, context = self.retrieve_information(layer_nodes, query, num_layers)

        if return_layer_information:
            layer_information = [
                {
                    "node_index": node.index,
                    "layer_number": self.tree_node_index_to_layer[node.index],
                }
                for node in selected_nodes
            ]
            return context, layer_information

        return context
