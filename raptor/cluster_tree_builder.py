import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .cluster_utils import RaptorClustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node
from .utils import (
    get_node_list,
    get_text,
)

logger = logging.getLogger(__name__)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RaptorClustering,
        clustering_params=None,
        **kwargs,
    ):
        if clustering_params is None:
            clustering_params = {}
        super().__init__(**kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")

        logger.info("Successfully initialized ClusterTreeBuilder with Config %s", config.log_config())

    def construct_tree(
        self,
        current_level_nodes: dict[int, Node],
        all_tree_nodes: dict[int, Node],
        layer_to_nodes: dict[int, list[Node]],
        use_multithreading: bool = False,
    ) -> tuple[dict[int, Node], int]:
        logger.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)
        num_layers_built = self.config.num_layers

        def process_cluster(cluster, new_level_nodes, next_node_index, summarization_length, lock):
            node_texts = get_text(cluster)

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logger.info(
                "Node Texts Length: %d, Summarized Text Length: %d",
                len(self.config.tokenizer.encode(node_texts)),
                len(self.config.tokenizer.encode(summarized_text)),
            )

            _, new_parent_node = self.create_node(next_node_index, summarized_text, {node.index for node in cluster})

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        lock = Lock()
        summarization_length = self.config.summarization_length

        for layer in range(self.config.num_layers):
            new_level_nodes = {}

            logger.info("Constructing Layer %d", layer)

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.config.reduction_dimension + 1:
                num_layers_built = layer
                logger.info("Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: %d", layer)
                break

            clusters = self.config.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.config.cluster_embedding_model,
                reduction_dimension=self.config.reduction_dimension,
                **self.config.clustering_params,
            )

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

        return current_level_nodes, num_layers_built
