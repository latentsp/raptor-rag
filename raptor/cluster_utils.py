import logging

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture

from .tree_structures import Node

logger = logging.getLogger(__name__)

RANDOM_SEED = 224


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int | None = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def gmm_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(embeddings: np.ndarray, dim: int, threshold: float) -> list[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) - 2))
    global_clusters, n_global_clusters = gmm_cluster(reduced_embeddings_global, threshold)

    logger.debug("Global Clusters: %d", n_global_clusters)

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        cluster_member_embeddings = embeddings[np.array([i in gc for gc in global_clusters])]
        logger.debug("Nodes in Global Cluster %d: %d", i, len(cluster_member_embeddings))
        if len(cluster_member_embeddings) == 0:
            continue
        if len(cluster_member_embeddings) <= dim + 1:
            local_clusters = [np.array([0]) for _ in cluster_member_embeddings]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(cluster_member_embeddings, dim)
            local_clusters, n_local_clusters = gmm_cluster(reduced_embeddings_local, threshold)

        logger.debug("Local Clusters in Global Cluster %d: %d", i, n_local_clusters)

        for j in range(n_local_clusters):
            local_member_embeddings = cluster_member_embeddings[np.array([j in lc for lc in local_clusters])]
            indices = np.where((embeddings == local_member_embeddings[:, None]).all(-1))[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

        total_clusters += n_local_clusters

    logger.debug("Total Clusters: %d", total_clusters)
    return all_local_clusters


class RaptorClustering:
    @staticmethod
    def perform_clustering(
        nodes: list[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=None,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
    ) -> list[list[Node]]:
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])
        clusters = perform_clustering(embeddings, dim=reduction_dimension, threshold=threshold)

        node_clusters = []

        for label in np.unique(np.concatenate(clusters)):
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_length = sum(len(tokenizer.encode(node.text)) for node in cluster_nodes)

            if total_length > max_length_in_cluster:
                logger.debug("reclustering cluster with %d nodes", len(cluster_nodes))
                node_clusters.extend(
                    RaptorClustering.perform_clustering(cluster_nodes, embedding_model_name, max_length_in_cluster)
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
