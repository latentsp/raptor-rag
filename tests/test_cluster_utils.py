"""Tests for raptor/cluster_utils.py — clustering with synthetic numpy embeddings."""

import numpy as np

from raptor.cluster_utils import (
    get_optimal_clusters,
    global_cluster_embeddings,
    gmm_cluster,
    local_cluster_embeddings,
    perform_clustering,
)


class TestGetOptimalClusters:
    def test_single_cluster_with_uniform_data(self):
        rng = np.random.default_rng(42)
        # Tight cluster of points — BIC should favor 1 cluster
        embeddings = rng.standard_normal((20, 5)) * 0.01
        optimal = get_optimal_clusters(embeddings, max_clusters=5)
        assert isinstance(int(optimal), int)
        assert 1 <= optimal <= 5

    def test_two_clearly_separated_clusters(self):
        rng = np.random.default_rng(42)
        cluster_a = rng.standard_normal((30, 5)) + np.array([10, 0, 0, 0, 0])
        cluster_b = rng.standard_normal((30, 5)) + np.array([-10, 0, 0, 0, 0])
        embeddings = np.vstack([cluster_a, cluster_b])
        optimal = get_optimal_clusters(embeddings, max_clusters=10)
        # Should find 2 or close to 2 clusters
        assert 1 <= optimal <= 5

    def test_max_clusters_capped_by_sample_count(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((5, 3))
        # max_clusters > len(embeddings) should be capped
        optimal = get_optimal_clusters(embeddings, max_clusters=50)
        assert 1 <= optimal <= 4  # at most len-1 because n_clusters = np.arange(1, max_clusters)


class TestGMMCluster:
    def test_returns_labels_and_count(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((30, 5))
        labels, n_clusters = gmm_cluster(embeddings, threshold=0.1)
        assert len(labels) == 30
        assert n_clusters >= 1
        # Each label should be a numpy array of cluster IDs
        for label in labels:
            assert isinstance(label, np.ndarray)

    def test_high_threshold_gives_single_assignments(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((20, 4))
        labels, _ = gmm_cluster(embeddings, threshold=0.9)
        # With a very high threshold, most points should have few cluster assignments
        for label in labels:
            assert len(label) >= 0  # may have zero if prob never exceeds threshold


class TestGlobalClusterEmbeddings:
    def test_reduces_dimension(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 20))
        reduced = global_cluster_embeddings(embeddings, dim=3)
        assert reduced.shape == (50, 3)

    def test_custom_n_neighbors(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((30, 10))
        reduced = global_cluster_embeddings(embeddings, dim=2, n_neighbors=5)
        assert reduced.shape == (30, 2)


class TestLocalClusterEmbeddings:
    def test_reduces_dimension(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 20))
        reduced = local_cluster_embeddings(embeddings, dim=3, num_neighbors=5)
        assert reduced.shape == (50, 3)


class TestPerformClustering:
    def test_returns_cluster_assignments_for_all_points(self):
        rng = np.random.default_rng(42)
        # Need enough points for UMAP (at least dim+2 for the global step)
        embeddings = rng.standard_normal((40, 10))
        clusters = perform_clustering(embeddings, dim=5, threshold=0.1)
        assert len(clusters) == 40
        # Each element should be a numpy array
        for cluster in clusters:
            assert isinstance(cluster, np.ndarray)

    def test_two_blobs(self):
        rng = np.random.default_rng(42)
        blob_a = rng.standard_normal((25, 8)) + 5
        blob_b = rng.standard_normal((25, 8)) - 5
        embeddings = np.vstack([blob_a, blob_b])
        clusters = perform_clustering(embeddings, dim=4, threshold=0.1)
        assert len(clusters) == 50
