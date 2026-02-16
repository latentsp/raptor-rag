"""Tests for raptor/tree_structures.py — Node and Tree construction, pickle round-trip."""

import pickle
import tempfile

import numpy as np

from raptor.tree_structures import Node, Tree


class TestNode:
    def test_basic_construction(self, sample_node):
        assert sample_node.text == "The quick brown fox jumps over the lazy dog."
        assert sample_node.index == 0
        assert sample_node.children == set()
        assert isinstance(sample_node.embeddings, dict)

    def test_with_children(self, embedding_model_name, sample_embedding):
        node = Node(
            text="Parent node",
            index=10,
            children={0, 1, 2},
            embeddings={embedding_model_name: sample_embedding},
        )
        assert node.children == {0, 1, 2}
        assert node.index == 10

    def test_embeddings_multiple_models(self):
        emb_a = [1.0, 2.0, 3.0]
        emb_b = [4.0, 5.0, 6.0]
        node = Node(
            text="multi-model",
            index=0,
            children=set(),
            embeddings={"model_a": emb_a, "model_b": emb_b},
        )
        assert node.embeddings["model_a"] == emb_a
        assert node.embeddings["model_b"] == emb_b

    def test_empty_text(self, embedding_model_name, sample_embedding):
        node = Node(text="", index=99, children=set(), embeddings={embedding_model_name: sample_embedding})
        assert node.text == ""

    def test_node_pickle_roundtrip(self, sample_node):
        data = pickle.dumps(sample_node)
        restored = pickle.loads(data)
        assert restored.text == sample_node.text
        assert restored.index == sample_node.index
        assert restored.children == sample_node.children
        assert restored.embeddings == sample_node.embeddings


class TestTree:
    def test_basic_construction(self, sample_tree):
        assert sample_tree.num_layers == 1
        assert len(sample_tree.all_nodes) == 7  # 5 leaf + 2 parent
        assert len(sample_tree.root_nodes) == 2
        assert len(sample_tree.leaf_nodes) == 5
        assert 0 in sample_tree.layer_to_nodes
        assert 1 in sample_tree.layer_to_nodes

    def test_layer_to_nodes_contents(self, sample_tree):
        layer_0 = sample_tree.layer_to_nodes[0]
        layer_1 = sample_tree.layer_to_nodes[1]
        assert len(layer_0) == 5
        assert len(layer_1) == 2

    def test_all_nodes_contains_both_layers(self, sample_tree):
        for idx in range(7):
            assert idx in sample_tree.all_nodes

    def test_root_nodes_have_children(self, sample_tree):
        for node in sample_tree.root_nodes.values():
            assert len(node.children) > 0

    def test_leaf_nodes_have_no_children(self, sample_tree):
        for node in sample_tree.leaf_nodes.values():
            assert len(node.children) == 0

    def test_tree_pickle_roundtrip(self, sample_tree):
        data = pickle.dumps(sample_tree)
        restored = pickle.loads(data)

        assert restored.num_layers == sample_tree.num_layers
        assert len(restored.all_nodes) == len(sample_tree.all_nodes)
        assert len(restored.root_nodes) == len(sample_tree.root_nodes)
        assert len(restored.leaf_nodes) == len(sample_tree.leaf_nodes)

        for idx in sample_tree.all_nodes:
            assert idx in restored.all_nodes
            assert restored.all_nodes[idx].text == sample_tree.all_nodes[idx].text
            assert restored.all_nodes[idx].children == sample_tree.all_nodes[idx].children

    def test_tree_pickle_file_roundtrip(self, sample_tree):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(sample_tree, f)
            path = f.name

        with open(path, "rb") as f:
            restored = pickle.load(f)

        assert isinstance(restored, Tree)
        assert restored.num_layers == sample_tree.num_layers
        assert set(restored.all_nodes.keys()) == set(sample_tree.all_nodes.keys())

    def test_single_node_tree(self, embedding_model_name):
        rng = np.random.default_rng(0)
        node = Node("solo", 0, set(), {embedding_model_name: rng.standard_normal(4).tolist()})
        tree = Tree(
            all_nodes={0: node},
            root_nodes={0: node},
            leaf_nodes={0: node},
            num_layers=0,
            layer_to_nodes={0: [node]},
        )
        assert tree.num_layers == 0
        assert len(tree.all_nodes) == 1

    def test_three_layer_tree(self, embedding_model_name):
        """Build a 3-layer tree (leaves -> mid -> root) and verify structure."""
        rng = np.random.default_rng(789)
        leaves = []
        for i in range(4):
            leaves.append(Node(f"leaf-{i}", i, set(), {embedding_model_name: rng.standard_normal(4).tolist()}))

        mid0 = Node("mid-0", 4, {0, 1}, {embedding_model_name: rng.standard_normal(4).tolist()})
        mid1 = Node("mid-1", 5, {2, 3}, {embedding_model_name: rng.standard_normal(4).tolist()})
        root = Node("root", 6, {4, 5}, {embedding_model_name: rng.standard_normal(4).tolist()})

        all_nodes = {n.index: n for n in [*leaves, mid0, mid1, root]}
        tree = Tree(
            all_nodes=all_nodes,
            root_nodes={6: root},
            leaf_nodes={n.index: n for n in leaves},
            num_layers=2,
            layer_to_nodes={0: leaves, 1: [mid0, mid1], 2: [root]},
        )

        assert tree.num_layers == 2
        assert len(tree.all_nodes) == 7
        assert len(tree.leaf_nodes) == 4
        assert len(tree.root_nodes) == 1
        assert root.children == {4, 5}
