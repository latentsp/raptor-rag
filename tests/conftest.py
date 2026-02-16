"""Shared fixtures for the raptor-rag test suite."""

import pickle

import numpy as np
import pytest
import tiktoken

from raptor.tree_structures import Node, Tree


@pytest.fixture
def tokenizer():
    """Return the default cl100k_base tiktoken encoding."""
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture
def sample_text():
    """A multi-sentence paragraph suitable for splitting tests."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
        "The five boxing wizards jump quickly. "
        "Bright vixens jump; dozy fowl quack. "
        "Jinxed wizards pluck ivy from the big quilt."
    )


@pytest.fixture
def long_text():
    """A longer document with multiple paragraphs for more thorough splitting."""
    paragraphs = [
        "Machine learning is a branch of artificial intelligence that focuses on building systems "
        "that learn from data. It involves algorithms that improve through experience without being "
        "explicitly programmed. Modern machine learning draws on concepts from statistics, information "
        "theory, and computer science.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers. "
        "These networks can automatically learn representations of data at multiple levels of abstraction. "
        "Convolutional neural networks are commonly used for image recognition tasks.",
        "Natural language processing enables computers to understand and generate human language. "
        "Recent advances in transformer architectures have revolutionized the field. "
        "Large language models can now perform translation, summarization, and question answering.",
        "Reinforcement learning trains agents to make decisions by interacting with an environment. "
        "The agent receives rewards or penalties based on its actions. "
        "This approach has been successful in game playing and robotics.",
    ]
    return "\n".join(paragraphs)


@pytest.fixture
def embedding_model_name():
    """The name key used in node embeddings dictionaries."""
    return "test_model"


@pytest.fixture
def sample_embedding(embedding_model_name):
    """A deterministic 8-dimensional embedding vector."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(8).tolist()
    return vec


@pytest.fixture
def sample_node(embedding_model_name, sample_embedding):
    """A single leaf Node with no children."""
    return Node(
        text="The quick brown fox jumps over the lazy dog.",
        index=0,
        children=set(),
        embeddings={embedding_model_name: sample_embedding},
    )


@pytest.fixture
def sample_nodes(embedding_model_name):
    """A list of 5 leaf Nodes with random embeddings, suitable for tree construction."""
    rng = np.random.default_rng(123)
    nodes = []
    texts = [
        "Artificial intelligence is transforming many industries.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables machines to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning trains agents through trial and error.",
    ]
    for i, text in enumerate(texts):
        emb = rng.standard_normal(8).tolist()
        nodes.append(Node(text=text, index=i, children=set(), embeddings={embedding_model_name: emb}))
    return nodes


@pytest.fixture
def sample_tree(sample_nodes, embedding_model_name):
    """A two-layer tree: 5 leaf nodes and 2 parent nodes."""
    rng = np.random.default_rng(456)
    leaf_dict = {n.index: n for n in sample_nodes}

    # Create two parent nodes that group the leaves
    parent0 = Node(
        text="Summary of AI and deep learning concepts.",
        index=5,
        children={0, 1},
        embeddings={embedding_model_name: rng.standard_normal(8).tolist()},
    )
    parent1 = Node(
        text="Summary of NLP, vision, and RL concepts.",
        index=6,
        children={2, 3, 4},
        embeddings={embedding_model_name: rng.standard_normal(8).tolist()},
    )

    all_nodes = {**leaf_dict, 5: parent0, 6: parent1}
    root_nodes = {5: parent0, 6: parent1}
    layer_to_nodes = {0: sample_nodes, 1: [parent0, parent1]}

    return Tree(
        all_nodes=all_nodes,
        root_nodes=root_nodes,
        leaf_nodes=leaf_dict,
        num_layers=1,
        layer_to_nodes=layer_to_nodes,
    )


@pytest.fixture
def tree_pickle_path(sample_tree, tmp_path):
    """Write sample_tree to a temporary pickle file and return the path."""
    path = tmp_path / "tree.pkl"
    with open(path, "wb") as f:
        pickle.dump(sample_tree, f)
    return str(path)
