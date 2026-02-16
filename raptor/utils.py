import re
from abc import ABC, abstractmethod

import numpy as np
from scipy import spatial

from .tree_structures import Node


def reverse_mapping(layer_to_nodes: dict[int, list[Node]]) -> dict[int, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(text: str, tokenizer, max_tokens: int, overlap: int = 0):
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)

    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens, strict=False):
        if not sentence.strip():
            continue

        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts, strict=False):
                if sub_length + sub_token_count > max_tokens and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                    sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap) : len(sub_chunk)])

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap) : len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count

        else:
            current_chunk.append(sentence)
            current_length += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class BaseTextSplitter(ABC):
    @abstractmethod
    def split_text(self, text: str, tokenizer, max_tokens: int) -> list[str]:
        pass


class DefaultTextSplitter(BaseTextSplitter):
    def __init__(self, overlap: int = 0):
        self.overlap = overlap

    def split_text(self, text: str, tokenizer, max_tokens: int) -> list[str]:
        return split_text(text, tokenizer, max_tokens, overlap=self.overlap)


def distances_from_embeddings(
    query_embedding: list[float],
    embeddings: list[list[float]],
    distance_metric: str = "cosine",
) -> list[float]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [distance_metrics[distance_metric](query_embedding, embedding) for embedding in embeddings]
    return distances


def get_node_list(node_dict: dict[int, Node]) -> list[Node]:
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: list[Node], embedding_model: str) -> list:
    return [node.embeddings[embedding_model] for node in node_list]


def get_text(node_list: list[Node]) -> str:
    if not node_list:
        return ""
    return "\n\n".join(" ".join(node.text.splitlines()) for node in node_list) + "\n\n"


def indices_of_nearest_neighbors_from_distances(distances: list[float]) -> np.ndarray:
    return np.argsort(distances)
