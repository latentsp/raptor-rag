from abc import ABC, abstractmethod

import litellm

from .tree_structures import Node


class BaseReRanker(ABC):
    @abstractmethod
    def rerank(self, query: str, nodes: list[Node], top_k: int = 5) -> list[Node]:
        pass


class CrossEncoderReRanker(BaseReRanker):
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install raptor-rag[reranker]"
            ) from None
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, nodes: list[Node], top_k: int = 5) -> list[Node]:
        if not nodes:
            return []

        pairs = [[query, node.text] for node in nodes]
        scores = self.model.predict(pairs)

        scored_nodes = sorted(zip(scores, nodes, strict=False), key=lambda x: x[0], reverse=True)
        return [node for _, node in scored_nodes[:top_k]]


class LiteLLMReRanker(BaseReRanker):
    def __init__(self, model="cohere/rerank-english-v3.0"):
        self.model = model

    def rerank(self, query: str, nodes: list[Node], top_k: int = 5) -> list[Node]:
        if not nodes:
            return []

        documents = [node.text for node in nodes]
        response = litellm.rerank(model=self.model, query=query, documents=documents, top_n=top_k)

        return [nodes[result.index] for result in response.results]
