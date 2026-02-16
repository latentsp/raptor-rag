"""LlamaIndex retriever integration for RAPTOR RAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

if TYPE_CHECKING:
    from raptor.retrieval_augmentation import RetrievalAugmentation


class RaptorRetriever(BaseRetriever):
    """LlamaIndex retriever that wraps RAPTOR's RetrievalAugmentation.

    Retrieves nodes from a RAPTOR tree using collapsed tree or
    tree traversal retrieval, returning LlamaIndex NodeWithScore objects.
    """

    def __init__(
        self,
        ra: RetrievalAugmentation,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the RAPTOR retriever.

        Args:
            ra: A raptor.RetrievalAugmentation instance with a built tree.
            top_k: Maximum number of nodes to retrieve.
            max_tokens: Maximum total tokens across retrieved nodes.
            collapse_tree: Use collapsed tree retrieval (recommended).
            **kwargs: Additional keyword arguments passed to BaseRetriever.
        """
        super().__init__(**kwargs)
        self._ra = ra
        self._top_k = top_k
        self._max_tokens = max_tokens
        self._collapse_tree = collapse_tree

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve relevant nodes from the RAPTOR tree.

        Args:
            query_bundle: The query bundle containing the query string.

        Returns:
            A list of NodeWithScore objects wrapping TextNode instances
            with metadata containing node_index and layer_number.
        """
        _context, layer_info = self._ra.retrieve(
            query_bundle.query_str,
            top_k=self._top_k,
            max_tokens=self._max_tokens,
            collapse_tree=self._collapse_tree,
            return_layer_information=True,
        )

        results = []
        num_nodes = len(layer_info)
        for rank, info in enumerate(layer_info):
            node_index = info["node_index"]
            layer_number = info["layer_number"]
            node = self._ra.retriever.tree.all_nodes[node_index]

            text_node = TextNode(
                text=node.text,
                metadata={
                    "node_index": node_index,
                    "layer_number": layer_number,
                },
            )

            # Assign a relevance score based on retrieval rank (1.0 for first, decreasing).
            score = 1.0 - (rank / num_nodes) if num_nodes > 1 else 1.0
            results.append(NodeWithScore(node=text_node, score=score))

        return results
