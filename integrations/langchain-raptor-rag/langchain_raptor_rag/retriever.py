"""LangChain retriever integration for RAPTOR RAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun


class RaptorRetriever(BaseRetriever):
    """LangChain retriever that wraps RAPTOR's RetrievalAugmentation.

    Retrieves documents from a RAPTOR tree using collapsed tree or
    tree traversal retrieval, returning LangChain Document objects
    with node metadata.
    """

    ra: Any = Field(description="A raptor.RetrievalAugmentation instance with a built tree.")
    top_k: int = Field(default=10, description="Maximum number of nodes to retrieve.")
    max_tokens: int = Field(default=3500, description="Maximum total tokens across retrieved nodes.")
    collapse_tree: bool = Field(default=True, description="Use collapsed tree retrieval (recommended).")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents from the RAPTOR tree.

        Args:
            query: The query string to retrieve documents for.
            run_manager: Optional callback manager (unused).

        Returns:
            A list of LangChain Document objects with page_content set to
            the node text and metadata containing node_index and layer_number.
        """
        _context, layer_info = self.ra.retrieve(
            query,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            collapse_tree=self.collapse_tree,
            return_layer_information=True,
        )

        documents = []
        for info in layer_info:
            node_index = info["node_index"]
            layer_number = info["layer_number"]
            node = self.ra.retriever.tree.all_nodes[node_index]
            documents.append(
                Document(
                    page_content=node.text,
                    metadata={
                        "node_index": node_index,
                        "layer_number": layer_number,
                    },
                )
            )

        return documents
