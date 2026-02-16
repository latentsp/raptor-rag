import logging
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tiktoken
from tqdm import tqdm

from .embedding_models import BaseEmbeddingModel, LiteLLMEmbeddingModel
from .retrievers import BaseRetriever
from .utils import split_text

logger = logging.getLogger(__name__)


class FaissRetrieverConfig:
    def __init__(
        self,
        max_tokens=100,
        max_context_tokens=3500,
        use_top_k=False,
        embedding_model=None,
        question_embedding_model=None,
        top_k=5,
        tokenizer=None,
        embedding_model_string=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if max_context_tokens is not None and max_context_tokens < 1:
            raise ValueError("max_context_tokens must be at least 1 or None")

        if embedding_model is not None and not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError("embedding_model must be an instance of BaseEmbeddingModel or None")

        if question_embedding_model is not None and not isinstance(question_embedding_model, BaseEmbeddingModel):
            raise ValueError("question_embedding_model must be an instance of BaseEmbeddingModel or None")

        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.use_top_k = use_top_k
        self.embedding_model = embedding_model or LiteLLMEmbeddingModel()
        self.question_embedding_model = question_embedding_model or self.embedding_model
        self.tokenizer = tokenizer
        self.embedding_model_string = embedding_model_string or "default"

    def log_config(self):
        config_summary = f"""
        FaissRetrieverConfig:
            Max Tokens: {self.max_tokens}
            Max Context Tokens: {self.max_context_tokens}
            Use Top K: {self.use_top_k}
            Embedding Model: {self.embedding_model}
            Question Embedding Model: {self.question_embedding_model}
            Top K: {self.top_k}
            Tokenizer: {self.tokenizer}
            Embedding Model String: {self.embedding_model_string}
        """
        return config_summary


class FaissRetriever(BaseRetriever):
    """Flat FAISS retriever over text chunks with optional separate question encoder."""

    def __init__(self, config):
        self.config = config
        self.index = None
        self.embeddings = None
        self.context_chunks = None

    def build_from_text(self, doc_text):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss is required. Install with: pip install raptor-rag[faiss]") from None

        self.context_chunks = split_text(doc_text, self.config.tokenizer, self.config.max_tokens)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.config.embedding_model.create_embedding, context_chunk)
                for context_chunk in self.context_chunks
            ]

            self.embeddings = []
            for future in tqdm(futures, total=len(futures), desc="Building embeddings"):
                self.embeddings.append(future.result())

        self.embeddings = np.array(self.embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def build_from_leaf_nodes(self, leaf_nodes):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss is required. Install with: pip install raptor-rag[faiss]") from None

        self.context_chunks = [node.text for node in leaf_nodes]

        self.embeddings = np.array(
            [node.embeddings[self.config.embedding_model_string] for node in leaf_nodes],
            dtype=np.float32,
        )

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def sanity_check(self, num_samples=4):
        if self.index is None:
            raise ValueError("Index not built. Call build_from_text() or build_from_leaf_nodes() first.")

        indices = random.sample(range(len(self.context_chunks)), num_samples)

        for i in indices:
            original_embedding = self.embeddings[i]
            recomputed_embedding = self.config.embedding_model.create_embedding(self.context_chunks[i])
            if not np.allclose(original_embedding, recomputed_embedding):
                raise ValueError(f"Embeddings do not match for index {i}")

        logger.info("Sanity check passed for %d random samples.", num_samples)

    def retrieve(self, query: str) -> str:
        if self.index is None:
            raise ValueError("Index not built. Call build_from_text() or build_from_leaf_nodes() first.")

        query_embedding = np.array(
            [self.config.question_embedding_model.create_embedding(query)],
            dtype=np.float32,
        )

        if self.config.use_top_k:
            _, indices = self.index.search(query_embedding, self.config.top_k)
            selected = [self.context_chunks[indices[0][i]] for i in range(self.config.top_k)]
        else:
            search_count = int(self.config.max_context_tokens / self.config.max_tokens)
            _, indices = self.index.search(query_embedding, search_count)
            selected = []
            total_tokens = 0
            for i in range(search_count):
                chunk = self.context_chunks[indices[0][i]]
                chunk_tokens = len(self.config.tokenizer.encode(chunk))
                if total_tokens + chunk_tokens > self.config.max_context_tokens:
                    break
                selected.append(chunk)
                total_tokens += chunk_tokens

        return "".join(selected)
