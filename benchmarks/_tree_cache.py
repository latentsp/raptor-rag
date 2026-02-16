"""
Pickle-based tree caching to avoid rebuilding expensive RAPTOR trees.

Key: hash of (doc_id, model, tb_num_layers, tb_max_tokens, summarization_length)
Storage: <cache_dir>/<benchmark>/<hash>.pkl
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

from raptor.tree_structures import Tree

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"


class TreeCache:
    def __init__(self, benchmark_name: str, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR) / benchmark_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, doc_id: str, model: str, num_layers: int, max_tokens: int, summarization_length: int = 100) -> str:
        raw = f"{doc_id}|{model}|{num_layers}|{max_tokens}|{summarization_length}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def load(self, doc_id: str, model: str, num_layers: int, max_tokens: int, summarization_length: int = 100) -> Tree | None:
        key = self._key(doc_id, model, num_layers, max_tokens, summarization_length)
        path = self._path(key)
        if path.exists():
            logger.info("Loading cached tree for doc_id=%r", doc_id)
            with open(path, "rb") as f:
                tree = pickle.load(f)
            if isinstance(tree, Tree):
                return tree
            logger.warning("Cached object is not a Tree, ignoring: %s", path)
        return None

    def save(self, doc_id: str, model: str, num_layers: int, max_tokens: int, tree: Tree, summarization_length: int = 100) -> None:
        key = self._key(doc_id, model, num_layers, max_tokens, summarization_length)
        path = self._path(key)
        with open(path, "wb") as f:
            pickle.dump(tree, f)
        logger.info("Cached tree for doc_id=%r -> %s", doc_id, path)
