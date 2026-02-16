"""
Base benchmark runner class.

Subclasses implement load_dataset(), evaluate_single(), and aggregate_metrics().
The base class handles tree building/caching, progress tracking, and CLI args.
"""

from __future__ import annotations

import argparse
import logging
import time
from abc import ABC, abstractmethod

from ._progress import ProgressTracker
from ._tree_cache import TreeCache

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Abstract base for all RAPTOR benchmarks."""

    name: str = "base"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = args.model
        self.top_k = args.top_k
        self.max_tokens = args.max_tokens
        self.num_layers = args.num_layers
        self.summarization_length = args.summarization_length
        self.collapse_tree = args.collapse_tree == "true"
        self.reranker = self._make_reranker(args.reranker, args.reranker_model)
        self.cache = TreeCache(self.name, cache_dir=args.cache_dir)
        self.progress = ProgressTracker(args.output, no_resume=args.no_resume)

    @abstractmethod
    def load_dataset(self) -> tuple[list[dict], list[dict]]:
        """Return (documents, questions).

        Each document dict must have at least 'doc_id' and 'text'.
        Each question dict must have at least 'question_id', 'doc_id', and 'question'.
        """

    @abstractmethod
    def evaluate_single(self, question: dict, context: str, layer_info: list[dict]) -> dict:
        """Evaluate a single question. Return a result dict including 'question_id'."""

    @abstractmethod
    def aggregate_metrics(self, results: list[dict]) -> dict:
        """Compute aggregate metrics from individual results."""

    @staticmethod
    def _make_reranker(reranker_type, reranker_model):
        """Create a reranker instance based on CLI args."""
        if reranker_type is None:
            return None
        if reranker_type == "cross_encoder":
            from raptor import CrossEncoderReRanker

            model = reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            return CrossEncoderReRanker(model_name=model)
        if reranker_type == "litellm":
            from raptor import LiteLLMReRanker

            model = reranker_model or "cohere/rerank-english-v3.0"
            return LiteLLMReRanker(model=model)
        return None

    def _make_models(self):
        """Create summarization and QA model instances based on model name."""
        from raptor import LiteLLMQAModel, LiteLLMSummarizationModel

        return (
            LiteLLMSummarizationModel(model=self.model),
            LiteLLMQAModel(model=self.model),
        )

    def _make_qa_model(self):
        """Create a QA model instance based on model name."""
        from raptor import LiteLLMQAModel

        return LiteLLMQAModel(model=self.model)

    def build_or_load_tree(self, doc: dict):
        """Build a RAPTOR tree for a document, or load from cache."""
        from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

        doc_id = doc["doc_id"]
        text = doc["text"]
        summarization_model, qa_model = self._make_models()

        config = RetrievalAugmentationConfig(
            summarization_model=summarization_model,
            qa_model=qa_model,
            tb_num_layers=self.num_layers,
            tb_max_tokens=self.max_tokens,
            tb_summarization_length=self.summarization_length,
            tr_reranker=self.reranker,
        )

        cached = self.cache.load(doc_id, self.model, self.num_layers, self.max_tokens, self.summarization_length)
        if cached is not None:
            ra = RetrievalAugmentation(config=config, tree=cached)
            return ra

        logger.info("Building tree for doc_id=%r (%d chars)...", doc_id, len(text))
        ra = RetrievalAugmentation(config=config)
        ra.add_documents(text)
        self.cache.save(doc_id, self.model, self.num_layers, self.max_tokens, ra.tree, self.summarization_length)
        return ra

    def run(self) -> dict:
        """Orchestrate: load data -> build trees -> evaluate -> save."""
        logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

        logger.info("=== %s Benchmark ===", self.name)
        logger.info(
            "Model: %s, top_k: %d, max_tokens: %d, reranker: %s, collapse_tree: %s",
            self.model, self.top_k, self.max_tokens, self.args.reranker, self.collapse_tree,
        )

        docs, questions = self.load_dataset()
        logger.info("Loaded %d documents, %d questions", len(docs), len(questions))

        trees_by_doc_id: dict[str, object] = {}
        for doc in docs:
            doc_id = doc["doc_id"]
            if doc_id not in trees_by_doc_id:
                trees_by_doc_id[doc_id] = self.build_or_load_tree(doc)

        results = list(self.progress.results)  # start with resumed results
        total = len(questions)
        for q in questions:
            qid = q["question_id"]
            if self.progress.is_done(qid):
                continue

            doc_id = q["doc_id"]
            ra = trees_by_doc_id[doc_id]

            start = time.perf_counter()
            context, layer_info = ra.retrieve(
                q["question"],
                top_k=self.top_k,
                max_tokens=3500,
                collapse_tree=self.collapse_tree,
                return_layer_information=True,
            )
            retrieval_time = time.perf_counter() - start

            result = self.evaluate_single(q, context, layer_info)
            result["question_id"] = qid
            result["retrieval_time_s"] = round(retrieval_time, 3)

            results.append(result)
            self.progress.record(result)
            logger.info("[%d/%d] %s: %s", len(results), total, qid, _preview_metrics(result))

        metrics = self.aggregate_metrics(results)
        output = {
            "benchmark": self.name,
            "config": {
                "model": self.model,
                "top_k": self.top_k,
                "max_tokens": self.max_tokens,
                "num_layers": self.num_layers,
                "collapse_tree": self.collapse_tree,
                "reranker": self.args.reranker,
                "reranker_model": self.args.reranker_model,
                "num_docs": len(docs),
                "num_questions": len(questions),
            },
            "metrics": metrics,
            "results": results,
        }
        self.progress.save_final(output)
        logger.info("=== %s Complete ===", self.name)
        logger.info("Metrics: %s", metrics)
        return output

    @staticmethod
    def add_common_args(parser: argparse.ArgumentParser) -> None:
        """Add CLI args shared by all benchmarks."""
        parser.add_argument(
            "--model", type=str, default="gpt-4.1-nano", help="LLM model for summarization/QA (any LiteLLM model)"
        )
        parser.add_argument("--max_docs", type=int, default=None, help="Limit number of documents")
        parser.add_argument("--max_questions", type=int, default=None, help="Limit number of questions")
        parser.add_argument("--top_k", type=int, default=5, help="Top-k for retrieval")
        parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens per chunk for tree building")
        parser.add_argument("--num_layers", type=int, default=5, help="Number of tree layers")
        parser.add_argument("--summarization_length", type=int, default=100, help="Max tokens for summarization (increase for newer models like gpt-5-mini)")
        parser.add_argument("--cache_dir", type=str, default=None, help="Tree cache directory")
        parser.add_argument("--output", type=str, default=None, help="Output JSON path")
        parser.add_argument("--no_resume", action="store_true", help="Don't resume from partial results")
        parser.add_argument(
            "--reranker", type=str, default=None, choices=["cross_encoder", "litellm"],
            help="Reranker to apply after retrieval (default: none)",
        )
        parser.add_argument(
            "--reranker_model", type=str, default=None,
            help="Model for reranker (default: cross-encoder/ms-marco-MiniLM-L-6-v2 or cohere/rerank-english-v3.0)",
        )
        parser.add_argument(
            "--collapse_tree", type=str, default="true", choices=["true", "false"],
            help="Use collapsed tree retrieval (default: true)",
        )


def _preview_metrics(result: dict) -> str:
    """Format a one-line preview of a result dict's numeric values."""
    parts = []
    for k, v in result.items():
        if k in ("question_id", "question", "answer", "predicted", "gold", "gold_answers", "doc_id"):
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        elif isinstance(v, (int, bool)):
            parts.append(f"{k}={v}")
    return ", ".join(parts)
