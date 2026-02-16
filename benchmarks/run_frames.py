"""
FRAMES Benchmark — Multi-hop QA over Wikipedia articles.

Dataset: google/frames-benchmark (HuggingFace)
Metric: Accuracy (normalized string match)
Note: Not in original RAPTOR paper — newer benchmark for multi-hop RAG.

The dataset provides Wikipedia URLs per question. This benchmark fetches
article text via the MediaWiki API and caches it locally.

Usage:
    OPENAI_API_KEY=... python benchmarks/run_frames.py --max_questions 20
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests

from ._base import BaseBenchmark
from ._metrics import accuracy

logger = logging.getLogger(__name__)

WIKI_CACHE_DIR = Path(__file__).parent / ".cache" / "frames_wiki"
WIKI_API = "https://en.wikipedia.org/w/api.php"


def _title_from_url(url: str) -> str | None:
    """Extract Wikipedia article title from URL."""
    match = re.search(r"wikipedia\.org/wiki/(.+?)(?:#.*)?$", url)
    if match:
        return unquote(match.group(1)).replace("_", " ")
    return None


def _fetch_wikipedia_text(title: str) -> str:
    """Fetch plain text of a Wikipedia article via MediaWiki API. Cached."""
    cache_file = WIKI_CACHE_DIR / f"{title.replace('/', '_')[:100]}.txt"
    if cache_file.exists():
        return cache_file.read_text()

    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    headers = {"User-Agent": "raptor-rag-benchmark/0.2.0 (https://github.com/latentsp/raptor-rag)"}
    try:
        resp = requests.get(WIKI_API, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            if text:
                WIKI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(text)
                return text
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch Wikipedia article '{title}': {e}")

    return ""


def _get_wiki_texts(row: dict) -> list[tuple[str, str]]:
    """Get (title, text) pairs for all Wikipedia articles in a FRAMES row."""
    # Collect all Wikipedia URLs
    urls = []

    # From wiki_links field (string representation of a list)
    wiki_links = row.get("wiki_links", "")
    if wiki_links:
        try:
            parsed = ast.literal_eval(wiki_links)
            if isinstance(parsed, list):
                urls.extend(parsed)
        except (ValueError, SyntaxError):
            pass

    # From individual link columns
    for i in range(1, 11):
        link = row.get(f"wikipedia_link_{i}", "")
        if link:
            urls.append(link)
    link_11 = row.get("wikipedia_link_11+", "")
    if link_11:
        urls.append(link_11)

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            unique_urls.append(url)

    results = []
    for url in unique_urls:
        title = _title_from_url(url)
        if title:
            text = _fetch_wikipedia_text(title)
            if text:
                results.append((title, text))
            time.sleep(0.1)  # Be polite to Wikipedia API

    return results


class FRAMESBenchmark(BaseBenchmark):
    name = "frames"

    def __init__(self, args: argparse.Namespace):
        if args.output is None:
            args.output = "benchmarks/results_frames.json"
        super().__init__(args)

    def load_dataset(self) -> tuple[list[dict], list[dict]]:
        from datasets import load_dataset

        ds = load_dataset("google/frames-benchmark", split="test")

        docs: list[dict] = []
        questions: list[dict] = []

        for i, row in enumerate(ds):
            if self.args.max_questions and len(questions) >= self.args.max_questions:
                break

            # Fetch Wikipedia articles for this question
            articles = _get_wiki_texts(row)
            if not articles:
                logger.warning(f"Skipping question {i}: no Wikipedia articles fetched")
                continue

            # Concatenate all articles
            parts = []
            for title, text in articles:
                parts.append(f"# {title}\n\n{text}")
            combined_text = "\n\n---\n\n".join(parts)

            # One doc per question (unique articles per question)
            doc_id = f"frames_q{i}"
            docs.append({"doc_id": doc_id, "text": combined_text})

            answer = row.get("Answer", row.get("answer", ""))
            if not answer:
                continue

            questions.append(
                {
                    "question_id": doc_id,
                    "doc_id": doc_id,
                    "question": row.get("Prompt", row.get("prompt", "")),
                    "gold": str(answer),
                }
            )

        logger.info(f"FRAMES: {len(docs)} question-doc pairs, {len(questions)} questions")
        return docs, questions

    def evaluate_single(self, question: dict, context: str, layer_info: list[dict]) -> dict:
        qa = self._make_qa_model()
        answer = qa.answer_question(context, question["question"])
        gold = question["gold"]
        acc = accuracy(answer, gold)

        return {
            "answer": answer[:500],
            "gold": gold,
            "correct": acc,
        }

    def aggregate_metrics(self, results: list[dict]) -> dict:
        if not results:
            return {"accuracy": 0.0, "n": 0}

        correct = [r["correct"] for r in results]
        return {
            "accuracy": sum(correct) / len(correct),
            "n": len(results),
        }


def main():
    parser = argparse.ArgumentParser(description="Run FRAMES benchmark")
    BaseBenchmark.add_common_args(parser)
    args = parser.parse_args()

    bench = FRAMESBenchmark(args)
    bench.run()


if __name__ == "__main__":
    main()
