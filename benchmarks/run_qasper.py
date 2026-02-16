"""
QASPER Benchmark — QA on scientific NLP papers.

Dataset: allenai/qasper v0.3 (downloaded from S3)
Metric: Token F1 (max across annotators)
Paper result: 55.7% (GPT-4, RAPTOR)

Usage:
    OPENAI_API_KEY=... python benchmarks/run_qasper.py --max_docs 5 --max_questions 20
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import tarfile
from pathlib import Path

import requests

from ._base import BaseBenchmark
from ._metrics import max_token_f1

logger = logging.getLogger(__name__)

QASPER_TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
QASPER_CACHE_DIR = Path(__file__).parent / ".cache" / "qasper_data"


def _download_qasper_test() -> dict:
    """Download and cache QASPER test set JSON."""
    cache_file = QASPER_CACHE_DIR / "qasper-test-v0.3.json"
    if cache_file.exists():
        logger.info(f"Loading cached QASPER test data from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    logger.info(f"Downloading QASPER test set from {QASPER_TEST_URL}...")
    resp = requests.get(QASPER_TEST_URL, timeout=120)
    resp.raise_for_status()

    # Extract JSON from tarball
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".json") and "test" in member.name.lower():
                f = tar.extractfile(member)
                if f is not None:
                    data = json.load(f)
                    # Cache for next time
                    QASPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, "w") as out:
                        json.dump(data, out)
                    logger.info(f"Cached QASPER test data to {cache_file}")
                    return data

    raise RuntimeError("Could not find test JSON in QASPER tarball")


class QASPERBenchmark(BaseBenchmark):
    name = "qasper"

    def __init__(self, args: argparse.Namespace):
        if args.output is None:
            args.output = "benchmarks/results_qasper.json"
        super().__init__(args)

    def load_dataset(self) -> tuple[list[dict], list[dict]]:
        data = _download_qasper_test()

        docs: list[dict] = []
        questions: list[dict] = []

        for paper_id, paper in data.items():
            # Reconstruct full paper text
            parts = []
            if paper.get("title"):
                parts.append(paper["title"])
            if paper.get("abstract"):
                parts.append(paper["abstract"])
            for section in paper.get("full_text", []):
                section_name = section.get("section_name", "")
                if section_name:
                    parts.append(section_name)
                for para in section.get("paragraphs", []):
                    if para:
                        parts.append(para)

            text = "\n\n".join(parts)
            if not text.strip():
                continue

            docs.append({"doc_id": paper_id, "text": text})
            if self.args.max_docs and len(docs) > self.args.max_docs:
                docs.pop()
                break

            for qa in paper.get("qas", []):
                q_text = qa.get("question", "")
                q_id = qa.get("question_id", f"{paper_id}_{len(questions)}")

                answer_texts = []
                answer_type = "abstractive"
                for ans in qa.get("answers", []):
                    ans_data = ans.get("answer", {})
                    if ans_data.get("unanswerable"):
                        answer_texts.append("unanswerable")
                        answer_type = "unanswerable"
                    elif ans_data.get("yes_no") is not None:
                        answer_texts.append("yes" if ans_data["yes_no"] else "no")
                        answer_type = "yes_no"
                    elif ans_data.get("extractive_spans"):
                        answer_texts.append(" ".join(ans_data["extractive_spans"]))
                        answer_type = "extractive"
                    elif ans_data.get("free_form_answer"):
                        answer_texts.append(ans_data["free_form_answer"])
                        answer_type = "abstractive"

                if not answer_texts:
                    continue

                questions.append(
                    {
                        "question_id": q_id,
                        "doc_id": paper_id,
                        "question": q_text,
                        "gold_answers": answer_texts,
                        "answer_type": answer_type,
                    }
                )

                if self.args.max_questions and len(questions) >= self.args.max_questions:
                    break

            if self.args.max_questions and len(questions) >= self.args.max_questions:
                break

        logger.info(f"QASPER: {len(docs)} papers, {len(questions)} questions")
        return docs, questions

    def evaluate_single(self, question: dict, context: str, layer_info: list[dict]) -> dict:
        qa = self._make_qa_model()
        answer = qa.answer_question(context, question["question"])
        golds = question["gold_answers"]
        f1 = max_token_f1(answer, golds)

        return {
            "answer": answer[:500],
            "gold_answers": golds,
            "answer_type": question["answer_type"],
            "token_f1": f1,
        }

    def aggregate_metrics(self, results: list[dict]) -> dict:
        if not results:
            return {"token_f1": 0.0, "n": 0}

        f1s = [r["token_f1"] for r in results]

        # Per answer-type breakdown
        by_type: dict[str, list[float]] = {}
        for r in results:
            t = r.get("answer_type", "unknown")
            by_type.setdefault(t, []).append(r["token_f1"])

        type_metrics = {t: sum(v) / len(v) for t, v in by_type.items()}

        return {
            "token_f1": sum(f1s) / len(f1s),
            "n": len(results),
            "by_answer_type": type_metrics,
        }


def main():
    parser = argparse.ArgumentParser(description="Run QASPER benchmark")
    BaseBenchmark.add_common_args(parser)
    args = parser.parse_args()

    bench = QASPERBenchmark(args)
    bench.run()


if __name__ == "__main__":
    main()
