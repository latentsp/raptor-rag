"""
QuALITY Benchmark — Long-document multiple-choice QA.

Dataset: tasksource/QuALITY (HuggingFace)
Metric: Accuracy (overall + QuALITY-HARD subset)
Paper result: 82.6% (GPT-4, RAPTOR)

Usage:
    OPENAI_API_KEY=... python benchmarks/run_quality.py --max_docs 5 --max_questions 20
"""

from __future__ import annotations

import argparse
import logging

from ._base import BaseBenchmark
from ._metrics import accuracy

logger = logging.getLogger(__name__)

MC_PROMPT = """\
Based on the following context, answer the multiple-choice question.
Respond with ONLY the letter (A, B, C, or D).

Context:
{context}

Question: {question}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Answer (A, B, C, or D):"""


class QuALITYBenchmark(BaseBenchmark):
    name = "quality"

    def __init__(self, args: argparse.Namespace):
        if args.output is None:
            args.output = "benchmarks/results_quality.json"
        super().__init__(args)
        self.split = args.split
        self.hard_only = args.hard_only

    def load_dataset(self) -> tuple[list[dict], list[dict]]:
        from datasets import load_dataset

        ds = load_dataset("tasksource/QuALITY", split=self.split)

        # Group questions by article
        articles: dict[str, dict] = {}
        questions: list[dict] = []

        for row in ds:
            article_id = row["article_id"]
            if article_id not in articles:
                articles[article_id] = {
                    "doc_id": article_id,
                    "text": row["article"],
                }
                if self.args.max_docs and len(articles) > self.args.max_docs:
                    articles.pop(article_id)
                    continue

            if article_id not in articles:
                continue

            is_hard = row.get("difficult", 0) == 1
            if self.hard_only and not is_hard:
                continue

            options = row["options"]
            # gold_label is 1-indexed in QuALITY
            gold_idx = row["gold_label"] - 1
            letters = ["A", "B", "C", "D"]

            questions.append(
                {
                    "question_id": f"{article_id}_{row['question_unique_id']}",
                    "doc_id": article_id,
                    "question": row["question"],
                    "options": options,
                    "gold_letter": letters[gold_idx] if 0 <= gold_idx < 4 else "A",
                    "is_hard": is_hard,
                }
            )

            if self.args.max_questions and len(questions) >= self.args.max_questions:
                break

        docs = list(articles.values())
        logger.info("QuALITY: %d articles, %d questions (hard_only=%s)", len(docs), len(questions), self.hard_only)
        return docs, questions

    def evaluate_single(self, question: dict, context: str, layer_info: list[dict]) -> dict:
        qa = self._make_qa_model()

        options = question["options"]
        prompt = MC_PROMPT.format(
            context=context,
            question=question["question"],
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        )

        response = qa.answer_question(context="", question=prompt)
        predicted = _parse_letter(response)
        gold = question["gold_letter"]

        return {
            "predicted": predicted,
            "gold": gold,
            "is_hard": question["is_hard"],
            "correct": accuracy(predicted, gold),
        }

    def aggregate_metrics(self, results: list[dict]) -> dict:
        if not results:
            return {"accuracy": 0.0, "hard_accuracy": 0.0, "n": 0}

        all_correct = [r["correct"] for r in results]
        hard_correct = [r["correct"] for r in results if r.get("is_hard")]

        return {
            "accuracy": sum(all_correct) / len(all_correct),
            "hard_accuracy": sum(hard_correct) / len(hard_correct) if hard_correct else 0.0,
            "n": len(results),
            "n_hard": len(hard_correct),
        }


def _parse_letter(response: str) -> str:
    """Extract A/B/C/D from model response."""
    response = response.strip().upper()
    for letter in ["A", "B", "C", "D"]:
        if response.startswith(letter):
            return letter
    # Fallback: find first letter mention
    for char in response:
        if char in "ABCD":
            return char
    return "A"


def main():
    parser = argparse.ArgumentParser(description="Run QuALITY benchmark")
    BaseBenchmark.add_common_args(parser)
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--hard_only", action="store_true", help="Only evaluate QuALITY-HARD questions")
    args = parser.parse_args()

    bench = QuALITYBenchmark(args)
    bench.run()


if __name__ == "__main__":
    main()
