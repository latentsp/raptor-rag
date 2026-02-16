"""
NarrativeQA Benchmark — Book and script comprehension.

Dataset: deepmind/narrativeqa (HuggingFace)
Metrics: ROUGE-L, METEOR
Paper result: 30.87% ROUGE-L, 19.1 METEOR (GPT-4, RAPTOR)

Usage:
    OPENAI_API_KEY=... python benchmarks/run_narrativeqa.py --max_docs 5 --max_questions 20
"""

from __future__ import annotations

import argparse
import logging

from ._base import BaseBenchmark
from ._metrics import meteor, rouge_l

logger = logging.getLogger(__name__)


class NarrativeQABenchmark(BaseBenchmark):
    name = "narrativeqa"

    def __init__(self, args: argparse.Namespace):
        if args.output is None:
            args.output = "benchmarks/results_narrativeqa.json"
        super().__init__(args)
        self.use_summaries = args.use_summaries

    def load_dataset(self) -> tuple[list[dict], list[dict]]:
        from datasets import load_dataset

        ds = load_dataset("deepmind/narrativeqa", split="test")

        docs: dict[str, dict] = {}
        questions: list[dict] = []

        for row in ds:
            doc_id = row["document"]["id"]

            if doc_id not in docs:
                text = row["document"]["summary"]["text"] if self.use_summaries else row["document"]["text"]

                if not text or not text.strip():
                    continue

                docs[doc_id] = {"doc_id": doc_id, "text": text}
                if self.args.max_docs and len(docs) > self.args.max_docs:
                    docs.pop(doc_id)
                    continue

            if doc_id not in docs:
                continue

            # Each row has 2 reference answers
            answers = [a["text"] for a in row["answers"] if a.get("text")]
            if not answers:
                continue

            questions.append(
                {
                    "question_id": f"{doc_id}_{len(questions)}",
                    "doc_id": doc_id,
                    "question": row["question"]["text"],
                    "gold_answers": answers,
                }
            )

            if self.args.max_questions and len(questions) >= self.args.max_questions:
                break

        doc_list = list(docs.values())
        logger.info(
            "NarrativeQA: %d documents, %d questions (summaries=%s)",
            len(doc_list),
            len(questions),
            "yes" if self.use_summaries else "no",
        )
        return doc_list, questions

    def evaluate_single(self, question: dict, context: str, layer_info: list[dict]) -> dict:
        qa = self._make_qa_model()
        answer = qa.answer_question(context, question["question"])
        golds = question["gold_answers"]

        # Take max across references for each metric
        rouge = max(rouge_l(answer, g) for g in golds)
        met = max(meteor(answer, g) for g in golds)

        return {
            "answer": answer[:500],
            "gold_answers": golds,
            "rouge_l": rouge,
            "meteor": met,
        }

    def aggregate_metrics(self, results: list[dict]) -> dict:
        if not results:
            return {"rouge_l": 0.0, "meteor": 0.0, "n": 0}

        rouges = [r["rouge_l"] for r in results]
        meteors = [r["meteor"] for r in results]

        return {
            "rouge_l": sum(rouges) / len(rouges),
            "meteor": sum(meteors) / len(meteors),
            "n": len(results),
        }


def main():
    parser = argparse.ArgumentParser(description="Run NarrativeQA benchmark")
    BaseBenchmark.add_common_args(parser)
    parser.add_argument(
        "--use_summaries",
        action="store_true",
        default=True,
        help="Use document summaries instead of full text (default: True, much cheaper)",
    )
    parser.add_argument("--use_full_text", action="store_true", help="Use full document text instead of summaries")
    args = parser.parse_args()
    if args.use_full_text:
        args.use_summaries = False

    bench = NarrativeQABenchmark(args)
    bench.run()


if __name__ == "__main__":
    main()
