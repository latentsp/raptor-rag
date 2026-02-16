"""
Incremental result saving and resume support for interrupted benchmark runs.

Saves results after each question to a .partial.json file.
On restart, loads partial results and skips completed question IDs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProgressTracker:
    def __init__(self, output_path: str | Path, no_resume: bool = False):
        self.output_path = Path(output_path)
        self.partial_path = self.output_path.with_suffix(".partial.json")
        self.results: list[dict] = []
        self.completed_ids: set[str] = set()

        if not no_resume and self.partial_path.exists():
            self._load_partial()

    def _load_partial(self) -> None:
        try:
            with open(self.partial_path) as f:
                self.results = json.load(f)
            self.completed_ids = {r["question_id"] for r in self.results if "question_id" in r}
            logger.info("Resumed %d results from %s", len(self.results), self.partial_path)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to load partial results from %s, starting fresh", self.partial_path)
            self.results = []
            self.completed_ids = set()

    def is_done(self, question_id: str) -> bool:
        return question_id in self.completed_ids

    def record(self, result: dict) -> None:
        self.results.append(result)
        if "question_id" in result:
            self.completed_ids.add(result["question_id"])
        # Save incrementally
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.partial_path, "w") as f:
            json.dump(self.results, f, indent=2)

    def save_final(self, output: dict) -> None:
        """Write full JSON results and clean up partial file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", self.output_path)
        if self.partial_path.exists():
            self.partial_path.unlink()
            logger.info("Cleaned up %s", self.partial_path)
