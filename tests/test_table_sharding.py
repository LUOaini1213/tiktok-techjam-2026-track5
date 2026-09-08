"""Sharding is a CPU trick: a merged table must equal a single-process table."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import EVAL_PRESETS
from src.eval_robustness import evaluate_robustness, merge_tables
from tests.dataset_guard import needs_dataset

FIELDS = ["transform", "n", "acc", "auc", "auc_lo", "auc_hi", "tpr_at_1fpr", "tpr_at_5fpr"]


def _write(path: Path, names: list[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for n in names:
            w.writerow({"transform": n, "n": 2, "acc": "1.0000", "auc": "1.0000", "auc_lo": "1.0000",
                        "auc_hi": "1.0000", "tpr_at_1fpr": "1.0000", "tpr_at_5fpr": "1.0000"})
    return path


class MergeTests(unittest.TestCase):
    def test_merge_restores_preset_order(self):
        order = [n for n, _, _ in EVAL_PRESETS]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Shard the presets out of order across parts; merge must re-sort them.
            parts = [
                _write(td / "p0.csv", order[10:]),
                _write(td / "p1.csv", order[:5]),
                _write(td / "p2.csv", order[5:10]),
            ]
            dest = merge_tables(parts, td / "merged.csv")
            with dest.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["transform"] for r in rows], order)

    def test_merge_rejects_incomplete_shards(self):
        order = [n for n, _, _ in EVAL_PRESETS]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            parts = [_write(td / "p0.csv", order[:-1])]  # one transform never ran
            with self.assertRaises(RuntimeError):
                merge_tables(parts, td / "merged.csv")

    @needs_dataset
    def test_transform_filter_scores_only_requested_rows(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "part.csv"
            out = evaluate_robustness(
                max_images=2, use_tta=False, table_path=dest, bootstrap_reps=10,
                transforms=["clean", "jpeg_30"], scores_dir=Path(td) / "scores",
            )
            with out.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["transform"] for r in rows], ["clean", "jpeg_30"])

    def test_unknown_transform_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "table.csv"
            with self.assertRaises(ValueError):
                evaluate_robustness(max_images=2, table_path=dest, transforms=["clean", "jpeg_31"])
            # The rejection must happen before the destination is opened, so a bad call
            # cannot truncate an existing results table.
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
