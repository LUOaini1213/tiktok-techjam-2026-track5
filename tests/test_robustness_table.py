"""Drive the shipped robustness table writer on a tiny val slice."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_robustness import balanced_val_slice, evaluate_robustness
from tests.dataset_guard import needs_dataset


class RobustnessTableTests(unittest.TestCase):
    def test_balanced_val_slice_keeps_both_labels(self):
        rows = [{"label": 0, "path": "a"}] * 5 + [{"label": 1, "path": "b"}] * 5
        sliced = balanced_val_slice(rows, 4)
        labels = {int(r["label"]) for r in sliced}
        self.assertEqual(labels, {0, 1})
        self.assertEqual(len(sliced), 4)

    @needs_dataset
    def test_evaluate_robustness_writes_official_families(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "robustness_table.csv"
            # Keep bootstrap tiny so the test stays fast; the CI columns still get written.
            out = evaluate_robustness(
                max_images=2, use_tta=False, table_path=dest, bootstrap_reps=50
            )
            self.assertTrue(out.exists())
            with out.open(newline="", encoding="utf-8") as f:
                table = list(csv.DictReader(f))
        names = [row["transform"] for row in table]
        joined = " ".join(names)
        self.assertTrue(table)
        self.assertEqual(
            list(table[0].keys()),
            ["transform", "n", "acc", "auc", "auc_lo", "auc_hi"],
        )
        for family in ("clean", "jpeg", "blur", "resize", "noise", "jitter", "crop"):
            self.assertIn(family, joined, msg=f"missing family {family} in {names}")
        for row in table:
            float(row["acc"])
            float(row["auc"])
            float(row["auc_lo"])
            float(row["auc_hi"])
            self.assertGreaterEqual(int(row["n"]), 2)
