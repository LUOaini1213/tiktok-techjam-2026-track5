"""Pin the current README's V3/V4 numbers to the committed results.

Three layers, none of which needs torch or the dataset:

1. results/v3/summary.json and results/v4/summary.json are means over the per-condition
   rows in the matching evaluation.csv. Recompute them, so a hand-edited summary fails.
2. The generated README blocks (TLDR, V3_TABLE, V4_TABLE) equal a fresh render from those
   files by scripts/update_readme.py. CI also runs `update_readme.py --check`.
3. The hand-written sentences around the tables quote further figures. Each is anchored on
   a distinctive phrase and compared with the value recomputed from results/, at the
   rounding the README prints. A reworded sentence fails loudly instead of skipping.

Also: no tracked text file may carry an absolute local path (scripts/check_local_paths.py).
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unittest
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_local_paths  # noqa: E402
import update_readme  # noqa: E402

RESULTS = ROOT / "results"
README_RAW = (ROOT / "README.md").read_text(encoding="utf-8")
# Whitespace-flattened, so a re-wrapped paragraph does not break every pattern below.
README = re.sub(r"\s+", " ", README_RAW)


def rnd(value: float, places: int) -> float:
    quant = Decimal(1).scaleb(-places)
    return float(Decimal(repr(value)).quantize(quant, rounding=ROUND_HALF_UP))


def pct(value: float, places: int) -> float:
    return rnd(100 * value, places)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


V3 = json.loads((RESULTS / "v3" / "summary.json").read_text(encoding="utf-8"))["results"]
V4 = json.loads((RESULTS / "v4" / "summary.json").read_text(encoding="utf-8"))["metrics"]
V4_SEL = json.loads((RESULTS / "v4" / "selection.json").read_text(encoding="utf-8"))


def _block(name: str) -> str:
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    i = README_RAW.find(start)
    j = README_RAW.find(end, i)
    assert i >= 0 and j >= 0, f"README.md is missing the {name} markers"
    return README_RAW[i + len(start): j].replace("\r\n", "\n").strip("\n")


class SummaryMatchesPerConditionRows(unittest.TestCase):
    def test_v4_summary_is_mean_of_evaluation_rows(self):
        groups = defaultdict(list)
        for r in _rows(RESULTS / "v4" / "evaluation.csv"):
            groups[(r["model"], r["domain"], r["policy"])].append(r)
        self.assertTrue(groups)
        columns = {
            "auc": "mean_auc",
            "accuracy_at_05": "mean_accuracy_at_05",
            "fpr": "fpr",
            "recall": "recall",
            "false_clear_rate": "false_clear_rate",
            "coverage": "coverage",
            "review_rate": "review_rate",
        }
        for (model, domain, policy), rows in groups.items():
            self.assertEqual(len(rows), 19, (model, domain, policy))
            summary = V4[model][domain][policy]
            for column, key in columns.items():
                mean = sum(float(r[column]) for r in rows) / len(rows)
                self.assertAlmostEqual(mean, summary[key], places=9, msg=(model, domain, policy, key))

    def test_v3_summary_is_mean_of_evaluation_rows(self):
        groups = defaultdict(list)
        for r in _rows(RESULTS / "v3" / "evaluation.csv"):
            groups[(r["domain"], r["model"])].append(r)
        self.assertTrue(groups)
        for (domain, model), rows in groups.items():
            summary = V3[domain][model]
            for key in ("auc", "fpr", "recall", "review_rate", "accuracy_at_0_5", "fpr_at_0_5"):
                mean = sum(float(r[key]) for r in rows) / len(rows)
                self.assertAlmostEqual(mean, summary[key], places=9, msg=(domain, model, key))
            self.assertAlmostEqual(min(float(r["auc"]) for r in rows), summary["worst_auc"], places=12)


class GeneratedBlocksAreCurrent(unittest.TestCase):
    def test_tldr(self):
        self.assertEqual(_block("TLDR"), update_readme.tldr_section(RESULTS))

    def test_v3_table(self):
        self.assertEqual(_block("V3_TABLE"), update_readme.v3_table_section(RESULTS))

    def test_v4_table(self):
        self.assertEqual(_block("V4_TABLE"), update_readme.v4_table_section(RESULTS))

    def test_tldr_headline_numbers(self):
        # Independent of the renderer: the headline figures a reader quotes.
        tldr = re.sub(r"\s+", " ", _block("TLDR"))
        d = V4["v3_default"]
        self.assertIn(f"mean AUC **{d['all']['v3_frozen_strict']['mean_auc']:.4f}**", tldr)
        self.assertIn(f"**{d['SID']['v3_frozen_strict']['mean_auc']:.4f}** on fresh SID", tldr)
        self.assertIn(f"**{d['CF_StyleGANXL']['v3_frozen_strict']['mean_auc']:.4f}** on the held-out StyleGANXL", tldr)
        self.assertIn("0/192 real originals flagged", tldr)
        self.assertIn("0 of 8 new candidate heads passed", tldr)
        self.assertEqual(V4_SEL["eligible"], [])


class _Prose(unittest.TestCase):
    def _find(self, pattern: str) -> tuple[str, ...]:
        match = re.search(pattern, README)
        self.assertIsNotNone(
            match,
            f"README.md no longer contains the passage this test pins:\n  {pattern}\n"
            "Update the pattern together with the sentence, or restore the claim.",
        )
        return match.groups()


class HandWrittenV3Prose(_Prose):
    def test_binary_threshold_accuracy_and_fpr(self):
        acc2, acc3, fpr2, fpr3 = self._find(
            r"sealed mean accuracy rises from \*\*([0-9.]+)% to ([0-9.]+)%\*\* "
            r"and FPR falls from ([0-9.]+)% to ([0-9.]+)%\."
        )
        s = V3["all_sealed"]
        self.assertEqual(float(acc2), pct(s["v2"]["accuracy_at_0_5"], 1))
        self.assertEqual(float(acc3), pct(s["v3"]["accuracy_at_0_5"], 1))
        self.assertEqual(float(fpr2), pct(s["v2"]["fpr_at_0_5"], 1))
        self.assertEqual(float(fpr3), pct(s["v3"]["fpr_at_0_5"], 1))

    def test_three_way_policy(self):
        fpr, recall, review = self._find(
            r"mean FPR \*\*([0-9.]+)%\*\*, AI recall \*\*([0-9.]+)%\*\*, and review rate \*\*([0-9.]+)%\*\*"
        )
        s = V3["all_sealed"]["v3"]
        self.assertEqual(float(fpr), pct(s["fpr"], 4))
        self.assertEqual(float(recall), pct(s["recall"], 1))
        self.assertEqual(float(review), pct(s["review_rate"], 1))

    def test_regressions(self):
        t2, t3, r2, r3 = self._find(
            r"SID local-edit mean AUC decreases from ([0-9.]+) to ([0-9.]+); "
            r"reference FPR at 0\.5 increases from ([0-9.]+)% to ([0-9.]+)%"
        )
        tampered, ref = V3["SID-Set fresh (tampered)"], V3["WildFake reference"]
        self.assertEqual(float(t2), rnd(tampered["v2"]["auc"], 4))
        self.assertEqual(float(t3), rnd(tampered["v3"]["auc"], 4))
        self.assertEqual(float(r2), pct(ref["v2"]["fpr_at_0_5"], 1))
        self.assertEqual(float(r3), pct(ref["v3"]["fpr_at_0_5"], 1))

    def test_reference_quarter_scale(self):
        auc, fpr = self._find(
            r"quarter-scale condition still has AUC only ([0-9.]+) and shipped-policy FPR ([0-9.]+)%"
        )
        row = [
            r for r in _rows(RESULTS / "v3" / "evaluation.csv")
            if r["domain"] == "WildFake reference" and r["model"] == "v3" and r["transform"] == "resize_0.25"
        ]
        self.assertEqual(len(row), 1)
        self.assertEqual(float(auc), rnd(float(row[0]["auc"]), 4))
        self.assertEqual(float(fpr), pct(float(row[0]["fpr"]), 1))


class HandWrittenV4Prose(_Prose):
    def test_development_trade_off(self):
        ext0, ext1, tam0, tam1 = self._find(
            r"raises external mean AUC from ([0-9.]+) to ([0-9.]+), but lowers SID "
            r"real-versus-local-edit mean AUC from ([0-9.]+) to ([0-9.]+)\."
        )
        trials = V4_SEL["trials"]
        self.assertEqual(float(ext0), rnd(trials["v3_recal"]["external_auc"], 4))
        self.assertEqual(float(ext1), rnd(trials["lr_001"]["external_auc"], 4))
        self.assertEqual(float(tam0), rnd(trials["v3_recal"]["tampered_auc"], 4))
        self.assertEqual(float(tam1), rnd(trials["lr_001"]["tampered_auc"], 4))

    def test_any_view_false_clears(self):
        default_errors, default_n, opt_errors, opt_n, opt_pct = self._find(
            r"gives \*\*(\d+)/(\d+)\*\* for default V3 versus \*\*(\d+)/(\d+)\*\* for the opt-in "
            r"policy\. The latter is ([0-9.]+)%"
        )
        d = V4["v3_default"]["all"]["v3_frozen_strict"]["original"]
        o = V4["v3_recal"]["all"]["conditionwise"]["original"]
        self.assertEqual((int(default_errors), int(default_n)), (d["fake_any_view_clear_errors"], d["n_fake"]))
        self.assertEqual((int(opt_errors), int(opt_n)), (o["fake_any_view_clear_errors"], o["n_fake"]))
        self.assertEqual(float(opt_pct), pct(o["fake_any_view_false_clear"], 1))

    def test_upper_bound(self):
        (bound,) = self._find(r"one-sided 95% upper bound of \*\*([0-9.]+)%\*\*")
        o = V4["v3_default"]["all"]["v3_frozen_strict"]["original"]
        self.assertEqual(o["real_any_view_errors"], 0)
        self.assertEqual(float(bound), pct(o["real_any_view_fpr_upper"], 2))

    def test_generalisation(self):
        sid, cf = self._find(
            r"mean AUC is \*\*([0-9.]+) on fresh SID\*\*, but only \*\*([0-9.]+) on the named StyleGANXL"
        )
        d = V4["v3_default"]
        self.assertEqual(float(sid), rnd(d["SID"]["v3_frozen_strict"]["mean_auc"], 4))
        self.assertEqual(float(cf), rnd(d["CF_StyleGANXL"]["v3_frozen_strict"]["mean_auc"], 4))
        cleared, n = self._find(r"opt-in policy clears (\d+)/(\d+) CF AI originals")
        o = V4["v3_recal"]["CF_StyleGANXL"]["conditionwise"]["original"]
        self.assertEqual((int(cleared), int(n)), (o["fake_any_view_clear_errors"], o["n_fake"]))

    def test_paired_difference_is_zero(self):
        self.assertIn("all four reported domain intervals are [0, 0]", README)
        paired = json.loads((RESULTS / "v4" / "summary.json").read_text(encoding="utf-8"))["paired_auc"]
        self.assertEqual(len(paired), 4)
        for domain, entry in paired.items():
            self.assertEqual(entry["ci95"], [0.0, 0.0], domain)


class NoLocalPaths(unittest.TestCase):
    def test_tracked_files_have_no_absolute_local_paths(self):
        self.assertEqual(check_local_paths.offenders(), [])


if __name__ == "__main__":
    unittest.main()
