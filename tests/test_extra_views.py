"""Extra training views must land INSIDE each image's view block, not appended at the end.

`source_groups` derives CV groups positionally (row // views_per_image), so a misplaced
merge would silently split one image's views across GroupKFold folds -- i.e. leakage.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train import merge_extra_views, source_groups  # scripts/train.py

N_IMG, VPI, DIM = 6, 4, 5


def _base():
    # Feature value encodes (image, view) so a misplaced row is detectable.
    X = np.array([[i * 10 + v] * DIM for i in range(N_IMG) for v in range(VPI)], dtype=np.float64)
    y = np.array([i % 2 for i in range(N_IMG) for _ in range(VPI)], dtype=np.int64)
    sid = np.array([(i % 3) for i in range(N_IMG) for _ in range(VPI)], dtype=np.int64)
    return {"pre": X, "proj": X[:, :3].copy()}, y, sid, VPI


def _extra(tmp: Path, name: str, tag: float, labels=None) -> Path:
    X = np.array([[i * 10 + tag] * DIM for i in range(N_IMG)], dtype=np.float64)
    y = labels if labels is not None else np.array([i % 2 for i in range(N_IMG)], dtype=np.int64)
    path = tmp / f"{name}.npz"
    np.savez_compressed(
        path, X_pre=X, X_proj=X[:, :3].copy(), y=y,
        sid=np.array([(i % 3) for i in range(N_IMG)], dtype=np.int64),
        views_per_image=np.array([1]),
    )
    return path


class MergeExtraViewTests(unittest.TestCase):
    def test_extra_view_lands_in_its_own_image_block(self):
        Xs, y, sid, vpi = _base()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            Xs2, y2, sid2, vpi2 = merge_extra_views(Xs, y, sid, vpi, [_extra(td, "noise", 7.0)])

        self.assertEqual(vpi2, VPI + 1)
        self.assertEqual(len(y2), N_IMG * (VPI + 1))
        blocks = Xs2["pre"].reshape(N_IMG, VPI + 1, DIM)
        for i in range(N_IMG):
            # views 0..3 keep their original tags, and the extra view is view 4 of image i
            self.assertEqual(list(blocks[i][:, 0]), [i * 10 + v for v in range(VPI)] + [i * 10 + 7.0])
        # groups must still be one contiguous run per image
        groups = source_groups(len(y2), vpi2)
        self.assertEqual(list(groups[: VPI + 1]), [0] * (VPI + 1))
        self.assertEqual(len(np.unique(groups)), N_IMG)

    def test_labels_and_sids_follow_their_image(self):
        Xs, y, sid, vpi = _base()
        with tempfile.TemporaryDirectory() as td:
            _, y2, sid2, vpi2 = merge_extra_views(Xs, y, sid, vpi, [_extra(Path(td), "noise", 7.0)])
        for i in range(N_IMG):
            blk = slice(i * vpi2, (i + 1) * vpi2)
            self.assertEqual(len(set(y2[blk].tolist())), 1, "one label per image block")
            self.assertEqual(len(set(sid2[blk].tolist())), 1, "one sid per image block")

    def test_multiple_families_stack(self):
        Xs, y, sid, vpi = _base()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            parts = [_extra(td, "noise", 7.0), _extra(td, "jitter", 8.0), _extra(td, "crop", 9.0)]
            Xs2, y2, _, vpi2 = merge_extra_views(Xs, y, sid, vpi, parts)
        self.assertEqual(vpi2, VPI + 3)
        blocks = Xs2["pre"].reshape(N_IMG, vpi2, DIM)
        self.assertEqual(list(blocks[2][:, 0])[VPI:], [27.0, 28.0, 29.0])

    def test_misaligned_labels_are_rejected(self):
        Xs, y, sid, vpi = _base()
        with tempfile.TemporaryDirectory() as td:
            bad = _extra(Path(td), "bad", 7.0, labels=np.array([1 - (i % 2) for i in range(N_IMG)], dtype=np.int64))
            with self.assertRaises(SystemExit):
                merge_extra_views(Xs, y, sid, vpi, [bad])


if __name__ == "__main__":
    unittest.main()
