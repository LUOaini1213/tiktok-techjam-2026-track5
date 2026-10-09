"""Calibration operating points; each original image, not each view, is a trial."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta


def binomial_upper(errors: int, n: int, confidence: float = 0.95) -> float:
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")
    if n <= 0:
        raise ValueError("At least one independent source image is required")
    if not 0 <= errors <= n:
        raise ValueError("Invalid error count")
    return 1.0 if errors == n else float(beta.ppf(confidence, errors + 1, n - errors))


def _group_extremes(y, scores, groups, label, maximum):
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    if not (len(y) == len(scores) == len(groups)) or not np.isfinite(scores).all():
        raise ValueError("Labels, scores and groups must align and scores must be finite")
    values = []
    for group in np.unique(groups):
        idx = groups == group
        if len(np.unique(y[idx])) != 1:
            raise ValueError("An original image cannot have different labels across its views")
        if y[idx][0] == label:
            values.append(float(np.max(scores[idx]) if maximum else np.min(scores[idx])))
    if not values:
        raise ValueError(f"No calibration source images for label {label}")
    return np.asarray(values)


def select_policy(y, scores, groups, target_fpr=0.01, target_false_clear=0.05, confidence=0.95):
    """Order-statistic tolerance bounds on worst-view scores, calibrated without test data.

    A real source fails if ANY of its views is flagged; an AI source fails if ANY view is
    cleared. This avoids pretending correlated transform rows are independent trials.
    Bounds apply to the calibration distribution, never to arbitrary deployment data.
    """
    if not 0 < target_fpr < 1 or not 0 < target_false_clear < 1:
        raise ValueError("Error-rate targets must lie strictly between zero and one")
    real = np.sort(_group_extremes(y, scores, groups, 0, True))[::-1]
    fake = np.sort(_group_extremes(y, scores, groups, 1, False))
    def allowed(n, target):
        valid = [k for k in range(n) if binomial_upper(k, n, confidence) <= target]
        return max(valid) if valid else 0, bool(valid)
    kr, certified_r = allowed(len(real), target_fpr)
    kf, certified_f = allowed(len(fake), target_false_clear)
    high = float(np.nextafter(real[kr], np.inf))
    low = float(np.nextafter(fake[kf], -np.inf))
    # If the two safe regions overlap, shrink both of them to leave a review band.
    # Increasing the flag threshold / decreasing the clear threshold cannot add errors.
    if low >= high:
        center, quarter = (low + high) / 2, (low - high) / 4
        low, high = center - quarter, center + quarter
    er, ef = int((real >= high).sum()), int((fake <= low).sum())
    return {
        "aigc_threshold": high, "authentic_threshold": low,
        "target_fpr": target_fpr, "target_false_clear": target_false_clear,
        "confidence": confidence, "n_real_groups": len(real), "n_fake_groups": len(fake),
        "cal_real_group_errors": er, "cal_fake_group_errors": ef,
        "cal_fpr_upper": binomial_upper(er, len(real), confidence),
        "cal_false_clear_upper": binomial_upper(ef, len(fake), confidence),
        "cal_fpr_bound_met": certified_r, "cal_false_clear_bound_met": certified_f,
        "unit": "original-image worst view", "scope": "calibration distribution only",
    }


def verdict(score: float, policy: dict | None = None) -> str:
    if not np.isfinite(score):
        return "Needs review"
    if not policy:
        return "Likely AIGC" if score >= 0.5 else "Low AIGC signal"
    if score >= policy["aigc_threshold"]:
        return "Likely AIGC"
    if score <= policy["authentic_threshold"]:
        return "Low AIGC signal"
    return "Needs review"


def operating_metrics(y, scores, policy):
    y, s = np.asarray(y), np.asarray(scores)
    flagged = s >= policy["aigc_threshold"]
    cleared = s <= policy["authentic_threshold"]
    return {"fpr": float(flagged[y == 0].mean()), "recall": float(flagged[y == 1].mean()),
            "false_clear_rate": float(cleared[y == 1].mean()),
            "coverage": float((flagged | cleared).mean()),
            "review_rate": float((~(flagged | cleared)).mean())}
