
"""Threshold and classification metrics for Rawaj Pay."""
from __future__ import annotations

import math
import operator

import numpy as np


def _validated_inputs(y_true, scores):
    try:
        y = np.asarray(y_true)
        s = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid labels or scores") from exc

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")
    if len(y) != len(s):
        raise ValueError("Inputs must have equal lengths")
    if len(y) == 0:
        raise ValueError("Inputs must not be empty")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if np.isnan(s).any():
        raise ValueError("Scores must not contain NaN")

    return y.astype(int), s


def confusion_at(y_true, scores, threshold: float) -> dict:
    y, s = _validated_inputs(y_true, scores)
    flagged = s >= threshold
    positive = y == 1

    return {
        "tp": int(np.sum(flagged & positive)),
        "fp": int(np.sum(flagged & ~positive)),
        "tn": int(np.sum(~flagged & ~positive)),
        "fn": int(np.sum(~flagged & positive)),
    }


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    c = confusion_at(y_true, scores, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    y, s = _validated_inputs(y_true, scores)

    try:
        k = operator.index(k)
    except TypeError as exc:
        raise ValueError("k must be an integer") from exc

    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, len(y))
    order = np.argsort(-s, kind="stable")[:k]
    return float(np.mean(y[order] == 1))


def expected_cost(
    y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float
) -> float:
    c = confusion_at(y_true, scores, threshold)
    return float(c["fp"] * cost_fp + c["fn"] * cost_fn)


def best_threshold(
    y_true, scores, *, cost_fp: float, cost_fn: float
) -> float:
    y, s = _validated_inputs(y_true, scores)
    candidates = list(np.unique(s))
    candidates.append(float(math.inf))

    best = float(math.inf)
    best_cost = expected_cost(
        y, s, best, cost_fp=cost_fp, cost_fn=cost_fn
    )

    for candidate in candidates:
        candidate = float(candidate)
        cost = expected_cost(
            y, s, candidate, cost_fp=cost_fp, cost_fn=cost_fn
        )

        if cost < best_cost or (cost == best_cost and candidate > best):
            best = candidate
            best_cost = cost

    return best


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    y, s = _validated_inputs(y_true, scores)

    for candidate in np.sort(np.unique(s)):
        precision, _ = precision_recall_at(y, s, float(candidate))
        if precision >= min_precision:
            return float(candidate)

    return None
