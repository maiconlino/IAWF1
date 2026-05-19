"""
Canonical IAWF1 — strict implementation of the algorithm described in the
manuscript text and Algorithm 1, with the conceptual ambiguities resolved.

Differences from the GitHub version (IAWF1_metric.py):
  1. Weights follow 1/sqrt(n_i) (paper text) — NOT n_i/sum(n_i) (code bug).
  2. n_i is the TRUE class size from ground truth — NOT the predicted count.
  3. Class sizes are passed explicitly OR derived from the CM using the
     standard sklearn convention [[TN,FP],[FN,TP]], where rows = true class.
  4. Precision and recall are computed with correct semantics (the code's
     labels were swapped; F1 was correct only by symmetry).

This module is the single source of truth for IAWF1 going forward.
"""

from __future__ import annotations
import numpy as np
from typing import Iterable, Optional, Tuple


# -----------------------------------------------------------------------------
# Per-class F1 from a multi-class confusion matrix
# -----------------------------------------------------------------------------
def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """
    Compute the F1-score of each class from a sklearn-style multi-class
    confusion matrix where rows = true class, columns = predicted class.

    Parameters
    ----------
    cm : np.ndarray of shape (k, k)
        Confusion matrix in sklearn convention.

    Returns
    -------
    np.ndarray of shape (k,)
        F1-score for each class, indexed by class label 0..k-1.
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape {cm.shape}")
    k = cm.shape[0]

    f1 = np.zeros(k)
    for i in range(k):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP    # predicted as i but actually some other class
        FN = cm[i, :].sum() - TP    # actually i but predicted as some other class
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1[i] = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


# -----------------------------------------------------------------------------
# True class sizes from a confusion matrix
# -----------------------------------------------------------------------------
def class_sizes_from_cm(cm: np.ndarray) -> np.ndarray:
    """
    Return the actual (true) class sizes from a sklearn-convention CM.
    Row sums are true class sizes (rows = true class in sklearn).
    """
    cm = np.asarray(cm, dtype=float)
    return cm.sum(axis=1)


# -----------------------------------------------------------------------------
# Canonical IAWF1
# -----------------------------------------------------------------------------
class IAWF1:
    """
    Canonical Imbalance-Aware Weighted F1 metric.

    Two operating modes:
      * Automatic (default): w_i = 1 / sqrt(n_i), then normalize sum to 1.
      * Custom: caller supplies a weight vector. By default it is normalized
        to sum to 1 to keep the metric in [0, 1]. (This resolves the paper's
        internal contradiction about custom-weight normalization.)

    Parameters
    ----------
    custom_weights : iterable of floats, optional
        If provided, used directly instead of the 1/sqrt(n) scheme.
    normalize_custom : bool, default True
        If True (recommended), custom weights are normalized to sum to 1.
        Set to False only if the caller has already normalized them.
    """

    def __init__(
        self,
        custom_weights: Optional[Iterable[float]] = None,
        normalize_custom: bool = True,
    ):
        self.custom_weights = (
            np.asarray(custom_weights, dtype=float)
            if custom_weights is not None else None
        )
        self.normalize_custom = normalize_custom

    # ---------- Score from CM (most common entry point) ----------
    def score(
        self,
        confusion_matrix: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute IAWF1 from a sklearn-convention confusion matrix.

        Returns
        -------
        score : float
            IAWF1 value in [0, 1].
        final_weights : np.ndarray
            The normalized weights actually used.
        """
        cm = np.asarray(confusion_matrix, dtype=float)
        f1 = per_class_f1(cm)
        n = class_sizes_from_cm(cm)
        return self.score_from_components(f1, n)

    # ---------- Score from already-computed per-class F1 and class sizes ----------
    def score_from_components(
        self,
        f1_scores: Iterable[float],
        class_sizes: Iterable[float],
    ) -> Tuple[float, np.ndarray]:
        """
        Compute IAWF1 directly from per-class F1-scores and true class sizes.
        Useful when the CM is unavailable or when class sizes are known
        from the dataset description.
        """
        f1 = np.asarray(f1_scores, dtype=float)
        n  = np.asarray(class_sizes, dtype=float)

        if f1.shape != n.shape or f1.ndim != 1:
            raise ValueError("f1_scores and class_sizes must be 1-D, same shape")

        if self.custom_weights is not None:
            w = self.custom_weights.copy()
            if w.shape != f1.shape:
                raise ValueError("custom_weights shape must match number of classes")
            # CRITICAL: classes absent from the test set must be excluded BEFORE
            # applying the user's custom weight vector; otherwise, a class with
            # n_i = 0 (no samples in the test set) would receive positive weight
            # if the user-supplied w_i^c were positive, which contradicts the
            # intended treatment of absent classes.
            w = np.where(n > 0, w, 0.0)
            if self.normalize_custom and w.sum() > 0:
                w = w / w.sum()
            elif w.sum() == 0:
                # All custom weights zero (or all classes absent). The metric is
                # undefined in this configuration; return NaN rather than 0 so
                # the caller is alerted.
                return float("nan"), w
        else:
            # Paper formula: w_i_initial = 1/sqrt(n_i) for n_i >= 1, else 0.
            initial = np.where(n >= 1, 1.0 / np.sqrt(np.maximum(n, 1e-12)), 0.0)
            total = initial.sum()
            if total == 0:
                # All classes absent: metric undefined.
                return float("nan"), np.zeros_like(initial)
            w = initial / total

        score = float(np.sum(w * f1))
        return score, w


# -----------------------------------------------------------------------------
# Convenience wrappers
# -----------------------------------------------------------------------------
def iawf1_score(
    confusion_matrix: np.ndarray,
    custom_weights: Optional[Iterable[float]] = None,
) -> float:
    """One-shot functional API."""
    return IAWF1(custom_weights=custom_weights).score(confusion_matrix)[0]


__all__ = [
    "IAWF1",
    "iawf1_score",
    "per_class_f1",
    "class_sizes_from_cm",
]
