"""Fused classifier: WVD energies + per-mode Riemannian tangent vectors, RBF SVM.

The reference point of each tangent space is learned on the training subjects
only. Features are then z-scored within each subject, test subject included,
with its own statistics (unsupervised, no labels used).
"""
from __future__ import annotations

import numpy as np
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC

from .features import Features


def zscore_within(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Centre and scale each column within each group, with that group's own statistics.

    Columns that are constant within a group are only centred.
    """
    groups = np.asarray(groups)
    out = np.empty_like(X, dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        sd = X[m].std(0)
        out[m] = (X[m] - X[m].mean(0)) / np.where(sd < 1e-8, 1.0, sd)
    return out


class FusedClassifier:
    """RBF SVM on per-subject z-scored features.

    use : 'fused' (default), 'wvd' or 'riemann'.
    C : SVM regularisation parameter.

    `predict` z-scores the segments it receives within each subject, so it
    needs all the segments of a subject at once, not one segment at a time.
    """

    def __init__(self, use: str = "fused", C: float = 1.0):
        if use not in ("fused", "wvd", "riemann"):
            raise ValueError(f"use must be 'fused', 'wvd' or 'riemann', got {use!r}")
        self.use, self.C = use, C

    def _vectors(self, f: Features) -> np.ndarray:
        parts = []
        if self.use in ("fused", "wvd"):
            parts.append(f.wvd)
        if self.use in ("fused", "riemann"):
            parts.append(np.concatenate(
                [ts.transform(f.covs[:, k]) for k, ts in enumerate(self.ts_)], axis=1))
        return np.concatenate(parts, axis=1)

    def fit(self, f: Features, y: np.ndarray, groups: np.ndarray) -> FusedClassifier:
        self.ts_ = []
        if self.use in ("fused", "riemann"):
            self.ts_ = [TangentSpace(metric="riemann").fit(f.covs[:, k]) for k in range(f.covs.shape[1])]
        svm = SVC(kernel="rbf", C=self.C, gamma="scale", class_weight="balanced")
        self.svm_ = svm.fit(self.transform(f, groups), y)
        return self

    def transform(self, f: Features, groups: np.ndarray) -> np.ndarray:
        """Feature vectors given to the SVM, z-scored within each subject. Uses no labels."""
        return zscore_within(self._vectors(f), groups)

    def predict(self, f: Features, groups: np.ndarray) -> np.ndarray:
        return self.svm_.predict(self.transform(f, groups))
