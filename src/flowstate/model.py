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
    out = np.empty_like(X, dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        sd = X[m].std(0)
        out[m] = (X[m] - X[m].mean(0)) / np.where(sd < 1e-8, 1.0, sd)
    return out


class FusedClassifier:
    """use: 'fused' (default), 'wvd' or 'riemann'."""

    def __init__(self, use: str = "fused", C: float = 1.0):
        self.use, self.C = use, C

    def _vectors(self, f: Features) -> np.ndarray:
        parts = []
        if self.use in ("fused", "wvd"):
            parts.append(f.wvd)
        if self.use in ("fused", "riemann"):
            parts.append(np.concatenate(
                [ts.transform(f.covs[:, k]) for k, ts in enumerate(self.ts_)], axis=1))
        return np.concatenate(parts, axis=1)

    def fit(self, f: Features, y: np.ndarray, groups: np.ndarray):
        self.ts_ = []
        if self.use in ("fused", "riemann"):
            self.ts_ = [TangentSpace(metric="riemann").fit(f.covs[:, k]) for k in range(f.covs.shape[1])]
        X = zscore_within(self._vectors(f), groups)
        self.svm_ = SVC(kernel="rbf", C=self.C, gamma="scale", class_weight="balanced").fit(X, y)
        return self

    def predict(self, f: Features, groups: np.ndarray) -> np.ndarray:
        return self.svm_.predict(zscore_within(self._vectors(f), groups))
