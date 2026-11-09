"""Leave-one-subject-out evaluation and a within-subject permutation test."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from .features import Features


def _subset(f: Features, idx) -> Features:
    return Features(wvd=f.wvd[idx], covs=f.covs[idx])


def loso(make_model, f: Features, y: np.ndarray, groups: np.ndarray) -> dict:
    """Train on all subjects but one, test on the held-out one, for every subject.

    Returns the macro-F1 pooled over all folds and the per-subject scores.
    """
    y_true, y_pred, per_subject = [], [], {}
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        model = make_model().fit(_subset(f, tr), y[tr], groups[tr])
        yp = model.predict(_subset(f, te), groups[te])
        per_subject[int(g)] = f1_score(y[te], yp, average="macro", zero_division=0)
        y_true.extend(y[te]); y_pred.extend(yp)
    return {"f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "per_subject": per_subject}


def permutation_test(make_model, f: Features, y, groups, n_permutations: int = 200,
                     seed: int = 0, n_jobs: int = -1) -> dict:
    """Null distribution of the LOSO F1 with labels shuffled within each subject.

    Shuffling inside subjects keeps each subject's class balance, so the null
    only removes the link between the signal and the label.
    p = (1 + #{null >= observed}) / (n_permutations + 1).
    """
    rng = np.random.default_rng(seed)
    observed = loso(make_model, f, y, groups)["f1"]
    perms = []
    for _ in range(n_permutations):
        yp = y.copy()
        for g in np.unique(groups):
            m = np.flatnonzero(groups == g)
            yp[m] = rng.permutation(y[m])
        perms.append(yp)
    null = np.array(Parallel(n_jobs=n_jobs)(delayed(lambda yy: loso(make_model, f, yy, groups)["f1"])(yy) for yy in perms))
    return {"f1": observed, "null": null, "p_value": (1 + np.sum(null >= observed)) / (n_permutations + 1)}
