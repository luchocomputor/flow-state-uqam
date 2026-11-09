"""End-to-end run on synthetic EEG: features, leave-one-subject-out, permutation test.

    python examples/demo_synthetic.py
"""
import time

import flowstate as fs

X, y, groups = fs.make_dataset(n_subjects=6, n_per_class=20)
print(f"{len(X)} segments, {X.shape[1]} channels, {X.shape[2]} samples, {len(set(groups))} subjects")

t0 = time.time()
feats = fs.extract(X, sfreq=125.0, n_imfs=6, n_dirs=16)
print(f"features: WVD {feats.wvd.shape}, covariances {feats.covs.shape} ({time.time() - t0:.0f} s)")

for use in ("wvd", "riemann", "fused"):
    res = fs.loso(lambda: fs.FusedClassifier(use=use), feats, y, groups)
    print(f"{use:8s} LOSO macro-F1 = {res['f1']:.3f}")

perm = fs.permutation_test(lambda: fs.FusedClassifier(use="wvd"), feats, y, groups, n_permutations=50)
print(f"permutation test (wvd): F1 = {perm['f1']:.3f}, null = {perm['null'].mean():.3f}, p = {perm['p_value']:.3f}")
