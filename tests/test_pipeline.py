import numpy as np
import pytest
from sklearn.metrics import f1_score

import flowstate as fs
from flowstate.model import zscore_within

# pyriemann warns when the gradient of a Riemannian mean is still above 1e-8 after 50 iterations.
# On these small sets it stalls near 3e-8, by which point the reference matrix has stopped moving.
pytestmark = pytest.mark.filterwarnings("ignore:Convergence not reached")


def make_model():
    return fs.FusedClassifier()


@pytest.fixture(scope="module")
def small():
    """3 subjects, 8 segments per class, 4 channels, 2 s."""
    X, y, groups = fs.make_dataset(n_subjects=3, n_per_class=8, n_channels=4, duration=2.0, effect=1.0)
    feats = fs.extract(X, sfreq=125.0, n_imfs=3, n_dirs=4, n_jobs=1)
    return X, y, groups, feats


def test_memd_modes_sum_to_signal():
    rng = np.random.default_rng(0)
    t = np.arange(500) / 125
    x = np.stack([np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 2 * t) + 0.1 * rng.normal(size=t.size)
                  for _ in range(4)])
    imfs = fs.memd(x, n_imfs=4, n_dirs=8)
    assert imfs.shape == (4, 4, 500)
    assert np.allclose(imfs.sum(0), x)


def test_memd_separates_scales():
    t = np.arange(1000) / 125
    x = np.stack([np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 3 * t)] * 3)
    imfs = fs.memd(x, n_imfs=3, n_dirs=8)
    # the first mode carries the fast oscillation
    spec = np.abs(np.fft.rfft(imfs[0, 0]))
    freqs = np.fft.rfftfreq(1000, 1 / 125)
    assert abs(freqs[spec.argmax()] - 20) < 1


def test_memd_is_deterministic():
    x = np.random.default_rng(0).normal(size=(3, 300))
    assert np.array_equal(fs.memd(x, n_dirs=8), fs.memd(x, n_dirs=8))


def test_memd_degenerate_inputs():
    # nothing to sift: the signal comes back as the first mode and the others are zero
    for x in (np.ones((3, 200)), np.outer([1.0, 2.0, 3.0], np.arange(200.0)), np.ones((1, 2))):
        imfs = fs.memd(x, n_imfs=4)
        assert np.array_equal(imfs[0], x) and not imfs[1:].any()
    # a single channel
    x = np.random.default_rng(0).normal(size=(1, 300))
    assert np.allclose(fs.memd(x, n_imfs=4, n_dirs=4).sum(0), x)
    with pytest.raises(ValueError, match="n_channels, n_times"):
        fs.memd(np.zeros(100))
    with pytest.raises(ValueError, match="NaN"):
        fs.memd(np.full((2, 100), np.nan))


def test_wigner_ville_frequency_axis():
    sfreq, t = 125.0, np.arange(500) / 125.0
    W, freqs = fs.wigner_ville(np.sin(2 * np.pi * 12 * t)[None], sfreq)
    assert abs(freqs[W.mean(0).argmax()] - 12) < 1


def test_wigner_ville_has_no_smoothing():
    sfreq, t = 125.0, np.arange(500) / 125.0
    two_tones = np.sin(2 * np.pi * 8 * t) + np.sin(2 * np.pi * 20 * t)
    W, freqs = fs.wigner_ville(two_tones[None], sfreq)
    assert W.shape == (64, freqs.size) and freqs[0] == 1 and freqs[-1] == 40
    assert not W[[0, -1]].any()             # first and last instants: rows left at zero
    cross = np.abs(freqs - 14).argmin()     # the cross-term halfway between the tones takes both signs
    assert W[:, cross].min() < 0 < W[:, cross].max()


def test_extract_shapes_and_positive_definite(small):
    X, _, _, feats = small
    assert feats.wvd.shape == (len(X), 3 * 5)
    assert feats.covs.shape == (len(X), 3, 4, 4)
    assert np.linalg.eigvalsh(feats.covs).min() > 0


def test_extract_uses_one_segment_at_a_time(small):
    X, _, _, feats = small
    alone = fs.extract(X[5:6], sfreq=125.0, n_imfs=3, n_dirs=4, n_jobs=1)
    assert np.array_equal(alone.wvd[0], feats.wvd[5])
    assert np.array_equal(alone.covs[0], feats.covs[5])
    # and the same result in worker processes
    parallel = fs.extract(X[:4], sfreq=125.0, n_imfs=3, n_dirs=4, n_jobs=2)
    assert np.array_equal(parallel.wvd, feats.wvd[:4]) and np.array_equal(parallel.covs, feats.covs[:4])


def test_extract_standardises_each_channel(small):
    X, _, _, feats = small
    rescaled = X[:2].copy()
    rescaled[:, 1] *= 50.0
    rescaled[:, 2] += 7.0
    out = fs.extract(rescaled, sfreq=125.0, n_imfs=3, n_dirs=4, n_jobs=1)
    assert np.allclose(out.wvd, feats.wvd[:2], rtol=1e-4, atol=1e-6)
    assert np.allclose(out.covs, feats.covs[:2])


def test_extract_rejects_bad_input():
    X = np.random.default_rng(0).normal(size=(2, 3, 250))
    with pytest.raises(ValueError, match="n_segments, n_channels, n_times"):
        fs.extract(X[0], sfreq=125.0, n_jobs=1)
    X_nan = X.copy()
    X_nan[1, 0, 10] = np.nan
    with pytest.raises(ValueError, match=r"segments \[1\]"):
        fs.extract(X_nan, sfreq=125.0, n_jobs=1)
    X_flat = X.copy()
    X_flat[0] = 1.0
    with pytest.warns(RuntimeWarning, match="non-zero modes"):
        fs.extract(X_flat, sfreq=125.0, n_imfs=3, n_dirs=4, n_jobs=1)


def test_zscore_within_uses_each_subject_alone():
    rng = np.random.default_rng(0)
    X, groups = rng.normal(size=(30, 5)), np.repeat([0, 1, 2], 10)
    Z = zscore_within(X, groups)
    assert np.allclose(Z[groups == 1].mean(0), 0) and np.allclose(Z[groups == 1].std(0), 1)
    X_other = X.copy()
    X_other[groups != 1] += 100 * rng.normal(size=(20, 5))   # changing the other subjects changes nothing
    assert np.array_equal(zscore_within(X_other, groups)[groups == 1], Z[groups == 1])
    assert np.array_equal(zscore_within(X, list(groups)), Z)


def test_fold_ignores_labels_of_held_out_subject(small):
    _, y, groups, feats = small
    tr, te = groups != 0, groups == 0
    train = fs.Features(wvd=feats.wvd[tr], covs=feats.covs[tr])
    test = fs.Features(wvd=feats.wvd[te], covs=feats.covs[te])
    pred = make_model().fit(train, y[tr], groups[tr]).predict(test, groups[te])
    # the fold of subject 0 is a model fitted on the other subjects
    assert fs.loso(make_model, feats, y, groups)["per_subject"][0] == f1_score(y[te], pred, average="macro")
    # with the labels of subject 0 inverted, its fold makes the same predictions
    y_inverted = np.where(te, 1 - y, y)
    assert (fs.loso(make_model, feats, y_inverted, groups)["per_subject"][0]
            == f1_score(1 - y[te], pred, average="macro"))


def test_loso_subject_ids_and_single_class_subjects(small):
    _, y, groups, feats = small
    res = fs.loso(make_model, feats, y, groups)
    names = ["s%02d" % g for g in groups]
    named = fs.loso(make_model, feats, list(y), names)
    assert named["f1"] == res["f1"] and set(named["per_subject"]) == {"s00", "s01", "s02"}
    y_one = np.where(groups == 0, 0, y)     # subject 0 has a single class: it is not a test subject
    assert set(fs.loso(make_model, feats, y_one, groups)["per_subject"]) == {1, 2}
    with pytest.raises(ValueError, match="no fold"):
        fs.loso(make_model, feats, np.zeros_like(y), groups)
    with pytest.raises(ValueError, match="same length"):
        fs.loso(make_model, feats, y[:-1], groups)
    with pytest.raises(ValueError, match="use must be"):
        fs.FusedClassifier(use="both")


@pytest.mark.parametrize("use", ["wvd", "fused"])
def test_permutation_test_equals_loso_on_shuffled_labels(small, use):
    _, y, groups, feats = small

    def make():
        return fs.FusedClassifier(use=use)

    perm = fs.permutation_test(make, feats, y, groups, n_permutations=4, seed=3, n_jobs=1)
    assert perm["f1"] == fs.loso(make, feats, y, groups)["f1"]
    rng = np.random.default_rng(3)
    for null_score in perm["null"]:
        shuffled = y.copy()
        for g in np.unique(groups):
            m = np.flatnonzero(groups == g)
            shuffled[m] = rng.permutation(y[m])
            assert shuffled[m].sum() == y[m].sum()      # each subject keeps its class balance
        assert null_score == fs.loso(make, feats, shuffled, groups)["f1"]
    assert perm["p_value"] == (1 + np.sum(perm["null"] >= perm["f1"])) / 5


def test_pipeline_beats_chance_on_synthetic():
    X, y, groups = fs.make_dataset(n_subjects=4, n_per_class=12, n_channels=6, effect=1.0)
    feats = fs.extract(X, sfreq=125.0, n_imfs=4, n_dirs=8, n_jobs=1)
    assert feats.wvd.shape == (len(X), 4 * 5)
    assert feats.covs.shape == (len(X), 4, 6, 6)
    res = fs.loso(make_model, feats, y, groups)
    assert res["f1"] > 0.7


def test_pipeline_is_at_chance_without_effect():
    X, y, groups = fs.make_dataset(n_subjects=4, n_per_class=12, n_channels=6, effect=0.0)
    feats = fs.extract(X, sfreq=125.0, n_imfs=4, n_dirs=8, n_jobs=1)
    res = fs.loso(make_model, feats, y, groups)
    assert 0.3 < res["f1"] < 0.7
