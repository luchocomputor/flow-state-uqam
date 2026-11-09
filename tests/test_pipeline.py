import numpy as np

import flowstate as fs


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


def test_wigner_ville_frequency_axis():
    sfreq, t = 125.0, np.arange(500) / 125.0
    W, freqs = fs.wigner_ville(np.sin(2 * np.pi * 12 * t)[None], sfreq)
    assert abs(freqs[W.mean(0).argmax()] - 12) < 1


def test_pipeline_beats_chance_on_synthetic():
    X, y, groups = fs.make_dataset(n_subjects=4, n_per_class=12, n_channels=6, effect=1.0)
    feats = fs.extract(X, sfreq=125.0, n_imfs=4, n_dirs=8, n_jobs=1)
    assert feats.wvd.shape == (len(X), 4 * 5)
    assert feats.covs.shape == (len(X), 4, 6, 6)
    res = fs.loso(lambda: fs.FusedClassifier(), feats, y, groups)
    assert res["f1"] > 0.7
