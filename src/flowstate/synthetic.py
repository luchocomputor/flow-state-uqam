"""Synthetic multi-subject EEG, to run the pipeline without the private data."""
from __future__ import annotations

import numpy as np


def make_dataset(n_subjects: int = 6, n_per_class: int = 20, n_channels: int = 8,
                 sfreq: float = 125.0, duration: float = 4.0, effect: float = 0.6, seed: int = 0):
    """Two-class EEG-like segments with subject-specific mixing.

    Each segment mixes three latent rhythms (theta near 6 Hz, alpha near 10 Hz,
    beta near 20 Hz) through a mixing matrix drawn once per subject, then adds
    a random walk and white noise on each channel. In class 1 the theta and
    alpha amplitudes are multiplied by 1 + effect and 1 + 0.6 * effect; with
    effect = 0 the label carries no information.

    Returns X (n_segments, n_channels, n_times), y, groups.
    """
    rng = np.random.default_rng(seed)
    n = int(sfreq * duration)
    t = np.arange(n) / sfreq
    X, y, groups = [], [], []
    for s in range(n_subjects):
        mixing = rng.normal(size=(n_channels, 3)) + np.eye(n_channels, 3) * 2
        for label in (0, 1):
            for _ in range(n_per_class):
                amp = np.array([1.0, 1.0, 0.5]) + effect * label * np.array([1.0, 0.6, 0.0])
                amp *= rng.uniform(0.8, 1.2, 3)
                freqs = np.array([6.0, 10.0, 20.0]) + rng.normal(0, 0.5, 3)
                env = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5, (3, 1)) * t + rng.uniform(0, 6.3, (3, 1)))
                sources = amp[:, None] * env * np.sin(2 * np.pi * freqs[:, None] * t + rng.uniform(0, 6.3, (3, 1)))
                noise = np.cumsum(rng.normal(size=(n_channels, n)), axis=1) * 0.05
                noise -= noise.mean(axis=1, keepdims=True)
                X.append(mixing @ sources + noise + 0.3 * rng.normal(size=(n_channels, n)))
                y.append(label)
                groups.append(s)
    return np.asarray(X), np.asarray(y), np.asarray(groups)
