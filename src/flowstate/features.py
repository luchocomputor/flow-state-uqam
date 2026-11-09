"""Per-segment features: MEMD modes, Wigner-Ville band energies, per-mode covariances.

Everything here is computed segment by segment, without labels or other
segments, so it can be cached before cross-validation without leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import hilbert
from sklearn.covariance import ledoit_wolf

from .memd import memd

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 29), "gamma": (31, 40)}


def wigner_ville(mode: np.ndarray, sfreq: float, n_times: int = 64,
                 fmin: float = 1.0, fmax: float = 40.0):
    """Pseudo Wigner-Ville distribution of one mode, averaged over channels.

    mode : (n_channels, n_samples). Returns (power (n_times, n_freqs), freqs).
    The kernel z[n+tau] z*[n-tau] oscillates at 2f, so frequency m*fs/N is read
    at bin 2m of the full FFT.
    """
    C, N = mode.shape
    z = hilbert(mode, axis=1)
    fr = np.fft.rfftfreq(N, d=1.0 / sfreq)
    keep = (fr >= fmin) & (fr <= fmax)
    W = np.zeros((n_times, int(keep.sum())), dtype=np.float32)
    for i, n in enumerate(np.round(np.linspace(0, N - 1, n_times)).astype(int)):
        tm = min(n, N - 1 - n)
        if tm < 1:
            continue
        taus = np.arange(tm + 1)
        ip = (z[:, n + taus] * np.conj(z[:, n - taus])).mean(axis=0)
        iaf = np.zeros(N, dtype=complex)
        iaf[taus] = ip
        iaf[N - taus[1:]] = np.conj(ip[1:])
        W[i] = np.fft.fft(iaf).real[(2 * np.arange(fr.size)) % N][keep]
    return W, fr[keep]


def band_energies(imfs: np.ndarray, sfreq: float, bands=BANDS) -> np.ndarray:
    """Mean WVD energy of each mode in each band: (n_imfs * n_bands,)."""
    out = []
    for mode in imfs:
        W, freqs = wigner_ville(mode, sfreq)
        for lo, hi in bands.values():
            m = (freqs >= lo) & (freqs <= hi)
            out.append(W[:, m].mean() if m.any() else 0.0)
    return np.asarray(out, dtype=np.float32)


def mode_covariances(imfs: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf covariance of each mode across channels: (n_imfs, C, C)."""
    return np.stack([ledoit_wolf(mode.T)[0] for mode in imfs])


@dataclass
class Features:
    """Cached features of a set of segments."""
    wvd: np.ndarray    # (n_segments, n_imfs * n_bands)
    covs: np.ndarray   # (n_segments, n_imfs, C, C)


def _one(segment, sfreq, n_imfs, n_dirs):
    mu = segment.mean(axis=1, keepdims=True)
    sd = segment.std(axis=1, keepdims=True)
    imfs = memd((segment - mu) / np.where(sd < 1e-8, 1.0, sd), n_imfs=n_imfs, n_dirs=n_dirs)
    return band_energies(imfs, sfreq), mode_covariances(imfs)


def extract(X: np.ndarray, sfreq: float, n_imfs: int = 6, n_dirs: int = 32,
            n_jobs: int = -1, reg: float = 1e-4) -> Features:
    """MEMD, WVD band energies and per-mode covariances for every segment.

    X : (n_segments, n_channels, n_times), already filtered (e.g. 0.1-40 Hz).
    """
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(seg, sfreq, n_imfs, n_dirs) for seg in X)
    wvd = np.stack([r[0] for r in res])
    covs = np.stack([r[1] for r in res])
    C = covs.shape[-1]
    for k in range(covs.shape[1]):  # small ridge, keeps every matrix SPD
        covs[:, k] += reg * np.trace(covs[:, k].mean(0)) / C * np.eye(C)
    return Features(wvd=wvd, covs=covs)
