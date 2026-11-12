"""Multivariate empirical mode decomposition (Rehman & Mandic, 2010).

The signal is projected on many directions of the unit sphere; for each one,
the extrema of the projection give the instants where the multichannel upper
and lower envelopes are interpolated (cubic splines, mirror extension at the
edges as in Rilling et al., 2003). Their mean, averaged over directions, is the
local mean removed at each sifting step. All channels are sifted together, so
mode k is meant to capture the same oscillatory scale on every channel.

Two choices differ from the reference implementation. The directions are
normalised Gaussian draws (uniform on the sphere) with a fixed seed, not a
Hammersley sequence. Sifting of a mode stops when the normalised squared
difference between two iterations (Huang et al., 1998), computed over all
channels, falls below a threshold, or after a fixed number of iterations; the
reference implementation uses the envelope-based criterion of Rilling et al.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _boundary_conditions(indmin, indmax, x, nbsym=2):
    """Mirror extension of the extrema at both edges (0-based port of memd.m)."""
    T = len(x)
    last = T - 1
    nmin, nmax = len(indmin), len(indmax)
    rev = lambda a: np.asarray(a, dtype=int)[::-1]  # noqa: E731
    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax, lmin, lsym = rev(indmax[1:nbsym + 1]), rev(indmin[:nbsym]), indmax[0]
        else:
            lmax, lmin, lsym = rev(indmax[:nbsym]), np.r_[rev(indmin[:nbsym - 1]), 0], 0
    else:
        if x[0] < x[indmax[0]]:
            lmax, lmin, lsym = rev(indmax[:nbsym]), rev(indmin[1:nbsym + 1]), indmin[0]
        else:
            lmax, lmin, lsym = np.r_[rev(indmax[:nbsym - 1]), 0], rev(indmin[:nbsym]), 0
    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = rev(indmax[max(nmax - nbsym, 0):])
            rmin = rev(indmin[max(nmin - nbsym - 1, 0):nmin - 1])
            rsym = indmin[-1]
        else:
            rmax = np.r_[last, rev(indmax[max(nmax - nbsym + 1, 0):])]
            rmin = rev(indmin[max(nmin - nbsym, 0):])
            rsym = last
    else:
        if x[-1] > x[indmin[-1]]:
            rmax = rev(indmax[max(nmax - nbsym - 1, 0):nmax - 1])
            rmin = rev(indmin[max(nmin - nbsym, 0):])
            rsym = indmax[-1]
        else:
            rmax = rev(indmax[max(nmax - nbsym, 0):])
            rmin = np.r_[last, rev(indmin[max(nmin - nbsym + 1, 0):])]
            rsym = last
    if min(len(lmin), len(lmax), len(rmin), len(rmax)) == 0:
        return None
    if 2 * lsym - lmin[0] > 0 or 2 * lsym - lmax[0] > 0:
        if lsym == indmax[0]:
            lmax = rev(indmax[:nbsym])
        else:
            lmin = rev(indmin[:nbsym])
        if lsym == 0:
            return None
        lsym = 0
    if 2 * rsym - rmin[-1] < last or 2 * rsym - rmax[-1] < last:
        if rsym == indmax[-1]:
            rmax = rev(indmax[max(nmax - nbsym, 0):])
        else:
            rmin = rev(indmin[max(nmin - nbsym, 0):])
        if rsym == last:
            return None
        rsym = last
    return lmin, lmax, rmin, rmax, lsym, rsym


def _envelope_mean(h, proj, t, nbsym=2):
    """Mean of the upper and lower multichannel envelopes along one direction, (C, T) or None."""
    imax = np.flatnonzero((proj[1:-1] > proj[:-2]) & (proj[1:-1] > proj[2:])) + 1
    imin = np.flatnonzero((proj[1:-1] < proj[:-2]) & (proj[1:-1] < proj[2:])) + 1
    if len(imax) < 1 or len(imin) < 1 or len(imax) + len(imin) < 3:
        return None
    bc = _boundary_conditions(imin, imax, proj, nbsym)
    if bc is None:
        return None
    lmin, lmax, rmin, rmax, lsym, rsym = bc
    tmin = np.r_[2 * lsym - lmin, imin, 2 * rsym - rmin]
    tmax = np.r_[2 * lsym - lmax, imax, 2 * rsym - rmax]
    if np.any(np.diff(tmin) <= 0) or np.any(np.diff(tmax) <= 0):
        return None
    zmin = h[:, np.r_[lmin, imin, rmin]].T
    zmax = h[:, np.r_[lmax, imax, rmax]].T
    return 0.5 * (CubicSpline(tmin, zmin, axis=0)(t) + CubicSpline(tmax, zmax, axis=0)(t)).T


def memd(x: np.ndarray, n_imfs: int = 6, n_dirs: int = 32, max_sift: int = 15,
         sd_thr: float = 0.2, seed: int = 42) -> np.ndarray:
    """Decompose a multichannel signal into aligned modes.

    Parameters
    ----------
    x : array (n_channels, n_times)
    n_imfs : number of modes returned; the last one is the residue (slowest).
    n_dirs : number of projection directions on the sphere.
    max_sift : maximum number of sifting iterations per mode.
    sd_thr : sifting of a mode stops once
        sum((h_prev - h) ** 2) / sum(h_prev ** 2) < sd_thr, summed over channels.
    seed : seed of the directions. They depend only on (seed, n_dirs, n_channels),
        so every call with the same arguments uses the same directions.

    Returns
    -------
    imfs : array (n_imfs, n_channels, n_times), from fastest to slowest. They sum
        to x. If no direction has enough extrema left to sift (constant, monotonic
        or very short input), what remains is returned as the current mode and
        the following ones are zero.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must have shape (n_channels, n_times), got {x.shape}")
    if n_imfs < 1 or n_dirs < 1:
        raise ValueError(f"n_imfs and n_dirs must be at least 1, got {n_imfs} and {n_dirs}")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or infinite values")
    C, T = x.shape
    t = np.arange(T, dtype=float)
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_dirs, C))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    imfs, residue = [], x.astype(np.float64).copy()
    for _ in range(n_imfs - 1):
        h = residue.copy()
        for _ in range(max_sift):
            projs = dirs @ h
            mean_env, n_valid = np.zeros((C, T)), 0
            for proj in projs:
                env = _envelope_mean(h, proj, t)
                if env is not None:
                    mean_env += env
                    n_valid += 1
            if n_valid == 0:
                break
            prev = h.copy()
            h = h - mean_env / n_valid
            if np.sum((prev - h) ** 2) / (np.sum(prev ** 2) + 1e-10) < sd_thr:
                break
        imfs.append(h)
        residue = residue - h
    imfs.append(residue)
    return np.asarray(imfs)
