# flow-state-uqam

[![tests](https://github.com/luchocomputor/flow-state-uqam/actions/workflows/tests.yml/badge.svg)](https://github.com/luchocomputor/flow-state-uqam/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline-dark.svg">
  <img width="420" alt="Pipeline: EEG, multivariate EMD into 6 joint modes, Wigner-Ville band energies and Riemannian tangent vectors, fused in an SVM" src="assets/pipeline-light.svg">
</picture>
</p>

Code for **EEG decoding of in-game attentional engagement**: multivariate empirical mode decomposition, Wigner-Ville band energies and per-mode Riemannian geometry, evaluated leave-one-subject-out.

Louis Almairac, Mounir Boukadoum. *EEG Decoding of In-Game Attentional Engagement.* Submitted to IEEE ISCAS 2027. Research internship at UQAM (MicroPro lab, Montréal), 2026.

## The study

26 skilled players, about 4 hours of *Apex Legends* each, recorded at Ubisoft La Forge with a 16-channel EEG headset (OpenBCI, 125 Hz). Flow state has no ground truth, so the labels come from behaviour: small targets appeared at the edge of the screen and players pressed a button when they saw one. A long run of missed targets marks a player fully absorbed in the game; lobby rest between matches is the contrast.

Absorbed play vs. rest is decoded across subjects with macro-F1 = 0.76 (leave-one-subject-out), on par with an end-to-end EEG-Conformer. Control analyses (eye-tracking, within-subject permutation tests, self-supervised probes) indicate what the model reads: a broad play-vs-rest arousal shift rather than a marker of flow itself, hence the framing as attentional engagement.

## What is and is not in this repository

In the repository:

- `flowstate`, a small Python package with the feature pipeline of the paper: MEMD, Wigner-Ville band energies, per-mode covariances, the tangent-space SVM, leave-one-subject-out evaluation and the permutation test.
- A synthetic data generator, a script and a notebook that run the pipeline end to end, and tests.

Not in the repository:

- The recordings and the labels. They belong to the Ubisoft La Forge study and are not public, so the result above cannot be reproduced from this repository. The scores printed by the demo and the notebook describe synthetic data and say nothing about real EEG.
- The steps before the features: filtering, segmentation and labelling of the recordings.
- The EEG-Conformer baseline and the control analyses.
- The per-subject recentring of the covariance matrices that the paper applies before the tangent-space projection. Here the tangent vectors are z-scored within each subject, without that step.

## The method

| Step | What it does | Code |
|---|---|---|
| Multivariate EMD | Sifts all channels jointly along 32 random directions of the sphere, with cubic-spline envelopes and mirror extension at the edges, so that mode *k* captures the same oscillatory scale on every channel. Five modes and the residue. | `flowstate.memd` |
| Wigner-Ville band energies | Wigner-Ville distribution of each mode, averaged over channels; its mean in 5 bands (delta to gamma) gives 6 × 5 = 30 features. | `flowstate.wigner_ville`, `band_energies` |
| Per-mode covariances | Ledoit-Wolf covariance between channels for each mode, one SPD matrix per mode. | `flowstate.mode_covariances` |
| Tangent space | Each mode's matrices are mapped to the tangent space at their Riemannian mean, learned on the training subjects only. | `flowstate.FusedClassifier` |
| Classifier | Both feature sets are z-scored within each subject and fused in an RBF SVM. | `flowstate.FusedClassifier` |
| Evaluation | Leave-one-subject-out, and a permutation test that shuffles labels within each subject. | `flowstate.loso`, `permutation_test` |

Everything in `extract` is computed segment by segment, without labels, so features can be cached before cross-validation without leakage. The choices behind each step are listed in [Implementation notes](#implementation-notes).

## Quick start

```bash
git clone https://github.com/luchocomputor/flow-state-uqam
cd flow-state-uqam
pip install -e .
python examples/demo_synthetic.py
```

```
240 segments, 8 channels, 500 samples, 6 subjects
features: WVD (240, 30), covariances (240, 6, 8, 8) (10 s)
wvd      LOSO macro-F1 = 0.871
riemann  LOSO macro-F1 = 0.900
fused    LOSO macro-F1 = 0.925
permutation test (wvd): F1 = 0.871, null = 0.505, p = 0.020
```

The demo uses 16 MEMD directions and 50 permutations to stay short; the defaults are 32 and 200.

On your own data, segments of shape `(n_segments, n_channels, n_times)`, already band-pass filtered:

```python
import flowstate as fs

feats = fs.extract(X, sfreq=125.0)            # MEMD, WVD energies, per-mode covariances
res = fs.loso(lambda: fs.FusedClassifier(), feats, y, groups)
print(res["f1"], res["per_subject"])

perm = fs.permutation_test(lambda: fs.FusedClassifier(), feats, y, groups, n_permutations=500)
print(perm["p_value"])
```

`y` holds the two class labels and `groups` the subject of each segment. `FusedClassifier(use="wvd")` and `use="riemann"` run each branch alone.

## Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luchocomputor/flow-state-uqam/blob/main/notebooks/demo.ipynb)

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) shows what each step does on synthetic data: the modes, the Wigner-Ville map, the tangent space before and after per-subject z-scoring, the leave-one-subject-out scores and the permutation test. It runs in under two minutes on a laptop.

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/demo.ipynb
```

<p align="center">
<img width="720" alt="MEMD modes of one synthetic segment on three channels, with the amplitude spectrum of each mode" src="assets/demo-memd-modes.png">
</p>

MEMD of one synthetic segment. The label on the right is the peak of the spectrum averaged over the 8 channels. Within a row the channels have peaks at the same frequencies, but not always the same dominant one: theta (6 Hz) and alpha (10 Hz) are less than an octave apart, and MEMD does not fully separate them here.

<p align="center">
<img width="560" alt="PCA of the tangent vectors of mode 3, coloured by subject and by class, before and after per-subject z-scoring" src="assets/demo-tangent-space.png">
</p>

Tangent vectors of mode 3 (PCA, synthetic data): one cluster per subject before per-subject z-scoring (left), classes separated after (right).

## Implementation notes

Choices a reader of the code should know about. They are those of the pipeline behind the paper, so they are stated here rather than changed.

- **Standardisation.** `extract` centres and scales each channel of each segment before MEMD. The features do not depend on the amplitude of a channel.
- **MEMD directions.** The projection directions are normalised Gaussian vectors drawn with a fixed seed, so every segment is decomposed along the same directions. Rehman and Mandic (2010) use a Hammersley sequence.
- **MEMD stopping criterion.** Sifting of a mode stops when the normalised squared difference between two iterations falls below 0.2, or after 15 iterations. The reference implementation uses the envelope-based criterion of Rilling et al. (2003).
- **Sixth mode.** `n_imfs=6` means five modes and the residue, kept as the slowest mode.
- **Wigner-Ville.** The distribution has no smoothing window, in time or in lag. It is computed at 64 instants per segment; at each instant the lags are limited by the distance to the nearest edge of the segment, so the frequency resolution is best at the centre, and the first and last instants are left at zero.
- **Band energies.** A feature is the mean of the distribution over a band and over time. The distribution is signed, so a feature can be negative: about a quarter of them on the synthetic demo, for about 1 % of the summed absolute values. The bands are 1-4, 4-8, 8-13, 13-29 and 31-40 Hz, edges included; nothing between 29 and 31 Hz is used.
- **Covariances.** Ledoit-Wolf shrinkage, then a ridge of 1e-4 times the mean eigenvalue of each matrix.
- **Per-subject z-scoring.** Features are z-scored within each subject, the held-out subject included, with that subject's own mean and standard deviation. This uses the unlabelled segments of the test subject, never its labels. `predict` therefore needs all the segments of a subject at once.
- **Synthetic scores.** With other seeds for the MEMD directions (four tried), the demo scores move by up to 0.03, which is comparable to the gaps between the three variants: their order on the demo carries no information.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests check that the modes add up to the signal, that MEMD puts the fast oscillation in the first mode, that the Wigner-Ville frequency axis is right, that a segment gives the same features alone or in a batch, that a fold does not depend on the labels of its held-out subject, that the permutation test gives the same scores as rerunning the evaluation on shuffled labels, and that the pipeline is above chance on synthetic data with an effect and at chance without one. They also cover degenerate inputs (flat or very short segments, a single channel, NaN, a subject with a single class).

## Citation

The paper is under review. Until it is published:

> L. Almairac and M. Boukadoum, "EEG Decoding of In-Game Attentional Engagement", submitted to IEEE ISCAS 2027.

The same reference is in [CITATION.cff](CITATION.cff).

Methods this code builds on:

- N. E. Huang et al., "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis", *Proc. R. Soc. Lond. A*, 1998.
- G. Rilling, P. Flandrin and P. Gonçalves, "On empirical mode decomposition and its algorithms", *IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing*, 2003.
- N. Rehman and D. P. Mandic, "Multivariate empirical mode decomposition", *Proc. R. Soc. A*, 2010. The mirror extension in `memd.py` is ported from their MATLAB implementation, `memd.m`.
- A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass brain-computer interface classification by Riemannian geometry", *IEEE Trans. Biomed. Eng.*, 2012.
- O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices", *J. Multivariate Anal.*, 2004.

## License

MIT, see [LICENSE](LICENSE).
