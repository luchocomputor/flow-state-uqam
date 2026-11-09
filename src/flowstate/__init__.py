"""EEG decoding of in-game attentional engagement: MEMD, Wigner-Ville and Riemannian features."""
from .evaluation import loso, permutation_test
from .features import Features, band_energies, extract, mode_covariances, wigner_ville
from .memd import memd
from .model import FusedClassifier
from .synthetic import make_dataset

__all__ = ["memd", "wigner_ville", "band_energies", "mode_covariances", "extract", "Features",
           "FusedClassifier", "loso", "permutation_test", "make_dataset"]
__version__ = "0.1.0"
