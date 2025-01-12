from sklearn.ensemble.tforest import TForestClassifier
from sklearn.ensemble.tforest import StablePathBundle
import numpy as np


def get_poisoned_loss(
    forest: TForestClassifier,
    bundle: StablePathBundle,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    error: int,
):
    poisoned = bundle.compute_error(forest, xtr, ytr, xv, yv)
    return poisoned - error
