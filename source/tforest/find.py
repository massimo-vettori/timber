from sklearn.ensemble.tforest import TForestClassifier, StablePathBundle
from .retrain import get_poisoned_loss
import numpy as np
from ..logger import LOGGER_NAME
from logging import getLogger


LOGGER = getLogger(LOGGER_NAME)
MINIMUM_VALID_LOSS = 0


def find_candidate(
    xtrain: np.ndarray,
    ytrain: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    mod: int,
    div: int,
    bundles: list[StablePathBundle],
    forest: TForestClassifier,
    early_stop: bool,
    results: list[tuple[int, int | None, int]],
):
    xtc, ytc = xtrain.copy(), ytrain.copy()
    target = None
    tries = 0
    loss = None
    original = forest.compute_error(xv, yv)

    for i, bundle in enumerate(bundles):
        if early_stop:
            for r in results:
                if r[1] is not None and r[1] > MINIMUM_VALID_LOSS:
                    # Another thread has already found a valid candidate
                    LOGGER.debug(f"[Thread {mod}]: Another thread has already found a valid candidate")
                    return

        if i % div != mod:
            continue

        l = get_poisoned_loss(forest, bundle, xtc, ytc, xv, yv, original)
        tries += 1

        if l > MINIMUM_VALID_LOSS and early_stop:
            target = bundle.get_target(forest)
            loss = l
            break

        if loss is None or l > loss:
            target = bundle.get_target(forest)
            loss = l

    if target is None or loss is None:
        LOGGER.debug(f"[Thread {mod}]: No valid candidate found")
        return

    LOGGER.debug(f"[Thread {mod}]: Found candidate {target} with loss {loss} after {tries} tries")
    results[mod] = (target, loss, tries)
    return
