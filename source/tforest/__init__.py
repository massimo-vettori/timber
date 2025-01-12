from sklearn.ensemble.tforest import TForestClassifier, StablePathBundle
from ..benchmark.types import PoisonedInstance, TargetClass
from ..data import train_validation_split
from .find import find_candidate
from ..logger import LOGGER_NAME
from logging import getLogger
from threading import Thread
from time import time_ns
from tqdm import tqdm
import numpy as np


LOGGER = getLogger(LOGGER_NAME)
THREAD_COUNT = 16
INT32_MAX = int(np.iinfo(np.int32).max)


def get_best_result(
    results: list[tuple[int, int | None, int]],
) -> tuple[int, int | None, int]:
    filtered = [(iid, loss if loss is not None else -INT32_MAX, tries) for (iid, loss, tries) in results]

    best = filtered[0]
    bi, bl, __ = best

    for iid, loss, tries in filtered:
        if loss > bl:
            best = (iid, loss, tries)
            bi, bl, __ = best
            continue

        if loss == bl and iid < bi:
            best = (iid, loss, tries)
            bi, bl, __ = best

    return best


def rtimber(
    x: np.ndarray,
    y: np.ndarray,
    budget: int,
    tree_params: dict,
    target: TargetClass,
    early_stop: bool = False,
    verbose: bool = False,
):
    xp, xv, yp, yv = train_validation_split(x, y, random_state=tree_params["random_state"])

    selected = []
    instances = []

    runtime = 0

    prevscore = None
    lastloss = None

    for k in tqdm(range(budget), leave=False):
        start = time_ns()

        forest = TForestClassifier(**tree_params)
        forest.fit(xp, yp)
        error = forest.compute_error(xv, yv)

        if prevscore is not None and lastloss is not None:
            if error - prevscore != lastloss:
                LOGGER.warning(f"Score mismatch: {error} - {prevscore} != {lastloss}")

        prevscore = error
        bundles = forest.get_stable_path_bundles(yp)
        bundles = StablePathBundle.sort(bundles) if early_stop else bundles
        validbundles = [b for b in bundles if b.get_target(forest) not in selected and b.label == target]

        if not validbundles:
            LOGGER.debug(f"No valid bundles found at iteration {k}")
            break

        threads = []
        results = [(-1, None, 0) for _ in range(THREAD_COUNT)]

        for i in range(THREAD_COUNT):
            thread = Thread(
                target=lambda: find_candidate(xp, yp, xv, yv, i, THREAD_COUNT, validbundles, forest, early_stop, results)  # type: ignore
            )

            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        elapsed = time_ns() - start
        tgt, loss, _ = get_best_result(results)  # type: ignore

        if tgt < 0 or tgt >= y.size or loss is None:
            LOGGER.debug(f"No valid candidate found at iteration {k}")
            break

        selected.append(tgt)
        yp[tgt] = 1 - yp[tgt]
        lastloss = loss

        instances.append(
            PoisonedInstance(
                instanceid=tgt,
                label=1 - yp[tgt],
                trainloss=loss,
                testloss=0,
                depth=0,
                nsamples=0,
                runtime=elapsed,
            )
        )

        LOGGER.debug(f"Iteration {k}. Selected instance (id:{tgt}) with loss {lastloss}")
        LOGGER.debug(f"This instance was found in {sum([r[2] for r in results])} tries.\n")
        runtime += elapsed

    return instances, runtime, 0
