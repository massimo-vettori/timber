from ..benchmark.types import TargetClass, PoisonedInstance
from sklearn.ensemble.tforest import RandomForestClassifier
from ..data import train_validation_split
from ..logger import LOGGER_NAME
from logging import getLogger
from datetime import datetime
from threading import Thread
from time import time_ns
from tqdm import tqdm
import numpy as np

INT32_MAX = int(np.iinfo(np.int32).max)
MINIMUM_VALID_LOSS = 0
THREAD_COUNT = 16
DEFAULT_TARGET_CLASS = 0
LOGGER = getLogger(LOGGER_NAME)


def retrain(params: dict, candidate: int, xpc: np.ndarray, ypc: np.ndarray, xv: np.ndarray, yv: np.ndarray) -> int:
    seq_params = params.copy()
    seq_params["n_jobs"] = 1

    trf = RandomForestClassifier(**seq_params)

    ypc[candidate] = 1 - ypc[candidate]
    trf.fit(xpc, ypc)
    ypc[candidate] = 1 - ypc[candidate]

    return (trf.predict(xv) != yv).sum()


def find_candidate(
    xp: np.ndarray,
    yp: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    target: TargetClass,
    mod: int,
    div: int,
    candidates: np.ndarray,
    params: dict,
    early_stop: bool,
    results: list[tuple[int, int | None, int]],
):
    xpc = xp.copy()
    ypc = yp.copy()

    trf = RandomForestClassifier(**params)
    trf.fit(xpc, ypc)
    orig = int((trf.predict(xv) != yv).sum())

    tries = 0
    best = None
    tgt = None

    for i, candidate in enumerate(candidates):
        candidate = int(candidate)
        if early_stop:
            for r in results:
                if r[1] is not None and r[1] > MINIMUM_VALID_LOSS:
                    LOGGER.debug(f"[Thread {mod}]: Another thread has already found a valid candidate")
                    return

        if i % div != mod or (yp[candidate] != target and target is not None):
            continue

        curr = retrain(params, candidate, xpc, ypc, xv, yv)
        tries += 1
        loss = curr - orig

        if loss > MINIMUM_VALID_LOSS and early_stop:
            tgt = candidate
            best = loss
            break

        if best is None or loss > best:
            best = loss
            tgt = candidate

    if tgt is None or best is None:
        LOGGER.debug(f"[Thread {mod}]: No valid candidate found")
        return

    LOGGER.debug(f"[Thread {mod}]: Found candidate {tgt} with loss {best} in {tries} tries")
    results[mod] = (tgt, best, tries)
    return


def get_best_result(results: list[tuple[int, int | None, int]]) -> tuple[int, int | None, int]:
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


def rgreedy(
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
        trf = RandomForestClassifier(**tree_params)
        trf.fit(xp, yp)

        score = (trf.predict(xv) != yv).sum()
        if prevscore is not None and lastloss is not None:
            if score - prevscore != lastloss:
                LOGGER.warning(f"Discrepancy detected: {score - prevscore} != {lastloss}")
        prevscore = score

        results = [(-1, None, 0) for _ in range(THREAD_COUNT)]  # type: list[tuple[int, int|None, int]]
        candidates = np.arange(yp.size)
        candidates = np.setdiff1d(candidates, selected)
        threads = []

        for i in range(THREAD_COUNT):
            thread = Thread(
                target=lambda: find_candidate(
                    xp, yp, xv, yv, target, i, THREAD_COUNT, candidates, tree_params, early_stop, results
                )
            )

            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        c, l, _ = get_best_result(results)
        total_tries = sum([x[2] for x in results if x[2] > 0])

        if c is None or c < 0 or c > yp.size or l is None:
            LOGGER.info("Terminating early due to no valid candidate found")
            break

        elapsed = time_ns() - start
        runtime += elapsed
        selected.append(c)
        yp[c] = 1 - yp[c]

        instances.append(
            PoisonedInstance(
                instanceid=c,
                label=1 - yp[c],
                trainloss=l,
                testloss=0,
                depth=0,
                nsamples=0,
                runtime=elapsed,
            )
        )

        LOGGER.debug(f"Iteration {k}. Selected instance (id:{c}) with loss {l}")
        LOGGER.debug(f"This instance was found in {total_tries} tries\n")
        lastloss = l

    return instances, runtime, budget
