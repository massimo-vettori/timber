from ..benchmark.types import PoisonedInstance, TargetClass
from sklearn.tree._stable_path import DecodedStablePath
from sklearn.tree.ttree import TTreeClassifier
from ..data import train_validation_split
from ..logger import LOGGER_NAME
from logging import getLogger
from threading import Thread
from time import time_ns
from tqdm import tqdm
import numpy as np

LOGGER = getLogger(LOGGER_NAME)
MINIMUM_VALID_LOSS = 0
THREAD_COUNT = 16
DEFAULT_TARGET_CLASS = 1
INT32_MAX = int(np.iinfo(np.int32).max)


def find_candidate(
    xp: np.ndarray,
    yp: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    mod: int,
    div: int,
    paths: list[DecodedStablePath],
    params: dict,
    early_stop: bool,
    results: list[tuple[int, int | None, int]],
):
    xpc, ypc = xp.copy(), yp.copy()

    target = None
    loss = None
    tries = 0

    tree = TTreeClassifier(**params)
    tree.fit(xpc, ypc)
    errs = tree.compute_error(None, xpc, ypc, xv, yv)

    for i, path in enumerate(paths):
        if early_stop:
            for r in results:
                if r[1] is not None and r[1] > MINIMUM_VALID_LOSS:
                    # Another thread has already found a valid candidate
                    LOGGER.debug(f"[Thread {mod}]: Another thread has already found a valid candidate")
                    return

        if i % div != mod:
            continue

        if path.depth >= params["max_depth"]:
            # When the depth is max_depth is just an artifact of the depth calculation. In order to safely retrain the tree
            # we can set the depth to 0 and the samples to the size of the training set
            # THIS BEHAVIOUR HAS NOT BEEN DETECTED YET, but it serves as a safety measure
            path = DecodedStablePath(path.vulnerable, 0, ypc.size, path.target, 0, path.label, "*")

        e = tree.compute_error(path, xpc, ypc, xv, yv)
        l = e - errs
        tries += 1

        if l > MINIMUM_VALID_LOSS and early_stop:
            target = path.target
            loss = l
            break

        if loss is None or l > loss:
            target = path.target
            loss = l

    if target is None or loss is None:
        LOGGER.debug(f"[Thread {mod}]: No valid candidate found")
        return

    LOGGER.debug(f"[Thread {mod}]: Found candidate {target} with loss {loss} after {tries} tries")
    results[mod] = (target, loss, tries)


def sort_paths(tree_params: dict, pathlist: list[DecodedStablePath]) -> list[DecodedStablePath]:
    sortkey = lambda p: p.samples if (p.depth < tree_params["max_depth"] and p.vulnerable) else float("inf")
    return sorted(pathlist, key=sortkey)


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


def timber(
    x: np.ndarray,
    y: np.ndarray,
    budget: int,
    tree_params: dict,
    target: TargetClass,
    early_stop: bool = False,
    verbose: bool = False,
) -> tuple[list[PoisonedInstance], int, int]:
    # yp, xp = y.copy(), x.copy()
    # xv, yv = xp.copy(), yp.copy()
    xp, xv, yp, yv = train_validation_split(x, y, random_state=tree_params["random_state"])

    selected = []
    instances = []

    runtime = 0

    prevscore = None
    lastloss = None

    for k in tqdm(range(budget), leave=False):
        tree = TTreeClassifier(**tree_params)
        tree.set_compute_stable_paths(True)
        tree.fit(xp, yp)

        score = tree.compute_error(None, xp, yp, xv, yv)
        if prevscore is not None and lastloss is not None:
            if score - prevscore != lastloss:
                LOGGER.warning(f"Score mismatch: {score} - {prevscore} != {lastloss}")
                break

        prevscore = score
        paths = tree.decode_paths(yp)
        valid = [p for p in paths if (p.target not in selected) and (p.label == target or target is None)]

        if not valid:
            LOGGER.debug(f"No valid paths found at iteration {k}")
            break

        if early_stop:
            valid = sort_paths(tree_params, valid)

        threads = []
        results = [(-1, None, 0) for _ in range(THREAD_COUNT)]  # type: list[tuple[int, int|None, int]]

        start = time_ns()
        for i in range(THREAD_COUNT):
            thread = Thread(
                target=lambda: find_candidate(xp, yp, xv, yv, i, THREAD_COUNT, valid, tree_params, early_stop, results)
            )

            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        elapsed = time_ns() - start
        tgt, loss, _ = get_best_result(results)

        if tgt < 0 or tgt >= yp.size or loss is None:
            LOGGER.debug(f"No valid candidate found at iteration {k}")
            break

        selected.append(tgt)
        yp[tgt] = 1 - yp[tgt]
        lastloss = loss

        instances.append(
            PoisonedInstance(
                instanceid=tgt,
                label=paths[tgt].label,
                trainloss=lastloss,
                testloss=0,
                depth=paths[tgt].depth,
                nsamples=paths[tgt].samples,
                runtime=elapsed,
            )
        )

        LOGGER.debug(
            f"Iteration {k}. Selected instance (id:{tgt}) with loss {lastloss} and absolute depth {paths[tgt].depth}"
        )
        LOGGER.debug(f"This instance was found in {sum([r[2] for r in results])} tries.\n")
        runtime += elapsed

    return instances, runtime, 0
