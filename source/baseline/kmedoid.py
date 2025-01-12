from sklearn.metrics import pairwise_distances
from pandas import DataFrame
from time import time_ns
import numpy as np
from source.benchmark.types import PoisonedInstance
from .annotate import annotate_iterations
from ..benchmark.types import TargetClass
from ..data import train_validation_split


TARGET = 1


def euclidean(medoid: np.ndarray, instance: np.ndarray) -> float:
    return float(np.linalg.norm(medoid - instance))


def medoids(x: np.ndarray, y: np.ndarray):
    xpos = x[y == 1]
    xneg = x[y == 0]

    dstpos = pairwise_distances(xpos, xpos)
    dstneg = pairwise_distances(xneg, xneg)

    medpos = xpos[dstpos.sum(axis=1).argmin()]
    medneg = xneg[dstneg.sum(axis=1).argmin()]

    return medpos, medneg


def kmedoid(
    x: np.ndarray,
    y: np.ndarray,
    budget: int,
    tree_params: dict,
    target: TargetClass = TARGET,
    early_stop: bool = False,
    verbose: bool = False,
):
    xp, xv, yp, yv = train_validation_split(x, y, random_state=tree_params["random_state"])
    start = time_ns()
    candidates = np.where(yp == target)[0]
    weights = np.zeros(yp.size)

    mpos, mneg = medoids(xp, yp)

    for i in candidates:
        if target == 1:
            weights[i] = euclidean(xp[i], mneg)
        else:
            weights[i] = euclidean(xp[i], mpos)

    weights = weights[candidates]
    sorted = np.argsort(weights)
    runtime = time_ns() - start

    selected = candidates[sorted[:budget]]
    candidates = [
        PoisonedInstance(
            instanceid=i,
            label=target if target is not None else TARGET,
            trainloss=0,
            testloss=0,
            depth=0,
            nsamples=0,
            runtime=int(runtime / budget),
        )
        for i in selected
    ]

    # candidates = annotate_iterations(candidates, x, y, tree_params)
    return candidates, runtime, 0
