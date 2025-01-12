import numpy as np
from pandas import DataFrame
from time import time_ns
from source.benchmark.types import PoisonedInstance
from .annotate import annotate_iterations
from ..benchmark.types import TargetClass
from ..data import train_validation_split

TARGET = 1


def normalize(features: DataFrame) -> DataFrame:
    mean = features.mean()
    std = features.std()
    return (features - mean) / std


def minmax(features: DataFrame) -> DataFrame:
    min = features.min()
    max = features.max()
    return (features - min) / (max - min)


def pij(features: DataFrame) -> np.ndarray:
    xnorm = normalize(features).to_numpy()
    output = np.zeros_like(xnorm)

    for i in range(xnorm.shape[0]):
        for j in range(xnorm.shape[1]):
            col = xnorm[:, j]
            output[i, j] = xnorm[i, j] / col.sum()

    return output


def ej(p: np.ndarray) -> np.ndarray:
    output = np.zeros(p.shape[1])

    for j in range(p.shape[1]):
        col = p[:, j]
        k = 1 / np.log2(p.shape[0])
        output[j] = -np.sum(col * np.log2(col + 0.0000000001)) * k

    return output


def Wj(e: np.ndarray) -> np.ndarray:
    total = np.sum(1 - e)
    return (1 - e) / total


def si(features: DataFrame) -> np.ndarray:
    p = pij(features)
    e = ej(p)
    w = Wj(e)

    output = np.zeros(features.shape[0])

    for i in range(features.shape[0]):
        row = p[i, :] * w
        output[i] = row.sum()

    return output


def entropy_method(
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
    features = DataFrame(xp)
    weights = si(features)
    runtime = time_ns() - start

    candidates = np.where(yp == target)[0] if target is not None else np.arange(yp.size)
    sorted = np.argsort(weights[candidates])
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

    return candidates, runtime, 0
