from time import time_ns
import numpy as np
from source.benchmark.types import PoisonedInstance
from ..benchmark.types import TargetClass
from ..data import train_validation_split

TARGET = 1

def random(
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

    selected = np.random.choice(candidates, budget, replace=False)
    candidates = [
        PoisonedInstance(
            instanceid=i,
            label=target if target is not None else TARGET,
            trainloss=0,
            testloss=0,
            depth=0,
            nsamples=0,
            runtime=0,
        )
        for i in selected
    ]

    elapsed = time_ns() - start
    return candidates, elapsed, 0