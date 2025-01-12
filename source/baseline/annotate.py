from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.tforest import TForestClassifier
from ..benchmark.types import PoisonedInstance
import numpy as np
from tqdm import tqdm


def create_model(params: dict):
    return TForestClassifier(**params) if "n_estimators" in params.keys() else DecisionTreeClassifier(**params)


def annotate_iterations(
    iterations: list[PoisonedInstance], xtr: np.ndarray, ytr: np.ndarray, params: dict
) -> list[PoisonedInstance]:
    is_forest = "n_estimators" in params.keys()
    print(f"{is_forest=}")

    yp = ytr.copy()
    tree = create_model(params)
    if not is_forest:
        tree.set_compute_stable_paths(True)  # type: ignore
    tree.fit(xtr, yp)
    pre = np.sum(tree.predict(xtr) != yp)
    instances = []

    for it in tqdm(range(len(iterations)), leave=False, desc="Annotating iterations"):
        i = iterations[it]
        yp[i.instanceid] = 1 - yp[i.instanceid]
        tree = create_model(params)

        if not is_forest:
            tree.set_compute_stable_paths(True)  # type: ignore

        tree.fit(xtr, yp)
        post = np.sum(tree.predict(xtr) != ytr)
        paths = tree.decode_paths(ytr) if not is_forest else None
        bundles = tree.get_stable_path_bundles(ytr) if is_forest else None

        if is_forest:
            depth = 0
            nsamples = bundles[i.instanceid].weight  # type: ignore
        else:
            depth = paths[i.instanceid].depth  # type: ignore
            nsamples = paths[i.instanceid].samples  # type: ignore

        instances.append(
            PoisonedInstance(
                instanceid=i.instanceid,
                label=i.label,
                trainloss=post - pre,
                testloss=0,
                depth=depth,
                nsamples=nsamples,
                runtime=i.runtime,
            )
        )

        pre = post

    return instances
