from source.benchmark.synthesis import RunSynthesis, PoisonedInstance, SYNTHESIS_PATH
from source.defence.bagging import bagging_defence, fit_model
from source.defence.knn import knn_based_defence
from source.defence.base import no_defence
from source.logger import getLogger, LOGGER_NAME
from source.data import load, split

from sklearn.metrics import f1_score, accuracy_score
from itertools import product
from pandas import DataFrame
from numpy import ndarray
from os.path import join
from json import loads

LOGGER = getLogger(LOGGER_NAME)

def flip_labels(
    x: ndarray, y: ndarray, 
    instances: list[PoisonedInstance]
) -> tuple[ndarray, ndarray]:
    xcp, ycp = x.copy(), y.copy()
    
    for instance in instances:
        ycp[instance.instanceid] = 1 - ycp[instance.instanceid]
    
    return xcp, ycp

def apply_defence(
    kind: str,
    params: dict,
    synthesis: RunSynthesis
):
    x, y = load(synthesis.dataset)
    xtr, xte, ytr, yte = split(x, y, synthesis.test_size, synthesis.split_seed)
    xpo, ypo = flip_labels(xtr, ytr, synthesis.poisoned)

    match kind:
        case "bagging":
            orig_model = bagging_defence(xtr, ytr, synthesis.modelparams, **params)
            pois_model = bagging_defence(xpo, ypo, synthesis.modelparams, **params)
        case "knn":
            orig_model = knn_based_defence(xtr, ytr, synthesis.modelparams, **params)
            pois_model = knn_based_defence(xpo, ypo, synthesis.modelparams, **params)
        case "no_defence":
            orig_model = no_defence(xtr, ytr, synthesis.modelparams)
            pois_model = no_defence(xpo, ypo, synthesis.modelparams)
        case _:
            raise ValueError(f"Unknown defence kind: {kind}")

    orig_acc = accuracy_score(yte, orig_model.predict(xte))
    orig_f1s = f1_score(yte, orig_model.predict(xte))
    pois_acc = accuracy_score(yte, pois_model.predict(xte))
    pois_f1s = f1_score(yte, pois_model.predict(xte))

    return (orig_acc, orig_f1s), (pois_acc, pois_f1s)

def defence(
    kind: str,
    pgrid: dict,
    synthesis: RunSynthesis
) -> DataFrame:
    outcome = DataFrame(columns=["kind", *pgrid.keys(), "orig_acc", "orig_f1s", "pois_acc", "pois_f1s"])

    no_defence_params = tuple("" for _ in pgrid.keys())

    orig, pois = apply_defence("no_defence", dict(zip(pgrid.keys(), no_defence_params)), synthesis)
    outcome.loc[len(outcome)] = ["NO DEFENCE", *no_defence_params, *orig, *pois]

    for params in product(*pgrid.values()):
        orig, pois = apply_defence(kind, dict(zip(pgrid.keys(), params)), synthesis)
        outcome.loc[len(outcome)] = [kind, *params, *orig, *pois]

    return outcome


def get_syntheses():
    from os import listdir

    return [
        RunSynthesis.partial_load(join(SYNTHESIS_PATH, synth))
        for synth in listdir(SYNTHESIS_PATH)
    ]

def main():
    syntheses = get_syntheses()
    
    with open("pipeline.json", "r") as f:
        pipeline = loads(f.read())
        defences = pipeline["defences"]
        LOGGER.debug(f"Successfully read pipeline.json")

    for syn in syntheses:
        for method in defences:
            results = defence(method["kind"], method["param_grid"], syn)
            results.to_csv(join(SYNTHESIS_PATH, syn.name, f"{method['kind']}.def.csv"), index=False)

if __name__ == "__main__":
    main()
