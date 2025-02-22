from json import load as json_load

from pandas import DataFrame
from source.logger import getLogger, LOGGER_NAME
import numpy as np
from typing import Any
from attrs import define

from source.benchmark.synthesis import RunSynthesis, PoisonedInstance, _get_subdir
from sklearn.ensemble.tforest import TForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from source.data import load, split, DATASETS
from source.benchmark import Bench, Poisoning
from sklearn.metrics import f1_score
from tqdm import tqdm

LOGGER = getLogger(LOGGER_NAME)
PIPELINE_CFG = "pipeline.json"


def get_method(name: str) -> Poisoning:
    from source.tforest import rtimber
    from source.gforest import rgreedy
    from source.baseline.entropy import entropy_method
    from source.baseline.kmedoid import kmedoid
    from source.baseline.random import random

    match name:
        case "timber":
            return rtimber
        case "greedy":
            return rgreedy
        case "ges":
            return rgreedy
        case "tes":
            return rtimber
        case "entropy":
            return entropy_method
        case "kmedoid":
            return kmedoid
        case "random":
            return random
        case _:
            raise ValueError(f"Method {name} not found")

def is_early_stopping(method: str) -> bool:
    return method in ["tes", "ges"]


class Annotator:
    @define(slots=True, frozen=True)
    class AnnotatedInstance(PoisonedInstance):
        train_f1: float
        test_f1: float
        train_acc: float
        test_acc: float

        def get_keys(self) -> list:
            return [*self.to_dict().keys()]

        def to_dict(self) -> dict[str, Any]:
            return {
                **super().to_dict(),
                "train_f1": self.train_f1,
                "test_f1": self.test_f1,
                "train_acc": self.train_acc,
                "test_acc": self.test_acc,
            }
        
        def to_list(self) -> list:
            return [v for v in self.to_dict().values()]
        
        def add_self_to_df(self, df: DataFrame) -> None:
            df.loc[len(df)] = self.to_list()
        


    @staticmethod
    def annotate(synthesis: RunSynthesis, dataset: "LoadedDataset"):
        clf = TForestClassifier(**synthesis.modelparams)
        clf.set_compute_stable_paths(False)
        clf.fit(dataset.xtr, dataset.ytr)

        xorig, yorig = dataset.xtr.copy(), dataset.ytr.copy()
        xpois, ypois = dataset.xtr.copy(), dataset.ytr.copy()

        LOGGER.debug(f"[Annotator]: Annotating {len(synthesis.poisoned)} instances for {synthesis.name}")
        progress = tqdm(total=len(synthesis.poisoned), desc="Annotating", unit="instance")

        for index, iteration in enumerate(synthesis.poisoned):
            ypois[iteration.instanceid] = 1 - ypois[iteration.instanceid]
            clf.fit(xpois, ypois)

            trn_acc = clf.score(xpois, ypois)
            trn_f1s = float(f1_score(y_true=yorig, y_pred=clf.predict(xorig)))
            tst_acc = clf.score(dataset.xv, dataset.yv)
            tst_f1s = float(f1_score(y_true=dataset.yv, y_pred=clf.predict(dataset.xv)))

            annotated = Annotator.AnnotatedInstance(train_f1=trn_f1s, test_f1=tst_f1s, train_acc=trn_acc, test_acc=tst_acc, **iteration.to_dict())
            synthesis.poisoned[index] = annotated
            progress.update(1)

        progress.close()

    @staticmethod
    def summarize(synthesis: RunSynthesis):
        df = DataFrame(columns=["step", "train_f1", "train_acc", "test_f1", "test_acc"])
        # Divides the iterations into 10 equal parts, with the remainder on the last group
        size = len(synthesis.poisoned) // 10

        LOGGER.debug(f"[Annotator]: Summarizing {len(synthesis.poisoned)} instances into 10 groups for {synthesis.name}")
        progress = tqdm(total=10, desc="Summarizing", unit="data point")

        # Add an initial line to the summary dataframe, to indiate the performance of the model before
        # poisoning
        first: Annotator.AnnotatedInstance = synthesis.poisoned[0] # type: ignore
        df.loc[len(df)] = [0, first.train_f1, first.train_acc, first.test_f1, first.test_acc]

        for chunk in range(10):
            start = chunk * size
            end = start + size if chunk != 9 else len(synthesis.poisoned)

            if end > len(synthesis.poisoned):
                end = len(synthesis.poisoned)

            subset = synthesis.poisoned[start:end]
            last: Annotator.AnnotatedInstance = subset[-1] # type: ignore

            trnf1s = last.train_f1
            tstf1s = last.test_f1
            trnacc = last.train_acc
            tstacc = last.test_acc

            df.loc[len(df)] = [end/len(synthesis.poisoned), trnf1s, trnacc, tstf1s, tstacc]
            progress.update(1)

        # Save the DataFrame into a csv file at the same directory of the
        # synthesis
        df.to_csv(f"{_get_subdir(synthesis)}/summary.csv", index=False)
        progress.close()

    @staticmethod
    def compute_weight_distribution(synthesis: RunSynthesis, dataset: "LoadedDataset"):
        wdist = np.zeros((dataset.ytr.size, len(synthesis.poisoned)))
        

@define()
class Pipeline:
    budget: float
    test_size: float
    random_state: int
    decorate_syntheses: bool
    summarize: bool
    verbose: bool
    methods: list[str]

    datasets: list["LoadedDataset"] = []

    def fit_best_params(self, dataset: "LoadedDataset", grid: dict[str, list[Any]]) -> dict[str, Any]:
        clf = RandomForestClassifier()
        grid_search = GridSearchCV(clf, grid, n_jobs=-1, verbose=self.verbose)
        grid_search.fit(dataset.xtr, dataset.ytr)
        return grid_search.best_params_
    
    def run(self):
        grid = get_param_grid()
        for dataset in self.datasets:
            params = self.fit_best_params(dataset, grid)
            for method in self.methods:
                bench = Bench(get_method(method), method)
                bench.setup(dataset.x, dataset.y, self.budget, self.test_size, is_early_stopping(method), dataset.name, 1, "forest", params)
                bench.run([self.random_state], verbose=self.verbose)
                bench.dump()
                bench.log()

                if self.decorate_syntheses:
                    syn = bench.syntheis[0]
                    Annotator.annotate(syn, dataset)

                    # Create a new dataframe to store and then save the annotated
                    # instances in the same location as the synthesis
                    annotated = DataFrame(columns=[*syn.poisoned[0].get_keys()])

                    # Populate the dataframe with the annotated instances
                    for instance in syn.poisoned:
                        instance.add_self_to_df(annotated)

                    # Save the annotated instances into a csv file
                    annotated.to_csv(f"{_get_subdir(syn)}/annotated.csv", index=False)

                    # Notice that the summarize method is called only
                    # after the annotation is done. If no annotation is
                    # performed, then the summarize method will not be called
                    # since it depends on the annotated instances
                    if self.summarize:
                        Annotator.summarize(syn)
                    
                bench.clean()

@define()
class LoadedDataset:
    name: DATASETS
    x: np.ndarray
    y: np.ndarray
    xtr: np.ndarray
    ytr: np.ndarray
    xv: np.ndarray
    yv: np.ndarray

    @staticmethod
    def append_to(id: DATASETS, classes: list[int], pipeline: Pipeline):
        x, y = load(id, *classes)
        xtr, xv, ytr, yv = split(x, y, test_size=pipeline.test_size, random_state=pipeline.random_state, stratified=True)
        pipeline.datasets.append(LoadedDataset(name=id, x=x, y=y, xtr=xtr, ytr=ytr, xv=xv, yv=yv))





def get_param_grid() -> dict[str, list]:
    with open(PIPELINE_CFG, "r") as f:
        total = json_load(f)
        params = total["forest_param_grid"]
    return params

def get_pipeline() -> Pipeline:
    with open(PIPELINE_CFG, "r") as f:
        config = json_load(f)
        pipeline = config["attack_pipeline"]
        datasets = config["datasets"]
    
    pip = Pipeline(**pipeline)
    for dataset in datasets:
        LoadedDataset.append_to(**dataset, pipeline=pip)
    return pip



def main():
    pip = get_pipeline()
    pip.run()

if __name__ == "__main__":
    main()
