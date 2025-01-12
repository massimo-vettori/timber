from ._forest import RandomForestClassifier, _generate_sample_indices, check_random_state
from sklearn.tree._stable_path import DecodedStablePath
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.ttree import TTreeClassifier
from ._base import _set_random_states
import numpy as np


class StablePathBundle:
    @staticmethod
    def sort(paths: list["StablePathBundle"]) -> list["StablePathBundle"]:
        return sorted(paths, key=lambda x: x.weight)

    def __init__(self, bundle: tuple[DecodedStablePath | None]) -> None:
        self.bundle = tuple(bundle)
        self._target = None

    def __len__(self) -> int:
        return len(self.bundle)

    def __iter__(self):
        return iter(self.bundle)

    def __getitem__(self, index: int) -> DecodedStablePath | None:
        return self.bundle[index]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        string = "StablePathBundle(\n"
        for path in self.bundle:
            string += f"\t{path}\n"
        string += ")"
        return string

    @property
    def weight(self) -> int:
        w = 0
        for path in self.bundle:
            if path is not None:
                w += path.samples
        return w

    @property
    def label(self) -> int:
        for path in self.bundle:
            if path is not None:
                return path.label
        return -1

    def get_target(self, forest: "TForestClassifier"):
        if self._target is not None:
            return self._target

        samples = forest.estimators_samples_
        for i, path in enumerate(self.bundle):
            if path is not None:
                target = path.target
                self._target = target
                return samples[i][target]

    def _soft_majority_vote(
        self,
        forest: "TForestClassifier",
        xtr: np.ndarray,
        ytr: np.ndarray,
        xv: np.ndarray,
        yv: np.ndarray,
    ):
        preds = np.zeros((yv.size, forest.n_classes_))  # type: ignore
        for i in range(len(self.bundle)):
            estimator = forest.estimators[i]
            preds += estimator.compute_poisoned_proba(self.bundle[i], xtr, ytr, xv, yv)

        preds = np.argmax(preds, axis=1)
        return preds

    def compute_error(
        self,
        forest: "TForestClassifier",
        xtr: np.ndarray,
        ytr: np.ndarray,
        xv: np.ndarray,
        yv: np.ndarray,
    ):
        preds = self._soft_majority_vote(forest, xtr, ytr, xv, yv)
        return np.sum(preds != yv)


class TForestClassifier(RandomForestClassifier):
    compute_stable_paths: bool = False

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,  # type: ignore
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=check_random_state(random_state).randint(np.iinfo(np.int32).max),
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )

    def _make_estimator(self, append=True, random_state=None):
        # Cast random seed to int64
        tree = TTreeClassifier(
            criterion="entropy",
            splitter="best",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,  # type: ignore
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            ccp_alpha=self.ccp_alpha,
            class_weight=self.class_weight,
        )
        tree.set_compute_stable_paths(True)
        return tree

    @property
    def estimators(self) -> list[TTreeClassifier]:
        return self.estimators_ if hasattr(self, "estimators_") else []

    def get_stable_path_bundles(self, ytrue: np.ndarray) -> list[StablePathBundle]:
        bundles = np.empty((self.n_estimators, ytrue.size), dtype=object)
        # Fill the bundles vector with None
        for i in range(self.n_estimators):
            bundles[i] = [None] * ytrue.size

        samples = self.estimators_samples_
        for i, estimator in enumerate(self.estimators_):
            indices = np.unique(samples[i])
            ysubset = ytrue[indices]
            paths = estimator.decode_paths(ysubset)
            for j, path in enumerate(paths):
                bundles[i][indices[j]] = path

        # Each bundle is represented by a column
        bundles = bundles.T
        bundles = [StablePathBundle(bundle) for bundle in bundles]
        return bundles

    def compute_error(self, xtr: np.ndarray, ytrue: np.ndarray) -> int:
        preds = self.predict(xtr)
        return np.sum(preds != ytrue)
    
    def set_compute_stable_paths(self, compute: bool):
        for estimator in self.estimators:
            estimator.set_compute_stable_paths(compute)
        self.compute_stable_paths = compute
