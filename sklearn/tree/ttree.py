from ._classes import DecisionTreeClassifier, _fit_context
from ._classes import DecodedStablePath
import numpy as np
from logging import getLogger

UNCACHED_SUBSET = (None, None, None)
LOGGER = getLogger("poisoning")


class TTreeClassifier(DecisionTreeClassifier):
    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            monotonic_cst=monotonic_cst,
            ccp_alpha=ccp_alpha,
        )

        self._compute_stable_paths = False
        self._decision_paths = None
        self._subset_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]] = []
        self._validation_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]] = []
        self._fitted = False

    def get_subset(
        self,
        node_id: int,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self._subset_cache) == 0:
            self._subset_cache = [UNCACHED_SUBSET for _ in range(self.tree_.node_count)]

        cache = self._subset_cache[node_id]
        if cache[0] is not None and cache[1] is not None and cache[2] is not None:
            # Return the cached subset (copying the y in order to avoid race conditions)
            return cache[0], cache[1], cache[2].copy()

        mask = self.get_subtree_sample_mask(node_id, x)
        samples = np.where(mask)[0]
        xsub = x[mask]
        ysub = y[mask]

        self._subset_cache[node_id] = (samples, xsub, ysub)
        # Return the newly cached subset (copying the y in order to avoid race conditions)
        return samples, xsub, ysub.copy()

    def get_validation_set(
        self,
        node_id: int,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self._validation_cache) == 0:
            self._validation_cache = [UNCACHED_SUBSET for _ in range(self.tree_.node_count)]

        cache = self._validation_cache[node_id]
        if cache[0] is not None and cache[1] is not None and cache[2] is not None:
            return cache

        subset = self.decision_path(x.astype(np.float32)).toarray()
        mask = subset[:, node_id].astype(bool)
        samples = np.where(mask == True)[0]
        xval = x[mask]
        yval = y[mask]

        self._validation_cache[node_id] = (samples, xval, yval)
        return samples, xval, yval

    def set_compute_stable_paths(self, value):
        self._compute_stable_paths = value

    def decode_paths(self, ytrue: np.ndarray) -> list[DecodedStablePath]:
        """
        Decodes the vulnerable paths from the splitter object and returns them as a str list.
        Each path is composed of either `L` or `R` characters, representing a left or right step down the path
        If a path terminates with `*`, then it is **vulnerable**, thus the last touched node
        will change when the instance is flipped
        """

        def translate_path(long: np.uint64, label: int, target: int, graph) -> DecodedStablePath:
            path = ""
            node = 0
            depth = 0

            while long != 0:
                step = long & 0b11

                if step == 0b10:
                    next = graph.children_left[node]
                    node = next if next != -1 else node
                    depth += 1
                    path += "L"

                elif step == 0b01:
                    next = graph.children_right[node]
                    node = next if next != -1 else node
                    depth += 1
                    path += "R"

                elif step == 0b11:
                    path += "*"
                    break

                long >>= 2  # type: ignore

            vulnerable = path.endswith("*")
            max_depth = graph.max_depth if graph.max_depth else np.iinfo(np.int32).max
            samples = graph.n_node_samples[node]

            if not vulnerable:
                value = graph.value[node][0]
                # If we are not at the max_depth and the leaf is pure, then the path is vulnerable if the label is equal to the majority class
                if depth < max_depth and np.argmax(value) == label:
                    vulnerable = True
                    path += "*"

                # If we reached the max_depth, then we need to check if the target label class is equal in size to the opposite class or equal minus 1
                # to the majority class
                elif depth == max_depth:
                    samples = graph.n_node_samples[node]
                    values = (value.copy().flatten() * samples).astype(int)
                    if values[label] == (samples // 2) or values[label] == (samples // 2) - 1:
                        vulnerable = True
                        path += "*"

            return DecodedStablePath(path.endswith("*"), node, int(samples), target, depth, label, path)

        if not self._compute_stable_paths:
            raise ValueError("LVI was not computed during training")
        if not self._splitter_obj:
            raise ValueError("Splitter object is not available")

        raw = self._splitter_obj.paths  # type: ignore
        graph = self.tree_

        return [translate_path(long, label, i, graph) for i, (long, label) in enumerate(zip(raw, ytrue))]

    def get_subtree_sample_mask(self, node_id: int, x: np.ndarray) -> np.ndarray:
        if self._decision_paths is None:
            self._decision_paths = self.decision_path(x.astype(np.float32)).toarray()

        return self._decision_paths[:, node_id].astype(bool)

    def compute_poisoned_proba(
        self,
        path: DecodedStablePath | None,
        xtr: np.ndarray,
        ytr: np.ndarray,
        xv: np.ndarray,
        yv: np.ndarray,
    ):
        proba = self.predict_proba(xv)
        if path is None:
            return proba

        samples, xsub, ysub = self.get_subset(path.node_id, xtr, ytr)
        validation, xval, _ = self.get_validation_set(path.node_id, xv, yv)

        if validation.size == 0:
            return proba

        if not path.target in samples:
            return proba

        target = samples.tolist().index(path.target)
        params = path.compute_subtree_params(self.get_params())

        if params["max_depth"] < 1:
            return proba

        ysub[target] = 1 - ysub[target]
        clf = DecisionTreeClassifier(**params)
        clf.fit(xsub, ysub)
        ysub[target] = 1 - ysub[target]

        subproba = clf.predict_proba(xval)
        proba[validation] = subproba
        return proba

    def compute_error(
        self,
        path: DecodedStablePath | None,
        xtr: np.ndarray,
        ypo: np.ndarray,
        xv: np.ndarray,
        yv: np.ndarray,
    ):
        preds = self.compute_poisoned_proba(path, xtr, ypo, xv, yv)
        return np.sum(np.argmax(preds, axis=1) != yv)
