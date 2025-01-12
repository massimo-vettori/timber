from .base import Model, fit_model
from numpy import ndarray, ceil, array, nonzero, argmax, bincount

class BaggingHashedClassifier:
    estimators: list[Model]
    hashedsets: list[tuple[ndarray, ndarray]]
    npartition: int
    baseparams: dict
    psize: int

    def __init__(
        self,
        npartition: int,
        psize: int,
        baseparams: dict,
    ):
        self.npartition = npartition
        self.psize = psize
        self.baseparams = baseparams
        self.estimators = []
        self.hashedsets = []

    def _create_hashed_sets(self, x: ndarray, y: ndarray) -> None:
        self.hashedsets = []

        overlap = int(ceil(self.npartition / int(1 / self.psize))) - 1
        prehash = x.copy()

        final = []

        for time in range(overlap + 1):
            partitions = int(1 / self.psize)
            hasheddata = [hash(str(data) + str(time)) % partitions for data in prehash]

            if time != overlap:
                indices = [
                    nonzero((hasheddata == array(i)))[0]
                    for i in range(partitions)
                ]
            
            else:
                indices = [
                    nonzero((hasheddata == array(i)))[0]
                    for i in range(self.npartition - overlap * partitions)
                ]

            final += indices

        for hashedset in final:
            xhash, yhash = x[hashedset], y[hashedset]
            self.hashedsets.append((xhash, yhash))

    def fit(self, x: ndarray, y: ndarray) -> None:
        self._create_hashed_sets(x, y)
        for xh, yh in self.hashedsets:
            if xh.shape[0] == 0:
                continue
            est = fit_model(xh, yh, self.baseparams)
            self.estimators.append(est)

    def predict(self, x: ndarray) -> ndarray:
        preds = array([est.predict(x) for est in self.estimators])
        return array([
            argmax(bincount(preds[:, i])) for i in range(x.shape[0])
        ])



def bagging_defence(
    x: ndarray, y: ndarray,
    model_params: dict,
    /,
    npartition: int,
    psize: int
) -> BaggingHashedClassifier:
    bagger = BaggingHashedClassifier(npartition, psize, model_params)
    bagger.fit(x, y)
    return bagger