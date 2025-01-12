from sklearn.neighbors import KNeighborsClassifier
from .base import Model, fit_model
from numpy import ndarray, array, where

def knn_based_defence(
    x: ndarray, y: ndarray, 
    model_params: dict, 
    /,
    nneighbors: int,
    rounds: int,
    nu: float,
    **dparams
) -> Model:
    xcp, ycp = x.copy(), y.copy()
    for _ in range(rounds):
        knn = KNeighborsClassifier(n_neighbors=nneighbors)
        knn.fit(xcp, ycp)

        preds = array(knn.predict_proba(xcp))
        ycp = where(preds.max(axis=1) >= nu, preds.argmax(axis=1), ycp)

    return fit_model(x, ycp, model_params)
