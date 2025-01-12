from typing import Protocol
from sklearn.ensemble.tforest import TForestClassifier
from numpy import ndarray

Model = TForestClassifier

def fit_model(
    x: ndarray, y: ndarray, 
    model_params: dict
) -> Model:
    model = Model(**model_params)
    model.fit(x, y)
    return model


class Defence(Protocol):
    def __call__(
        self, 
        x: ndarray, y: ndarray, 
        model_params: dict, 
        /,
        **dparams
    ): ...

def no_defence(
    x: ndarray, y: ndarray, 
    model_params: dict, 
    /,
    **dparams
) -> Model:
    return fit_model(x, y, model_params)
