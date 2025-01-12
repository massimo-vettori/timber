from sklearn.model_selection import train_test_split
from pandas import read_csv
from typing import Literal
from os.path import join
from os import getcwd
import numpy as np

DATASETS = Literal["digits", "winecolor", "breast_cancer", "spam", "chess", "ionosphere", "musk2", "sonar"]
RawDataset = tuple[np.ndarray, np.ndarray]


def get_data_path(name: str) -> str:
    return join(getcwd(), f"source/data/csv/{name}.csv")


def load_external(name: str) -> RawDataset:
    path = get_data_path(name)
    data = read_csv(path)
    x = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])
    return x, y


def load_sklearn(load_fn) -> RawDataset:
    raw = load_fn()
    x = raw.data  # type: ignore
    y = raw.target  # type: ignore
    return x, y


def load(name: DATASETS, positive_class: int = 1, negative_class: int = 0):
    if name in ["iris", "digits", "wine", "breast_cancer"]:
        match name:
            case "digits":
                from sklearn.datasets import load_digits

                raw = load_sklearn(load_digits)
            case "wine":
                from sklearn.datasets import load_wine

                raw = load_sklearn(load_wine)
            case "breast_cancer":
                from sklearn.datasets import load_breast_cancer

                raw = load_sklearn(load_breast_cancer)
            case _:
                raise ValueError("Invalid dataset name")

    else:
        raw = load_external(name)

    x, y = raw

    mask = np.isin(y, [positive_class, negative_class])
    x = x[mask]
    y = y[mask]

    y = np.where(y == positive_class, 1, 0)
    return x, y


def split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratified: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Centralized split function, in order to avoid code duplication and to
    # ensure that the same parameters are used in all the experiments.
    xtr, xts, ytr, yts = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratified else None,
    )

    return xtr, xts, ytr, yts


def train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int | None = None,
    stratified: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # xtr, xval, ytr, yval = split(x, y, test_size=VALIDATION_SPLIT, random_state=random_state, stratified=stratified)
    # return xtr, xval, ytr, yval
    return x.copy(), x.copy(), y.copy(), y.copy()
