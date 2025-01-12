from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.tforest import TForestClassifier
from ..data import train_validation_split
from dataclasses import dataclass
from ..logger import LOGGER_NAME
from logging import getLogger
from typing import Any
import numpy as np

from pandas import DataFrame
from typing import Literal
from os import makedirs


LOGGER = getLogger(LOGGER_NAME)
MODEL_TYPE = Literal["forest", "tree"]


@dataclass
class Statistics:
    train_score: float
    test_score: float
    validation_score: float
    train_loss: float
    test_loss: float
    validation_loss: float
    train_eff: float
    test_eff: float

    poisoned: float
    runtime: float
    budget: float
    ges: float
    model: MODEL_TYPE

    @staticmethod
    def generate(
        x: np.ndarray,
        y: np.ndarray,
        xt: np.ndarray,
        yt: np.ndarray,
        budget: float,
        selected: list[int],
        runtime: float,
        ges: float,
        tree_params: dict,
        model: MODEL_TYPE,
    ):
        xp, xv, yp, yv = train_validation_split(x, y, random_state=tree_params["random_state"])
        clf = DecisionTreeClassifier(**tree_params) if model == "tree" else TForestClassifier(**tree_params)
        clf.fit(xp, yp)

        trs = float(clf.score(xp, yp))
        vds = float(clf.score(xv, yv))
        tes = float(clf.score(xt, yt))

        for iid in selected:
            yp[iid] = 1 - yp[iid]

        clf = DecisionTreeClassifier(**tree_params) if model == "tree" else TForestClassifier(**tree_params)
        clf.fit(xp, yp)

        trl = float(trs - clf.score(x, y))
        vdl = float(vds - clf.score(xv, yv))
        tel = float(tes - clf.score(xt, yt))
        used = len(selected) / yp.size

        treff = trl / used
        teeff = tel / used
        vdeff = vdl / used

        return Statistics(
            trs,
            tes,
            vds,
            trl,
            tel,
            vdl,
            treff,
            teeff,
            len(selected),
            runtime,
            budget,
            ges,
            model,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_score": self.train_score,
            "test_score": self.test_score,
            "validation_score": self.validation_score,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "validation_loss": self.validation_loss,
            "train_eff": self.train_eff,
            "test_eff": self.test_eff,
            "runtime": self.runtime,
            "budget": self.budget,
            "poisoned": self.poisoned,
            "ges": self.ges,
        }

    def dump(self, df: DataFrame):
        json = self.to_dict()
        cols = list(json.keys())
        df.loc[df.size] = [json[col] for col in cols]
