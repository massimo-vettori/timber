from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from attrs import define
from pandas import DataFrame
from typing import Literal

TargetClass = Literal[1, 0, None]

@define(slots=True, frozen=True)
class PoisonedInstance:
    """
    A class containing useful information about a poisoned instance.

    Attributes
    ----------
    instanceid : int
        The id of the instance.
    label : int
        The label of the instance.
    trainloss : int
        The number of additional errors on the training set prediction, after the label flip.
    testloss : int
        The number of additional errors on the test set prediction, after the label flip.
    depth : int
        The depth of the subtree that was modified after the label flip.
    nsamples : int
        The number of samples in the attacked subtree.
    """


    instanceid: int
    label: int
    trainloss: int
    testloss: int
    depth: int
    nsamples: int
    runtime: int

    def to_list(self) -> list:
        """
        Convert the instance to a list.

        Returns
        -------
        list
            The instance as a list.
        """
        return [v for v in self.to_dict().values()]
                
    def to_dict(self) -> dict:
        """
        Convert the instance to a dictionary.

        Returns
        -------
        dict
            The instance as a dictionary.
        """
        return {
            "instanceid": self.instanceid,
            "label": self.label,
            "trainloss": self.trainloss,
            "testloss": self.testloss,
            "depth": self.depth,
            "nsamples": self.nsamples,
            "runtime": self.runtime
        }
    
    def get_keys(self) -> list:
        """
        Get the keys of the dictionary in sorted order.

        Returns
        -------
        list
            The keys of the dictionary in sorted order.
        """
        return [k for k in self.to_dict().keys()]

    def add_self_to_df(self, df: DataFrame) -> None:
        """
        Add the instance to a DataFrame.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to which the instance will be added.
        """
        df.loc[len(df)] = self.to_list()

Outcome = tuple[list[PoisonedInstance], int, int]


class Poisoning(Protocol):
    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        budget: int,
        tree_params: dict,
        target: TargetClass,
        early_stop: bool = False,
        verbose: bool = False,
    ) -> Outcome: ...
