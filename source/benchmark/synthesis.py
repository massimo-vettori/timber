from .types import Poisoning, Outcome, PoisonedInstance, TargetClass
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ..data import load, split, DATASETS
from dataclasses import dataclass
from ..logger import LOGGER_NAME
from .stats import Statistics
from logging import getLogger
from datetime import datetime
from attrs import define
from typing import Any
import numpy as np
from pandas import DataFrame, read_csv
from os.path import join
from os import makedirs
from json import dumps, loads

SYNTHESIS_PATH = ".synthesis/"

makedirs(SYNTHESIS_PATH, exist_ok=True)

def change_synthesis_path(path: str):
    global SYNTHESIS_PATH
    SYNTHESIS_PATH = path
    makedirs(SYNTHESIS_PATH, exist_ok=True)


def _get_train_nsamples(dsname: DATASETS, test_size: float):
    _, y = load(dsname)
    return int(y.size * (1 - test_size))


def _get_subdir(synth: "RunSynthesis"):
    subdirpath = join(SYNTHESIS_PATH, synth.name)
    return subdirpath


@define(slots=True)
class RunSynthesis:
    algorithm: str
    split_seed: int
    test_size: float
    budget: float
    poisoned: list[PoisonedInstance]
    runtime: int
    modelparams: dict[str, Any]
    modeltype: str
    dataset: DATASETS
    target: TargetClass
    subdirname: str = ""
    generate_subdir: bool = True

    # Create the directory on post init
    def __attrs_post_init__(self):
        if self.generate_subdir:
            getLogger(LOGGER_NAME).info(f"Creating directory for {self.name}")
            makedirs(_get_subdir(self), exist_ok=True)

    @property
    def name(self) -> str:
        if self.subdirname == "":
            timestamp = datetime.now().strftime("%d%H%M%S")
            self.subdirname = f"{self.algorithm}_{timestamp}_class{self.target}_{self.dataset}_{self.split_seed}_{int(self.test_size * 100)}_{int(self.budget * 100)}"
        return self.subdirname

    def dump(self):
        subdir = _get_subdir(self)
        filename = "raw.csv"
        filename = join(subdir, filename)

        df = DataFrame([p.to_list() for p in self.poisoned], columns=self.poisoned[0].get_keys())
        config = {
            "algorithm": self.algorithm,
            "split_seed": int(self.split_seed) if self.split_seed is not None else "NO SEED",
            "test_size": float(self.test_size),
            "budget": float(self.budget),
            "runtime": int(self.runtime),
            "modelparams": self.modelparams,
            "dataset": self.dataset,
            "target": int(self.target) if self.target is not None else "NO TARGET",
            "modeltype": self.modeltype,
        }

        with open(join(subdir, "config.json"), "w") as f:
            f.write(dumps(config, indent=3, sort_keys=True))

        df.to_csv(filename, index=False)

    @staticmethod
    def partial_load(name: str) -> "RunSynthesis":
        subdir = name
        configfile = join(subdir, "config.json")
        rawfile = join(subdir, "raw.csv")

        with open(configfile, "r") as f:
            config = loads(f.read())

        with open(rawfile, "r") as f:
            raw = read_csv(f)

        if config["split_seed"] == "NO SEED":
            config["split_seed"] = None

        x, y = load(config["dataset"])
        _, _, ytr, _ = train_test_split(x, y, test_size=config["test_size"], random_state=config["split_seed"])

        iids = raw["instanceid"]
        poisoned = [
            PoisonedInstance(iid, ytr[iid], 0, 0, 0, 0, 0)
            for iid in iids
        ]

        # Remove the SYNTHESIS_PATH from the name
        synname = name.split("/")[1:]
        synname = "/".join(synname)
        
        return RunSynthesis(
            config["algorithm"] if "algorithm" in config else "UNKNOWN",
            config["split_seed"],
            config["test_size"],
            config["budget"],
            poisoned,
            config["runtime"],
            config["modelparams"],
            config["modeltype"],
            config["dataset"],
            config["target"],
            subdirname=synname,
            generate_subdir=False,
        )
        
    def plot(self):
        from matplotlib import pyplot as plt

        ftrn, axtrn = plt.subplots(1, 1, figsize=(12, 10))  # Train cumulative loss
        ftes, axtes = plt.subplots(1, 1, figsize=(12, 10))  # Test cumulative loss
        fdep, axdep = plt.subplots(1, 1, figsize=(12, 10))  # Depth distribution
        fnsa, axnsa = plt.subplots(1, 1, figsize=(12, 10))  # Number of samples distribution
        frtm, axrtm = plt.subplots(1, 1, figsize=(12, 10))  # Cumulative runtime

        trainnsamples = _get_train_nsamples(self.dataset, self.test_size)
        nsampledist = np.array([p.nsamples for p in self.poisoned])
        ctrainloss = np.cumsum([p.trainloss for p in self.poisoned]) / trainnsamples
        ctestloss = np.cumsum([p.testloss for p in self.poisoned]) / trainnsamples
        depthdist = np.array([p.depth for p in self.poisoned])
        runtime = np.cumsum([p.runtime for p in self.poisoned]) / 1e6

        xaxis = np.linspace(0, 1, depthdist.size)
        axtrn.plot(xaxis, ctrainloss, label="Train Loss", linewidth=3)
        axtrn.fill_between(xaxis, ctrainloss, alpha=0.1)
        axtrn.set_title("Train Loss")
        axtrn.set_xlabel("Used budget (%)")
        axtrn.set_ylabel("Cumulative Loss")
        axtrn.axhline(self.budget, color="black", linestyle="--", label="Fixed Budget (%)")
        axtrn.legend()
        axtrn.grid()

        axtes.plot(xaxis, ctestloss, label="Test Loss", linewidth=3)
        axtes.fill_between(xaxis, ctestloss, alpha=0.1)
        axtes.set_title("Test Loss")
        axtes.set_xlabel("Used budget (%)")
        axtes.set_ylabel("Cumulative Loss")
        axtes.axhline(self.budget, color="black", linestyle="--", label="Fixed Budget (%)")
        axtes.legend()
        axtes.grid()

        axdep.bar(depthdist, np.arange(depthdist.size))
        axdep.set_title("Depth Distribution")
        axdep.set_xlabel("Depth")
        axdep.set_ylabel("Frequency")
        axdep.grid()

        axnsa.bar(nsampledist, np.arange(nsampledist.size))
        axnsa.set_title("Number of Samples Distribution")
        axnsa.set_xlabel("Number of Samples")
        axnsa.set_ylabel("Frequency")
        axnsa.grid()

        axrtm.plot(xaxis, runtime, label="Runtime (ms)", linewidth=3)
        axrtm.fill_between(xaxis, runtime, alpha=0.1)
        axrtm.set_title("Cumulative Runtime (ms)")
        axrtm.set_xlabel("Used budget (%)")
        axrtm.set_ylabel("Cumulative Runtime (ms)")
        axrtm.legend()
        axrtm.grid()

        ftrn.savefig(join(_get_subdir(self), "train_loss.png"))
        ftes.savefig(join(_get_subdir(self), "test_loss.png"))
        fdep.savefig(join(_get_subdir(self), "depth_distribution.png"))
        fnsa.savefig(join(_get_subdir(self), "nsample_distribution.png"))
        frtm.savefig(join(_get_subdir(self), "runtime.png"))

        plt.close(ftrn)
        plt.close(ftes)
        plt.close(fdep)
        plt.close(fnsa)
        plt.close(frtm)

    # def plot(self):
    #     from matplotlib import pyplot as plt
    #     figname = self.name + ".png"
    #     figname = join(SYNTHESIS_PLOT_PATH, figname)

    #     trainnsamples = _get_train_nsamples(self.dataset, self.test_size)

    #     nsampledist = np.array([p.nsamples for p in self.poisoned])
    #     ctrainloss  = np.cumsum([p.trainloss for p in self.poisoned]) / trainnsamples
    #     ctestloss   = np.cumsum([p.testloss for p in self.poisoned]) / trainnsamples
    #     depthdist   = np.array([p.depth for p in self.poisoned])

    #     fig, ax = plt.subplots(2, 2, figsize=(24, 22))
    #     trax = ax[0, 0]
    #     teax = ax[0, 1]
    #     nsax = ax[1, 0]
    #     dpax = ax[1, 1]

    #     xaxis = np.linspace(0, 100, depthdist.size)

    #     trax.plot(xaxis, ctrainloss, label="Train Loss")
    #     trax.fill_between(xaxis, ctrainloss, alpha=0.1)
    #     trax.set_title("Train Loss")
    #     trax.set_xlabel("Used budget (%)")
    #     trax.set_ylabel("Cumulative Loss")
    #     trax.axvline(self.budget, color="black", linestyle="--", label="Fixed Budget (%)")
    #     trax.legend()
    #     trax.grid()

    #     teax.plot(xaxis, ctestloss, label="Test Loss")
    #     teax.fill_between(xaxis, ctestloss, alpha=0.1)
    #     teax.set_title("Test Loss")
    #     teax.set_xlabel("Used budget (%)")
    #     teax.set_ylabel("Cumulative Loss")
    #     teax.axvline(self.budget, color="black", linestyle="--", label="Fixed Budget (%)")
    #     teax.legend()
    #     teax.grid()

    #     dpax.bar(np.arange(depthdist.size), depthdist)
    #     dpax.set_title("Depth Distribution")
    #     dpax.set_xlabel("Poisoned Instance")
    #     dpax.set_ylabel("Depth")
    #     dpax.grid()

    #     nsax.bar(np.arange(nsampledist.size), nsampledist)
    #     nsax.set_title("Number of Samples Distribution")
    #     nsax.set_xlabel("Poisoned Instance")
    #     nsax.set_ylabel("Number of Samples")
    #     nsax.grid()

    #     plt.savefig(figname)
    #     plt.close(fig)


# @dataclass
# class RunSynthesis:
#     split_seed: int
#     test_size: float
#     budget: float
#     poisoned_samples: list[int]
#     runtime: int
#     tree_params: dict[str, Any]
#     dataset: DATASETS

#     @staticmethod
#     def generate(
#         split_seed: int,
#         test_size: float,
#         budget: float,
#         poisoned_samples: list[int],
#         runtime: int,
#         tree_params: dict[str, Any],
#         dataset: DATASETS,
#     ):
#         return RunSynthesis(
#             int(split_seed),
#             float(test_size),
#             float(budget),
#             [int(i) for i in poisoned_samples],
#             int(runtime),
#             tree_params,
#             dataset
#         )

#     def to_dict(self) -> dict[str, Any]:
#         return {
#             "split_seed": self.split_seed,
#             "test_size": self.test_size,
#             "budget": self.budget,
#             "poisoned_samples": self.poisoned_samples,
#             "runtime": self.runtime,
#             "tree_params": self.tree_params,
#             "dataset": self.dataset,
#         }

#     @staticmethod
#     def from_dict(data: dict[str, Any]) -> "RunSynthesis":
#         return RunSynthesis(
#             data["split_seed"],
#             data["test_size"],
#             data["budget"],
#             data["poisoned_samples"],
#             data["runtime"],
#             data["tree_params"],
#             data["dataset"],
#         )

#     def replicate(self) -> Statistics:
#         x, y = load(self.dataset)
#         xtr, xte, ytr, yte = split(x, y, test_size=self.test_size, random_state=self.split_seed)

#         return Statistics.generate(
#             xtr,
#             ytr,
#             xte,
#             yte,
#             self.budget,
#             self.poisoned_samples,
#             self.runtime,
#             0.0,
#             self.tree_params
#         )
