from .types import Poisoning, TargetClass
from ..data import DATASETS
from .synthesis import RunSynthesis
from ..logger import LOGGER_NAME
from .stats import Statistics, MODEL_TYPE
from logging import getLogger
from os.path import join
from ..data import split
from os import makedirs
from typing import Any
import numpy as np
from json import dumps

from pandas import DataFrame

LOGGER = getLogger(LOGGER_NAME)

class Bench:
    def __init__(self, poisoning: Poisoning, name: str):
        self.runs: list[Statistics] = []
        self.syntheis: list[RunSynthesis] = []
        self.poisoning = poisoning
        self.ready = False
        self.name = name

    def setup(
        self,
        x: np.ndarray,
        y: np.ndarray,
        budget: float,
        test_size: float,
        early_stopping: bool,
        dataset: DATASETS,
        target: TargetClass,
        model_type: MODEL_TYPE,
        model_params: dict[str, Any],
    ):
        self.x = x
        self.y = y
        self.budget = budget
        self.test_size = test_size
        self.early_stopping = early_stopping
        self.model_params = model_params
        self.target = target
        self.dataset = dataset
        self.ready = True
        self.model_type = model_type

        LOGGER.info(f"Set up bench for {self.name}")

    def clean(self):
        self.ready = False
        self.runs = []

    def run(self, seeds: list[int] | np.ndarray, verbose: bool = False):
        if not self.ready:
            LOGGER.error("Bench not set up")
            return

        budget = int(self.budget * ((1 - self.test_size) * self.y.size))

        for seed in seeds:
            LOGGER.debug("-" * 50)
            LOGGER.debug(f"Running {self.name} with seed {seed}")
            LOGGER.debug("-" * 50)

            xtr, xte, ytr, yte = split(self.x, self.y, test_size=self.test_size, random_state=seed)
            model_params = self.model_params
            model_params["random_state"] = int(seed) if seed is not None else None

            selected, timens, ges = self.poisoning(
                xtr, ytr, budget, model_params, self.target, early_stop=self.early_stopping, verbose=verbose
            )

            instanceids = [s.instanceid for s in selected]

            stats = Statistics.generate(
                xtr, ytr, xte, yte, self.budget, instanceids, timens, ges, model_params, self.model_type  # type: ignore
            )
            synthesis = RunSynthesis(
                self.name,
                seed,
                self.test_size,
                self.budget,
                selected,
                timens,
                model_params,
                self.model_type,
                self.dataset,
                self.target,
            )
            self.runs.append(stats)
            self.syntheis.append(synthesis)

    def dump(self):
        from datetime import datetime
        from os.path import join

        if not self.ready or len(self.runs) == 0:
            LOGGER.error("Bench not set up or no runs")
            return

        columns = list(self.runs[0].to_dict().keys())
        df = DataFrame(columns=columns)

        for run in self.runs:
            run.dump(df)

        for syn in self.syntheis:
            syn.dump()
            # syn.plot()


    def log(self):
        # Prints average and std of each metric of the statistics
        if not self.ready or len(self.runs) == 0:
            LOGGER.error("Bench not set up or no runs")
            return

        train_size = (1 - self.test_size) * self.y.size
        full_budget = self.budget * train_size

        LOGGER.info("")
        LOGGER.info("----------------------------------------")
        LOGGER.info(f"Logging results for {self.name}")
        LOGGER.info("----------------------------------------")
        LOGGER.info(f"Number of runs:     {len(self.runs)}")
        LOGGER.info(f"Budget:             {self.budget:.2%}")
        LOGGER.info(f"Early stopping:     {self.early_stopping}")
        LOGGER.info(f"Tree params:        {self.model_params}")
        LOGGER.info("----------------------------------------")

        train_scores = [run.train_score for run in self.runs]
        test_scores = [run.test_score for run in self.runs]
        poisoned_train_scores = [run.train_loss for run in self.runs]
        poisoned_test_scores = [run.test_loss for run in self.runs]
        runtimes = [run.runtime for run in self.runs]
        used_budgets = [(run.poisoned / full_budget) for run in self.runs]

        LOGGER.info(f"Train score:      {np.mean(train_scores):.3%} ~ {np.std(train_scores):.3%}")
        LOGGER.info(f"Test score:       {np.mean(test_scores):.3%} ~ {np.std(test_scores):.3%}")
        LOGGER.info(f"Train loss:       {np.mean(poisoned_train_scores):.3%} ~ {np.std(poisoned_train_scores):.3%}")
        LOGGER.info(f"Test loss:        {np.mean(poisoned_test_scores):.3%} ~ {np.std(poisoned_test_scores):.3%}")
        LOGGER.info(f"Runtime:          {np.mean(runtimes) / 1_000_000:.2f}ms ~ {np.std(runtimes) / 1_000_000:.2f}ms")
        LOGGER.info(f"Used budget:      {np.mean(used_budgets):.3%} ~ {np.std(used_budgets):.3%}")
        LOGGER.info("----------------------------------------")
        LOGGER.info("")
