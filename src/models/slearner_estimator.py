from typing import Tuple
import polars as pl
import pandas as pd
import numpy as np

from causalml.inference.meta import BaseTRegressor
from causalml.inference.meta import XGBTRegressor

from src.models.preprocessing import MetaLearnerPreProcessing
from src.data.experiment_setup import ExperimentSetup
from src.data.data_formatter import BaseFormater


class SLearner:

    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup, learner: BaseTRegressor=None, bootstrap_samples: int=100):
        self.learner = XGBTRegressor(ate_alpha=0.10) if learner is None else learner
        self.preprocessing = MetaLearnerPreProcessing(formatter, experiment_setup)
        self.bootstrap_samples = bootstrap_samples
        self.T = ''
        self.y = ''
        self.X = []

    def _store_variables(self, pandas_data: pd.DataFrame) -> None:
        self.T = self.preprocessing.T_variable
        self.y = self.preprocessing.y_variable
        self.X = pandas_data.columns.drop([self.preprocessing.default_date_col, self.T, self.y])

    def fit(self, data: pl.DataFrame) -> None:
        # Process the data
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        # Store variables
        self._store_variables(pandas_data)
        # Train T-Learner
        self.learner.fit(X=pandas_data[self.X], treatment=pandas_data[self.T], y=pandas_data[self.y])


    def predict(self, data: pl.DataFrame) -> pd.Series:
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        return self.learner.predict(X=pandas_data[self.X], treatment=pandas_data[self.T])

    def fit_predict(self, data: pl.DataFrame) -> pd.Series:
        self.fit(data)
        return self.predict(data)

    def estimate_ate(self, data: pl.DataFrame) -> float:
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        self.fit(data)
        avg, _, _ = self.learner.estimate_ate(
            X=pandas_data[self.X],
            treatment=pandas_data[self.T],
            y=pandas_data[self.y],
            bootstrap_ci=True,
            n_bootstraps=self.bootstrap_samples,
            bootstrap_size=pandas_data.shape[0],
            pretrain=True)
        std = self.preprocessing.get_treated_stats()[1]
        return float(avg * std)

    def estimate_ate_distribution(self, data: pd.DataFrame) -> Tuple[float]:
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        self.fit(data)
        _, lower_bound, upper_bound = self.learner.estimate_ate(X=pandas_data[self.X], treatment=pandas_data[self.T], y=pandas_data[self.y],
            bootstrap_ci=True,
            n_bootstraps=self.bootstrap_samples,
            bootstrap_size=pandas_data.shape[0], pretrain=True)
        std = self.preprocessing.get_treated_stats()[1]
        lower_bound, upper_bound = float(lower_bound * std), float(upper_bound * std)
        return (upper_bound - lower_bound)/1.645, lower_bound, upper_bound