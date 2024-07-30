from typing import Tuple
import pandas as pd
import polars as pl
import numpy as np

from src.models.preprocessing import DifferenceInDifferencesPreProcessing
from src.data.experiment_setup import ExperimentSetup
from src.data.data_formatter import BaseFormater

import arviz as az
import causalpy as cp


class DifferenceInDifferencesEstimator:
    """
    Implements DiD using CausalPy
    """

    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup):
        self.formatter = formatter
        self.experiment_setup = experiment_setup
        self.preprocessing = DifferenceInDifferencesPreProcessing(formatter, experiment_setup)
        self.post_treatment_col = "post_treatment"
        self.treated_location_col = "treated_location"
        self.formula = f"value ~ 1 + {self.preprocessing.default_date_col} +  {self.post_treatment_col} * {self.treated_location_col}"
        self.weighted_sum_fitter_kwargs = {"target_accept": 0.95, "random_seed": 42}
        self.result = None

    def fit(self, data: pl.DataFrame) -> None:
        # Transform Data and store the variables
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        self.result = cp.pymc_experiments.DifferenceInDifferences(
            pandas_data,
            formula=self.formula,
            time_variable_name=self.preprocessing.default_date_col,
            group_variable_name="treated_location",
            model=cp.pymc_models.LinearRegression(
                self.weighted_sum_fitter_kwargs
            ),
        )
        self.ate_samples = self.result.causal_impact * self.preprocessing.get_treated_stats()[1]

    def predict(self, data: pl.DataFrame) -> pd.Series:
        raise NotImplementedError

    def fit_predict(self, data: pl.DataFrame) -> pd.Series:
        self.fit(data)
        return self.predict(data)

    def estimate_ate(self, data: pl.DataFrame) -> float:
        return float(np.mean(self.ate_samples))
  
    def estimate_ate_distribution(self, data: pl.DataFrame) -> Tuple[float]:
        return float(np.std(self.ate_samples)), np.percentile(self.ate_samples, 5), np.percentile(self.ate_samples, 95)