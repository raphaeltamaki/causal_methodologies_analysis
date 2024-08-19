from typing import Tuple
import pandas as pd
import polars as pl
import numpy as np

from src.models.preprocessing import SyntheticControlPreProcessing, SyntheticControlLinRegPreProcessing
from src.data.experiment_setup import ExperimentSetup
from src.data.data_formatter import BaseFormater

import arviz as az
import causalpy as cp


class SyntheticControlEstimator:
    """
    Implements Synthetic Control using CausalPy
    """

    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup):
        self.formatter = formatter
        self.experiment_setup = experiment_setup
        self.preprocessing = SyntheticControlPreProcessing(formatter, experiment_setup)
        self.formula = ''
        self.columns_to_ignore = [self.preprocessing.default_date_col, self.preprocessing.treated_units_name]
        self.weighted_sum_fitter_kwargs = {"target_accept": 0.90, "random_seed": 42, "chains": 2}
        self.result = None
        self.pymc_model = cp.pymc_models.WeightedSumFitter

    def fit(self, data: pl.DataFrame) -> None:
        # Transform Data and store the variables
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        self.formula = f"target ~ 0 + {' + '.join([col for col in pandas_data.columns if col not in self.columns_to_ignore])}"
        pandas_data = (
            pandas_data
            # .assign(Time=lambda x: pd.to_datetime(x[self.preprocessing.default_date_col]))
            .set_index(self.preprocessing.default_date_col)
            )

        self.result = cp.pymc_experiments.SyntheticControl(
            pandas_data,
            self.experiment_setup.treatment_start_date,
            formula=self.formula,
            model=self.pymc_model(
                sample_kwargs=self.weighted_sum_fitter_kwargs
            ),
        )
        self.ate_samples = np.mean(self.result.post_impact, axis=2)

    def predict(self, data: pl.DataFrame) -> pd.Series:
        raise NotImplementedError

    def fit_predict(self, data: pl.DataFrame) -> pd.Series:
        self.fit(data)
        return self.predict(data)
    
    def estimate_ate(self, data: pl.DataFrame) -> float:
        raise NotImplementedError
    
    def estimate_ate_distribution(self, data: pl.DataFrame) -> float:
        raise NotImplementedError

    def estimate_att(self, data: pl.DataFrame) -> float:
        avg = self.preprocessing.get_treated_stats()[0]
        return float(self.ate_samples.mean()) * avg
  
    def estimate_att_distribution(self, data: pl.DataFrame) -> Tuple[float]:
        avg = self.preprocessing.get_treated_stats()[0]
        return float(np.std(self.ate_samples)) * avg, np.percentile(self.ate_samples, 5)* avg, np.percentile(self.ate_samples, 95)* avg

class SyntheticControlLinRegEstimator(SyntheticControlEstimator):
    """
    Implements Synthetic Control using CausalPy. But uses a linear regression model instead of the WeightedSumFitter
    which forces parameters to be between 0 and 1 and the sum of parameters to be 1, uses the LinearRegression model
    that has no restrictions to the parameters
    """
    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup):
        super().__init__(formatter, experiment_setup)
        self.pymc_model = cp.pymc_models.LinearRegression
        self.preprocessing = SyntheticControlLinRegPreProcessing(formatter, experiment_setup)

    def estimate_att(self, data: pl.DataFrame) -> float:
        std = self.preprocessing.get_treated_stats()[1]
        return float(self.ate_samples.mean()) * std
  
    def estimate_att_distribution(self, data: pl.DataFrame) -> Tuple[float]:
        std = self.preprocessing.get_treated_stats()[1]
        return float(np.std(self.ate_samples)) * std, np.percentile(self.ate_samples, 5)* std, np.percentile(self.ate_samples, 95)* std
