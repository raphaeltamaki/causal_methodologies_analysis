from typing import Tuple
from joblib import Parallel, delayed
import pandas as pd
import polars as pl
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.models.preprocessing import DoublyRobustPreProcessing
from src.data.experiment_setup import ExperimentSetup
from src.data.data_formatter import BaseFormater

class DoublyRobustEstimator:
    """
    Implements Doubly Robust Estimator
    Reference for the code: https://matheusfacure.github.io/python-causality-handbook/12-Doubly-Robust-Estimation.html
    """

    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup, mu_reg_control: BaseEstimator=None, mu_reg_treat: BaseEstimator=None, treatment_clasifier: BaseEstimator=None,
                 bootstrap_samples: int=500, n_jobs: int=4):
        self.formatter = formatter
        self.experiment_setup = experiment_setup
        self.preprocessing = DoublyRobustPreProcessing(formatter, experiment_setup)
        

        self.mu_reg_control = LinearRegression() if mu_reg_control is None else mu_reg_control
        self.mu_reg_treat = LinearRegression() if mu_reg_treat is None else mu_reg_treat
        self.treatment_clasifier = LogisticRegression() if treatment_clasifier is None else treatment_clasifier
        self.bootstrap_samples = bootstrap_samples
        self.n_jobs = n_jobs

    def fit(self, data: pl.DataFrame) -> None:
        # Transform Data and store the variables
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        self.T = self.preprocessing.T_variable
        self.y = self.preprocessing.y_variable
        self.X = self.preprocessing.X_variables

        self.treatment_clasifier.fit(pandas_data[self.X], pandas_data[self.T])
        self.mu_reg_control.fit(pandas_data.query(f"{self.T}==0")[self.X], pandas_data.query(f"{self.T}==0")[self.y])
        self.mu_reg_treat.fit(pandas_data.query(f"{self.T}==1")[self.X], pandas_data.query(f"{self.T}==1")[self.y])

    def predict(self, data: pl.DataFrame, filter_treated: bool=False) -> pd.Series:
        pandas_data = self.preprocessing.transform(data).to_pandas()
        if filter_treated:
            pandas_data = pandas_data.query(f"{self.T}==1")
        ps = self.treatment_clasifier.predict_proba(pandas_data[self.X])[:, 1]
        mu0 = self.mu_reg_control.predict(pandas_data[self.X])
        mu1 = self.mu_reg_treat.predict(pandas_data[self.X])
        return (
            (pandas_data[self.T]*(pandas_data[self.y] - mu1)/ps + mu1) -
            ((1-pandas_data[self.T])*(pandas_data[self.y] - mu0)/(1-ps) + mu0)
        )

    def fit_predict(self, data: pl.DataFrame) -> pd.Series:
        self.fit(data)
        return self.predict(data)

    def estimate_ate(self, data: pl.DataFrame) -> float:
        std = self.preprocessing.get_treated_stats()[1]
        return self.predict(data).mean() * std
  
    def estimate_ate_distribution(self, data: pl.DataFrame) -> Tuple[float]:
        ates = Parallel(n_jobs=self.n_jobs)(delayed(self.estimate_ate)(data.sample(fraction=1, with_replacement=True))
                            for _ in range(self.bootstrap_samples))
        ates = np.array(ates)
        return np.std(ates), np.percentile(ates, 5), np.percentile(ates, 95)

    def estimate_att(self, data: pl.DataFrame) -> float:
        std = self.preprocessing.get_treated_stats()[1]
        return self.predict(data, filter_treated=True).mean() * std
    
    def estimate_att_distribution(self, data: pl.DataFrame) -> Tuple[float]:
        atts = Parallel(n_jobs=self.n_jobs)(delayed(self.estimate_att)(data.sample(fraction=1, with_replacement=True))
                            for _ in range(self.bootstrap_samples))
        atts = np.array(atts)
        return np.std(atts), np.percentile(atts, 5), np.percentile(atts, 95)