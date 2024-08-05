from typing import Tuple
from joblib import Parallel, delayed # for parallel processing
import polars as pl
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from src.models.preprocessing import MetaLearnerPreProcessing
from src.data.experiment_setup import ExperimentSetup
from src.data.data_formatter import BaseFormater


class TLearner:

    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup, bootstrap_samples: int=100, n_jobs: int=4):
        self.learner_control = RandomForestRegressor()
        self.learner_treat = RandomForestRegressor() 
        self.preprocessing = MetaLearnerPreProcessing(formatter, experiment_setup)
        self.bootstrap_samples = bootstrap_samples
        self.n_jobs = n_jobs
        self.T = ''
        self.y = ''
        self.X = []

    def _store_variables(self, pandas_data: pd.DataFrame) -> None:
        self.T = self.preprocessing.T_variable
        self.y = self.preprocessing.y_variable
        self.X = list(pandas_data.columns.drop([self.preprocessing.default_date_col, self.T, self.y]))


    def _train_learners(self, pandas_data: pd.DataFrame) -> None:
        treated_units = pandas_data[self.T] == 1
        self.learner_treat.fit(X=pandas_data[treated_units][self.X], y=pandas_data[treated_units][self.y])
        self.learner_control.fit(X=pandas_data[~treated_units][self.X], y=pandas_data[~treated_units][self.y])

    def fit(self, data: pl.DataFrame) -> None:
        # Process the data
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        # Store variables
        self._store_variables(pandas_data)
        # Train T-Learner
        self._train_learners(pandas_data)

    def _predict_learners(self, pandas_data: pd.DataFrame) -> pd.Series:
        return self.learner_treat.predict(X=pandas_data[self.X])  - self.learner_control.predict(X=pandas_data[self.X])

    def predict(self, data: pl.DataFrame) -> pd.Series:
        pandas_data = self.preprocessing.transform(data).to_pandas()
        return self._predict_learners(pandas_data)

    def fit_predict(self, data: pl.DataFrame) -> pd.Series:
        self.fit(data)
        return self.predict(data)

    def estimate_ate(self, data: pl.DataFrame) -> float:
        std = self.preprocessing.get_treated_stats()[1]
        avg = self.predict(data).mean()
        return float(avg * std)

    def _bootstrap_ate(self, pandas_data: pd.DataFrame) -> pd.DataFrame:
        idxs = np.random.choice(np.arange(0, pandas_data.shape[0]), size=pandas_data.shape[0])
        self._train_learners(pandas_data.iloc[idxs])
        # Predict on the original dataset, based on the model trained on sampled data
        avg = self._predict_learners(pandas_data).mean()
        std = self.preprocessing.get_treated_stats()[1]
        return float(avg * std)

    def estimate_ate_distribution(self, data: pd.DataFrame) -> Tuple[float]:
        pandas_data = self.preprocessing.transform(data).to_pandas()
        # Store variables
        self._store_variables(pandas_data)
        # Train T-Learner
        ates = Parallel(n_jobs=self.n_jobs)(delayed(self._bootstrap_ate)(pandas_data)
                            for _ in range(self.bootstrap_samples))
        return np.std(ates), np.percentile(ates, 5), np.percentile(ates, 95)


    def estimate_att(self, data: pl.DataFrame) -> float:
        pandas_data = self.preprocessing.transform(data).to_pandas()
        # get only data about Treated unites 
        pandas_data = pandas_data.query(f"{self.T}==1")
        std = self.preprocessing.get_treated_stats()[1]
        avg = self._predict_learners(pandas_data).mean()
        return float(avg * std)

    def _bootstrap_att(self, pandas_data: pd.DataFrame) -> pd.DataFrame:
        idxs = np.random.choice(np.arange(0, pandas_data.shape[0]), size=pandas_data.shape[0])
        self._train_learners(pandas_data.iloc[idxs])
        # Predict on the original dataset, based on the model trained on sampled data
        pandas_treated_data = pandas_data.query(f"{self.T}==1")
        avg = self._predict_learners(pandas_treated_data).mean()
        std = self.preprocessing.get_treated_stats()[1]
        return float(avg * std)

    def estimate_att_distribution(self, data: pd.DataFrame) -> Tuple[float]:
        pandas_data = self.preprocessing.transform(data).to_pandas()
        
        # Store variables
        self._store_variables(pandas_data)
        # Train T-Learner
        atts = Parallel(n_jobs=self.n_jobs)(delayed(self._bootstrap_att)(pandas_data)
                            for _ in range(self.bootstrap_samples))
        return np.std(atts), np.percentile(atts, 5), np.percentile(atts, 95)