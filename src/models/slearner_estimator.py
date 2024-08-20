from typing import Tuple
from joblib import Parallel, delayed # for parallel processing
import polars as pl
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.data.experiment_setup import ExperimentSetup
from src.data.data_formatter import BaseFormater
from src.models.preprocessing import MetaLearnerPreProcessing, GranularMetaLearnerPreProcessing
from src.models.MLPreprocessing import RandomForestPreprocessing


class SLearner:

    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup, learner: RandomForestRegressor=None, bootstrap_samples: int=100, n_jobs: int=4, random_state: int=None):
        self.learner = RandomForestRegressor(random_state=random_state) if learner is None else learner
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
        self.learner.fit(X=pandas_data[self.X + [self.T]], y=pandas_data[self.y])

    def fit(self, data: pl.DataFrame) -> None:
        # Process the data
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        # Store variables
        self._store_variables(pandas_data)
        # Train S-Learner
        self._train_learners(pandas_data)

    def _predict_learners(self, pandas_data: pd.DataFrame) -> pd.Series:
        X = pandas_data[self.X + [self.T]]
        return (
            self.learner.predict(X.assign(**{self.T: 1})) - # predict under treatment
            self.learner.predict(X.assign(**{self.T: 0}))   # predict under control
            )

    def predict(self, data: pl.DataFrame) -> pd.Series:
        pandas_data = self.preprocessing.transform(data).to_pandas()
        return self._predict_learners(pandas_data)

    def fit_predict(self, data: pl.DataFrame) -> pd.Series:
        self.fit(data)
        return self.predict(data)

    def estimate_ate(self, data: pl.DataFrame, filter_treated: bool=False, do_bootstrap: bool=False) -> float:
        avg = self.predict(data).mean()
        std = self.preprocessing.get_treated_stats()[1]
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
        ates = Parallel(n_jobs=self.n_jobs)(delayed(self._bootstrap_att)(pandas_data)
                            for _ in range(self.bootstrap_samples))
        return np.std(ates), np.percentile(ates, 5), np.percentile(ates, 95)
    

class GranularSLearner(SLearner):

    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup, learner: RandomForestRegressor=None, bootstrap_samples: int=100, n_jobs: int=4, random_state: int=None, sample_size: int=None):
        super().__init__(formatter, experiment_setup ,learner, bootstrap_samples, n_jobs, random_state)
        self.learner = HistGradientBoostingRegressor(random_state=random_state) if learner is None else learner
        self.preprocessing = GranularMetaLearnerPreProcessing(formatter, experiment_setup)
        self.ml_prepocessing = RandomForestPreprocessing()
        self.sample_size = 5000 if sample_size is None else sample_size

    def _store_variables(self, pandas_data: pd.DataFrame) -> None:
        self.T = self.preprocessing.T_variable
        self.y = self.preprocessing.y_variable
        self.X = list(pandas_data.columns.drop([self.preprocessing.default_date_col, self.T, self.y]))
        self.X = [col for col in self.X if col not in ['user_id', 'id']]

    def _train_learners(self, pandas_data: pd.DataFrame) -> None:
        X = pandas_data[self.X + [self.T]].copy()
        X = self.ml_prepocessing.fit_transform(X)
        self.learner.fit(X=X, y=pandas_data[self.y])

    def _predict_learners(self, pandas_data: pd.DataFrame) -> pd.Series:
        X = pandas_data[self.X + [self.T]]
        X = self.ml_prepocessing.transform(X)
        pandas_data['prediction'] = (
            self.learner.predict(X.assign(**{self.T: 1})) - # predict under treatment
            self.learner.predict(X.assign(**{self.T: 0}))   # predict under control
            )
        agg_predictions = (
            pandas_data
            .groupby([self.preprocessing.default_id_col, self.preprocessing.default_date_col])
            ['prediction'].sum()
            .reset_index()
            )
        return agg_predictions['prediction'].to_numpy()

    def _bootstrap_ate(self, pandas_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
    
    def _bootstrap_att(self, pandas_data: pd.DataFrame) -> pd.DataFrame:
        idxs = np.random.choice(np.arange(0, pandas_data.shape[0]), size=self.sample_size)
        self._train_learners(pandas_data.iloc[idxs])
        # Predict on the original dataset, based on the model trained on sampled data
        pandas_treated_data = pandas_data.query(f"{self.T}==1")
        avg = self._predict_learners(pandas_treated_data).mean()
        std = self.preprocessing.get_treated_stats()[1]
        return float(avg * std)