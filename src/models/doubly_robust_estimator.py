from typing import Tuple
from joblib import Parallel, delayed
import pandas as pd
import polars as pl
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


from src.models.preprocessing import DoublyRobustPreProcessing, GranularMetaLearnerPreProcessing
from src.models.MLPreprocessing import RandomForestPreprocessing
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


    def _store_variables(self, pandas_data: pd.DataFrame) -> None:
        self.T = self.preprocessing.T_variable
        self.y = self.preprocessing.y_variable
        self.X = list(pandas_data.columns.drop([self.preprocessing.default_date_col, self.T, self.y]))
    
    def _train_learners(self, pandas_data: pd.DataFrame) -> None:
        self.treatment_clasifier.fit(pandas_data[self.X], pandas_data[self.T])
        self.mu_reg_control.fit(pandas_data.query(f"{self.T}==0")[self.X], pandas_data.query(f"{self.T}==0")[self.y])
        self.mu_reg_treat.fit(pandas_data.query(f"{self.T}==1")[self.X], pandas_data.query(f"{self.T}==1")[self.y])

    def fit(self, data: pl.DataFrame) -> None:
        # Transform Data and store the variables
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        # Store variables
        self._store_variables(pandas_data)
        # Train learners
        self._train_learners(pandas_data)


    def _predict_learners(self, pandas_data: pd.DataFrame) -> pd.Series:
        ps = self.treatment_clasifier.predict_proba(pandas_data[self.X])[:, 1]
        mu0 = self.mu_reg_control.predict(pandas_data[self.X])
        mu1 = self.mu_reg_treat.predict(pandas_data[self.X])
        return (
            (pandas_data[self.T]*(pandas_data[self.y] - mu1)/ps + mu1) -
            ((1-pandas_data[self.T])*(pandas_data[self.y] - mu0)/(1-ps) + mu0)
        )
        
    def predict(self, data: pl.DataFrame) -> pd.Series:
        pandas_data = self.preprocessing.transform(data).to_pandas()
        return self._predict_learners(pandas_data)

    def fit_predict(self, data: pl.DataFrame) -> pd.Series:
        self.fit(data)
        return self.predict(data)

    def estimate_ate(self, data: pl.DataFrame) -> float:
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
  
    def estimate_ate_distribution(self, data: pl.DataFrame) -> Tuple[float]:
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
    

class GranularDoublyRobustEstimator(DoublyRobustEstimator):
    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup, mu_reg_control: BaseEstimator=None, mu_reg_treat: BaseEstimator=None, treatment_clasifier: BaseEstimator=None,
                 bootstrap_samples: int=500, n_jobs: int=4, sample_size: int=5000):
        super().__init__(formatter, experiment_setup, mu_reg_control, mu_reg_treat, treatment_clasifier, bootstrap_samples, n_jobs)
        
        self.preprocessing = GranularMetaLearnerPreProcessing(formatter, experiment_setup)
        self.ml_prepocessing = RandomForestPreprocessing()
        self.mu_reg_control = HistGradientBoostingRegressor(min_samples_leaf=40) if mu_reg_control is None else mu_reg_control
        self.mu_reg_treat = HistGradientBoostingRegressor(min_samples_leaf=40) if mu_reg_treat is None else mu_reg_treat
        self.treatment_clasifier = HistGradientBoostingClassifier(l2_regularization=0.5, min_samples_leaf=40) if treatment_clasifier is None else treatment_clasifier
        self.sample_size = sample_size

    def _store_variables(self, pandas_data: pd.DataFrame) -> None:
        self.T = self.preprocessing.T_variable
        self.y = self.preprocessing.y_variable
        self.X = list(pandas_data.columns.drop([self.preprocessing.default_date_col, self.T, self.y]))
        self.X = [col for col in self.X if col not in ['user_id', 'id']]

    def _train_learners(self, pandas_data: pd.DataFrame) -> None:
        treated_units = pandas_data[self.T] == 1
        X = pandas_data[self.X].copy()
        X = self.ml_prepocessing.fit_transform(X)
        self.treatment_clasifier.fit(X, pandas_data[self.T])
        self.mu_reg_treat.fit(X=X[treated_units], y=pandas_data[treated_units][self.y])
        self.mu_reg_control.fit(X=X[~treated_units], y=pandas_data[~treated_units][self.y])

    def _predict_learners(self, pandas_data: pd.DataFrame) -> np.ndarray:
        X = pandas_data[self.X]
        X = self.ml_prepocessing.transform(X)

        ps = self.treatment_clasifier.predict_proba(X)[:, 1]
        mu0 = self.mu_reg_control.predict(X)
        mu1 = self.mu_reg_treat.predict(X)

        pandas_data['prediction']  = (
            (pandas_data[self.T]*(pandas_data[self.y] - mu1)/ps + mu1) -
            ((1-pandas_data[self.T])*(pandas_data[self.y] - mu0)/(1-ps) + mu0)
        )
        agg_predictions = (
            pandas_data
            .groupby([self.preprocessing.default_id_col, self.preprocessing.default_date_col])
            ['prediction'].sum()
            .reset_index()
            )
        return agg_predictions['prediction'].to_numpy()

    
    def _bootstrap_ate(self, pandas_data: pd.DataFrame) -> pd.DataFrame:
        idxs = np.random.choice(np.arange(0, pandas_data.shape[0]), size=self.sample_size)
        self._train_learners(pandas_data.iloc[idxs])
        # Predict on the original dataset, based on the model trained on sampled data
        avg = self._predict_learners(pandas_data).mean()
        std = self.preprocessing.get_treated_stats()[1]
        return float(avg * std)
    
    def _bootstrap_att(self, pandas_data: pd.DataFrame) -> float:
        idxs = np.random.choice(np.arange(0, pandas_data.shape[0]), size=self.sample_size)
        self._train_learners(pandas_data.iloc[idxs])
        # Predict on the original dataset, based on the model trained on sampled data
        pandas_treated_data = pandas_data.query(f"{self.T}==1")
        avg = self._predict_learners(pandas_treated_data).mean()
        std = self.preprocessing.get_treated_stats()[1]
        return float(avg * std)
    

    