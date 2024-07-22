from typing import Tuple
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

class DoublyRobustEstimator:
    """
    Implements Doubly Robust Estimator
    Reference for the code: https://matheusfacure.github.io/python-causality-handbook/12-Doubly-Robust-Estimation.html
    """

    def __init__(self, mu_reg_control, mu_reg_treat, treatment_clasifier, X, y, T):
        self.mu_reg_control = mu_reg_control
        self.mu_reg_treat = mu_reg_treat
        self.treatment_clasifier = treatment_clasifier
        self.T = T
        self.y = y
        self.X = X

    def fit(self, data: pd.DataFrame) -> None:
        self.treatment_clasifier.fit(data[self.X], data[self.T])
        self.mu_reg_control.fit(data.query(f"{self.T}==0")[self.X], data.query(f"{self.T}==0")[self.y])
        self.mu_reg_treat.fit(data.query(f"{self.T}==1")[self.X], data.query(f"{self.T}==1")[self.y])

    def predict(self, data: pd.DataFrame) -> pd.Series:
        ps = self.treatment_clasifier.predict_proba(data[self.X])[:, 1]
        mu0 = self.mu_reg_control.predict(data[self.X])
        mu1 = self.mu_reg_treat.predict(data[self.X])
        return (
            np.mean(data[self.T]*(data[self.y] - mu1)/ps + mu1) -
            np.mean((1-data[self.T])*(data[self.y] - mu0)/(1-ps) + mu0)
        )

    def fit_predict(self, data: pd.DataFrame) -> pd.Series:
        self.fit(data)
        return self.predict(data)

    def estimate_ate(self, data: pd.DataFrame) -> float:
        return self.predict(data).mean()
  
    def estimate_ate_distribution(self, data: pd.DataFrame, bootstrap_samples: int=1000, n_jobs: int=4) -> Tuple[float]:
        ates = Parallel(n_jobs=n_jobs)(delayed(self.fit_predict)(data.sample(frac=1, replace=True))
                            for _ in range(bootstrap_samples))
        return np.array(ates)