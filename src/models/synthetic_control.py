from typing import AnyStr, Any, Dict, List, Tuple
import pandas as pd
import polars as pl
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
import numpy as np
import stan

class SyntheticControlRegressor():

    def __init__(
            self,
            setup,
            model,
            data: pl.DataFrame,
            confidence_interval: float=0.89,
            random_seed: int=42,
            mapie_n_resamplings: int=10,
            mapie_n_blocks: int=10,
            ):
        self.setup = setup
        self.data = data
        self.feature_cols = [col for col in data.columns if (col not in ["date"] + self.setup.experiment_setup.treated_groups)]
        self.time_col = "date"
        self.target_col = "target"
        self.model = model
        self.confidence_interval = confidence_interval
        

        self.cv_mapiets = BlockBootstrap(
            n_resamplings=mapie_n_resamplings, n_blocks=mapie_n_blocks, overlapping=False, random_state=random_seed
        )
        self.mapie_enbpi = MapieTimeSeriesRegressor(
            self.model, method="enbpi", cv=self.cv_mapiets, agg_function="mean", n_jobs=-1
        )
        self.X, self.y = None, None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

    
    def _extract_train_test_data(self):
        data_filter = pl.col(self.time_col) < pl.lit(self.setup.experiment_setup.treatment_start_date)
        train_data = self.data.filter(data_filter)
        test_data = self.data.filter(~data_filter)

        X_train, y_train = train_data.select(self.feature_cols), train_data.select(self.target_col)
        X_test, y_test = test_data.select(self.feature_cols), test_data.select(self.target_col)


        X, y = self.data.select(self.feature_cols).to_pandas().fillna(0), self.data.select(self.target_col).to_pandas().fillna(0)
        X_train, y_train = X_train.to_pandas().fillna(0), y_train.to_pandas().fillna(0)
        X_test, y_test = X_test.to_pandas().fillna(0), y_test.to_pandas().fillna(0)
        return X, y, X_train, y_train, X_test, y_test

    def fit(self) -> None:
        self.X, self.y, self.X_train, self.y_train, self.X_test, self.y_test = self._extract_train_test_data()
        self.model.fit(self.X_train, self.y_train)
        self.mapie_enbpi = self.mapie_enbpi.fit(self.X_train, self.y_train)

    def predict(self) -> Tuple[np.ndarray, List]:
        y_pred, y_intervals = self.mapie_enbpi.predict(self.X, alpha=1-self.confidence_interval)
        return y_pred, y_intervals[:, :, 0]

    def fit_predict(self):
        self.fit()
        return self.predict()
    



class BayesianSyntheticControlRegressor():

    def __init__(
            self,
            setup,
            data: pl.DataFrame,
            confidence_interval: float=0.89,
            chains: int=3,
            n_samples: int=1000
            ):
        self.setup = setup
        self.data = data
        self.feature_cols = [col for col in data.columns if (col not in ["date"] + self.setup.experiment_setup.treated_groups)]
        self.time_col = "date"
        self.target_col = "target"
        self.model = self._get_stan_model()
        self.confidence_interval = confidence_interval
        self.chains = chains
        self.n_samples = n_samples
    
        self.X, self.y = None, None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        self.posterior = None
        self.samples = None

    @staticmethod
    def _get_stan_model():
        return """
        data {
            int<lower=1> N; // number of observations
            int<lower=1> cols; // number of features
            vector[N] y; // target variable
            matrix[N, cols] X; // design matrix
        }
        parameters {
            vector[cols] coefs; // coefficients
            real<lower=0.0001> sigma;
            real bias;
        }
        model {
            sigma ~ cauchy(0, 2);
            treatment_coef ~ normal(0, 2);
            coefs ~ double_exponential(0, 2);
            y ~ normal(X * coefs + bias, sigma);
        }
        """

    def _transform2stan_data(self, data: pd.DataFrame) -> Dict[AnyStr, Any]:
        X = np.matrix(data[self.feature_cols])
        y = np.array(data[self.target_col])
        return {"X": X, "y": y, "N": X.shape[0], "cols": X.shape[1]}
    
    def _get_posteriors(self, stan_data: Dict[AnyStr, Any]):
        return stan.build(self.model, data=stan_data)
    
    def _estimate_effect(self, post_period_data: pd.DataFrame):
        return None


    def fit(self) -> None:
        data = self.data.to_pandas().fillna(0)
        stan_data = self._transform2stan_data(data)
        self.posterior = self._get_posteriors(stan_data)

    def predict(self):
        return self.posterior.sample(num_chains=self.chains, num_samples=self.n_samples)

    def fit_predict(self):
        self.fit()
        return self.predict()