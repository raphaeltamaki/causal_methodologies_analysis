from src.models.preprocessing import DoWhyPreProcessing
from src.data.experiment_setup import ExperimentSetup
from src.data.data_formatter import BaseFormater
from dowhy import CausalModel
from typing import Tuple
import pandas as pd
import polars as pl

class DoWhyEstimator():
    def __init__(self, formatter: BaseFormater, experiment_setup: ExperimentSetup):
        self.formatter = formatter
        self.experiment_setup = experiment_setup
        self.preprocessing = DoWhyPreProcessing(formatter, experiment_setup)
        self.model = None
        self.estimate = None

    def fit(self, data: pl.DataFrame, target_units: str='att') -> None:
        pandas_data = self.preprocessing.fit_transform(data).to_pandas()
        self.model= CausalModel(
            data=pandas_data,
            treatment=[self.preprocessing.T_variable],
            outcome=self.preprocessing.y_variable,
            common_causes=[self.preprocessing.default_id_col],
            effect_modifiers=self.preprocessing.X_variables)
        
        identified_estimand = self.model.identify_effect(proceed_when_unidentifiable=False)
        self.estimate = self.model.estimate_effect(
            identified_estimand,
            confidence_intervals=True,
            method_name="backdoor.propensity_score_matching",
            target_units=target_units)

    def predict(self, data: pl.DataFrame) -> pd.Series:
        raise NotImplementedError

    def fit_predict(self, data: pl.DataFrame) -> pd.Series:
        self.fit(data)
        return self.predict(data)

    def estimate_ate(self, data: pl.DataFrame) -> float:
        self.fit(data, target_units='ate')
        return self.estimate.value * self.preprocessing.get_treated_stats()[1]

    def estimate_ate_distribution(self, data: pl.DataFrame, ci: float=0.9) -> Tuple[float]:
        self.fit(data, target_units='ate')
        std = self.preprocessing.get_treated_stats()[1]
        return self.estimate.get_standard_error() * std, self.estimate.get_confidence_intervals(ci)[0] * std, self.estimate.get_confidence_intervals(ci)[1] * std
    
    def estimate_att(self, data: pl.DataFrame) -> float:
        self.fit(data, target_units='att')
        return self.estimate.value * self.preprocessing.get_treated_stats()[1]

    def estimate_att_distribution(self, data: pl.DataFrame, ci: float=0.9) -> Tuple[float]:
        self.fit(data, target_units='att')
        std = self.preprocessing.get_treated_stats()[1]
        return self.estimate.get_standard_error() * std, self.estimate.get_confidence_intervals(ci)[0] * std, self.estimate.get_confidence_intervals(ci)[1] * std
