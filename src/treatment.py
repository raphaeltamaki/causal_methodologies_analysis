import polars as pl
from typing import List
import numpy as np


class Treatment:
    def __init__(
        self,
        norm_expected_value: float,
        norm_noise: float,
        metric_col: str = "value",
        seed: int = None,
    ) -> None:
        """
        Class responsible to apply treatment to time-series data
        Inputs
            - norm_expected_value: normalized true expected value of treatment effect
            - norm_noise: standard deviation of white noise
        """
        self.norm_expected_value = norm_expected_value
        self.norm_noise = norm_noise
        self.rng = np.random.default_rng(seed)

        self.id_col = "id"
        self.value_col = "value"
        self.treatment_period_flag_col = "treatment_period"

    def _get_effect_samples(self, n: int):
        return self.rng.normal(self.norm_expected_value, self.norm_noise, size=n)

    def apply_treatment(
        self, preprocessed_data: pl.DataFrame, treated_groups: List[str]
    ) -> pl.DataFrame:
        """
        Apply treatment to treated groups
        Inputs
            - preprocessed_data: time series data to apply treatment. Expects it to have gone through pre-processing step
            - treated_groups: groups (based in the ID column) that received treatment
        """
        treated = pl.col(self.id_col).is_in(treated_groups) * pl.col(
            self.treatment_period_flag_col
        )
        row_number = preprocessed_data.select(pl.count()).item()
        treatment_effect = pl.Series(self._get_effect_samples(row_number))

        return preprocessed_data.with_columns(
            pl.col(self.value_col) * (pl.lit(1) + treated * treatment_effect)
        )
