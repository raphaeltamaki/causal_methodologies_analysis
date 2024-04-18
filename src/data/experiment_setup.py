from datetime import datetime
import polars as pl
import kaggle
import os
from pathlib import Path
from typing import Callable, Dict, List


class ExperimentSetup:
    def __init__(
            self,
            date_col: str,
            target_col: str,
            treatment_start_date: datetime,
            treatment_end_date: datetime,
            treatment_variable: str,
            treated_groups: List[str]) -> None:
        
        """
        TODO: document
        """
        self.date_col = date_col
        self.target_col = target_col
        self.treatment_start_date = treatment_start_date
        self.treatment_end_date = treatment_end_date
        self.treatment_variable = treatment_variable
        self.treated_groups = treated_groups

        # On hold paramets
        self.treated_units = None
        self.treatment_dates = None
        self.treatment_effect = None

    def _find_treatment_dates(self, data: pl.DataFrame) -> pl.DataFrame:
        return (data[self.date_col] >= self.treatment_start_date) & (data[self.date_col] <= self.treatment_end_date)

    def apply_treatment(self, data: pl.DataFrame, treatment_effect_method: Callable) -> pl.DataFrame:
        """
        Apply the treatment effect method based on the self.treatment_variable directly on target_col, if
        the row is within the period defined.
        """
        self.treatment_dates = self._find_treatment_dates(data)
        self.treated_units = (
            self.treatment_dates &
            (data[self.treatment_variable].is_in(self.treated_groups))
            )
        self.treatment_effect = data.select(pl.col("City").map_elements(treatment_effect_method))
        data = data.with_columns((pl.col(self.target_col) * (1 + pl.Series(name="t", values=self.treatment_effect))).alias(self.target_col))
        return data