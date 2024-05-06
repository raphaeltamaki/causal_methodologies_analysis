from datetime import datetime
import polars as pl
import kaggle
import os
from pathlib import Path
from typing import Callable, Dict, List
from src.data.data_formatter import BaseFormater


class ExperimentSetup:
    def __init__(
            self,
            data_formatter: BaseFormater,
            treatment_start_date: datetime,
            treatment_end_date: datetime,
            lift_size: float,
            treated_groups: List[str],
            treatment_date_format: str="%Y-%m-%d") -> None:
        
        """
        TODO: document
        """
        self.data_formatter = data_formatter
        self.date_col = data_formatter.date_col
        self.target_col = data_formatter.target_col
        self.lift_size = lift_size
        self.treatment_variable = data_formatter.treatment_col

        # cast dates to datetime if they are stings
        self.treatment_start_date = treatment_start_date if isinstance(treatment_start_date, datetime) else datetime.strptime(treatment_start_date, treatment_date_format)
        self.treatment_end_date = treatment_end_date if isinstance(treatment_end_date, datetime) else datetime.strptime(treatment_end_date, treatment_date_format)
        self.treated_groups = treated_groups

        # On hold paramets
        self.treated_units = None
        self.treatment_dates = None
        self.treatment_effect = None

    def _find_treatment_dates(self, data: pl.DataFrame) -> pl.DataFrame:
        return (data[self.date_col] >= self.treatment_start_date) & (data[self.date_col] <= self.treatment_end_date)
    
    def _get_treatment_effect(self, data: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def apply_treatment(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Apply the treatment effect method based on the self.treatment_variable directly on target_col, if
        the row is within the period defined.
        """

        # find which rows refer to treatment period, and which rows (i.e. units) got the treatment
        self.treatment_dates = self._find_treatment_dates(data)
        self.treated_units = (
            self.treatment_dates &
            (data[self.treatment_variable].is_in(self.treated_groups))
            )
        
        # apply the treatment effect
        self.treatment_effect = self._get_treatment_effect(data) * self.treated_units
        output_data = data.with_columns((pl.col(self.target_col) * (1 + pl.Series(name="t", values=self.treatment_effect))).alias(self.target_col))
        return output_data
    

class ConstantLiftExperiment(ExperimentSetup):
    def _get_treatment_effect(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select(pl.col(self.treatment_variable).map_elements(lambda x: self.lift_size if x in self.treated_groups else 0.0))
