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
        self.true_ate = None
        self.reference_value = None

    def _find_treatment_dates(self, data: pl.DataFrame) -> pl.DataFrame:
        return (data[self.date_col] >= self.treatment_start_date) & (data[self.date_col] <= self.treatment_end_date)
    
    def _get_treatment_effect(self, data: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def store_true_ate(self, treated_data: pl.DataFrame) -> None:
        """
        Calculates and stores the (true) Average Treatment Effect
        """
        self.reference_value = (
            treated_data
            .filter(self.treated_units)
            .group_by([self.date_col])
            .agg(pl.col(self.target_col).sum())
            .select(pl.col(self.target_col).mean())
            .item() 
        )
        self.true_ate = self.reference_value * self.lift_size / (1+self.lift_size)

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
        # Stores the true ATE
        self.store_true_ate(output_data)
        return output_data

    def get_true_ate(self):
        """
        Returns the (true) ATE if treatment was applied
        """
        return self.true_ate

    def get_reference_value(self):
        """
        Returns the reference value. That is, the average target balue if there was no treatment applied
        """
        return self.reference_value

        

class ConstantLiftExperiment(ExperimentSetup):
    def _get_treatment_effect(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select(pl.when(pl.col(self.treatment_variable).is_in(self.treated_groups)).then(self.lift_size).otherwise(0.0))
