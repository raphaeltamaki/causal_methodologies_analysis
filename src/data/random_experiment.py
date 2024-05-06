import numpy as np
import pandas as pd
import polars as pl
import random
from src.data.data_formatter import (
    SupermarketSalesFormatter,
    IowaLicorSalesFormatter,
    WallmartSalesFormatter,
    SuperstoreSalesFormatter,
    LifetimeValueFormatter,
    EarthTemperatureFormatter,
    Nifty50StockMarketFormatter,
    CoronaFormatter,
    )
from src.data.experiment_setup import ConstantLiftExperiment


class RandomConstantLiftExperiment():

    def __init__(
            self,
            dataset_name: str,
            raw_dataset: pl.DataFrame,
            eligible_treatment_percentage: float=None,
            seed: int=None) -> None:
        
        self.dataset_name = dataset_name
        self.raw_dataset = raw_dataset
        self.eligible_treatment_percentage = 0.5 if eligible_treatment_percentage is None else eligible_treatment_percentage
        # define the seed for the pseudo-random
        random.seed(seed)
        

        self.dataset_formatters = {
            'supermarket_sales': SupermarketSalesFormatter(),
            'iowa_licor_sales': IowaLicorSalesFormatter(),
            'wallmart_sales': WallmartSalesFormatter(),
            'superstore_sales': SuperstoreSalesFormatter(),
            'lifetime_value': LifetimeValueFormatter(),
            'climate_change_earth_temperature': EarthTemperatureFormatter(),
            'nifty50_stock_market': Nifty50StockMarketFormatter(),
            'corona': CoronaFormatter(),
            }
        
        # Resulting parameters
        self.data_formatter = self._get_data_formatter()
        self.dataset = self.data_formatter.fit_transform(raw_dataset)
        self.time_start, self.time_end = self._extract_time_limits()
        self.time_range = pd.date_range(self.time_start, self.time_end, freq=self.data_formatter.date_discretization)
        self.treatment_options = self._extract_possible_treatment_options()

        self.experiment_setup = None
        
    
    def _get_data_formatter(self):
        return self.dataset_formatters[self.dataset_name]

    def _extract_possible_treatment_options(self):
        return list(self.dataset[self.data_formatter.treatment_col].unique())

    def _extract_time_limits(self):
        return self.dataset[self.data_formatter.date_col].min(), self.dataset[self.data_formatter.date_col].max()

    def _get_treatment_group(self):
        eligible_groups_n = int(len(self.treatment_options) * self.eligible_treatment_percentage)
        groups_n = random.randint(1, eligible_groups_n)
        return random.sample(self.treatment_options, groups_n)

    def _get_treatment_time(self):
        return self.time_range[random.randint(1, len(self.time_range) - 1)]

    @staticmethod
    def _get_lift() -> float:
        return random.random()

    def get_experiment(self) -> ConstantLiftExperiment:

        treament_group = self._get_treatment_group()
        treament_time = self._get_treatment_time()
        lift_size = self._get_lift()

        return ConstantLiftExperiment(
            self.data_formatter,
            treament_time,
            self.time_end,
            lift_size,
            treament_group
            )
    
    def get_treated_data(self) -> pl.DataFrame:
        self.experiment_setup = self.get_experiment()
        return self.experiment_setup.apply_treatment(self.dataset)
        