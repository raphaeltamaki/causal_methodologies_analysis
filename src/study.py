from typing import  List
from pathlib import Path
import polars as pl
import pandas as pd
from pathlib import Path
import polars as pl
import pandas as pd

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

from src.data.load import DataLoader

class Study():

    def __init__(
            self,
            datasets_names: List[str],
            estimators_classes: List,
            experiment_class,
            save_file_path: Path=Path("./results_data.csv"),
            dataset_store_path: Path=Path("data"),
            n_experiments: int=100,
            eligible_treatment_percentage: float=0.5,
            eligible_treatment_start_range: List[float]=[0.5, 0.5],
            lift_limit: float=1.0,
            last_n_days: int=100) -> None:
        self.datasets_names = datasets_names
        self.estimators_classes = estimators_classes
        self.experiment_class = experiment_class
        self.save_file_path = save_file_path if isinstance(save_file_path, Path) else Path(save_file_path)
        # Number and configuration of experiments
        self.n_experiments = n_experiments
        self.eligible_treatment_percentage = eligible_treatment_percentage
        self.eligible_treatment_start_range = eligible_treatment_start_range
        self.lift_limit = lift_limit
        self.last_n_days = last_n_days
        

        # To be used
        self.save_data = pd.DataFrame(data={
            "it": [],
            "dataset": [],
            "treated_groups": [],
            "lift_size": [],
            "reference_value": [],
            "model": [],
            "true_ate": [],
            "estimated_ate": [],
            "estimated_std": [],
            "ci": [],
            "estimated_lower_bound": [],
            "estimated_upper_bound": [],
        })

        # Constants
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
        self.loader = DataLoader(dataset_store_path)
    
    def get_dataset_formatter(self, dataset_name: str):
        return self.dataset_formatters[dataset_name]
    
    @staticmethod
    def filter_last_n_days(data: pl.DataFrame, formatter, n: int=0):
        """
        Filter only the last n days of the dataset.
        This might be important for some large datasets (like the stock market), because some algorithms take a long time to run
        """
        # if n below 0, we don't filter any date
        if n < 0:
            return data
        
        return (
            data
            .with_columns(
                pl.col(formatter.date_col).str.strptime(pl.Date, format=formatter.date_format).cast(pl.Date).alias("formated_date_col")
            )
            .with_columns(pl.col("formated_date_col").max().alias("last_date"))
            .filter(pl.col("formated_date_col") > (pl.col("last_date") - n))
            .drop(["last_date", "formated_date_col"])
            )
        return data

    def run(self):
        for dataset_name in self.datasets_names:
            print(f"Using {dataset_name} dataset")
            # Load data
            data = self.loader.load_dataset(dataset_name)
            formatter = self.get_dataset_formatter(dataset_name)
            # Filter the last n days of the dataset
            data = self.filter_last_n_days(data, formatter, self.last_n_days)
            # Run experiments N types for all classes for each dataset
            for i in range(self.n_experiments):
                for estimator_class in self.estimators_classes:
                    # Create a new experiment
                    setup = self.experiment_class(
                        dataset_name,
                        data,
                        eligible_treatment_percentage=self.eligible_treatment_percentage,
                        eligible_treatment_period=self.eligible_treatment_start_range,
                        lift_limit=self.lift_limit
                    )

                    treated_data = setup.get_treated_data()

                    reg = estimator_class(setup.data_formatter, setup.experiment_setup)
                    reg.fit(treated_data)

                    stats = reg.estimate_ate_distribution(treated_data)
                    results_data = pd.DataFrame(data={
                        "it": [i],
                        "dataset": [dataset_name],
                        "treated_groups": ['_'.join(setup.experiment_setup.treated_groups)],
                        "lift_size": [setup.experiment_setup.lift_size],
                        "model": [reg.__class__.__name__],
                        "reference_value": [setup.experiment_setup.get_reference_value()],
                        "true_ate": [setup.experiment_setup.get_true_ate()],
                        "estimated_ate": [reg.estimate_ate(treated_data)],
                        "estimated_std": [stats[0]],
                        "ci": [0.9],
                        "estimated_lower_bound": [stats[1]],
                        "estimated_upper_bound": [stats[2]],
                    })

                    self.save_data = pd.concat([self.save_data, results_data], axis=0)
                    self.save_data.to_csv(self.save_file_path,index = False)
    
    def save_results(self):
        pass