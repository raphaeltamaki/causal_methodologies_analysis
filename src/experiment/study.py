from typing import  Dict, List
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
            last_n_days: int=100,
            req_segments_frequency: float=0.5) -> None:
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
        self.req_segments_frequency = req_segments_frequency
        self.infrequency_group_names = 'aggregated_places'
        
        

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
        
    def _group_infrequent_segments(self, data: pl.DataFrame, formatter):
        """
        Group segments that don't appear at least req_frequency days in the dataset.
        This is necessary to guarantee that the treated locations appear enough times in the dataset for a estimation to be possible
        All segments that appear less than self.req_segments_frequency % of days in the dataset will be grouped together.
        """
        frequent_segments = (
            formatter._format_date_col(data)
            .with_columns(((pl.col(formatter.date_col).max() - pl.col(formatter.date_col).min()).dt.total_days() + 1).alias("total_days")) # +1 needed to make count include on dates
            .group_by([formatter.treatment_col, "total_days"])
            .agg((pl.col(formatter.date_col).n_unique()).alias("unique_dates"))
            .with_columns(
                (pl.col("unique_dates") / pl.col("total_days")).alias("frequency")
            )
            .sort("frequency")
            # .drop(["unique_dates", "total_days"])
            # .filter(pl.col('frequency') >= pl.lit(self.req_segments_frequency))
        )
        return frequent_segments


    @staticmethod
    def _execute_experiment(estimator, treated_data: pl.DataFrame, ci: float=0.9) -> Dict[str, str]:
        estimator.fit(treated_data)
        std, lower_bound, upper_bound = estimator.estimate_ate_distribution(treated_data)
        return pd.DataFrame(
            data={
                "model": [estimator.__class__.__name__],
                "estimated_ate": [estimator.estimate_ate(treated_data)],
                "estimated_std": [std],
                "estimated_lower_bound": [lower_bound],
                "estimated_upper_bound": [upper_bound],
                "ci": [ci],
                }
            )
    @staticmethod
    def _get_experiment_stats(it: int, dataset_name: str, setup) -> Dict[str, str]:
        return {
                "it": [it],
                "dataset": [dataset_name],
                "treated_groups": ['_'.join([str(group) for group in setup.experiment_setup.treated_groups])],
                "lift_size": [setup.experiment_setup.lift_size],
                "reference_value": [setup.experiment_setup.get_reference_value()],
                "true_ate": [setup.experiment_setup.get_true_ate()]
            }

    @staticmethod
    def _combine_stats(experiment_stats: Dict, estimates_stats: Dict) -> pd.DataFrame:
        return pd.DataFrame(data={**experiment_stats, **estimates_stats})

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
                # Create a new experiment
                setup = self.experiment_class(
                    dataset_name,
                    data,
                    eligible_treatment_percentage=self.eligible_treatment_percentage,
                    eligible_treatment_period=self.eligible_treatment_start_range,
                    lift_limit=self.lift_limit
                )
                treated_data = setup.get_treated_data()
                for estimator_class in self.estimators_classes:
                    
                    reg = estimator_class(setup.data_formatter, setup.experiment_setup)
                    # Combine the experiment stats
                    experiment_stats = self._get_experiment_stats(i, dataset_name, setup)
                    estimates = self._execute_experiment(reg, treated_data)
                    results_data = self._combine_stats(experiment_stats, estimates)
                    # Save the data
                    self.save_data = pd.concat([self.save_data, results_data], axis=0)
                self.save_data.to_csv(self.save_file_path, index=False)
    
    def save_results(self):
        pass