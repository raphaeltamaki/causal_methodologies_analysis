import polars as pl
import kaggle
import os
from pathlib import Path
from typing import Dict


class DataLoader:
    """
    This class is responsible for downloading the data that is required for the analysis
    It is meant to simply download all the relevant datasets in the local enviroment and read it
    when asked.
    There are currently 5 datasets being used, all from Kaggle:
        - Iowa Licor Sales
        - Wallmart Dataset
        - Supermarket Sales
        - Superstore Sales Dataset
        - Lifetime Value
        - Novel Corona 2019 Dataset
    """

    def __init__(self, data_path: str = None):

        self.data_path = (
            Path(data_path)
            if data_path is not None
            else Path(os.path.dirname(os.path.abspath(__file__)))
        )
        # Constants
        self.remote_data_addresses = {
            "iowa_licor_sales": "residentmario/iowa-liquor-sales",
            "wallmart_sales": "yasserh/walmart-dataset",
            "supermarket_sales": "aungpyaeap/supermarket-sales",
            "superstore_sales": "rohitsahoo/sales-forecasting",
            "lifetime_value": "baetulo/lifetime-value",
            "corona": "sudalairajkumar/novel-corona-virus-2019-dataset",
            "climate_change_earth_temperature": "berkeleyearth/climate-change-earth-surface-temperature-data",
            "nifty50_stock_market": "rohanrao/nifty50-stock-market-data",
        }
        self.raw_data = {}

    def get_data(self, force: bool = False) -> None:
        """
        Get the data from the remote repositores and store it in self.data_path
        If data already exists, just load the data from local repository
        If force=True, redownload the data and overwrite what exists
        """
        kaggle.api.authenticate()
        for dataset_name, dataset_address in self.remote_data_addresses.items():
            print(f"Getting {dataset_name} dataset")

            download_folder = self.data_path / Path(dataset_name)
            if not os.path.exists(download_folder):
                os.makedirs(download_folder)

            # download data if we are forcing or if it doesn't exist (folder is empty)
            if force or not os.listdir(download_folder):
                print(
                    f"Downloading {dataset_name} dataset. Storing it in {download_folder}"
                )
                kaggle.api.dataset_download_files(
                    dataset_address, path=download_folder, unzip=True
                )
            else:
                print(f"Dataset {dataset_name} already present\n")

    def load_dataset(self, dataset_name: str) -> pl.DataFrame:
        """
        Loads the dataset of the given name.
        If dataset doesn't exists, raise error and points to possible inputs
        Assumes that one file exists in the path. If more than one exists, loads only the first one
        """
        if dataset_name not in self.remote_data_addresses:
            raise ValueError(
                f"This is not a valid dataset name. Please use one of the following: {', '.join(self.remote_data_addresses.keys())}"
            )

        # finds all files that are present in the path and get the first one if more than one exists
        file_path = self.data_path / Path(dataset_name)
        file_name = list(file_path.glob('*.csv'))[0].resolve()
        print(file_name)
        return pl.read_csv(file_path / file_name)

    def load_all_datasets(self) -> Dict[str, pl.DataFrame]:
        """
        Read all datasets and return a dictionary containing the Polars DataFrame of each one of them
        """
        output = {}
        for dataset_name in self.remote_data_addresses:
            output[dataset_name] = self.load_dataset(dataset_name)
        return output
