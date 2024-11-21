import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol

from .kaggle_datasets import KaggleBenchmarkDataset
from .utils import CreateLocalDirectoryIfNotExists

import kaggle
import polars as pl


class DataPuller(Protocol):
    """Protocol for classes that implement pull_data()"""
    def pull_data(self, data_path: Path) -> None:
        """Pulls data from a remote repository"""

@dataclass
class KaggleDataPuller:
    """Pulls a dataset from Kaggle"""

    def pull_kaggle_data(self, data_path: Path, kaggle_dataset_address: str, unzip: bool=True) -> None:
        """Pulls data from kaggle and stores it the folder path"""
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
                    kaggle_dataset_address, path=data_path, unzip=unzip
                )

class DataLoader(Protocol):
    """Protocol for classes that implement load_data()"""
    def load_data(self) -> pl.DataFrame:
        """Outputs the data"""


class LocalDataLoader:
    """Loads data stored locally"""
   
    def load_local_data(self, data_path: Path, data_format: str="csv") -> pl.DataFrame:
        """Load dataset from the local file path."""
        if data_format == 'csv':
            return pl.read_csv(data_path)
        if data_format == 'ipc':
            return pl.read_ipc(data_path)
        if data_format == 'json':
            return pl.read_json(data_path)

        raise ValueError(
            "{self.data_format} is not a valid data format. The accepted formats are 'csv', 'delta', 'json', and 'ipc'"
            )


@dataclass
class KaggleDataPullerLoader(KaggleDataPuller, LocalDataLoader):
    """Downlods a dataset from Kaggle in a local directory, and loads it"""
    dataset: KaggleBenchmarkDataset
    data_folder_path: Path

    def _get_dataset_folder_path(self) -> Path:
        return self.data_folder_path / Path(self.dataset.dataset_name())
    
    def _get_dataset_main_file_path(self) -> Path:
        return self.data_folder_path / Path(self.dataset.dataset_name()) / Path(self.dataset.main_file_name())
    
    def pull_data(self) -> None:
        """Pull dataset from Kaggle in the folder path declared when class was initialized"""
        dataset_path = self._get_dataset_folder_path()
        CreateLocalDirectoryIfNotExists().create_path_if_not_exists(dataset_path)
        self.pull_kaggle_data(dataset_path, self.dataset.kaggle_dataset_address)

    def load_data(self) -> pl.DataFrame:
        """Loads dataset from Kaggle in the folder path declared when class was initialized. Must have download dataset to work or dataset must already be in the path"""
        data_file_path = self._get_dataset_main_file_path()
        self.load_local_data(data_file_path, self.dataset.data_format())
            
@dataclass
class OpenDatasetsBenchmarksLoader:

    def __init__(self, kaggle_datasets: List[KaggleBenchmarkDataset], data_folder_path: Path):
        self.kaggle_datasets = kaggle_datasets
        self.data_folder_path = data_folder_path
        self.kaggle_data_loaders = self._get_loaders(kaggle_datasets, data_folder_path)

    def _get_loaders(self, kaggle_datasets: List[KaggleBenchmarkDataset], data_folder_path: Path):
        return [KaggleDataPullerLoader(dataset, data_folder_path) for dataset in kaggle_datasets]
   
    

class DataLoader2:
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

    @staticmethod
    def get_file_name(dataset_name) -> Path:
        """
        Get the correct file to load, given that some datasets actually consists of multiple files
        """
        if dataset_name == "nifty50_stock_market":
            return Path("NIFTY50_all.csv")
        elif dataset_name == "climate_change_earth_temperature":
            return Path("GlobalLandTemperaturesByState.csv")
        elif dataset_name == "corona":
            return Path("time_series_covid_19_confirmed.csv")
        elif dataset_name == "lifetime_value":
            return Path("test.csv")
        elif dataset_name == "wallmart_sales":
            return Path("Walmart.csv")
        elif dataset_name == "superstore_sales":
            return Path("train.csv")
        elif dataset_name == "supermarket_sales":
            return Path("supermarket_sales - Sheet1.csv")
        elif dataset_name == "iowa_licor_sales":
            return Path("Iowa_Liquor_Sales.csv")

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
        file_name = self.get_file_name(dataset_name)
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
