import polars as pl
import numpy as np
import kaggle
import os
from pathlib import Path

class DataLoader():

    def __init__(
            self,
            data_path: str=None
            ):
        
        self.data_path = Path(data_path) if data_path is not None else Path(os.path.dirname(os.path.abspath(__file__)))
        self.remote_data_addresses = {
            "iowa_licor_sales": "residentmario/iowa-liquor-sales",
            "wallmart_sales": "yasserh/walmart-dataset",
            "supermarket_sales": "aungpyaeap/supermarket-sales",
            "superstore_sales": "rohitsahoo/sales-forecasting",
            "lifetime_value": "baetulo/lifetime-value"
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
            download_folder = self.data_path / Path(dataset_name)
            if not os.path.exists(download_folder):
                os.makedirs(download_folder)
            
            # download data if we are forcing or if it doesn't exist (folder is empty)
            if force or not os.listdir(download_folder):
                kaggle.api.dataset_download_files(dataset_address, path=download_folder, unzip=True)

