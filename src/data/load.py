import polars as pl
import numpy as np
import kaggle


class DataLoader():

    def __init__(
            self,
            data_path: str,
            id_col :str,
            date_col: str,
            metric_col: str
            ):
        
        self.data_path = data_path

        self.remote_data_addresses = {
            "iowa_licor_sales": "residentmario/iowa-liquor-sales",
            "wallmart_sales": "yasserh/walmart-dataset",
            "supermarket_sales": "aungpyaeap/supermarket-sales",
            "superstore_sales": "rohitsahoo/sales-forecasting",
            "lifetime_value": "baetulo/lifetime-value"

        }
        import kaggle.cli
# Authenticate with Kaggle
kaggle.cli.login()
# Download a dataset
kaggle.cli.download_dataset()
        ## Datasets


    def get_data(self, force: bool = False) -> None:
        """
        Get the data from the remote repositores and store it in self.data_path
        If data already exists, just load the data from local repository
        If force=True, redownload the data and overwrite what exists
        """

