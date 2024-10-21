from dataclasses import dataclass
from typing import Protocol


@dataclass
class BenchmarkDataset(Protocol):
    def dataset_name(self) -> str:
        """Return name of the dataset"""

    def data_format(self) -> str:
        """Return the format used to store the dataset"""
    
    def main_file_name(self) -> str:
        """Return the name of the main file of the dataset"""

@dataclass
class KaggleBenchmarkDataset:
    _dataset_name: str
    _data_format: str
    _main_file_name: str
    _kaggle_dataset_address: str

    def dataset_name(self) -> str:
        return self._dataset_name

    def data_format(self) -> str:
        return self._data_format
    
    def main_file_name(self) -> str:
        return self._main_file_name
    
    def kaggle_dataset_address(self) -> str:
        return self._kaggle_dataset_address

@dataclass
class CoronaDataset(KaggleBenchmarkDataset):
    _dataset_name: str="corona"
    _data_format: str="csv"
    _main_file_name: str="time_series_covid_19_confirmed.csv"
    _kaggle_dataset_address: str="sudalairajkumar/novel-corona-virus-2019-dataset"

@dataclass
class EarthTemperatureDataset(KaggleBenchmarkDataset):
    _dataset_name: str="climate_change_earth_temperature"
    _data_format: str="csv"
    _main_file_name: str="GlobalLandTemperaturesByState.csv"
    _kaggle_dataset_address: str="berkeleyearth/climate-change-earth-surface-temperature-data"

@dataclass
class IowaLicorSalesDataset(KaggleBenchmarkDataset):
    _dataset_name: str="iowa_licor_sales"
    _data_format: str="csv"
    _main_file_name: str="Iowa_Liquor_Sales.csv"
    _kaggle_dataset_address: str="residentmario/iowa-liquor-sales"

@dataclass
class LifetimeValueDataset(KaggleBenchmarkDataset):
    _dataset_name: str="lifetime_value"
    _data_format: str="csv"
    _main_file_name: str="train.csv"
    _kaggle_dataset_address: str="baetulo/lifetime-value"

@dataclass
class Nifty50StockMarketValueDataset(KaggleBenchmarkDataset):
    _dataset_name: str="nifty50_stock_market"
    _data_format: str="csv"
    _main_file_name: str="NIFTY50_all.csv"
    _kaggle_dataset_address: str="rohanrao/nifty50-stock-market-data"

@dataclass
class SupermarketSalesDataset(KaggleBenchmarkDataset):
    _dataset_name: str="supermarket_sales"
    _data_format: str="csv"
    _main_file_name: str="supermarket_sales - Sheet1.csv"
    _kaggle_dataset_address: str="aungpyaeap/supermarket-sales"

@dataclass
class SuperstoreSalesDataset(KaggleBenchmarkDataset):
    _dataset_name: str="superstore_sales"
    _data_format: str="csv"
    _main_file_name: str="train.csv"
    _kaggle_dataset_address: str="rohitsahoo/sales-forecasting"
    
@dataclass
class WallmartSalesDataset(KaggleBenchmarkDataset):
    _dataset_name: str="wallmart_sales"
    _data_format: str="csv"
    _main_file_name: str="Walmart.csv"
    _kaggle_dataset_address: str="yasserh/walmart-dataset"
