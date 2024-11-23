from datetime import datetime
from typing import List, Protocol
from dataclasses import dataclass
from .data_formatter import 
import polars as pl



class DataFiller(Protocol):

    def fill_missing_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Fill missing (i.e. NULL) data"""

class  DataFillerConstant():
    def __init__(self, numeric_constant: float=None, categorical_constant: str=None):
        self.numeric_constant = numeric_constant if numeric_constant is not None else 0.0
        self.categorical_constant = categorical_constant if categorical_constant is not None else "None"

    @staticmethod
    def _get_numerical_columns(data: pl.DataFrame) -> List[str]:
        numeric_types = {pl.Int32, pl.Int64, pl.Float32, pl.Float64}
        # Find numeric columns
        return [name for name, dtype in data.schema.items() if dtype in numeric_types]
    
    def _get_non_numerical_columns(self, data: pl.DataFrame) -> List[str]:
        numeric_types = {pl.Int32, pl.Int64, pl.Float32, pl.Float64}
        # Find numeric columns
        return [name for name, dtype in data.schema.items() if dtype not in numeric_types]
    
    @staticmethod
    def _fill_missing_data(data: pl.DataFrame, columns_to_fill: List[str], fill_value: str) -> pl.DataFrame:
        return data.with_columns(
                [
                    pl.when(pl.col(column).is_null())
                    .then(fill_value)
                    .otherwise(pl.col(column))
                    .alias(column)
                    for column in columns_to_fill
                ]
            )

    def fill_missing_data(self, data: pl.DataFrame) -> pl.DataFrame:
        numerical_cols = self._get_numerical_columns(data)
        non_numerical_cols = self._get_non_numerical_columns(data)

        data = self._fill_missing_data(data, numerical_cols, self.numeric_constant)
        data = self._fill_missing_data(data, non_numerical_cols, self.categorical_constant)
        return data


class BaseDataCleaner():

    def __init__(self, data_formater: DataFormatter, data_filler: DataFillerConstant):
        self.data_formater = data_formater
        self.data_filler = data_filler
 
    def fit(self, X: pl.DataFrame, y:pl.Series=None) -> None:
        pass

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        X = self.data_formater.DataFormatter(X)
        X = self.data_filler.fill_missing_data(X)

        return X

    def fit_transform(self, X: pl.DataFrame, y: pl.Series=None) -> pl.DataFrame:
        self.fit(X, y)
        return self.transform(X)


class SupermarketSalesFormatter(BaseFormater):
    def __init__(self,
                 id_col: str="Invoice ID",
                 treatment_col: str="City",
                 date_col: str="Date",
                 target_col: str="Total",
                 feature_cols: List[str] = ["Branch", "Customer type", "Gender", "Product line", "Payment"],
                 date_format: str = '%m/%d/%Y') -> None:
        super().__init__(id_col, treatment_col, date_col, target_col, feature_cols, date_format)


class IowaLicorSalesFormatter(BaseFormater):
    def __init__(self,
                 id_col: str="Invoice/Item Number",
                 treatment_col: str="County",
                 date_col: str="Date",
                 target_col: str="Sale (Dollars)",
                 feature_cols: List[str] = ["Category", "Bottle Volume (ml)"],
                 date_format: str = '%m/%d/%Y',
                 date_discretization: str=MONTH_DISCRETIZATION) -> None:
        super().__init__(id_col, treatment_col, date_col, target_col, feature_cols, date_format, date_discretization)

    def _transform_target(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns(
            pl.col(self.target_col)
            .map_elements(lambda value: float(value.replace("$", "")), return_dtype=pl.Float64)
        )


class WallmartSalesFormatter(BaseFormater):
    def __init__(self, 
                 id_col: str="Store",
                 treatment_col: str="Store",
                 date_col: str="Date",
                 target_col: str="Weekly_Sales",
                 feature_cols: List[str] = ["Temperature", "Fuel_Price", "CPI", "Unemployment"],
                 date_format: str = '%d-%m-%Y') -> None:
        super().__init__(id_col, treatment_col, date_col, target_col, feature_cols, date_format)


class SuperstoreSalesFormatter(BaseFormater):
    def __init__(self,
                 id_col: str="Order ID",
                 treatment_col: str="City",
                 date_col: str="Order Date",
                 target_col: str="Sales",
                 feature_cols: List[str] = ["State", "Category", "Sub-Category"],
                 date_format: str = '%d/%m/%Y',
                 date_discretization: str=MONTH_DISCRETIZATION) -> None:
        super().__init__(id_col, treatment_col, date_col, target_col, feature_cols, date_format, date_discretization)


class LifetimeValueFormatter(BaseFormater):
    def __init__(self,
                 id_col: str="user_id",
                 treatment_col: str="country_segment",
                 date_col: str="join_date",
                 target_col: str="STV",
                 feature_cols: List[str] = ["product", "product_type", "credit_card_level"],
                 date_format: str = '%Y-%m-%d %H:%M:%S') -> None:
        super().__init__(id_col, treatment_col, date_col, target_col, feature_cols, date_format)


class EarthTemperatureFormatter(BaseFormater):
    def __init__(self,
                 id_col: str="Country",
                 treatment_col: str="Country",
                 date_col: str="dt",
                 target_col: str="AverageTemperature",
                 feature_cols: List[str] = [],
                 date_format: str = '%Y-%m-%d') -> None:
        super().__init__(id_col, treatment_col, date_col, target_col, feature_cols, date_format)

    def _filter_data(self, data: pl.DataFrame) -> pl.DataFrame:
        
        return (
            data
            .filter(pl.col(self.date_col) >= pl.lit(datetime(1990, 1, 1)))
            )
    
class Nifty50StockMarketFormatter(BaseFormater):
    def __init__(self,
                 id_col: str="Symbol",
                 treatment_col: str="Symbol",
                 date_col: str="Date",
                 target_col: str="Close",
                 feature_cols: List[str] = [],
                 date_format: str = '%Y-%m-%d') -> None:
        super().__init__(id_col, treatment_col, date_col, target_col, feature_cols, date_format)

class CoronaFormatter(BaseFormater):
    def __init__(self,
                 id_col: str="Province/State",
                 treatment_col: str="Country/Region",
                 date_col: str="ObservationDate",
                 target_col: str="Confirmed",
                 feature_cols: List[str] = [],
                 date_format: str = '%m/%d/%Y') -> None:
        super().__init__(id_col, treatment_col, date_col, target_col, feature_cols, date_format)
    
    def _filter_data(self, data: pl.DataFrame) -> pl.DataFrame:
        return (
            data
            .filter(pl.col(self.date_col) > pl.lit(datetime(2020, 10, 1)))
            )
    
    def _transform_target(self, data: pl.DataFrame) -> pl.DataFrame:
        return (
            data
            .sort([self.date_col])
            .with_columns(
                (pl.col(self.target_col) - pl.coalesce(pl.col(self.target_col).shift().over([self.treatment_col]), pl.lit(0))).alias(self.target_col)
            )
        )