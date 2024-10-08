from datetime import datetime
from typing import List
import polars as pl

DAY_DISCRETIZATION = "d"
MONTH_DISCRETIZATION = "m"
YEAR_DISCRETIZATION = "y"

class BaseFormater():

    def __init__(
            self,
            id_col :str,
            treatment_col :str,
            date_col: str,
            target_col: str,
            feature_cols: List[str]=None,
            date_format: str='%Y-%m-%d',
            date_discretization: str=DAY_DISCRETIZATION,
            missing_number_fill: float=0.0,
            ) -> None:
        """
        Inputs
            - id_col: name of the column that, together with date, is unique in the dataset.
            - treatment_col: column that indicates the dimension where the treatment can be applied
            - date_col: name of the column indicating when (date) an event happened
            - target_col: the dimension we are interested in predicting/measuring the effect the treatment has
            - feature_cols: features that can improve the accuracy of the predictions. Must not be dependent on the treatment_col
                (e.g. if the treatment is Country, then City and State cannot be used as features)
        """
        self.id_col = id_col
        self.treatment_col = treatment_col
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = [] if feature_cols is None else feature_cols
        self.date_format = date_format
        self.date_discretization = date_discretization
        self.missing_number_fill = missing_number_fill

        # Resulting parameters
        self.selected_cols = list(set([self.id_col, self.treatment_col, self.date_col, self.target_col] + self.feature_cols)) # drop duplicates

    @staticmethod
    def get_year_month(column_name):
        """
        Get the the year-date of the date in
        yyyy-MM format
        """
        year = pl.col(column_name).dt.year()
        month = pl.col(column_name).dt.month()
        month = pl.when(month < 10).then(pl.concat_str(pl.lit("0"), month)).otherwise(month)
        return pl.concat_str([year, pl.lit("-"), month]).str.to_datetime("%Y-%m", strict=True).alias(column_name)

    def _discretize_date(self, column_name):
        """
        Discretize the date column to yearly, monthly, or daily
        """
        if self.date_discretization == YEAR_DISCRETIZATION:
            yield pl.col(column_name).dt.year().str.to_datetime("%Y", strict=True).alias(column_name)
        elif self.date_discretization == MONTH_DISCRETIZATION:
            yield self.get_year_month(column_name).alias(column_name)
        elif self.date_discretization == DAY_DISCRETIZATION:
            yield pl.col(column_name).cast(pl.Date).cast(pl.Datetime).alias(column_name)
        else:
            raise ValueError(f"Invalid value for discretization, please use either {DAY_DISCRETIZATION}, {MONTH_DISCRETIZATION}, {YEAR_DISCRETIZATION}")

    def _format_date_col(self, data: pl.DataFrame) -> pl.DataFrame:
        if not isinstance(data.select(pl.col(self.date_col)).dtypes[0], pl.Datetime):
            data = (
                data
                .with_columns(
                    pl.col(self.date_col).str.to_datetime(self.date_format, strict=True)
                    )
            )
        return (
            data
            .with_columns(self._discretize_date(self.date_col))
        )
    def _format_id_col(self, data: pl.DataFrame) ->pl.DataFrame:
        return data.with_columns(pl.col(self.id_col).str.replace_all("[^0-9a-zA-Z]+", "").alias(self.id_col))
    
    def _filter_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Filter data to include only the appropriate rows for a quasi-experiment analysis.
        For example: for a dataset with a long time spam, but with a lot of missing dates, we filter only for the period
        where all ids are present almost all days
        """
        return data
    
    def _remove_extra_cols(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select(self.selected_cols)
    
    def _fill_missing_values(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.fill_nan(self.missing_number_fill).fill_null(self.missing_number_fill)

    def _transform_target(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the values of the target to a numerical value, if they need formatting
        Example:
            "$142.23" -> "142.23"
        """
        return data
    
    def fit(self, X: pl.DataFrame, y:pl.Series=None) -> None:
        pass

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        X = self._format_date_col(X)
        X = self._format_id_col(X)
        X = self._remove_extra_cols(X)
        X = self._transform_target(X)
        X = self._filter_data(X)
        X = self._fill_missing_values(X)
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