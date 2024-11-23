from typing import List, Protocol
import polars as pl


class DataFiller(Protocol):

    def fill_missing_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Fill missing (i.e. NULL) data"""


class DataFillerConstant:
    def __init__(
        self, numeric_constant: float = None, categorical_constant: str = None
    ):
        self.numeric_constant = (
            numeric_constant if numeric_constant is not None else 0.0
        )
        self.categorical_constant = (
            categorical_constant if categorical_constant is not None else "None"
        )

    @staticmethod
    def _get_numerical_columns(data: pl.DataFrame) -> List[str]:
        numeric_types = {pl.Int32, pl.Int64, pl.Float32, pl.Float64}
        # Find numeric columns
        return [name for name, dtype in data.schema.items() if dtype in numeric_types]

    def _get_non_numerical_columns(self, data: pl.DataFrame) -> List[str]:
        numeric_types = {pl.Int32, pl.Int64, pl.Float32, pl.Float64}
        # Find numeric columns
        return [
            name for name, dtype in data.schema.items() if dtype not in numeric_types
        ]

    @staticmethod
    def _fill_missing_data(
        data: pl.DataFrame, columns_to_fill: List[str], fill_value: str
    ) -> pl.DataFrame:
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
        data = self._fill_missing_data(
            data, non_numerical_cols, self.categorical_constant
        )
        return data
