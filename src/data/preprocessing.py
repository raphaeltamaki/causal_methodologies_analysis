import polars as pl
import numpy as np
from datetime import timedelta

class DataPreProcessing:

    def __init__(
            self,
            data: pl.DataFrame,
            id_col :str,
            date_col: str,
            metric_col: str,
            date_format: str='yyyy-MM-dd'
            ) -> None:
        
        # Required parameters
        self.data = data
        self.id_col = id_col
        self.date_col = date_col
        self.metric_col = metric_col

        # Facultative parameters
        self.date_format = date_format

        # Constants
        self.date_start = None
        self.date_end = None

    def _cast_date_column(self, data) -> None:
        """
        Cast the date column to date
        """
        data = (
            data
            .with_columns(
                pl.col(self.date_col).str.to_datetime(self.date_format)
                )
        )
        return data

    def _get_date_time(self, data: pl.DataFrame) -> None:
        """
        Get start and end date of the dataset
        """
        self.date_start = data[self.date_col].min()
        self.date_end = data[self.date_col].max()

    def _group_data(self, data: pl.DataFrame) -> None:
        """
        Group data based on the ID and date columns to remove duplicates (or to just regroup on a new granularity)
        """
        return (
            data
            .groupby([self.id_col, self.date_col])
            .agg(pl.col(self.metric_col).sum())
        )
    
    def _normalize_data(self, data: pl.DataFrame, pre_treatment_share: float):
        """
        Normalize the data based in the pre-treatment period
        """
        post_treatment_date_start = self.date_start + timedelta(
            days=np.round(pre_treatment_share * (self.date_end - self.date_start).days)
            )
        data_stats = (
            data
            .filter(pl.col(self.date_col) < post_treatment_date_start)
            .groupby([self.id_col])
            .agg(
                pl.col(self.metric_col).mean().alias('avg'),
                pl.col(self.metric_col).std().alias('std')
            )
        )
        return (
            data
            .join(data_stats, on=[self.id_col], how='inner')
            .with_columns(
                ((pl.col(self.metric_col) - pl.col('avg')) / pl.col('std')).alias(self.metric_col)
            )
            .with_columns((pl.col(self.date_col) >= pl.lit(post_treatment_date_start)).alias('treatment_period'))
            .drop(['avg', 'std'])
        )


    def _apply_default_names(self, data: pl.DataFrame) -> None:
        return (
            data
            .rename({
                self.id_col: 'id',
                self.date_col: 'date',
                self.metric_col: 'value'
                })
        )

    def get_preprocessed_data(self):
        """
        Apply all the steps to load and pre-process data
        1) Load data from storage
        2) Group data in the desired granularity
        3) Get the start and end dates
        4) Normalize data based on pre-treatment period
        5) Rename columns to the default of names for ID, Date, and Metric columns

        Observation:
        pre_treatment_share=1 as default, because this method is not meant to be used to analyze the effect of a treatment,
        so there is no problem of information leakage if we normalize again (using a fraction of the dataset) afterwards 
        when we apply the treatment effect
        """
        # Prepare the data 
        self.data = self._cast_date_column(self.data)
        self._get_date_time(self.data)

        # # group and normalize data
        self.data = self._group_data(self.data)
        self.data = self._normalize_data(self.data, 1.0)
        self.data = self._apply_default_names(self.data)
        return self.data