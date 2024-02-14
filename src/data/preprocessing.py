import polars as pl
import numpy as np

class DataPreProcessing():

    def __init__(
            self, 
            data_path: str,
            id_col :str,
            date_col: str,
            metric_col: str
            ) -> None:
        
        self.data_path = data_path
        self.id_col = id_col
        self.date_col = date_col
        self.metric_col = metric_col

        self.data = None
        self.date_start = None
        self.date_end = None
    
    def _load_data(self) -> None:
        """
        Load data from a csv based on the path given
        """
        return pl.read_csv(self.data_path)

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
            .group_by([self.id_col, self.date_col])
            .agg(pl.col(self.metric_col).sum())
        )
    
    def _normalize_data(self, data: pl.DataFrame, pre_treatment_share: float):
        """
        Normalize the data based in the pre-treatment period
        """
        post_treatment_date_start = self.date_start + np.round(pre_treatment_share * (self.date_end - self.date_start).days)
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

    def get_preprocessed_data(self, pre_treatment_share: float=0.8):
        """
        Apply all the steps to load and pre-process data
        1) Load data from storage
        2) Group data in the desired granularity
        3) Get the start and end dates
        4) Normalize data based on pre-treatment period
        5) Rename columns to the default of names for ID, Date, and Metric columns
        """
        data = self._load_data()
        self._get_date_time(data)

        # group and normalize data
        data =self._group_data(data)
        data = self._normalize_data(data, pre_treatment_share)
        data = self._apply_default_names(data)
        return data