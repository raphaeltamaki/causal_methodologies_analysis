import polars as pl
import numpy as np

class DataPreProcessing():

    def __init__(
            self, 
            data_path: str,
            id_col :str,
            date_col: str,
            metric_col: str,
            pre_treatment_share: float=0.8
            ) -> None:
        
        self.data_path = data_path
        self.id_col = id_col
        self.date_col = date_col
        self.metric_col = metric_col
        self.pre_treatment_share = pre_treatment_share

        self.data = None
        self.date_start = None
        self.date_end = None
    
    def _load_data(self) -> None:
        """
        Load data from a csv based on the path given
        """
        self.data = pl.read_csv(self.data_path)

    def _get_date_time(self) -> None:
        """
        Get start and end date of the dataset
        """
        self.date_start = self.data[self.date_col].min()
        self.date_end = self.data[self.date_col].max()

    def _group_data(self) -> None:
        """
        Group data based on the ID and date columns to remove duplicates (or to just regroup on a new granularity)
        """
        self.data = (
            self.data
            .group_by([self.id_col, self.date_col])
            .agg(pl.col(self.metric_col).sum())
        )
    
    def normalize_data(self):
        """
        Normalize the data based in the pre-treatment period
        """
        post_treatment_date_start = self.date_start + np.round(self.pre_treatment_share * (self.date_end - self.date_start).days)
        data_stats = (
            self.data
            .filter(pl.col(self.date_col) < post_treatment_date_start)
            .groupby([self.id_col])
            .agg(
                pl.col(self.metric_col).mean().alias('avg'),
                pl.col(self.metric_col).std().alias('std')
            )
        )
        self.data = (
            self.data
            .join(data_stats, on=[self.id_col], how='inner')
            .with_columns(
                (pl.col(self.metric_col) - pl.col('avg')) / pl.col('std')
            )
            .drop(['avg', 'std'])
        )


