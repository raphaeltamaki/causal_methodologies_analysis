from datetime import timedelta

import numpy as np
import polars as pl


class SyntheticControlPreProcessing:

    def __init__(
            self,
            id_col :str,
            date_col: str,
            metric_col: str            ) -> None:
        
        # Required parameters
        self.id_col = id_col
        self.date_col = date_col
        self.metric_col = metric_col
        
        # Constants
        self.date_start = None
        self.date_end = None

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

    def _verify_uniqueness(self, data: pl.DataFrame) -> bool:
        """
        Check if values in the data are unique per (self.date_col, self.id_col)
        If they are not, they should be grouped together
        Return True if values are unique, False otherwise
        """
        return data.n_unique(subset=[self.id_col, self.date_col]) == data.shape[0]


    def _set_mean_and_std(self, data: pl.DataFrame) -> None:
        """
        Extract the mean and standard deviation of each segment in the data
        Also calculate an 'global' value to be used in case of new segments
        """
        # Get mean and average of all segments in the data
        self.segments_stats = (
            data
            .groupby([self.id_col])
            .agg(
                pl.col(self.metric_col).mean().alias('avg'),
                pl.col(self.metric_col).std().alias('std')
            )
        )
        # to be used in case there is a new segmet
        self.global_stat = (
            data
            .select(
                global_avg = pl.col(self.metric_col).mean(),
                global_std = pl.col(self.metric_col).std()
            )
        )


    def _normalize_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize the values based on the mean and standard deviation
        previously extracted
        """
        return  (
            data
            .join(self.segments_stats, on=[self.id_col], how='left')
            .join(self.global_stat, how='cross')
            # use segment specific std and mean if possible
            # otherwise, use global values
            .with_columns(
                pl.when(pl.col('std').is_not_nan())
                .then((pl.col(self.metric_col) - pl.col('avg')) / pl.col('std'))
                .otherwise((pl.col(self.metric_col) - pl.col('global_avg')) / pl.col('global_std'))
                .alias(self.metric_col)
            )
            .drop(['avg', 'std', 'global_avg', 'global_std'])
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

    def fit(self, X: pl.DataFrame, y: pl.Series=None) -> None:
        """
        Use the provided that to extract parameters needed to transform the data 
        """
        if not self._verify_uniqueness(X):
            X = self._group_data(X)
        self._set_mean_and_std(X)

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the data based on the parameters defined when initializing class and the parameters extracted during [fit]
        """
        if not self._verify_uniqueness(X):
            X = self._group_data(X)
        X = self._normalize_data(X)
        X = self._apply_default_names(X)
        return X

    def fit_transform(self, X: pl.DataFrame, y: pl.Series=None) -> pl.DataFrame:
        """
        Apply fit on the data and then transform it
        """
        self.fit(X)
        return self.transform(X)