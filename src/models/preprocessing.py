from datetime import timedelta

import numpy as np
import polars as pl
from src.data.experiment_setup import ExperimentSetup
from src.data.data_formatter import BaseFormater

class SyntheticControlPreProcessing:

    def __init__(
            self,
            data_formatter: BaseFormater,
            experiment_setup: ExperimentSetup,
        ) -> None:
        
        # Required parameters
        self.experiment_setup = experiment_setup
        self.treatment_col = data_formatter.treatment_col
        self.id_col = data_formatter.id_col
        self.date_col = data_formatter.date_col
        self.metric_col = data_formatter.target_col
        
        # Constants
        self.treatment_date_start = None
        self.treatment_date_end = None

        self.segments_stats = None
        self.global_stat = None

    def _group_data(self, data: pl.DataFrame) -> None:
        """
        Group data based on the ID and date columns to remove duplicates (or to just regroup on a new granularity)
        """
        return (
            data
            .groupby([self.treatment_col, self.date_col])
            .agg(pl.col(self.metric_col).sum())
        )

    def _verify_uniqueness(self, data: pl.DataFrame) -> bool:
        """
        Check if values in the data are unique per (self.date_col, self.treatment_col)
        If they are not, they should be grouped together
        Return True if values are unique, False otherwise
        """
        return data.n_unique(subset=[self.treatment_col, self.date_col]) == data.shape[0]


    def _set_mean_and_std(self, data: pl.DataFrame) -> None:
        """
        Extract the mean and standard deviation of each segment in the data
        Also calculate an 'global' value to be used in case of new segments
        """
        # Get mean and average of all segments in the data
        self.segments_stats = (
            data
            .groupby([self.treatment_col])
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
            .join(self.segments_stats, on=[self.treatment_col], how='left')
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
                self.treatment_col: 'id',
                self.date_col: 'date',
                self.metric_col: 'value'
                })
        )
    
    def _filter_for_pretreatment(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Filter data to be only for the pre-treatment period
        """
        return X.filter(~(self.experiment_setup.treatment_dates))

    def fit(self, X: pl.DataFrame, y: pl.Series=None) -> None:
        """
        Use the provided that to extract parameters needed to transform the data 
        """
        X = self._filter_for_pretreatment(X)
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
        X = X.pivot(
            index="date",
            columns="id",
            values="value",
            aggregate_function="sum"
        )
        return X

    def fit_transform(self, X: pl.DataFrame, y: pl.Series=None) -> pl.DataFrame:
        """
        Apply fit on the data and then transform it
        """
        self.fit(X)
        return self.transform(X)
    
