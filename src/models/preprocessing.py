from datetime import timedelta
from typing import List, AnyStr
import numpy as np
import polars as pl
from src.data.experiment_setup import ExperimentSetup
from src.data.data_formatter import BaseFormater


class BasePreProcessing:

    def __init__(
            self,
            data_formatter: BaseFormater,
            experiment_setup: ExperimentSetup,
            std_min_bound: float=0.001
        ) -> None:

        # Required parameters
        self.experiment_setup = experiment_setup
        self.treatment_col = data_formatter.treatment_col
        self.id_col = data_formatter.id_col
        self.date_col = data_formatter.date_col
        self.metric_col = data_formatter.target_col
        self.std_min_bound = std_min_bound

        # Constants
        self.treatment_date_start = None
        self.treatment_date_end = None

        self.segments_stats = None
        self.global_stat = None
        self.treated_units_name = "target"
        self.intervention_col = "intervention"
        self.default_id_col = "id"
        self.default_date_col = "date"
        self.default_metric_col = "value"

        self.X_variables = None
        self.T_variable = None
        self.y_variable = None

    def _rename_treatment_units(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Rename treated units to be all called "treated", so that they are summarized together
        """
        return (
            data
            .with_columns(
                pl.when(
                    pl.col(self.treatment_col).is_in(self.experiment_setup.treated_groups)
                    ).then(pl.lit(self.treated_units_name)
                           ).otherwise(pl.col(self.treatment_col))
                           .alias(self.treatment_col)
                )
            )
    def _group_data(self, data: pl.DataFrame) -> None:
        """
        Group data based on the ID and date columns to remove duplicates (or to just regroup on a new granularity)
        """
        return (
            data
            .group_by([self.treatment_col, self.date_col])
            .agg(pl.col(self.metric_col).sum())
        )


    def _create_intervention_label(self, data):
      return (
          data
          .with_columns(pl.when(
              (pl.col(self.date_col) >= self.experiment_setup.treatment_start_date) & (pl.col(self.treatment_col) == pl.lit("target"))
              ).then(1).otherwise(0).alias(self.intervention_col))
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
            .group_by([self.treatment_col])
            .agg(
                pl.col(self.metric_col).mean().alias('avg'),
                pl.col(self.metric_col).std().alias('std')
            )
            .with_columns(
                pl.max_horizontal(pl.col("std"), pl.lit(self.std_min_bound)),
                pl.lit(1).alias('id_num')
            ) # as, must be done in two separates with_columns
            .with_columns(pl.col('id_num').sort_by("avg").cum_sum())
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

    def get_treated_stats(self):
        """
        Gets the mean and standard deviation of the treated units during the treatment period
        """
        stats = (
            self.segments_stats
            .filter(pl.col(self.treatment_col) == self.treated_units_name)
            .select(pl.col('avg'), pl.col('std'))
        )
        return stats.select(pl.col("avg")).item(), stats.select(pl.col("std")).item()

    def _apply_default_names(self, data: pl.DataFrame) -> None:
        return (
            data
            .rename({
                self.treatment_col: self.default_id_col,
                self.date_col: self.default_date_col,
                self.metric_col: self.default_metric_col
                })
        )


    def _filter_for_pretreatment(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Filter data to be only for the pre-treatment period
        """
        return X.filter(~(self.experiment_setup.treatment_dates))

    def _fill_missing_values(self, X: pl.DataFrame) -> pl.DataFrame:
        return X.fill_null(0).fill_nan(0)

    def fit(self, X: pl.DataFrame, y: pl.Series=None) -> None:
        """
        Use the provided that to extract parameters needed to transform the data
        """
        X = self._filter_for_pretreatment(X)
        X = self._fill_missing_values(X)
        X = self._rename_treatment_units(X)
        if not self._verify_uniqueness(X):
            X = self._group_data(X)
        self._set_mean_and_std(X)

    def post_processings(self, X: pl.DataFrame) -> pl.DataFrame:
      """
      Applies some last processing steps, depending on the required format, columns names the algorithms required
      """
      return X.rename(lambda column_name: column_name.replace(' ', ''))
    
    def store_variables(self, data: pl.DataFrame):
        self.X_variables = [col for col in data.columns if col not in ["intervention", self.default_id_col, self.default_date_col, self.default_metric_col]]
        self.T_variable = "intervention"
        self.y_variable = self.default_metric_col

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the data based on the parameters defined when initializing class and the parameters extracted during [fit]
        """
        X = self._rename_treatment_units(X)
        X = self._fill_missing_values(X)
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
        X = self.post_processings(X)
        self.store_variables(X)
        return X

    def fit_transform(self, X: pl.DataFrame, y: pl.Series=None) -> pl.DataFrame:
        """
        Apply fit on the data and then transform it
        """
        self.fit(X)
        return self.transform(X)

    def get_variables(self):
        """
        Return common causal variables X, treatment variable T and target variable y
        """
        return self.X_variables, self.T_variable, self.y_variable


class DoWhyPreProcessing(BasePreProcessing):
    """
    Preprocessing for GCM from DoWhy Package
    """
    def post_processings(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Applies some last processing steps, depending on the required format, columns names the algorithms required
        """
        return (
            X
            .drop(["id_num"])
            .rename(lambda column_name: column_name.replace(' ', ''))
        )

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the data based on the parameters defined when initializing class and the parameters extracted during [fit]
        """
        X = self._rename_treatment_units(X)
        X = self._fill_missing_values(X)
        if not self._verify_uniqueness(X):
            X = self._group_data(X)
        X = self._create_intervention_label(X)
        X = self._normalize_data(X)
        X = self._apply_default_names(X)
        X = self.post_processings(X)
        self.store_variables(X)
        return X


class SyntheticControlPreProcessing(BasePreProcessing):
    """
    Preprocessing for Synthetic Control using CausalPy with pymc models
    """
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
                .then((pl.col(self.metric_col) / pl.col('avg')))
                .otherwise((pl.col(self.metric_col) / pl.col('global_avg')))
                .alias(self.metric_col)
            )
            .drop(['avg', 'std', 'global_avg', 'global_std'])
        )
    def post_processings(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Applies some last processing steps, depending on the required format, columns names the algorithms required
        """
        return X.rename(lambda column_name: column_name.replace(' ', ''))
    

    

class SyntheticControlLinRegPreProcessing(BasePreProcessing):
    """
    Preprocessing for Synthetic Control using CausalPy with pymc models
    But for the one using LinRegression as base model instead of WeightedSumFitter
    """

    def post_processings(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Applies some last processing steps, depending on the required format, columns names the algorithms required
        """
        return X.rename(lambda column_name: column_name.replace(' ', ''))
    

    

class DifferenceInDifferencesPreProcessing(BasePreProcessing):
    """
    Preprocessing for Difference-In-Differences Algorithm from CausalPy with pymc models
    Limiting to only using the same features as Synthetic Control
    """

    def post_processings(self, X: pl.DataFrame) -> pl.DataFrame:
      """
      Applies some last processing steps, depending on the required format, columns names the algorithms required
      """
      return (
          X
          .rename(lambda column_name: column_name.replace(' ', ''))
          .rename({'id': 'unit'})
          .with_columns(
              pl.when(pl.col('date') >= self.experiment_setup.treatment_start_date).then(pl.lit(1)).otherwise(pl.lit(0))
              .alias('post_treatment'),
              pl.when(pl.col('unit') == 'target').then(pl.lit(1)).otherwise(pl.lit(0))
              .alias('treated_location'),
              (pl.col('date') - pl.col('date').min()).dt.total_days().alias('date')
          )
          .group_by(['date', 'unit', 'treated_location', 'post_treatment'])
          .agg(pl.col('value').mean().alias('value'))
      )

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the data based on the parameters defined when initializing class and the parameters extracted during [fit]
        """
        X = self._rename_treatment_units(X)
        X = self._fill_missing_values(X)
        if not self._verify_uniqueness(X):
            X = self._group_data(X)
        X = self._normalize_data(X)
        X = self._apply_default_names(X)
        X = self.post_processings(X)
        self.store_variables(X)
        return X


class DoublyRobustPreProcessing(BasePreProcessing):
    """
    Preprocessing for Doubly Robust Estimation
    Limiting to only using the same features as Synthetic Control
    """

    def post_processings(self, X: pl.DataFrame) -> pl.DataFrame:
      """
      Applies some last processing steps, depending on the required format, columns names the algorithms required
      """
      return (
          X
          .rename(lambda column_name: column_name.replace(' ', ''))
          .to_dummies(columns=['id'])
      )

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the data based on the parameters defined when initializing class and the parameters extracted during [fit]
        """
        X = self._rename_treatment_units(X)
        X = self._fill_missing_values(X)
        if not self._verify_uniqueness(X):
            X = self._group_data(X)
        X = self._create_intervention_label(X)
        X = self._normalize_data(X)
        X = self._apply_default_names(X)
        X = self.post_processings(X)
        self.store_variables(X)
        return X

class MetaLearnerPreProcessing(BasePreProcessing):
    """
    Preprocessing for Meta_learners from CausalML
    Limiting to only using the same features as Synthetic Control
    """

    def post_processings(self, X: pl.DataFrame) -> pl.DataFrame:
      """
      Applies some last processing steps, depending on the required format, columns names the algorithms required
      """
      return (
          X
          .with_columns(pl.col("id_num").alias("id"))
          .drop(["id_num"])
          .rename(lambda column_name: column_name.replace(' ', ''))
      )

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the data based on the parameters defined when initializing class and the parameters extracted during [fit]
        """
        X = self._rename_treatment_units(X)
        X = self._fill_missing_values(X)
        if not self._verify_uniqueness(X):
            X = self._group_data(X)
        X = self._create_intervention_label(X)
        X = self._normalize_data(X)
        X = self._apply_default_names(X)
        X = self.post_processings(X)
        self.store_variables(X)
        return X
    
class GranularMetaLearnerPreProcessing(BasePreProcessing):
    """
    Preprocessing for Meta_learners from CausalML
    Limiting to only using the same features as Synthetic Control
    """

    def _normalize_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize the values based on the mean and standard deviation
        previously extracted
        """
        return  (
            data
            .join(self.segments_stats, on=[self.treatment_col], how='left')
            .join(self.global_stat, how='cross')
            .drop(['avg', 'std', 'global_avg', 'global_std'])
        )

    def get_treated_stats(self):
        return 0, 1
        
    def post_processings(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Applies some last processing steps, depending on the required format, columns names the algorithms required
        """
        return (
            X
            .with_columns(pl.col("id_num").alias("id"))
            .drop(["id_num"])
            .rename(lambda column_name: column_name.replace(' ', ''))
        )

    def fit(self, X: pl.DataFrame, y: pl.Series=None) -> None:
        """
        Use the provided that to extract parameters needed to transform the data
        """
        X = self._filter_for_pretreatment(X)
        X = self._fill_missing_values(X)
        X = self._rename_treatment_units(X)
        # if not self._verify_uniqueness(X):
        #     X = self._group_data(X)
        self._set_mean_and_std(X)

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the data based on the parameters defined when initializing class and the parameters extracted during [fit]
        """
        X = self._rename_treatment_units(X)
        X = self._fill_missing_values(X)
        # if not self._verify_uniqueness(X):
        #     X = self._group_data(X)
        X = self._create_intervention_label(X)
        X = self._normalize_data(X)
        X = self._apply_default_names(X)
        X = self.post_processings(X)
        self.store_variables(X)
        return X
