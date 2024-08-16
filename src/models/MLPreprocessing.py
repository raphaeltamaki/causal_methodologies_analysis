import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import List


MISSING_CATEGORICAL_VALUE_CONST = "missing_cat_value"
UNSEEN_CATEGORICAL_VALUE_CONST = -1


class SpecialCharactersCleaner(BaseEstimator, TransformerMixin):
    """
    Simple sklearn Transformer to remove special characters on all non-numeric columns of a data frame
    """

    @staticmethod
    def _remove_special_chars(string_to_clean: str):
        """Remove any special character from string"""
        return "".join(e for e in string_to_clean if e.isalnum())

    def fit(self, X: pd.DataFrame, y=None):
        """
        Pass, doesn't do anything
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove any special character from the categorical variables of the dataset
        """
        non_numeric_cols = [
            column for column in list(X.columns) if not is_numeric_dtype(X[column])
        ]
        for column in non_numeric_cols:
            X[column] = X[column].astype(str).apply(self._remove_special_chars)

        return X


class TreeBasedModelPreprocessing(BaseEstimator, TransformerMixin):
    """
    Base class to apply basic preprocessing (no feature creation) to tree based models, such as RandomForestRegressor from sklearn or CatBoostRegressor from catboost
    This clas just structures the processing executed by its children in
    1) processing numerical variables
    2) processing non-numerical variables
    """

    def __init__(
        self,
        cat_na_fill: str = MISSING_CATEGORICAL_VALUE_CONST,
        numeric_na_fill_strategy=None,
        numeric_na_fill_constant: float = -1,
        numeric_na_fill_multiplier: float = 1000,
    ) -> None:

        super().__init__()

        # parameters defined from inputs
        self.cat_na_fill = cat_na_fill
        self.numeric_na_fill_strategy = numeric_na_fill_strategy
        self.numeric_na_fill_constant = numeric_na_fill_constant
        self.numeric_na_fill_multiplier = numeric_na_fill_multiplier

        # declaration of variables defined in other methods
        self.numeric_cols = None
        self.non_numeric_cols = None

    def _numeric_na_fill(self, data: pd.Series):
        """
        Defines how to numeric columns are going to have their NA filled, when we have some data to support it
        Uses user-defined method if provided, else estimates a numeric value that is 'far enough' from any other
        value in the data, so that the Tree can consider it as it's own class
        """

        if self.numeric_na_fill_strategy:
            return self.numeric_na_fill_strategy(data)
        else:
            return np.abs(np.max(data)) * self.numeric_na_fill_multiplier

    @staticmethod
    def _divide_columns_whether_numeric(data: pd.DataFrame):
        non_numeric_cols = [
            column
            for column in list(data.columns)
            if not is_numeric_dtype(data[column])
        ]
        numeric_cols = [
            column for column in list(data.columns) if is_numeric_dtype(data[column])
        ]
        return numeric_cols, non_numeric_cols

    def fit(self, X: pd.DataFrame, y=None) -> None:
        self.numeric_cols, self.non_numeric_cols = self._divide_columns_whether_numeric(
            X
        )
        return self

    def _non_numerical_data_processing(
        self, data: pd.DataFrame, non_numerical_cols: list
    ):
        raise NotImplementedError

    def _numerical_data_processing(self, data: pd.DataFrame, numerical_cols: list):
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        For the numeric variables, fill missing values with the (_numeric_na_fill) or constant value.
        For categorical variables, cast them as 'categorical', encode them, and then
        remove the original columns
        """
        X = self._numerical_data_processing(X.copy(), self.numeric_cols)
        X = self._non_numerical_data_processing(X.copy(), self.non_numeric_cols)

        return X


class RandomForestPreprocessing(TreeBasedModelPreprocessing):
    """
    Class to apply basic preprocessing (no feature creation) so that the data can be used in sklearn.ensemble classes
    The processing of the data executed by this clas consists of
    1) casting all numeric columns to float
    2) fill NA values of numeric columns using some method to fill missing values (ex: using median()).Default method for filling missing values is 100*max(), so that they can in practice be interpreted as a different value.
       Method needs to take a pd.Series and return a single value
    3) if no NA fill is possible (ex: all values are NULL), then fill with a constant value (ex: -1).
    4) cast all non-numeric columns to "category"
    5) create columns that contains the encoded category, create a mapping of categories code to original value, and drop original column from dataframe
    6) fill NA with a special method (ex: mode(), median()) or with a constant value (ex: MISSING_CATEGORICAL_VALUE_CONST)
    """

    def __init__(
        self,
        cat_na_fill: str = MISSING_CATEGORICAL_VALUE_CONST,
        numeric_na_fill_strategy=None,
        numeric_na_fill_constant: float = -1,
        unseen_cat_fill: int = UNSEEN_CATEGORICAL_VALUE_CONST,
    ) -> None:

        super().__init__(
            cat_na_fill, numeric_na_fill_strategy, numeric_na_fill_constant
        )

        # map to connect enconded categories to their original values
        self.categorical_labels_map = {}
        self.unseen_cat_fill = unseen_cat_fill

    def _create_category_map(
        self, data: pd.DataFrame, non_numerical_cols: List[str]
    ) -> None:
        """
        Create the mapping that maps each category 'original' value to its categorical code (and the inverse)
        """
        for column in non_numerical_cols:
            cat_values = data[column].copy().fillna(self.cat_na_fill)
            codes = cat_values.astype("category").cat.codes.unique()
            values = cat_values.unique()
            self.categorical_labels_map[column] = {
                "value_to_code": dict(zip(values, codes)),
                "code_to_value": dict(zip(codes, values)),
            }

    def get_cat_code(self, column: str, value: str):
        return self.categorical_labels_map[column]["value_to_code"][value]

    def get_code_cat(self, column: str, code: str):
        return self.categorical_labels_map[column]["code_to_value"][code]

    def _encode_cat_column(self, data: pd.DataFrame, column: str) -> pd.Series:
        return (
            data[column]
            .map(self.categorical_labels_map[column]["value_to_code"])
            # .fillna(-1)
        )  # map categories and fill if the category is new

    def _decode_cat_column(self, data: pd.DataFrame, column: str) -> pd.Series:
        return data[column].map(
            self.categorical_labels_map[column]["code_to_value"],
            na_action=self.categorical_labels_map[column]["value_to_code"][
                self.cat_na_fill
            ],
        )

    def fit(self, X: pd.DataFrame, y=None) -> None:
        super().fit(X, y)
        self._create_category_map(
            X, self.non_numeric_cols
        )  # self.non_numeric_cols defined on the super().fit(X, y)
        return self

    def _non_numerical_data_processing(
        self, data: pd.DataFrame, non_numerical_cols: list
    ):
        """
        For all non-numerical columns (defined by non_numeric_cols) in the inputted dataframe
        fill all NA values with previously defined value (default: MISSING_CATEGORICAL_VALUE_CONST)
        and then cast all as 'categpry' type.
        Afterwards, encode the categories and creates a mapping of the codes to the original values
        """

        for column in non_numerical_cols:
            data[column] = data[column].fillna(self.cat_na_fill)
            data[column] = data[column].astype(
                "category"
            )  # We transform as category type
            data[column] = self._encode_cat_column(data, column)

        return data

    def _numerical_data_processing(self, data: pd.DataFrame, numerical_cols: list):
        """
        For all numerical columns (defined by numeric_cols) in the inputted dataframe
        - Cast them as "float32"
        - Try to fill NA values using provided method from input [numeric_na_fill_strategy]
        - If there are still NA values, fill them with a custom constant
        """

        for column in numerical_cols:
            data[column] = data[column].astype("float32")
            data[column] = data[column].fillna(self._numeric_na_fill(data[column]))
            data[column] = data[column].fillna(self.numeric_na_fill_constant)

        return data


class BoostingPreprocessing(TreeBasedModelPreprocessing):
    """
    Class to preprocess the raw data before it can be used in a boosting method such as XGBoost, CatBoost or LightGBM
    """

    def __init__(
        self,
        cat_na_fill: str = MISSING_CATEGORICAL_VALUE_CONST,
        numeric_na_fill_strategy=None,
        numeric_na_fill_constant: float = -1,
    ) -> None:

        super().__init__(
            cat_na_fill, numeric_na_fill_strategy, numeric_na_fill_constant
        )

    def _non_numerical_data_processing(
        self, data: pd.DataFrame, non_numerical_cols: list
    ):
        """
        For all non-numerical columns (defined by non_numeric_cols) in the inputted dataframe
        fill all NA values with previously defined value (default: MISSING_CATEGORICAL_VALUE_CONST)
        and then cast all as 'categpry' type
        """

        for column in non_numerical_cols:
            data[column] = data[column].fillna(self.cat_na_fill)
            data[column] = data[column].astype("category")

        return data

    def _numerical_data_processing(self, data: pd.DataFrame, numerical_cols: list):
        """
        For all numerical columns (defined by numeric_cols) in the inputted dataframe
        - Cast them as "float32"
        - Try to fill NA values using provided method from input [numeric_na_fill_strategy]
        - If there are still NA values, fill them with a custom constant
        """

        for column in numerical_cols:
            data[column] = data[column].astype("float32")
            data[column] = data[column].fillna(self._numeric_na_fill(data[column]))
            data[column] = data[column].fillna(self.numeric_na_fill_constant)

        return data


class LinearRegressionPreprocessing(BaseEstimator, TransformerMixin):
    """
    Class to apply necessary preprocessing to run a linear regression.
    This class functionality consists of
    - 'Fitting' the data to know how to normalize continuous variables
    -
    """

    def __init__(
        self,
        numeric_imputer: SimpleImputer = SimpleImputer(strategy="median"),
        categorical_imputer: SimpleImputer = SimpleImputer(
            strategy="constant", fill_value=MISSING_CATEGORICAL_VALUE_CONST
        ),
        scaler: StandardScaler = StandardScaler(),
        str_cleaner: TransformerMixin = SpecialCharactersCleaner(),
        one_hot_encoder: OneHotEncoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=10
        ),
    ):
        """
        Class to apply the basic but necessary data preprocessing to train linear models such as Lasso, Ridge, LinearRegressor, etc
        This class basically defines a sklearn.pipeline.Pipelien to separarely process numerical and non-numerical columns
        - Numerical:
            - Inputs missing values with some strategy. Default: median
            - Normalize features

        - Non-numerical
            - Inputs missing value with some strategy. Default is a constant value MISSING_CATEGORICAL_VALUE_CONST
            - One-hot-encode

        The class is thus basically a wrapper to call that Pipeline
        """

        super().__init__()
        self.numeric_imputer = numeric_imputer
        self.categorical_imputer = categorical_imputer
        self.scaler = scaler
        self.str_cleaner = str_cleaner
        self.one_hot_encoder = one_hot_encoder

        self.num_pipeline = Pipeline(
            [("imputer", self.numeric_imputer), ("scaler", self.scaler)]
        )
        self.cat_pipeline = Pipeline(
            [
                ("cleaner", self.str_cleaner),
                ("imputer", self.categorical_imputer),
                ("one_hot_encoder", self.one_hot_encoder),
            ]
        )

        self.pipeline = None
        self.non_numeric_cols = None
        self.numeric_cols = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the components of preprocessing pipeline to the data. The pieleine consists of the numerical and non-numerical parts
        Numerical
            1) Fill missing values with the median value
            2) Standard scale features
        Categorical
            1) Clean the string to remove any possible special character
            2) One-hot encode
        """

        self.non_numeric_cols = [
            c for c in list(X.columns) if not (is_numeric_dtype(X[c]))
        ]
        self.numeric_cols = list(set(X.columns) - set(self.non_numeric_cols))

        self.pipeline = ColumnTransformer(
            [
                ("numerical", self.num_pipeline, self.numeric_cols),
                ("categorical", self.cat_pipeline, self.non_numeric_cols),
            ]
        )
        self.pipeline.fit(X)

        filled_X = self.numeric_imputer.fit_transform(X[self.numeric_cols])
        self.scaler.fit(filled_X)
        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply preprocessing transformations to the X using the class pipeline
        """
        return self.pipeline.transform(X)
