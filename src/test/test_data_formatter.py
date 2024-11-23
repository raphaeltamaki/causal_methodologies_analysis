from datetime import datetime
import pytest
import polars as pl
from ..data.data_formatter import BenchmarkDataFormat, DataFormatter, DAY_DISCRETIZATION, MONTH_DISCRETIZATION, YEAR_DISCRETIZATION

@pytest.fixture
def sample_data():
    return pl.DataFrame({
        "id": ["&%^48230895", "$%^%d2*9()", "3"],
        "treatment": ["A", "B", "A"],
        "date": ["2023-01-01", "2023-02-21", "2024-03-31"],
        "target": [100, 200, 300]
    })

@pytest.fixture
def benchmark_data_format():
    return BenchmarkDataFormat(
        unique_id_col="id",
        treatment_discriminator_col="treatment",
        date_col="date",
        target_col="target",
        feature_cols=["extra_column_a", "extra_column_b"],
        date_format="%Y-%m-%d",
        date_discretization=MONTH_DISCRETIZATION
    )

@pytest.fixture
def formatter(benchmark_data_format):
    return DataFormatter(benchmark_data_format)

def test_get_unique_id_col(benchmark_data_format):
    assert benchmark_data_format.get_unique_id_col() == "id"

def test_get_treatment_discriminator_col(benchmark_data_format):
    assert benchmark_data_format.get_treatment_discriminator_col() == "treatment"

def test_get_date_col(benchmark_data_format):
    assert benchmark_data_format.get_date_col() == "date"

def test_get_target_col(benchmark_data_format):
    assert benchmark_data_format.get_target_col() == "target"

def test_get_date_discretization(benchmark_data_format):
    assert benchmark_data_format.get_date_discretization() == MONTH_DISCRETIZATION

def test_get_date_format(benchmark_data_format):
    assert benchmark_data_format.get_date_format() == "%Y-%m-%d"

def test_get_feature_cols(benchmark_data_format):
    assert benchmark_data_format.get_feature_cols() == ["extra_column_a", "extra_column_b"]

def test_format_date_col_type(sample_data, formatter):
    formatted_data = formatter.format_date_col_type(sample_data)
    assert isinstance(formatted_data.select('date').dtypes[0], pl.Datetime)

def test_discretize_date_yearly(sample_data, formatter):
    formatter.data_format.date_discretization = YEAR_DISCRETIZATION # overwrite format to yearly
    formatted_data = formatter.format_date_col_type(sample_data)
    formatted_data = formatter.discretize_date(formatted_data)
    assert formatted_data['date'].to_list() == [datetime(2023, 1, 1),
                                                datetime(2023, 1, 1),
                                                datetime(2024, 1, 1)
                                                ]

def test_discretize_date_monthly(sample_data, formatter):
    formatted_data = formatter.format_date_col_type(sample_data)
    formatted_data = formatter.discretize_date(formatted_data)
    assert formatted_data['date'].to_list() == [datetime(2023, 1, 1),
                                                datetime(2023, 2, 1),
                                                datetime(2024, 3, 1)
                                                ]
    
def test_discretize_date_daily(sample_data, formatter):
    formatter.data_format.date_discretization = DAY_DISCRETIZATION # overwrite format to daily
    formatted_data = formatter.format_date_col_type(sample_data)
    formatted_data = formatter.discretize_date(formatted_data)
    assert formatted_data['date'].to_list() == [datetime(2023, 1, 1),
                                                datetime(2023, 2, 21),
                                                datetime(2024, 3, 31)
                                                ]

# def test_format_id_col(sample_data, formatter):
    formatted_data = formatter.format_id_col(sample_data)
    assert formatted_data["id"].to_list() == ["48230895", "d29", "3"]

def test_pipeline(sample_data, formatter):
    processed_data = formatter.pipeline(sample_data)
    assert processed_data.select(pl.col("date")).dtypes[0] == pl.Datetime
    assert processed_data["id"].to_list() == ["48230895", "d29", "3"]