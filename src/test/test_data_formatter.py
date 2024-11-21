from ..data.data_formatter import BenchmarkDataFormat, MONTH_DISCRETIZATION


def test_benchmark_data_format():
    """Test methods to the methods that the class BenchmarkDataFormat must have"""
    feature_cols =  ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
    data_format = BenchmarkDataFormat("Store", "Region", "Date", "Weekly_Sales", feature_cols, '%d-%m-%Y', MONTH_DISCRETIZATION)
    assert data_format.get_unique_id_col() == 'Store', f"Method get_unique_id_col() didn't return expected value. Expected 'Store' got {data_format.get_unique_id_col()}"
    assert data_format.get_treatment_discriminator_col() == 'Region', f"Method get_treatment_discriminator_col() didn't return expected value. Expected 'Region' got {data_format.get_treatment_discriminator_col()}"
    assert data_format.get_date_col() == 'Date', f"Method get_date_col() didn't return expected value. Expected 'Date' got {data_format.get_date_col()}"
    assert data_format.get_target_col() == 'Weekly_Sales', f"Method get_target_col() didn't return expected value. Expected 'Weekly_Sales' got {data_format.get_target_col()}"
    assert data_format.get_feature_cols() ==  feature_cols, f"Method get_feature_cols() didn't return expected value. Expected f'{feature_cols}' got {data_format.get_feature_cols()}"
    assert data_format.get_date_format() == '%d-%m-%Y', f"Method get_date_format() didn't return expected value. Expected '%d-%m-%Y' got {data_format.get_date_format()}"
    assert data_format.get_date_discretization() == MONTH_DISCRETIZATION, f"Method get_date_discretization() didn't return expected value. Expected f'{MONTH_DISCRETIZATION}' got {data_format.get_date_discretization()}"
