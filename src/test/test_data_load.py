import os
import shutil
import polars as pl
import pytest
from ..data.load import KaggleDataPuller, LocalDataLoader
from ..data.utils import CreateLocalDirectoryIfNotExists

@pytest.fixture
def kaggle_test_directory(tmp_path):
    """Fixture to create and clean up a temporary directory for Kaggle API tests."""
    local_path = tmp_path / "pytest_kaggle_api"
    CreateLocalDirectoryIfNotExists().create_path_if_not_exists(str(local_path))
    yield local_path
    shutil.rmtree(str(local_path), ignore_errors=True)

def test_kaggle_api(kaggle_test_directory):
    """Tests if Kaggle API can download data and create local directories."""
    dataset = 'fedesoriano/heart-failure-prediction'
    data_name = 'heart.csv'
    local_path = kaggle_test_directory

    assert len(os.listdir(local_path)) == 0, "Folder is not empty, so test won't properly be able to validate execution"

    KaggleDataPuller().pull_kaggle_data(str(local_path), dataset)

    assert len(os.listdir(local_path)) > 0, "Error: No data was downloaded"
    assert data_name in os.listdir(local_path), f"Error: cannot find {data_name}"

def test_local_data_loader():
    """Test methods to load local data in various formats."""
    data_types = ['csv', 'json', 'ipc']
    data_path = os.path.join(os.getcwd(), "src", "test", "data", "load_data.")

    for data_type in data_types:
        data = LocalDataLoader().load_local_data(data_path=data_path + data_type, data_format=data_type)
        assert isinstance(data, pl.DataFrame), f"Not able to load {data_type} format file"
        assert data.shape == (10, 4), "Data loaded didn't have the expected number of columns and/or rows"