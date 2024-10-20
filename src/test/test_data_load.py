from ..data.load import KaggleDataPuller, LocalDataLoader
from ..data.utils import CreateLocalDirectoryIfNotExists
import polars as pl
import os
import shutil


def test_kaggle_api():
    """Tests if methods can create local directories"""
    local_path = "./pytest_kaggle_api"
    dataset = 'fedesoriano/heart-failure-prediction' # very small but highly maintained dataset
    data_name = 'heart.csv'
    CreateLocalDirectoryIfNotExists().create_path_if_not_exists(local_path)
    assert len(os.listdir(local_path)) == 0, "Folder is not empty, so test won't properly be able to validate execution"
    
    KaggleDataPuller(dataset).pull_data(local_path)
    assert len(os.listdir(local_path)) > 0, "Error: No data was downloaded"
    assert data_name in os.listdir(local_path), f"Error: cannot find {data_name}"

    shutil.rmtree(local_path)
    assert not os.path.exists(local_path), "Error: Temporary folder for testing Kaggle API was not deleted. This will affect future tests"



def test_local_data_loader():
    """Test methods to load local data"""
    
    data_types = ['csv', 'json', 'ipc',]
    data_path = os.getcwd() + "/src/test/data/load_data."
    for data_type in data_types:
        data = LocalDataLoader().load_local_data(data_path=data_path + data_type, data_format=data_type)
        assert isinstance(data, pl.DataFrame), f"Not able to load {data_type} format file"
        assert data.shape == (10, 4), "Data loaded didn't have the expected number of columns and/or columns"