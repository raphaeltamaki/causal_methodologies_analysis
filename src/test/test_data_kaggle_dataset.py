from ..data.kaggle_datasets import KaggleBenchmarkDataset


def test_kaggle_dataset():
    """Test methods to load local data"""
    test_dataset = KaggleBenchmarkDataset('MyDatasetName', 'MyDatasetFormat', 'MyDatasetMainFile', 'MyDatasetAddressInKaggle')
    assert test_dataset.dataset_name() == 'MyDatasetName', "Method dataset_name() didn't return expected value"
    assert test_dataset.data_format() == 'MyDatasetFormat', "Method data_format() didn't return expected value"
    assert test_dataset.main_file_name() == 'MyDatasetMainFile', "Method main_file_name() didn't return expected value"
    assert test_dataset.kaggle_dataset_address() == 'MyDatasetAddressInKaggle', "Method kaggle_dataset_address() didn't return expected value"