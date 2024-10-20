import os
from ..data.utils import CreateLocalDirectoryIfNotExists


def test_directory_creation():
    """Tests if methods can create local directories"""
    test_path = "./pytest_path_creation_test"
    assert not os.path.exists(
        test_path
    ), "Directory for test already exists, it should not, otherwise the test is give false positives"
    CreateLocalDirectoryIfNotExists().create_path_if_not_exists(test_path)
    assert os.path.exists(
        test_path
    ), "Directory was not created, or not created at the correct place"
    os.rmdir(test_path)
