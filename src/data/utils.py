from pathlib import Path
import os
from typing import Protocol

class CreatePathIfNotExists(Protocol):
    """Protocol to for classes that implement create_path_if_not_exists()"""
    def create_path_if_not_exists(self, path: Path) -> None:
        """Check if a directory exists. Creates it, if it doesn't"""

class CreateLocalDirectoryIfNotExists:
    """Mixin to create a directory if it doesn't exists"""
    def create_path_if_not_exists(self, path: Path) -> None:
        if not os.path.exists(path):
            os.makedirs(path)