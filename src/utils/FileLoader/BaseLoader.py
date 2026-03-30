from abc import ABC, abstractmethod
from logging import Logger
import os
from typing import Any
import magic


class LoaderException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class BaseLoader(ABC):
    def __init__(self, logger: Logger) -> None:
        """Initializes the BaseLoader with a logger."""
        self.logger = logger

    @abstractmethod
    def read_file(self, path: str) -> Any:
        pass

    @staticmethod
    def check_type(file_path: str, expected: str) -> bool:
        real_path = os.path.realpath(file_path)
        mime = magic.from_file(real_path, mime=True)
        valid = {"application/json", "text/plain"}
        return mime in valid
