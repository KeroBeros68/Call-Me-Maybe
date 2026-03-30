from abc import ABC, abstractmethod
from logging import Logger
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
    def check_type(filepath: str, expected: str) -> bool:
        mime = magic.from_file(filepath, mime=True)
        return mime == expected
