from abc import ABC, abstractmethod
from logging import Logger
from typing import Any


class LoaderException(Exception):
    """Raised when a file-loader operation fails."""

    def __init__(self, message: str) -> None:
        """Initialize with an error message.

        Args:
            message (str): Human-readable description of the failure.
        """
        super().__init__(message)


class BaseLoader(ABC):
    """Abstract base class for file-loading utilities.

    Defines the interface that all concrete loaders must implement:
    reading a file, writing a file, and checking a file's MIME type.
    """
    def __init__(self, logger: Logger) -> None:
        """Initializes the BaseLoader with a logger."""
        self.logger = logger

    @abstractmethod
    def read_file(self, path: str) -> Any:
        """Read and return the content of a file at ``path``.

        Args:
            path (str): Absolute or relative path to the file.

        Returns:
            Any: The parsed or raw file content.
        """
        pass

    @abstractmethod
    def write_file(self, output_files: str, content: Any) -> None:
        """Serialise ``content`` and write it to ``output_files``.

        Args:
            output_files (str): Destination file path.
            content (Any): Data to serialise and persist.
        """
        pass

    @abstractmethod
    def check_type(self, file_path: str, expected: list[str]) -> bool:
        """Verify that the file at ``file_path`` has an expected MIME type.

        Args:
            file_path (str): Path to the file to inspect.
            expected (list[str]): Acceptable MIME type strings.

        Returns:
            bool: ``True`` if the file's MIME type is in ``expected``,
                ``False`` otherwise.
        """
        pass
