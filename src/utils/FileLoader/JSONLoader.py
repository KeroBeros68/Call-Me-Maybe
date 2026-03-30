import json
from logging import Logger
from typing import Any

from .BaseLoader import BaseLoader, LoaderException


class JSONLoader(BaseLoader):
    """Utility class to load the contents of a JSON file from disk."""

    def __init__(self, logger: Logger) -> None:
        """Initializes the JSONLoader with a logger."""
        self.logger = logger

    def read_file(self, path: str) -> Any:
        """
        Reads and returns the full text content of a file.

        Args:
            path (str): Absolute or relative path to the file to read.

        Returns:
            str: The plain text content of the file.

        Raises:
            FileNotFoundError: If no file exists at the given path.
            PermissionError: If the file cannot be read due to permissions.
        """
        self.logger.info(f"File to open and read: '{path}'")
        if not self.check_type(path, "application/JSON"):
            raise LoaderException("ERROR: Invalide MIME type")
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            raise
