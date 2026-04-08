import json
from logging import Logger
from pathlib import Path
from typing import Any

from .BaseLoader import BaseLoader, LoaderException


class JSONLoader(BaseLoader):
    """Utility class to load the contents of a JSON file from disk."""

    def __init__(self, logger: Logger) -> None:
        """Initializes the JSONLoader with a logger."""
        super().__init__(logger)

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
        if not self.check_type(path, "application/json"):
            raise LoaderException("ERROR: Invalide MIME type")
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            raise

    def write_file(self, output_files: str, content: Any) -> None:
        output_path = Path(output_files)

        folder_parent = output_path.parent

        folder_parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_files, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=4)
        except Exception:
            raise
