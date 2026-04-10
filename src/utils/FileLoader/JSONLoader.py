import json
from logging import Logger
import os
from pathlib import Path
from typing import Any

import magic

from .BaseLoader import BaseLoader, LoaderException


class JSONLoader(BaseLoader):
    """Utility class to load the contents of a JSON file from disk."""

    def __init__(self, logger: Logger) -> None:
        """Initializes the JSONLoader with a logger."""
        super().__init__(logger)

    def read_file(self, path: str) -> Any:
        """
        Parse and return the content of a JSON file.

        Args:
            path (str): Absolute or relative path to the ``.json`` file.

        Returns:
            Any: The deserialised Python object (dict, list, etc.).

        Raises:
            LoaderException: If the file's MIME type is not JSON or plain text.
            FileNotFoundError: If no file exists at the given path.
            PermissionError: If the file cannot be read due to permissions.
            json.JSONDecodeError: If the file contents are not valid JSON.
        """
        self.logger.info(f"File to open and read: '{path}'")
        if not self.check_type(path, ["application/json", "text/plain"]):
            raise LoaderException("ERROR: Invalide MIME type")
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            raise

    def write_file(self, output_files: str, content: Any) -> None:
        """Serialise ``content`` as indented JSON and write it to disk.

        Creates any missing parent directories before writing.

        Args:
            output_files (str): Destination file path (will be created or
                overwritten).
            content (Any): JSON-serialisable Python object to persist.

        Raises:
            Exception: Any I/O error raised by :func:`open` or
                :func:`json.dump` is re-raised unchanged.
        """
        output_path = Path(output_files)

        folder_parent = output_path.parent

        folder_parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_files, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=4)
        except Exception:
            raise

    def check_type(self, file_path: str, expected: list[str]) -> bool:
        """Verify that ``file_path`` has a ``.json`` extension and an
        accepted MIME type.

        Resolves the path to its real location (following symlinks) before
        checking both the file extension and the MIME type reported by
        ``libmagic``.

        Args:
            file_path (str): Path to the file to inspect.
            expected (list[str]): Acceptable MIME type strings (e.g.
                ``["application/json", "text/plain"]``).

        Returns:
            bool: ``True`` if the extension is ``.json`` and the MIME type
                is contained in ``expected``, ``False`` otherwise.
        """
        real_path = os.path.realpath(file_path)

        if not real_path.lower().endswith('.json'):
            return False

        mime = magic.from_file(real_path, mime=True)
        return mime in expected
