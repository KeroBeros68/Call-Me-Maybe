import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.FileLoader.BaseLoader import BaseLoader, LoaderException
from src.utils.FileLoader.JSONLoader import JSONLoader


class TestJSONLoader:
    """
    Test suite for JSONLoader.

    Verifies that JSONLoader correctly reads and parses JSON files,
    and properly raises exceptions for invalid content, wrong MIME types,
    missing files, and insufficient permissions.

    python-magic is mocked via patch.object where MIME type validation
    is not the focus of the test, allowing isolated testing of file
    reading and parsing logic.
    """

    def setup_method(self) -> None:
        """
        Instantiate a fresh JSONLoader with a mocked logger before each test.
        """
        self.loader: BaseLoader = JSONLoader(MagicMock())

    def test_valid_json(self, tmp_path: Path) -> None:
        """Test that a valid JSON object is correctly read and parsed."""
        file = tmp_path / "test.json"
        file.write_text(json.dumps({"key": "value"}))

        with patch.object(self.loader, "check_type", return_value=True):
            result = self.loader.read_file(str(file))
        assert result == {"key": "value"}

    def test_valid_empty_json(self, tmp_path: Path) -> None:
        """Test that an empty JSON object ({}) is correctly read and parsed."""
        file = tmp_path / "test.json"
        file.write_text(json.dumps({}))

        with patch.object(self.loader, "check_type", return_value=True):
            result = self.loader.read_file(str(file))
        assert result == {}

    def test_valid_empty_json2(self, tmp_path: Path) -> None:
        """Test that an empty JSON array ([]) is correctly read and parsed."""
        file = tmp_path / "test.json"
        file.write_text(json.dumps([]))

        with patch.object(self.loader, "check_type", return_value=True):
            result = self.loader.read_file(str(file))
        assert result == []

    def test_invalid_json_content(self, tmp_path: Path) -> None:
        """
        Test that a file with malformed JSON content raises a JSONDecodeError.
        """
        file = tmp_path / "bad.json"
        file.write_text("not valid json {{{")

        with patch.object(self.loader, "check_type", return_value=True):
            with pytest.raises(json.JSONDecodeError):
                self.loader.read_file(str(file))

    def test_invalid_mime_type(self, tmp_path: Path) -> None:
        """
        Test that a file with an invalid MIME type raises a
        json.JSONDecodeError.
        """
        file = tmp_path / "bad.txt"
        file.write_text("not valid json {{{")

        with pytest.raises(LoaderException):
            self.loader.read_file(str(file))

    def test_file_not_found(self) -> None:
        """Test that a non-existent file path raises a FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.loader.read_file("test.json")

    def test_permission_error(self, tmp_path: Path) -> None:
        """
        Test that a file with no read permissions raises a PermissionError.

        Skipped when running as root, as root bypasses file permission checks.
        """
        if os.getuid() == 0:
            pytest.skip("Running as root, permission test irrelevant")
        file = tmp_path / "protected.json"
        file.write_text(json.dumps({"key": "value"}))
        file.chmod(0o000)

        with patch.object(self.loader, "check_type", return_value=True):
            with pytest.raises(PermissionError):
                self.loader.read_file(str(file))
