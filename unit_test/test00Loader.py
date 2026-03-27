import pytest

from src.FileLoader import FileLoader


class TestLoader:
    loader = FileLoader()

    def test_no_file(self) -> None:
        try:
            self.loader.read_file("")
            pytest.fail("INVALID[ No file ]")
        except FileNotFoundError:
            pass
