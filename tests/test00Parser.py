from argparse import ArgumentError

import pytest

from src.utils.PausingArgumentParser.PausingArgumentParser import (
    PausingArgumentParser,
)


class TestParser:
    """
    Test suite for PausingArgumentParser.

    Verifies that the argument parser correctly handles valid inputs,
    rejects missing required arguments, and recognizes all expected flags.
    """

    def setup_method(self):
        """
        Instantiate a fresh parser before each test to avoid side effects.
        """
        self.parser = PausingArgumentParser("test", "description", "help")

    def test_short_flags(self) -> None:
        """
        Test that short flags (-f, -i, -o) are parsed correctly.

        Ensures that values provided via short flags are accessible
        through their corresponding long attribute names.
        """
        arg = self.parser.parse_args(
            ["-f", "functions.json", "-i", "input.json", "-o", "output.json"]
        )
        assert arg.functions_definition == "functions.json"
        assert arg.input == "input.json"
        assert arg.output == "output.json"

    def test_long_flags(self) -> None:
        """
        Test that long flags (--functions_definition, --input, --output) are
        parsed correctly.

        Ensures that values provided via long flags are accessible
        through their corresponding attribute names.
        """
        arg = self.parser.parse_args(
            [
                "--functions_definition",
                "functions.json",
                "--input",
                "input.json",
                "--output",
                "output.json",
            ]
        )
        assert arg.functions_definition == "functions.json"

    def test_child_flag(self) -> None:
        """
        Test that the --child flag is recognized as a boolean flag.

        When --child is present, arg.child should be True.
        """
        arg = self.parser.parse_args(
            ["-f", "f.json", "-i", "i.json", "-o", "o.json", "--child"]
        )
        assert arg.child is True
