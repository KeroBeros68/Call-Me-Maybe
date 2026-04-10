import argparse


class PausingArgumentParser(argparse.ArgumentParser):
    """Argument parser that does not call ``sys.exit`` on parse errors.

    Extends :class:`argparse.ArgumentParser` with ``exit_on_error=False``
    and pre-registers all CLI arguments used by the application.
    """
    def __init__(self, name: str, description: str, epilog: str) -> None:
        """Create the parser with program metadata and register arguments.

        Args:
            name (str): Program name shown in the usage line.
            description (str): Short description shown above the help text.
            epilog (str): Text shown at the bottom of the help output.
        """
        super().__init__(
            exit_on_error=False,
            prog=name,
            description=description,
            epilog=epilog,
        )
        self._add_arguments()

    def _add_arguments(self) -> None:
        """Register all supported CLI arguments with the parser.

        Adds hidden flags (``--child``, ``--gui``) used for subprocess
        spawning, plus the user-facing options for function definitions
        path (``-f``), input path (``-i``), output path (``-o``), and
        model name (``-mn``).
        """
        self.add_argument(
            "--child", action="store_true", help=argparse.SUPPRESS
        )
        self.add_argument("--gui", action="store_true", help=argparse.SUPPRESS)
        self.add_argument(
            "-f",
            "--functions_definition",
            default="data/input/functions_definition.json",
        )
        self.add_argument(
            "-i",
            "--input",
            default="data/input/function_calling_tests.json",
        )
        self.add_argument(
            "-o", "--output", default="data/output/output.json"
        )

        self.add_argument(
            "-mn", "--model_name", default="Qwen/Qwen3-0.6B"
        )
