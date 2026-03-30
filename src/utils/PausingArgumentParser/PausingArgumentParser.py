import argparse


class PausingArgumentParser(argparse.ArgumentParser):
    def __init__(self, name: str, description: str, epilog: str) -> None:
        super().__init__(
            exit_on_error=False,
            prog=name,
            description=description,
            epilog=epilog,
        )
        self._add_arguments()

    def _add_arguments(self) -> None:
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
            default="data/input/functions_calling_tests.json",
        )
        self.add_argument(
            "-o", "--output", default="data/input/output.json"
        )
