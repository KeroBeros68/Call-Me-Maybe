"""Entry point for the Call Me Maybe application.

Handles environment bootstrapping, argument parsing, and delegates
execution to the Controller. When run directly (not as a child process),
spawns a new terminal window and re-launches itself as a child.
"""

import logging
import os
import subprocess
import sys
import time

from src.utils.RunSecurity import RunSecurity, RunEnvironmentError
from src.utils.Logger.Logger import setup_logger

TERMINAL: list[str] = ["gnome-terminal", "--"]
# TERMINAL: list[str] = ["konsole", "-e"]

PROG_NAME: str = "Call Me Maybe"
PROG_DESCRIPTION: str = "What the program does"  # a faire
PROG_HELP: str = "Text at the bottom of help"  # a faire


def main() -> None:
    """Run the main application flow.

    Sets up the runtime security check, parses CLI arguments, loads the
    LLM and its supporting components, then starts the generation loop
    via the Controller.  On any unrecoverable error the user is prompted
    to press Enter before the process exits.
    """
    logger = logging.getLogger(PROG_NAME)
    secure_env = RunSecurity()
    try:
        secure_env.check_process()
        time.sleep(0.2)
    except RunEnvironmentError as e:
        logger.error(f"{e}")
        logger.info("Programm exit")
        input("\n\nPress Enter to exit...")
        return
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        input("\n\nPress Enter to exit...")
        return

    try:
        from .Controller import Controller
        from .ConstrainedGenerator import ConstrainedGenerator
        from src.llm_custom import LLMCustom
        from src.utils import PausingArgumentParser

        from src.utils.FileLoader.JSONLoader import JSONLoader

        reader = JSONLoader(logger)
        parser = PausingArgumentParser(PROG_NAME, PROG_DESCRIPTION, PROG_HELP)
        cli_arg = parser.parse_args()
        llm = LLMCustom(reader=reader, model_name=cli_arg.model_name)
        controller = Controller(
            logger,
            parser,
            reader,
            llm,
            ConstrainedGenerator(llm)
        )
        controller.process()

    except Exception as e:
        logger.error(f"ERROR: {e}")
    input("\n\nPress Enter to exit...")
    return


if __name__ == "__main__":
    if "--child" not in sys.argv and "--gui" not in sys.argv:
        args = [sys.executable, "-m", "src", "--child"] + sys.argv[1:]
        subprocess.Popen(
            [TERMINAL[0], TERMINAL[1]] + args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        os._exit(0)
    setup_logger(PROG_NAME)
    main()
