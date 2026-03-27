
# TERMINAL: list[str] = ["gnome-terminal", "--"]
import logging
import os
import subprocess
import sys
import time

from FileLoader import FileLoader
from utils.check_env import RunSecurity, RunEnvironmentError
from utils.logger.Logger import setup_logger


TERMINAL: list[str] = ["konsole", "-e"]


def main() -> None:
    logger = logging.getLogger("Call-Me-Maybe")

    secure_env = RunSecurity()
    try:
        secure_env.check_process()
        print("\n[OK] Environment secure. Launching GUI...")
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
        from Controller import Controller, ControllerError

        controller = Controller(
            FileLoader(),
        )
        controller.process()

    except ControllerError:
        pass
    except Exception as e:
        logger.error(e)
        input("\n\nPress Enter to exit...")
        return


if __name__ == "__main__":
    if "--child" not in sys.argv and "--gui" not in sys.argv:
        args = [sys.executable, sys.argv[0], "--child"] + sys.argv[1:]
        subprocess.Popen(
            [TERMINAL[0], TERMINAL[1]] + args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        os._exit(0)
    setup_logger("Call-Me-Maybe")
    main()
