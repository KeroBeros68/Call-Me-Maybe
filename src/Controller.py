import logging

from FileLoader import FileLoader


class ControllerError(Exception):
    """
    Exception raised for errors in the Controller.
    """

    def __init__(self, message: str) -> None:
        """
        Initializes the ControllerError.

        Args:
            message (str): The error message.
        """
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        """
        Returns the string representation of the error.

        Returns:
            str: The formatted error message.
        """
        return f"[ControllerError] {self.message}"


class Controller():
    """
    """

    def __init__(
        self,
        reader: FileLoader,
    ) -> None:
        """
        Initializes the Controller with its required dependencies.

        Args:
            reader (FileLoader): Used to read JSON files from disk.
        """
        super().__init__()
        self.logger = logging.getLogger("Call-Me-Maybe")
        self.reader: FileLoader = reader

    def process(self) -> None:
        """
        Executes the main controller flow.
        """
        self.logger.info("Programm starting")

    def exit_program(self) -> None:
        """
        Safely halts execution and exits the program.

        Raises:
            ControllerError: To break execution explicitly.
        """
        self.logger.info("Programm exit")
        raise ControllerError("Programm exit")
