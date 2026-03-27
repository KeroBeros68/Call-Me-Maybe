import logging


class FileLoader:
    """Utility class to load the contents of a map file from disk."""

    def __init__(self) -> None:
        """Initializes the FileLoader with a logger."""
        self.logger = logging.getLogger("Fly-In")

    def read_file(self, path: str) -> str:
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
        try:
            with open(path) as f:
                content: str = f.read()
            return content
        except (FileNotFoundError, PermissionError) as e:
            raise e
