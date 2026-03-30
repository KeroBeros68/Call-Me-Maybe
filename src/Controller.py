import argparse
from logging import Logger

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models.FunctionDefinitionModel import FunctionDefinitionModel
from src.models.InputModel import InputModel

from .utils.FileLoader.BaseLoader import BaseLoader


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


class Controller:
    """ """

    def __init__(
        self,
        logger: Logger,
        parser: argparse.ArgumentParser,
        reader: BaseLoader,
        llm_model: Small_LLM_Model,
    ) -> None:
        """
        Initializes the Controller with its required dependencies.

        Args:
            reader (FileLoader): Used to read JSON files from disk.
        """
        super().__init__()
        self.logger: Logger = logger
        self.parser: argparse.ArgumentParser = parser
        self.reader: BaseLoader = reader
        self.llm_model: Small_LLM_Model = llm_model

    def process(self) -> None:
        """
        Executes the main controller flow.
        """
        self.logger.info("Programm starting")
        cli_args = self.parser.parse_args()
        self.logger.info(f"Inline ARG: {cli_args}")
        #  output_files = cli_args.output
        try:
            functions_definitions: FunctionDefinitionModel = (
                self.reader.read_file(cli_args.functions_definition)
            )
            prompt_list: InputModel = self.reader.read_file(cli_args.input)
        except ValueError:
            raise

        self.logger.info(functions_definitions)
        self.logger.info(prompt_list)

        for prompt in prompt_list:
            print(self.llm_model.encode(prompt["prompt"]).tolist())
            print(self.llm_model.cmm_encode(prompt["prompt"]))
            print()

        # for i in range(100):
        #     self.logger.warning(text)
        #     test1 = self.llm_model.encode(text).tolist()[0]
        #     test2 = self.llm_model.get_logits_from_input_ids(test1)
        #     text += self.llm_model.decode(
        #         [range(len(test2))[test2.index(max(test2))]]
        #     )
        # self.logger.warning(text)

    def exit_program(self) -> None:
        """
        Safely halts execution and exits the program.

        Raises:
            ControllerError: To break execution explicitly.
        """
        self.logger.info("Programm exit")
        raise ControllerError("Programm exit")
