import argparse
from logging import Logger
import time

from llm_sdk import Small_LLM_Model  # type: ignore
from src.models.OutputModel import OutputModel
from .ConstrainedGenerator import ConstrainedGenerator
from src.models.FunctionModel import FunctionModel
from src.models.InputModel import PromptModel

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
    """Orchestrates the end-to-end function-calling pipeline.

    Reads input files, validates them into typed models, drives constrained
    generation for each prompt, and persists the results to disk.
    """

    def __init__(
        self,
        logger: Logger,
        parser: argparse.ArgumentParser,
        reader: BaseLoader,
        llm_model: Small_LLM_Model,
        llm_manager: ConstrainedGenerator,
    ) -> None:
        """
        Initialize the Controller with its required dependencies.

        Args:
            logger (Logger): Logger instance used for progress and error
                reporting throughout the pipeline.
            parser (argparse.ArgumentParser): Argument parser whose
                ``parse_args()`` result supplies paths and model name.
            reader (BaseLoader): File-loader used to read and write JSON
                data from/to disk.
            llm_model (Small_LLM_Model): The underlying language model.
            llm_manager (ConstrainedGenerator): Wrapper that drives
                constrained token-by-token generation.
        """
        super().__init__()
        self.logger: Logger = logger
        self.cli_args = parser.parse_args()
        self.reader: BaseLoader = reader
        self.llm_model: Small_LLM_Model = llm_model
        self.llm_manager: ConstrainedGenerator = llm_manager

    def process(self) -> None:
        """
        Executes the main controller flow.
        """
        self.logger.info("Programm starting")
        self.logger.info(f"Inline ARG: {self.cli_args}")
        output_files = self.cli_args.output
        try:
            function_files = self.reader.read_file(
                self.cli_args.functions_definition
            )
            prompt_files = self.reader.read_file(self.cli_args.input)

            functions_definitions: list[FunctionModel] = [
                FunctionModel.model_validate(func)
                for func in function_files
            ]

            prompt_list: list[PromptModel] = [
                PromptModel.model_validate(prompt) for prompt in prompt_files
            ]
        except ValueError:
            raise

        self.logger.info(functions_definitions)
        self.logger.info(prompt_list)

        self.llm_manager.encode_function_name(functions_definitions)

        res: list[OutputModel] = []

        gen_start_time = time.time()
        for prompt in prompt_list:
            res.append(self.llm_manager.call_llm(
                functions_definitions, prompt.prompt
            ))

            self.logger.info(res)
        prompt_time = (time.time() - gen_start_time)
        self.process_time(prompt_time)
        data_to_save = [obj.model_dump() for obj in res]
        self.reader.write_file(output_files, data_to_save)

    @staticmethod
    def process_time(prompt_time: float) -> None:
        """Print the total generation time in HH:MM:SS format.

        Args:
            prompt_time (float): Elapsed time in seconds.
        """
        h = int(prompt_time // 3600)
        m = int((prompt_time % 3600) // 60)
        s = int(prompt_time % 60)
        print(f"\nTemps d'exécution totale : {h:02d}:{m:02d}:{s:02d}")

    def exit_program(self) -> None:
        """Log a program-exit message and return control to the caller."""
        self.logger.info("Programm exit")
        return
