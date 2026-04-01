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
            self.functions_definitions: FunctionDefinitionModel = (
                FunctionDefinitionModel.model_validate(
                    {
                        "function_list": self.reader.read_file(
                            cli_args.functions_definition
                        )
                    }
                )
            )
            self.prompt_list: InputModel = InputModel.model_validate(
                {"input_list": self.reader.read_file(cli_args.input)}
            )
        except ValueError:
            raise

        function_prompt = self.function_prompt()
        self.logger.info(self.functions_definitions.function_list)
        self.logger.info(self.prompt_list.input_list)

        for prompt in self.prompt_list.input_list:
            final_prompt = self.setup_final_prompt(
                function_prompt, prompt.prompt
            )

            input_ids: list[int] = self.llm_model.encode(final_prompt).tolist()[0]
            generated: list[int] = []

            while True:
                logits: list[float] = self.llm_model.get_logits_from_input_ids(
                    input_ids
                )

                # TODO: appliquer les contraintes ici (masquer les tokens invalides)

                next_token: int = logits.index(max(logits))
                generated.append(next_token)
                input_ids.append(next_token)

                decoded = self.llm_model.decode(generated)

                if decoded.strip().endswith("}"):
                    break

            self.logger.warning(f"Final JSON: {self.llm_model.decode(generated)}")

    def function_prompt(self) -> str:
        result: str = ""
        for func in self.functions_definitions.function_list:
            params_string: str = ""

            for params, params_type in func.parameters.items():
                params_string += "".join(f"{params}: {params_type.type}, ")

            function_string = (
                f" - {func.name}({params_string.strip(" ,")})"
                f" -> {func.returns.type}: {func.description}\n"
            )
            result += "".join(function_string)
        return result

    def setup_final_prompt(self, function_prompt, user_prompt) -> str:
        final_prompt = (
            "You are a function calling assistant.\n\n"
            "Available functions:"
            f"{function_prompt}\n\n"
            f'User request: "{user_prompt}"\n\n'
            'Reply with JSON: {"prompt": "<user request>", "name":'
            ' "<function name>", "parameters": {...}}}'
        )
        return final_prompt

    def exit_program(self) -> None:
        """
        Safely halts execution and exits the program.

        Raises:
            ControllerError: To break execution explicitly.
        """
        self.logger.info("Programm exit")
        raise ControllerError("Programm exit")
