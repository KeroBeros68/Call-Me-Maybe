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

        self.logger.info(self.functions_definitions.function_list)
        self.logger.info(self.prompt_list.input_list)
        print(self.llm_model.encode(self.prompt_list.input_list[0].prompt).tolist()[0])
        # function_prompt = self.function_prompt()
        # generate_state = "name"
        # for prompt in self.prompt_list.input_list:
        #     if generate_state == "name":

        #         res = self.call_llm(
        #             self.setup_final_prompt(
        #                 function_prompt,
        #                 prompt.prompt,
        #                 "give me the good function name to use",
        #             ),
        #             generate_state,
        #         )
        #         print(res)

    def call_llm(self, prompt: str, prefix: str):
        prefix_ids: list[int] = self.llm_model.encode(prefix).tolist()[0]

        input_ids: list[int] = (
            self.llm_model.encode(prompt).tolist()[0] + prefix_ids
        )
        generated = []

        while True:
            logits: list[float] = self.llm_model.get_logits_from_input_ids(
                input_ids
            )

            valid_tokens = self.get_valid_tokens(
                self.llm_model.decode(generated), prefix
            )

            for i in range(len(logits)):
                if i not in valid_tokens:
                    logits[i] = float("-inf")

            next_token: int = logits.index(max(logits))

            generated.append(next_token)
            input_ids.append(next_token)

            decoded = self.llm_model.decode(generated)
            print(decoded)
            if decoded.strip().endswith("EOF"):
                return decoded

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

    def setup_final_prompt(self, function_prompt, user_prompt, reply) -> str:
        final_prompt = (
            "Available functions:"
            f"{function_prompt} "
            f'User request: "{user_prompt} "'
            f"{reply} EOF"
        )
        return final_prompt

    def get_valid_tokens(self, generated_text: str, prefix: str) -> set[int]:
        if prefix == "name":
            function_names = [
                func.name for func in self.functions_definitions.function_list
            ]
            candidates = [
                name
                for name in function_names
                if name.startswith(generated_text)
            ]
            if len(candidates) == 1 and generated_text == candidates[0]:
                return {
                    self.llm_model.encode(" EOF").tolist()[0],
                }

            next_chars = {
                name[len(generated_text)]
                for name in candidates
                if len(name) > len(generated_text)
            }

            return {
                self.llm_model.vocab_files[c]
                for c in next_chars
                if c in self.llm_model.vocab_files
            }

        return set(self.llm_model.vocab_files.values())

    def exit_program(self) -> None:
        """
        Safely halts execution and exits the program.

        Raises:
            ControllerError: To break execution explicitly.
        """
        self.logger.info("Programm exit")
        raise ControllerError("Programm exit")
