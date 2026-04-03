import argparse
from logging import Logger

from llm_sdk.llm_sdk import Small_LLM_Model
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
            function_definitions = self.reader.read_file(
                cli_args.functions_definition
            )
            prompt_list = self.reader.read_file(cli_args.input)

            self.functions_definitions: list[FunctionModel] = [
                FunctionModel.model_validate(func)
                for func in function_definitions
            ]

            self.prompt_list: list[PromptModel] = [
                PromptModel.model_validate(prompt) for prompt in prompt_list
            ]
        except ValueError:
            raise

        self.logger.info(self.functions_definitions)
        self.logger.info(self.prompt_list)

        for prompt in self.prompt_list:
            res = self.call_llm(
                self.setup_final_prompt(self.functions_definitions, prompt)
            )
            print(res)

    def call_llm(self, prompt: str):
        prefix_ids: list[int] = []

        input_ids: list[int] = (
            self.llm_model.encode(prompt).tolist()[0] + prefix_ids
        )
        generated = []

        while True:
            logits: list[float] = self.llm_model.get_logits_from_input_ids(
                input_ids
            )
            # valid_tokens = self.get_valid_tokens(
            #     self.llm_model.decode(generated), prefix
            # )

            # for i in range(len(logits)):
            #     if i not in valid_tokens:
            #         logits[i] = float("-inf")

            next_token: int = logits.index(max(logits))

            generated.append(next_token)
            input_ids.append(next_token)

            decoded = self.llm_model.decode(generated)
            print(decoded)
            if "</tool_call>" in decoded and not decoded.strip().endswith(
                "<tool_call>"
            ):
                return decoded

    def setup_final_prompt(self, function_prompt, user_prompt) -> str:
        final_prompt = f"""<|im_start|>system
You are a function calling assistant. You MUST respond ONLY with a tool_call XML block. Never answer in plain text.
/no_think
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> JSON format:
<tools>
{function_prompt}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
    """
        return final_prompt

    # def get_valid_tokens(self, generated_text: str, prefix: str) -> set[int]:
    #     if prefix == "name":
    #         function_names = [
    #             func.name for func in self.functions_definitions.function_list
    #         ]
    #         candidates = [
    #             name
    #             for name in function_names
    #             if name.startswith(generated_text)
    #         ]
    #         if len(candidates) == 1 and generated_text == candidates[0]:
    #             return {
    #                 self.llm_model.encode("<|endoftext|>").tolist()[0][0],
    #             }

    #         next_chars = {
    #             name[len(generated_text)]
    #             for name in candidates
    #             if len(name) > len(generated_text)
    #         }

    #         return {
    #             self.llm_model.vocab_files[c]
    #             for c in next_chars
    #             if c in self.llm_model.vocab_files
    #         }

    #     return set(self.llm_model.vocab_files.values())

    def exit_program(self) -> None:
        """
        Safely halts execution and exits the program.

        Raises:
            ControllerError: To break execution explicitly.
        """
        self.logger.info("Programm exit")
        raise ControllerError("Programm exit")
