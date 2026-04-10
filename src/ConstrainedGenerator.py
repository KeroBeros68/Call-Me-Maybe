import re
import json
from typing import Any
import numpy as np

from src.llm_custom.LLMCustom import LLMCustom
from src.models.FunctionModel import FunctionModel
from src.models.OutputModel import OutputModel


class ConstrainedGeneratorError(Exception):
    """
    Raised when the ConstrainedGenerator encounters an unrecoverable error.
    """

    def __init__(self, message: str) -> None:
        """
        Initializes the ConstrainedGeneratorError.

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
        return f"[ConstrainedGeneratorError] {self.message}"


class ConstrainedGenerator:
    """Generates constrained LLM outputs structured as function-call JSON.

    Wraps a :class:`LLMCustom` instance and guides token-by-token generation
    so that the model always produces a valid JSON object matching one of the
    supplied function definitions, with correctly typed parameter values.
    """

    MAX_LOOP: int = 32

    def __init__(self, llm: LLMCustom):
        """Initialize the ConstrainedGenerator.

        Pre-computes and caches token sequences for structural JSON delimiters
        and numeric character sets used during constrained decoding.

        Args:
            llm (LLMCustom): The language model wrapper used for encoding,
                decoding, and logit inference.
        """
        self.llm: LLMCustom = llm
        self.encoded_func: dict[str, list[int]] = {}

        self.__PROMPT_TOKENS: tuple[str, list[int]] = (
            '{"prompt":"',
            self.llm.encode('{"prompt":"').tolist()[0],
        )

        self.__FUNC_TOKENS: tuple[str, list[int]] = (
            '","name":"',
            self.llm.encode('","name":"').tolist()[0],
        )

        self.__PARAMS_TARGET: tuple[str, list[int]] = (
            '","parameters":{',
            self.llm.encode('","parameters":{').tolist()[0],
        )

        NUM_TOKENS: list[int] = []

        for c in "0123456789.-,}":
            NUM_TOKENS.extend(self.llm.encode(c).tolist()[0])
        self.__NUMERIC_TOKENS: set[int] = set(NUM_TOKENS)

        self.all_tokens = set(self.llm.vocab_files.values())
        self.forbidden_int_token = set(self.llm.encode(".").tolist()[0])
        self.forbidden_float_token = set(self.llm.encode(",}").tolist()[0])
        self.forbidden_string_token = set(self.llm.encode('","').tolist()[0])
        self.__mask_buffer: np.ndarray = np.full(
            self.llm.vocab_size, -1e10, dtype=np.float32
        )
        self.__offset: int = 0
        self._rep_dict = {
            r"\'": "'",
            r"\d": r"\\d",
            r"\w": r"\\w",
            r"\s": r"\\s",
            r"\n": r"\\n",
        }
        self._rep_regex = re.compile(
            "|".join(re.escape(k) for k in self._rep_dict.keys())
        )

    def __setup_final_prompt(
        self, functions_as_dict: Any, user_prompt: str
    ) -> str:
        """Build the full ChatML prompt string sent to the model.

        Embeds the serialised function definitions inside a ``<tools>`` block
        and wraps everything in the ChatML ``im_start`` / ``im_end`` delimiters
        expected by the Qwen family of models.

        Args:
            functions_as_dict (Any): List of function definitions already
                serialised to plain Python dicts.
            user_prompt (str): The natural-language instruction from the user.

        Returns:
            str: The fully formatted prompt ready for tokenisation.
        """
        return f"""<|im_start|>system
You are a function calling assistant.
<tools>
{json.dumps(functions_as_dict, indent=2)}
</tools><|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

    def encode_function_name(
        self, functions_definitions: list[FunctionModel]
    ) -> None:
        """Pre-tokenise every function name and build a union token-ID set.

        The resulting :attr:`encoded_func` mapping and
        :attr:`_all_func_token_ids` set are used during generation to
        restrict the model to valid function-name tokens only.

        Args:
            functions_definitions (list[FunctionModel]): The function
                definitions whose names should be pre-encoded.
        """
        for fn in functions_definitions:
            fn_target = self.llm.encode(fn.name).tolist()[0]
            self.encoded_func[fn.name] = fn_target
        self._all_func_token_ids: set[int] = {
            tid for ids in self.encoded_func.values() for tid in ids
        }

    def _decode(self, generated: list[int]) -> None:
        """Incrementally decode newly generated tokens and print to stdout.

        Only the tokens appended since the last call are decoded, using the
        internal offset to avoid redundant work.

        Args:
            generated (list[int]): The full list of generated token IDs so far.
        """
        self.decoded += self.llm.decode(generated[self.__offset:])
        self.__offset = len(generated)
        print(f"{self.decoded.strip("\n")} \r", end="", flush=True)

    def call_llm(
        self, functions_definitions: list[FunctionModel], prompt: str
    ) -> OutputModel:
        """Run constrained generation for a single user prompt.

        Guides the model through three phases:
        1. Select a valid function name from ``functions_definitions``.
        2. Emit the ``parameters`` opening token.
        3. Generate each parameter value with type-specific token masking.

        Args:
            functions_definitions (list[FunctionModel]): The available
                function definitions the model may call.
            prompt (str): The natural-language instruction from the user.

        Returns:
            OutputModel: A validated object containing the chosen function
                name and its argument values.

        Raises:
            ConstrainedGeneratorError: If the maximum loop count is reached
                while selecting a function name.
            json.JSONDecodeError: If the final decoded output cannot be
                parsed as JSON.
        """
        self.decoded = ""
        self.__offset = 0
        functions_as_dict = [f.model_dump() for f in functions_definitions]

        final_prompt = self.__setup_final_prompt(functions_as_dict, prompt)

        input_ids: list[int] = self.llm.encode(final_prompt).tolist()[0]
        generated: list[int] = []

        prompt_line: list[int] = []
        prompt_line = self.__PROMPT_TOKENS[1][:]
        prompt_line.extend(self.llm.encode(prompt).tolist()[0])
        prompt_line.extend(self.__FUNC_TOKENS[1])
        generated.extend(prompt_line)
        input_ids.extend(prompt_line)

        func_in_decoded = ""
        self._decode(generated)
        current_loop = 0
        while True:
            input_ids, generated = self._get_logits(
                self._get_valid_function(), input_ids, generated
            )

            self._decode(generated)
            fn_in_decoded = re.search(r'"name":"([^"]*)', self.decoded)
            if fn_in_decoded is not None:
                func_in_decoded = fn_in_decoded.group(1)
            func = [
                f.name
                for f in functions_definitions
                if func_in_decoded in f.name
            ]
            if len(func) == 1:
                for _ in self.llm.encode(func_in_decoded).tolist()[0]:
                    input_ids.pop()
                    generated.pop()
                input_ids.extend(self.encoded_func[func[0]])
                generated.extend(self.encoded_func[func[0]])
                generated.extend(self.__PARAMS_TARGET[1])
                input_ids.extend(self.__PARAMS_TARGET[1])
                current_loop = 0
                break
            if current_loop == self.MAX_LOOP:
                raise ConstrainedGeneratorError("Infinite loop detected")
            else:
                current_loop += 1

        for fn in functions_definitions:
            if fn.name in func[0]:
                function = fn
                break
        args = list(function.parameters.items())

        for idx, arg in enumerate(args):
            target_arg_name = self.llm.encode(f'"{arg[0]}":').tolist()[0]
            if idx > len(args):
                break
            input_ids.extend(target_arg_name)
            generated.extend(target_arg_name)
            match arg[1].get("type"):
                case "integer":
                    input_ids, generated = self._get_arg_value_int(
                        input_ids, generated
                    )
                case "number":
                    input_ids, generated = self._get_arg_value_float(
                        arg[0], input_ids, generated
                    )
                case _:
                    input_ids, generated = self._get_arg_value_string(
                        arg[0], input_ids, generated
                    )
                    self._decode(generated)
                    if self.decoded.endswith(',"'):
                        input_ids.pop()
                        generated.pop()

        self._decode(generated)
        if not self.decoded.strip()[-3:].endswith("}}"):
            if self.decoded.strip()[-3:].count("}") == 1:
                input_ids, generated = self._get_logits(
                    self.llm.encode("}").tolist()[0], input_ids, generated
                )
                self._decode(generated)
            else:
                closing = self.llm.encode("}}").tolist()[0]
                input_ids.extend(closing)
                generated.extend(closing)
            self._decode(generated)
        try:
            res: OutputModel = OutputModel.model_validate(
                json.loads(self._fast_sanitize(self.decoded))
            )
            print()
            return res
        except json.JSONDecodeError:
            raise

    def _fast_sanitize(self, text: str) -> str:
        """Apply lightweight regex replacements to make the output valid JSON.

        Replaces common escape-sequence artefacts produced by the tokeniser
        (e.g. ``\'`` → ``'``, raw ``\\d`` → ``\\\\d``) so that
        :func:`json.loads` can parse the result without raising.

        Args:
            text (str): The raw decoded string from the model.

        Returns:
            str: The sanitised string ready to be parsed as JSON.
        """
        return self._rep_regex.sub(lambda m: self._rep_dict[m.group(0)], text)

    def _get_logits(
        self,
        valid_tokens: set[int],
        input_ids: list[int],
        generated: list[int],
    ) -> tuple[list[int], list[int]]:
        """Sample one token from a masked logit distribution.

        Fetches raw logits for the current ``input_ids``, masks all token IDs
        not present in ``valid_tokens`` with a large negative value, then
        selects the argmax as the next token.

        Args:
            valid_tokens (set[int]): The set of token IDs that are allowed
                at this position.
            input_ids (list[int]): The full sequence of token IDs fed to the
                model (prompt + generated so far).
            generated (list[int]): The list of token IDs generated so far
                (subset of ``input_ids`` after the prompt).

        Returns:
            tuple[list[int], list[int]]: Updated ``(input_ids, generated)``
                with the newly sampled token appended to both lists.
        """
        logits_raw = self.llm.get_logits_from_input_ids(input_ids)
        logits = np.array(logits_raw, dtype=np.float32)

        np.copyto(logits, self.__mask_buffer)

        valid_list = list(valid_tokens)
        if valid_list:
            logits[valid_list] = np.array(logits_raw, dtype=np.float32)[
                valid_list
            ]

        next_token = int(logits.argmax())

        generated.append(next_token)
        input_ids.append(next_token)

        return input_ids, generated

    def _get_valid_function(self) -> set[int]:
        """Return the union set of all pre-encoded function-name token IDs.

        Returns:
            set[int]: Token IDs that correspond to at least one character in
                any known function name.
        """
        return self._all_func_token_ids

    def _get_arg_value_float(
        self, arg_name: str, input_ids: list[int], generated: list[int]
    ) -> tuple[list[int], list[int]]:
        """Generate a floating-point argument value token by token.

        Restricts the model to numeric tokens (``0-9``, ``.``, ``-``, ``,``,
        ``}``) and enforces that the value always contains a decimal point by
        injecting ``.0`` when a comma or closing brace is emitted before any
        ``.`` has appeared.

        Args:
            arg_name (str): The parameter name being generated, used to
                isolate the current value from the decoded text.
            input_ids (list[int]): Current full token-ID sequence.
            generated (list[int]): Generated token IDs so far.

        Returns:
            tuple[list[int], list[int]]: Updated ``(input_ids, generated)``
                after the complete float value has been emitted.

        Raises:
            ConstrainedGeneratorError: If the maximum loop count is reached.
        """
        current_loop = 0
        while True:
            input_ids, generated = self._get_logits(
                self.__NUMERIC_TOKENS, input_ids, generated
            )
            self._decode(generated)
            current_value_str = self.decoded.split(f'"{arg_name}":')[
                -1
            ].strip()
            if "." not in current_value_str and "," in current_value_str:
                tmp = input_ids.pop()
                generated.pop()
                self.__offset -= 1
                self.decoded = self.decoded.removesuffix(',')
                self._decode(generated)
                input_ids.extend(self.llm.encode('.0').tolist()[0])
                generated.extend(self.llm.encode('.0').tolist()[0])
                input_ids.append(tmp)
                generated.append(tmp)
                self._decode(generated)
            if self.decoded.endswith(",") or self.decoded.endswith("}"):
                return input_ids, generated
            if current_loop == self.MAX_LOOP:
                raise ConstrainedGeneratorError("Infinite loop detected")
            else:
                current_loop += 1

    def _get_arg_value_int(
        self, input_ids: list[int], generated: list[int]
    ) -> tuple[list[int], list[int]]:
        """Generate an integer argument value token by token.

        Restricts the model to numeric tokens while excluding the ``.`` token
        so that no decimal point can appear in the value.

        Args:
            input_ids (list[int]): Current full token-ID sequence.
            generated (list[int]): Generated token IDs so far.

        Returns:
            tuple[list[int], list[int]]: Updated ``(input_ids, generated)``
                after the complete integer value has been emitted.

        Raises:
            ConstrainedGeneratorError: If the maximum loop count is reached.
        """
        current_loop = 0
        while True:
            input_ids, generated = self._get_logits(
                set(self.__NUMERIC_TOKENS) - self.forbidden_int_token,
                input_ids,
                generated,
            )
            self._decode(generated)
            if self.decoded.endswith(",") or self.decoded.endswith("}"):
                return input_ids, generated
            if current_loop == self.MAX_LOOP:
                raise ConstrainedGeneratorError("Infinite loop detected")
            else:
                current_loop += 1

    def _get_arg_value_string(
        self, arg_name: str, input_ids: list[int], generated: list[int]
    ) -> tuple[list[int], list[int]]:
        """Generate a quoted string argument value token by token.

        Allows any token except the ``","`` sequence that would prematurely
        close the current string.  Generation stops as soon as an unescaped
        closing double-quote is detected in the decoded output for this
        parameter.

        Args:
            arg_name (str): The parameter name being generated, used to
                locate the current value in the decoded text.
            input_ids (list[int]): Current full token-ID sequence.
            generated (list[int]): Generated token IDs so far.

        Returns:
            tuple[list[int], list[int]]: Updated ``(input_ids, generated)``
                after the closing quote of the string value has been emitted.
        """
        while True:
            input_ids, generated = self._get_logits(
                self.all_tokens - self.forbidden_string_token,
                input_ids,
                generated,
            )
            self._decode(generated)
            parts = self.decoded.split(f'"{arg_name}":')
            if len(parts) < 2:
                continue

            value_part = parts[-1].strip()
            if not value_part.startswith('"'):
                continue

            content = value_part[1:]
            i = 0
            while i < len(content):
                if content[i] == "\\":
                    i += 2
                    continue
                if content[i] == '"':
                    return input_ids, generated
                i += 1
