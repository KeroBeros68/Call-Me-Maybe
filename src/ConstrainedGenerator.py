import re
import json
import numpy as np

from src.llm_custom.LLMCustom import LLMCustom
from src.models.OutputModel import OutputModel


class ConstrainedGenerator:
    def __init__(self, llm: LLMCustom):
        self.llm: LLMCustom = llm
        self._cache: dict[str, bool] = {}
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

        self.__NUMERIC_TOKENS: list[int] = []
        for c in "0123456789.-,}":
            self.__NUMERIC_TOKENS.extend(self.llm.encode(c).tolist()[0])

        self.all_tokens = set(self.llm.vocab_files.values())
        self.forbidden_int_token = set(self.llm.encode(".").tolist()[0])
        self.forbidden_float_token = set(self.llm.encode(",}").tolist()[0])
        self.forbidden_string_token = set(self.llm.encode('","').tolist()[0])
        self.__mask_buffer: np.ndarray = np.full(
            self.llm.vocab_size, -1e10, dtype=np.float32
        )

    def __setup_final_prompt(self, functions_as_dict, user_prompt) -> str:
        return f"""<|im_start|>system
You are a function calling assistant.
<tools>
{json.dumps(functions_as_dict, indent=2)}
</tools><|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

    def encode_function_name(self, functions_definitions) -> None:
        for fn in functions_definitions:
            fn_target = self.llm.encode(fn.name).tolist()[0]
            self.encoded_func[fn.name] = fn_target
        self._all_func_token_ids: set[int] = {
            tid for ids in self.encoded_func.values() for tid in ids
        }

    def _decode(self, generated):
        self.decoded += self.llm.decode(generated[self.offset:])
        self.offset = len(generated)
        print(f"{self.decoded.strip("\n")} \r", end="", flush=True)

    def call_llm(self, functions_definitions, prompt) -> OutputModel:
        self.decoded = ""
        self.offset = 0
        functions_as_dict = [f.model_dump() for f in functions_definitions]

        final_prompt = self.__setup_final_prompt(functions_as_dict, prompt)

        input_ids: list[int] = self.llm.encode(final_prompt).tolist()[0]
        generated: list[int] = []
        self._cache = {}

        prompt_line: list[int] = []
        prompt_line = self.__PROMPT_TOKENS[1][:]
        prompt_line.extend(self.llm.encode(prompt).tolist()[0])
        prompt_line.extend(self.__FUNC_TOKENS[1])
        generated.extend(prompt_line)
        input_ids.extend(prompt_line)

        func_in_decoded = ""
        self._decode(generated)
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
                break

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
            match arg[1].type:
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
                        self.decoded, arg[0], input_ids, generated
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
            sanitized = self.decoded.replace("\\'", "'")
            sanitized = sanitized.replace("\\d", "\\\\d")
            sanitized = sanitized.replace("\\w", "\\\\w")
            sanitized = sanitized.replace("\\s", "\\\\s")
            sanitized = sanitized.replace("\\n", "\\\\n")
            res: OutputModel = OutputModel.model_validate(
                json.loads(sanitized)
            )
            print()
            return res
        except json.JSONDecodeError:
            raise

    def _get_logits(self, valid_tokens, input_ids, generated):
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

    def _get_valid_function(self):
        return self._all_func_token_ids

    def _get_arg_value_float(self, arg_name, input_ids, generated):
        while True:
            current_value_str = self.decoded.split(f'"{arg_name}":')[
                -1
            ].strip()
            if "." not in current_value_str:
                input_ids, generated = self._get_logits(
                    set(self.__NUMERIC_TOKENS) - self.forbidden_float_token,
                    input_ids,
                    generated,
                )
            else:
                input_ids, generated = self._get_logits(
                    self.__NUMERIC_TOKENS, input_ids, generated
                )
            self._decode(generated)
            if self.decoded.endswith(",") or self.decoded.endswith("}"):
                return input_ids, generated

    def _get_arg_value_int(self, input_ids, generated):
        while True:
            input_ids, generated = self._get_logits(
                set(self.__NUMERIC_TOKENS) - self.forbidden_int_token,
                input_ids,
                generated,
            )
            self._decode(generated)
            if self.decoded.endswith(",") or self.decoded.endswith("}"):
                return input_ids, generated

    def _get_arg_value_string(self, decoded, arg_name, input_ids, generated):
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
