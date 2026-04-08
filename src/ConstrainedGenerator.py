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

    def call_llm(self, functions_definitions, prompt) -> OutputModel:
        final_prompt = self.setup_final_prompt(functions_definitions, prompt)
        input_ids = self.llm.encode(final_prompt).tolist()[0]
        generated = []
        self._cache = {}
        step = 1

        prompt_line: list[int] = []
        prompt_line = self.__PROMPT_TOKENS[1][:]
        prompt_line.extend(self.llm.encode(prompt).tolist()[0])
        prompt_line.extend(self.__FUNC_TOKENS[1])
        generated.extend(prompt_line)
        input_ids.extend(prompt_line)

        while True:
            logits_raw = self.llm.get_logits_from_input_ids(input_ids)
            logits = np.array(logits_raw)

            valid_tokens = self.get_valid_tokens(
                generated, prompt_line, functions_definitions, step
            )

            # Application du masque (Corrected with full_like)
            mask = np.full_like(logits, -1e10)

            # Conversion de valid_tokens en liste d'indices pour NumPy
            valid_list = list(valid_tokens)
            if valid_list:
                mask[valid_list] = 0

            logits += mask
            next_token = int(logits.argmax())

            generated.append(next_token)
            input_ids.append(next_token)

            decoded = self.llm.decode(generated)

            if step == 1 and any(
                fn.name in decoded for fn in functions_definitions
            ):
                step = 2
            elif step == 2 and self.__PARAMS_TARGET[0] in decoded:
                step = 3

            if step == 3:
                try:
                    sanitized = decoded.replace("\\'", "'")
                    res: OutputModel = OutputModel.model_validate(
                        json.loads(sanitized)
                    )
                    return res
                except json.JSONDecodeError:
                    pass
            print(f" \r{decoded}", end="", flush=True)

    def setup_final_prompt(self, function_prompt, user_prompt) -> str:
        functions_as_dict = [f.model_dump() for f in function_prompt]
        return f"""<|im_start|>system
You are a function calling assistant.
<tools>
{json.dumps(functions_as_dict, indent=2)}
</tools><|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

    def encode_function_name(self, functions_definitions):
        for fn in functions_definitions:
            fn_target = self.llm.encode(fn.name).tolist()[0]
            self.encoded_func[fn.name] = fn_target

    def get_valid_tokens(self, generated, prompt, functions_definitions, step):
        if step == 1:
            valid_ids = []
            for func, encoded in self.encoded_func.items():
                valid_ids.extend(encoded)
            return set(valid_ids)

        elif step == 2:
            target = self.__PARAMS_TARGET[1]
            for length in range(len(target), 0, -1):
                if generated[-length:] == target[:length]:
                    if length == len(target):
                        return set(self.llm.vocab_files.values())
                    return {target[length]}
            return {target[0]}

        elif step == 3:
            return self._valid_arguments(generated, functions_definitions)

        return set(self.llm.vocab_files.values())

    def _valid_arguments(self, generated, functions_definitions):
        function = None
        decoded = self.llm.decode(generated)
        for fn in functions_definitions:
            if fn.name in decoded:
                function = fn
                break
        if function is None:
            return set(self.llm.vocab_files.values())

        args = list(function.parameters.items())

        for idx, (arg_name, arg_model) in enumerate(args):
            if self._cache.get(arg_name) is None:
                if f'"{arg_name}":' not in decoded:
                    target = self.llm.encode(f'"{arg_name}":').tolist()[0]
                    for length in range(len(target), 0, -1):
                        if generated[-length:] == target[:length]:
                            return {target[length]}
                    return {target[0]}

                else:
                    if (
                        (decoded.endswith(",") or decoded.endswith("}"))
                        and (
                            arg_model.type == "number"
                            or arg_model.type == "integer"
                        )
                    ) or decoded.endswith('"}'):
                        self._cache[arg_name] = True
                        continue
                    if arg_model.type == "integer":
                        return self.__NUMERIC_TOKENS
                    elif arg_model.type == "number":
                        current_value_str = decoded.split(f'"{arg_name}":')[
                            -1
                        ].strip()
                        if "." not in current_value_str:
                            forbidden_tokens = set(
                                self.llm.encode(",}").tolist()[0]
                            )
                            return (
                                set(self.__NUMERIC_TOKENS) - forbidden_tokens
                            )

                        return self.__NUMERIC_TOKENS
                    else:
                        all_tokens = set(self.llm.vocab_files.values())
                        return all_tokens

        return set(self.llm.encode("}").tolist()[0])
