import json

from src.llm_custom.LLMCustom import LLMCustom
from src.models.OutputModel import OutputModel


class ConstrainedGenerator:
    def __init__(self, llm: LLMCustom):
        self.llm: LLMCustom = llm
        self._cache: dict[str, bool] = {}

    def call_llm(self, functions_definitions, prompt) -> OutputModel:
        final_prompt = self.setup_final_prompt(functions_definitions, prompt)
        input_ids = self.llm.encode(final_prompt).tolist()[0]
        generated = []
        self._cache = {}
        step = 0

        while True:
            logits = self.llm.get_logits_from_input_ids(input_ids)

            valid_tokens = self.get_valid_tokens(
                generated, prompt, functions_definitions, step
            )

            for i in range(len(logits)):
                if i not in valid_tokens:
                    logits[i] = float("-inf")

            next_token = logits.index(max(logits))
            generated.append(next_token)
            input_ids.append(next_token)

            decoded = self.llm.decode(generated)

            if step == 0 and f'"prompt": "{prompt}", "name": "' in decoded:
                step = 1
            elif step == 1 and any(
                fn.name in decoded for fn in functions_definitions
            ):
                step = 2
            elif step == 2 and '", "parameters": {' in decoded:
                step = 3

            try:
                res: OutputModel = OutputModel.model_validate(
                    json.loads(decoded)
                )
                return res
            except json.JSONDecodeError:
                pass

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

    def _encode_function_name(self, functions_definitions):
        func_encoded = {}
        for fn in functions_definitions:
            fn_target = self.llm.encode(fn.name).tolist()[0]
            func_encoded[fn.name] = fn_target
        return func_encoded

    def get_valid_tokens(self, generated, prompt, functions_definitions, step):
        if step == 0:
            target = self.llm.encode(
                f'{{"prompt": "{prompt}", "name": "'
            ).tolist()[0]
            if len(generated) < len(target):
                return {target[len(generated)]}
            return set(self.llm.vocab_files.values())

        elif step == 1:
            valid_ids = []
            encoded_func = self._encode_function_name(functions_definitions)
            for func, encoded in encoded_func.items():
                valid_ids.extend(encoded)
            return set(valid_ids)

        elif step == 2:
            target = self.llm.encode('", "parameters": {').tolist()[0]
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

        for _, (arg_name, arg_model) in enumerate(args):
            if self._cache.get(arg_name) is None:
                if f'"{arg_name}": ' not in decoded:
                    target = self.llm.encode(f'"{arg_name}": ').tolist()[0]
                    for length in range(len(target), 0, -1):
                        if generated[-length:] == target[:length]:
                            return {target[length]}
                    return {target[0]}

                else:
                    if decoded.endswith(",") or decoded.endswith("}"):
                        self._cache[arg_name] = True
                        continue
                    if arg_model.type == "number":
                        num_tokens = set(
                            self.llm.encode("0123456789.-,}").tolist()[0]
                        )
                        return num_tokens
                    else:
                        all_tokens = set(self.llm.vocab_files.values())
                        return all_tokens

        return set(self.llm.encode("}").tolist()[0])
