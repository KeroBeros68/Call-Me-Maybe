import json

from src.llm_custom.LLMCustom import LLMCustom
from src.models.FunctionModel import FunctionModel


class ConstrainedGenerator:
    def __init__(self, llm: LLMCustom):
        self.llm: LLMCustom = llm

    def call_llm(
        self, functions_definitions: list[FunctionModel], prompt: str
    ):
        prefix_ids: list[int] = []

        final_prompt = self.setup_final_prompt(functions_definitions, prompt)

        input_ids: list[int] = (
            self.llm.encode(final_prompt).tolist()[0] + prefix_ids
        )
        generated: list[int] = []

        step = 0
        while True:
            logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)

            valid_tokens = self.get_valid_tokens(
                generated, prompt, functions_definitions, step
            )

            for i in range(len(logits)):
                if i not in valid_tokens:
                    logits[i] = float("-inf")

            next_token: int = logits.index(max(logits))

            generated.append(next_token)
            input_ids.append(next_token)

            decoded = self.llm.decode(generated)

            if ', "function": "' in decoded:
                step = 1
            if '", "arguments": {' in decoded:
                step = 2
            if '}, "return": {' in decoded:
                step = 3

            print(decoded, end="\r", flush=True)
            if "</tool_call>" in decoded and not decoded.strip().endswith(
                "<tool_call>"
            ):
                return decoded

    def setup_final_prompt(self, function_prompt, user_prompt) -> str:
        try:
            functions_as_dict = [f.model_dump() for f in function_prompt]
        except AttributeError:
            functions_as_dict = [vars(f) for f in function_prompt]

        final_prompt = f"""<|im_start|>system
You are a function calling assistant. You MUST respond ONLY with a tool_call XML block. Never answer in plain text.
/no_think
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> JSON format:
<tools>
{json.dumps(functions_as_dict, indent=2)}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{
    prompt: <user_prompt>,
    function: <function_name>,
    arguments: {{
        <type>: <value>,
        ...
    }},
    return: <{{
        <type>: <value>,
        ...
    }}
}}
</tool_call><|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
    """
        return final_prompt

    def get_valid_tokens(
        self, generated: list[int], prompt: str, functions_definitions, step
    ) -> set[int]:
        if step == 0:
            result = self._valid_prompt_line(prompt, generated)
        elif step == 1:
            result = self._valid_function_line(
                generated, functions_definitions
            )
        elif step == 2:
            result = self._valid_arguments_line(
                generated, functions_definitions
            )
        elif step == 3:
            result = self._valid_return_line(
                generated, functions_definitions
            )
        else:
            result = set(self.llm.vocab_files.values())

        # sécurité — ne jamais retourner None
        if result is None:
            return set(self.llm.vocab_files.values())
        return result

    def _valid_prompt_line(self, prompt: str, generated: list[int]):
        if not self.llm.decode(generated).startswith('{"prompt": "'):
            return set(self.llm.encode('{"prompt": "').tolist()[0])

        target = self.llm.encode(
            f'{{"prompt": "{prompt}", "function": "'
        ).tolist()[0]

        generated_len = len(generated)
        if generated_len < len(target):
            tokens = target[generated_len::]
            return set(tokens)

    def _valid_function_line(
        self, generated: list[int], functions_definitions
    ):
        generated_text = self.llm.decode(generated)

        if not any(fn.name in generated_text for fn in functions_definitions):
            valid_ids = []
            for fn in functions_definitions:
                tokens = self.llm.encode(f"{fn.name}").tolist()[0]
                valid_ids.extend(tokens)
            return set(valid_ids)
        return set(self.llm.encode('", "arguments": {').tolist()[0])

    def _valid_arguments_line(
        self, generated: list[int], functions_definitions
    ):
        generated_text = self.llm.decode(generated)

        for fn in functions_definitions:
            if fn.name in generated_text:
                function = fn
                break
        valid_ids = []
        for arg, type in function.parameters.items():
            tokens = self.llm.encode(f"{arg}").tolist()[0]
            valid_ids.extend(tokens)
            if type == "number":
                tokens = self.llm.encode("0123456789.-").tolist()[0]
            else:
                tokens = self.llm.vocab_files.values()
            valid_ids.extend(tokens)
            return set(valid_ids)

        return set(self.llm.encode('"}, "returns": {').tolist()[0])

    def _valid_return_line(
        self, generated: list[int], functions_definitions
    ):
        generated_text = self.llm.decode(generated)

        for fn in functions_definitions:
            if fn.name in generated_text:
                function = fn
                break
        valid_ids = []
        for ret in function.returns:
            tokens = self.llm.encode(f"{ret}").tolist()[0]
            valid_ids.extend(tokens)
            return set(valid_ids)

        return set(self.llm.encode('}}}}</tool_call>').tolist()[0])


"""
}</tool_call>
{
    prompt: <user_prompt>,
    function: <function_name>,
    arguments: {
        <type>: <value>,
        ...
    },
    return: <{
        <type>: <value>,
        ...
    }
}

"""
