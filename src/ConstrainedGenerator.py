import json

from src.llm_custom.LLMCustom import LLMCustom


class ConstrainedGenerator:
    def __init__(self, llm: LLMCustom):
        self.llm: LLMCustom = llm

    def call_llm(self, functions_definitions: str, prompt: str):
        prefix_ids: list[int] = []

        final_prompt = self.setup_final_prompt(functions_definitions, prompt)

        input_ids: list[int] = (
            self.llm.encode(final_prompt).tolist()[0] + prefix_ids
        )
        generated = []

        while True:
            logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
            valid_tokens = self.get_valid_tokens(self.llm.decode(generated), prompt)

            for i in range(len(logits)):
                if i not in valid_tokens:
                    logits[i] = float("-inf")

            next_token: int = logits.index(max(logits))

            generated.append(next_token)
            input_ids.append(next_token)

            decoded = self.llm.decode(generated)
            print(decoded)
            if "</tool_call>" in decoded and not decoded.strip().endswith(
                "<tool_call>"
            ):
                return decoded

    def setup_final_prompt(self, function_prompt, user_prompt) -> str:
        try:
            functions_as_dict = [f.model_dump() for f in function_prompt]
        except AttributeError:
            # Au cas où ce ne sont pas des modèles Pydantic mais des objets simples
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

    def get_valid_tokens(self, generated: str, prompt: str) -> set[int]:
        step = 0
        if not generated.startswith('{"prompt": "'):
            return set(self.llm.encode('{"prompt": "').tolist()[0])
        elif generated.startswith('{"prompt": "') and step == 0:
            return set(self.llm.encode(f'{prompt}"').tolist()[0])
        else:
            step = 1
        if step == 1 and ', "function":' not in generated:
            return set(self.llm.encode(', "function":').tolist()[0])
        else:
            step = 2
        if step == 2 and ', "arguments": {' not in generated:
            return set(self.llm.encode(', "arguments": {').tolist()[0])
        else:
            step = 3
        return set(self.llm.vocab_files.values())


"""

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
