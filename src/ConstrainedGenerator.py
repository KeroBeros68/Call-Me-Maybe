import json

from src.llm_custom.LLMCustom import LLMCustom


class ConstrainedGenerator:
    def __init__(self, llm: LLMCustom):
        self.llm: LLMCustom = llm

    def call_llm(self, prompt: str):
        prefix_ids: list[int] = []

        input_ids: list[int] = self.llm.encode(prompt).tolist()[0] + prefix_ids
        generated = []

        while True:
            logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
            # valid_tokens = self.get_valid_tokens(
            #     self.llm.decode(generated), prefix
            # )

            # for i in range(len(logits)):
            #     if i not in valid_tokens:
            #         logits[i] = float("-inf")

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
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
    """
        return final_prompt

    def get_valid_tokens(self) -> set[int]:
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

        return set(self.llm.vocab_files.values())
