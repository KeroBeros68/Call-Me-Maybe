from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model
from src.utils.FileLoader.BaseLoader import BaseLoader


class LLMCustom(Small_LLM_Model):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: Any | None = None,
        dtype: Any | None = None,
        trust_remote_code: bool = True,
        reader: BaseLoader,
    ):
        super().__init__(
            model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.reader: BaseLoader = reader
        self.vocab_files: dict[str, int] = self.reader.read_file(
            self.get_path_to_vocab_file()
        )

    def _pre_split(self, text: str) -> list[str]:
        list_str = text.split()

        result = []
        for i, token in enumerate(list_str):
            if token.endswith((".", "!", "?")):
                word, punct = token[:-1], token[-1]
                if word.isdigit():
                    result.append("\u0120")
                    for c in word:
                        result.append(c)
                    result.append(punct)
                else:
                    result.extend([word if i == 0 else "\u0120" + word, punct])
            elif token.isdigit():
                result.append("\u0120")
                for c in token:
                    result.append(c)
            else:
                result.append(token if i == 0 else "\u0120" + token)
        return result

    def cmm_encode(self, text: str) -> list[list[int]]:
        list_str = self._pre_split(text)

        list_ids = []
        for word in list_str:
            ids = self.vocab_files.get(word, None)
            if ids:
                list_ids.append(ids)
            else:
                merge = self.get_path_to_merges_file()
                print("choufleur")

        return [list_ids]
