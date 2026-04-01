import heapq
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model
from src.utils.FileLoader.BaseLoader import BaseLoader


class LLMCustom(Small_LLM_Model):

    SPECIAL_CHAR_REPLACE: dict[str, str] = {
        " ": " \u0120",
        "\t": " \u0109",
        "\n": " \u010a",
        "\r": " \u010d",
    }

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
        self.merge_file: list[str] = self._get_merge_file()

    def _get_merge_file(self) -> list[str]:
        try:
            with open(self.get_path_to_merges_file(), "r") as f:
                return f.read().split("\n")
        except FileNotFoundError:
            raise

    def _pre_split(self, text: str) -> list[str]:
        for c, replace_c in self.SPECIAL_CHAR_REPLACE.items():
            if c in text:
                text = text.replace(c, replace_c)

        tokens = text.split()
        return tokens

    def _bpe_algorithm(self, word: str):
        list_char = list(word)
        merge_priority: dict[str, int] = {
            pair: i for i, pair in enumerate(self.merge_file)
        }
        while True:
            queue_priority: list[tuple[int, int]] = []
            for i in range(len(list_char) - 1):
                pair = list_char[i] + " " + list_char[i + 1]
                if pair in merge_priority:
                    heapq.heappush(queue_priority, (merge_priority[pair], i))
            if not queue_priority:
                break
            priority, i = heapq.heappop(queue_priority)
            merged = list_char[i] + list_char[i + 1]
            list_char[i:i+2] = [merged]

        return [self.vocab_files.get(word, None) for word in list_char]

    def cmm_encode(self, text: str) -> list[list[int]]:
        list_str = self._pre_split(text)

        list_ids = []
        for word in list_str:
            ids = self.vocab_files.get(word, None)
            if ids:
                list_ids.append(ids)
            else:
                list_ids.extend(self._bpe_algorithm(word))
        return [list_ids]
