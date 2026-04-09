import json
from typing import Any

import torch

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
    ) -> None:
        super().__init__(
            model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.reader: BaseLoader = reader
        self.tokenizer_file: dict[str, Any] = self._get_tokenizer_file()

        _model = self.tokenizer_file.get("model")
        self._special_tokens: list[dict[str, Any]] = self.tokenizer_file.get(
            "added_tokens", []
        )
        self.vocab_files: dict[str, int] = (
            _model["vocab"] if isinstance(_model, dict) else {}
        )
        _raw_merges: list = (
            _model["merges"] if isinstance(_model, dict) else []
        )
        self.merge_file: list[str] = [
            " ".join(m) if isinstance(m, list) else m for m in _raw_merges
        ]

        self._custom_cache: dict[str, list[int]] = {}
        self.merge_priority: dict[str, int] = {
            pair: i for i, pair in enumerate(self.merge_file)
        }

        for token in self._special_tokens:
            self.vocab_files[token["content"]] = token["id"]

        self.reversed_vocab: dict[int, str] = {
            v: k for k, v in self.vocab_files.items()
        }
        self.sorted_special_tokens = sorted(
            self._special_tokens, key=lambda t: len(t["content"]), reverse=True
        )

    @property
    def vocab_size(self) -> int:
        num_embeddings = self._model.get_input_embeddings().num_embeddings
        assert isinstance(num_embeddings, int)
        return num_embeddings

    def _get_tokenizer_file(self) -> dict[str, Any]:
        try:
            with open(self.get_path_to_tokenizer_file(), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise

    def _pre_split(self, text: str) -> list[str]:
        for c, replace_c in self.SPECIAL_CHAR_REPLACE.items():
            if c in text:
                text = text.replace(c, replace_c)

        tokens = text.split()
        return tokens

    def _bpe_algorithm(self, word: str) -> list[int]:
        list_char = list(word)
        while len(list_char) > 1:
            best = None
            best_i = -1
            for i in range(len(list_char) - 1):
                pair = list_char[i] + " " + list_char[i + 1]
                p = self.merge_priority.get(pair)
                if p is not None and (best is None or p < best):
                    best = p
                    best_i = i
            if best is None:
                break
            list_char[best_i: best_i + 2] = [
                list_char[best_i] + list_char[best_i + 1]
            ]

        list_ids = [self.vocab_files.get(char, -1) for char in list_char]
        if -1 in list_ids:
            raise KeyError
        self._custom_cache[word] = list_ids
        return list_ids

    def encode(self, text: str) -> torch.Tensor:
        segments: list[tuple[str, bool]] = [(text, False)]

        for token in self.sorted_special_tokens:
            new_segments = []
            for segment, is_special in segments:
                if is_special:
                    new_segments.append((segment, True))
                    continue
                parts = segment.split(token["content"])
                for i, part in enumerate(parts):
                    if part:
                        new_segments.append((part, False))
                    if i < len(parts) - 1:
                        new_segments.append((token["content"], True))
            segments = new_segments

        list_ids = []
        for segment, is_special in segments:
            if is_special:
                token_id = self.vocab_files.get(segment)
                if token_id is not None:
                    list_ids.append(token_id)
            else:
                for word in self._pre_split(segment):
                    ids = self.vocab_files.get(word)
                    if ids is not None:
                        list_ids.append(ids)
                    elif word in self._custom_cache:
                        list_ids.extend(self._custom_cache[word])
                    else:
                        list_ids.extend(self._bpe_algorithm(word))
        return torch.tensor([list_ids], dtype=torch.long, device=self._device)

    def decode(self, list_ids: torch.Tensor | list[int]) -> str:
        if isinstance(list_ids, torch.Tensor):
            list_ids = list_ids.flatten().tolist()
        elif list_ids and isinstance(list_ids[0], list):
            list_ids = [item for sublist in list_ids for item in sublist]

        result = "".join(
            self.reversed_vocab.get(ids, "<unk>") for ids in list_ids
        )
        for c, replace_c in self.SPECIAL_CHAR_REPLACE.items():
            result = result.replace(replace_c.strip(), c)
        return result
