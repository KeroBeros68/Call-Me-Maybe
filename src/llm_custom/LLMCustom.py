import json
from typing import Any

import torch

from llm_sdk import Small_LLM_Model  # type: ignore
from src.utils.FileLoader.BaseLoader import BaseLoader


class LLMCustom(Small_LLM_Model):  # type: ignore[misc]
    """Custom LLM wrapper with a built-in BPE tokeniser.

    Extends :class:`Small_LLM_Model` with a pure-Python BPE encoder/decoder
    that reads merge rules and vocabulary directly from the model's
    ``tokenizer.json`` file, enabling deterministic, dependency-light
    tokenisation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: Any | None = None,
        dtype: Any | None = None,
        trust_remote_code: bool = True,
        reader: BaseLoader,
    ) -> None:
        """Initialize LLMCustom and load the BPE tokeniser from disk.

        After calling the parent ``__init__``, the tokeniser vocabulary,
        merge rules, and special tokens are read from the model's
        ``tokenizer.json`` file.  Several lookup structures (reverse vocab,
        merge priority map, translation tables) are pre-built to make
        :meth:`encode` and :meth:`decode` as fast as possible.

        Args:
            model_name (str): HuggingFace model identifier or local path.
                Defaults to ``"Qwen/Qwen3-0.6B"``.
            device: Target device (e.g. ``"cuda"`` or ``torch.device``).  If
                ``None`` the parent class selects automatically.
            dtype: Torch dtype override.  If ``None`` the parent class default
                is used.
            trust_remote_code (bool): Passed to the parent loader.  Defaults
                to ``True``.
            reader (BaseLoader): File-loader used to open ``tokenizer.json``.
        """
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
        _raw_merges: list[str] = (
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
        self._encode_table = str.maketrans({
            " ": "\u0120",
            "\t": "\u0109",
            "\n": "\u010a",
            "\r": "\u010d",
        })
        self._decode_table = str.maketrans({
            "\u0120": " ",
            "\u0109": "\t",
            "\u010a": "\n",
            "\u010d": "\r",
        })

    @property
    def vocab_size(self) -> int:
        """Return the number of embeddings in the model's embedding layer.

        Returns:
            int: The vocabulary size as reported by the model's input
                embedding matrix.
        """
        num_embeddings = self._model.get_input_embeddings().num_embeddings
        assert isinstance(num_embeddings, int)
        return num_embeddings

    def _get_tokenizer_file(self) -> Any:
        """Load and return the raw ``tokenizer.json`` content.

        Returns:
            Any: The parsed JSON object (typically a dict) from the tokeniser
                file associated with the current model.

        Raises:
            FileNotFoundError: If the tokeniser file path cannot be found.
        """
        try:
            with open(self.get_path_to_tokenizer_file(), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise

    def _pre_split(self, text: str) -> list[str]:
        """Translate whitespace characters and split text into BPE words.

        Replaces space, tab, newline, and carriage-return with their
        Unicode surrogate representations (e.g. ``\u0120`` for space)
        before splitting on whitespace boundaries, mirroring the
        pre-tokenisation step of the original GPT-2 / Qwen BPE implementation.

        Args:
            text (str): Raw input text to pre-tokenise.

        Returns:
            list[str]: A list of pre-tokenised word strings ready for
                BPE encoding.
        """
        return text.translate(self._encode_table).split()

    def _bpe_algorithm(self, word: str) -> list[int]:
        """Apply BPE merge rules to a single pre-tokenised word.

        Iteratively merges the highest-priority adjacent character pair
        (lowest index in :attr:`merge_file`) until no more merges are
        possible.  The result is cached in :attr:`_custom_cache` to
        avoid redundant work on repeated words.

        Args:
            word (str): A single pre-tokenised word (after whitespace
                translation by :meth:`_pre_split`).

        Returns:
            list[int]: Ordered list of vocabulary token IDs for the word.

        Raises:
            KeyError: If a merged sub-word is not present in the vocabulary.
        """
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
        """Tokenise a string into a 2-D token-ID tensor.

        Special tokens are identified first and kept as atomic units;
        the remaining text segments are handled by :meth:`_pre_split` and
        :meth:`_bpe_algorithm`.  Results are returned as a ``(1, seq_len)``
        :class:`torch.Tensor` on the model's device.

        Args:
            text (str): The input string to tokenise.

        Returns:
            torch.Tensor: A ``(1, N)`` long tensor of token IDs.
        """
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
        """Convert a sequence of token IDs back to a human-readable string.

        Accepts either a :class:`torch.Tensor` or a plain list of integers
        (including nested lists, which are flattened).  Each ID is looked up in
        :attr:`reversed_vocab`; unknown IDs are replaced with ``"<unk>"``.
        Whitespace surrogate characters are translated back to their original
        forms via :attr:`_decode_table`.

        Args:
            list_ids (torch.Tensor | list[int]): Token IDs to decode.

        Returns:
            str: The decoded human-readable string.
        """
        if isinstance(list_ids, torch.Tensor):
            list_ids = list_ids.flatten().tolist()
        elif list_ids and isinstance(list_ids[0], list):
            list_ids = [item for sublist in list_ids for item in sublist]

        result = "".join(
            self.reversed_vocab.get(ids, "<unk>") for ids in list_ids
        )
        return result.translate(self._decode_table)
