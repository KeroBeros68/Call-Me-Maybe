from unittest.mock import MagicMock, patch

from src.llm_custom.LLMCustom import LLMCustom


class TestEncodeDecode:

    def setup_method(self) -> None:
        with (
            patch(
                "src.llm_custom.LLMCustom.Small_LLM_Model.__init__",
                return_value=None,
            ),
            patch.object(
                LLMCustom,
                "_get_tokenizer_file",
                return_value={
                    "model": {"vocab": {}, "merges": []},
                    "added_tokens": [],
                },
            ),
        ):

            mock_reader = MagicMock()
            mock_reader.read_file.return_value = {
                "Hello": 9707,
                "ĠWorld": 1879,
            }
            self.llm_model = LLMCustom(reader=MagicMock())
            self.llm_model._device = "cpu"
            self.llm_model._tokenizer = MagicMock()
            self.llm_model.vocab_files = {"Hello": 9707, "ĠWorld": 1879}
            self.llm_model.merge_file = ["H ello", "Ġ W"]
            self._extended_tokken = [{"content": "coucou", "id": 18000}]
            self.llm_model.reversed_vocab = {
                v: k for k, v in self.llm_model.vocab_files.items()
            }

    def test_encode(self) -> None:
        result = self.llm_model.encode("Hello")
        assert result.tolist() == [[9707]]

    def test_decode(self) -> None:
        result = self.llm_model.decode([9707])
        assert result == "Hello"
