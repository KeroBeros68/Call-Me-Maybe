from unittest.mock import MagicMock, mock_open, patch

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
                "get_path_to_vocab_file",
                return_value="/fake/vocab.json",
            ),
            patch.object(
                LLMCustom,
                "get_path_to_merges_file",
                return_value="/fake/merges.txt",
            ),
            patch(
                "builtins.open",
                mock_open(read_data="#version: 0.2\nH ello\nĠ W"),
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
            self.llm_model.reversed_vocab = {
                v: k for k, v in self.llm_model.vocab_files.items()
            }

    def test_encode(self) -> None:
        result = self.llm_model.encode("Hello")
        assert result.tolist() == [[9707]]

    def test_decode(self) -> None:
        result = self.llm_model.decode([9707])
        assert result == "Hello"
