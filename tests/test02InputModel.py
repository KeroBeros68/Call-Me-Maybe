import pytest

from src.models.InputModel import PromptModel


class TestInputModel:
    def test_valid_input(self) -> None:
        result = PromptModel.model_validate(
            {"prompt": "Hello there!"},
        )
        assert result.model_dump() == {"prompt": "Hello there!"}

    def test_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            PromptModel.model_validate(
                {"hello": "Hello there!"},
            )
