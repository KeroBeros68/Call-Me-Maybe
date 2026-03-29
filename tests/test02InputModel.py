import pytest

from src.models.InputModel import InputModel


class TestInputModel:
    def test_valid_input(self) -> None:
        result = InputModel.model_validate(
            {
                "input_list": [
                    {"prompt": "Hello there!"},
                    {"prompt": "Hello there!"},
                    {"prompt": "Hello there!"},
                ]
            }
        )
        assert result.model_dump() == {
            "input_list": [
                {"prompt": "Hello there!"},
                {"prompt": "Hello there!"},
                {"prompt": "Hello there!"},
            ]
        }

    def test_empty_list_input(self) -> None:
        with pytest.raises(ValueError):
            InputModel.model_validate({"input_list": []})

    def test_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            InputModel.model_validate(
                {
                    "input_list": [
                        {"hello": "Hello there!"},
                        {"prompt": "Hello there!"},
                    ]
                }
            )
