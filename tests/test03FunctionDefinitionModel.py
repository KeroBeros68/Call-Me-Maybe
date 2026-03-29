import pytest

from src.models.FunctionDefinitionModel import (
    FunctionDefinitionModel,
    ParameterModel,
    ReturnModel,
)


class TestFunctionDefinitionModel:
    def test_valid_input(self) -> None:
        result = FunctionDefinitionModel.model_validate(
            {
                "function_list": [
                    {
                        "name": "fn_get_square_root",
                        "description": "Calculate the square root of "
                        "a number.",
                        "parameters": {"a": {"type": "number"}},
                        "returns": {"type": "number"},
                    }
                ]
            }
        )
        assert result.model_dump() == {
            "function_list": [
                {
                    "name": "fn_get_square_root",
                    "description": "Calculate the square root of a number.",
                    "parameters": {"a": {"type": "number"}},
                    "returns": {"type": "number"},
                }
            ]
        }

    def test_empty_list_function(self) -> None:
        with pytest.raises(ValueError):
            FunctionDefinitionModel.model_validate({"function_list": []})

    def test_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            ParameterModel.model_validate(
                {"type": "BANANA"},
            )

    def test_invalid_returns(self) -> None:
        with pytest.raises(ValueError):
            ReturnModel.model_validate(
                {"type": "BANANA"},
            )
