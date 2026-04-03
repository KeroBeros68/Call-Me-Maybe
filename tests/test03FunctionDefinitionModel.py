import pytest

from src.models.FunctionModel import (
    ArgModel,
    FunctionModel,
)


class TestFunctionDefinitionModel:
    def test_valid_input(self) -> None:
        result = FunctionModel(
            name="fn_get_square_root",
            description="Calculate the square root of a number.",
            parameters={"a": {"type": "number"}},
            returns={"type": "number"},
        )
        assert result.model_dump() == {
            "name": "fn_get_square_root",
            "description": "Calculate the square root of a number.",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "number"},
        }

    def test_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            ArgModel.model_validate(
                {"type": "BANANA"}
            )
