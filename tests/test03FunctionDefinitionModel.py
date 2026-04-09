from src.models.FunctionModel import (
    FunctionModel,
)


class TestFunctionDefinitionModel:
    def test_valid_input(self) -> None:
        result = FunctionModel(
            name="fn_get_square_root",
            description="Calculate the square root of a number.",
            parameters={"a": "number"},
            returns="number",
        )
        assert result.model_dump() == {
            "name": "fn_get_square_root",
            "description": "Calculate the square root of a number.",
            "parameters": {"a": "number"},
            "returns": "number",
        }
