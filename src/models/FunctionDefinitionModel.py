from typing import Literal

from pydantic import BaseModel, Field


class ParameterModel(BaseModel):
    type: Literal["string", "number"] = Field(
        description="The type of the parameter."
    )


class ReturnModel(BaseModel):
    type: Literal["number", "string"] = Field(
        description="The type of the return value."
    )


class DefinitionModel(BaseModel):
    name: str = Field(
        min_length=1,
        description="The name of the function.",
    )
    description: str = Field(
        min_length=1,
        description="A natural language description of what "
        "the function does.",
    )
    parameters: dict[str, ParameterModel] = Field(
        min_length=1,
        description="A mapping of parameter names to their type definitions.",
    )
    returns: ReturnModel = Field(
        description="The return type of the function.",
    )


class FunctionDefinitionModel(BaseModel):
    function_list: list[DefinitionModel] = Field(
        min_length=1,
        description="A non-empty list of function definitions.",
    )
