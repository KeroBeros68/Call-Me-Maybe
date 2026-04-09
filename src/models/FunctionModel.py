from typing import Literal

from pydantic import BaseModel, Field


class FunctionModel(BaseModel):
    name: str = Field(
        min_length=1,
        description="The name of the function.",
    )
    description: str = Field(
        min_length=1,
        description="A natural language description of what "
        "the function does.",
    )
    parameters: dict[
        str, Literal["string", "number", "integer", "boolean"]
    ] = Field(
        min_length=1,
        description="A mapping of parameter names to their type definitions.",
    )
    returns: Literal["string", "number", "integer", "boolean"] = Field(
        description="The return type of the function.",
    )
