from typing import Literal

from pydantic import BaseModel, Field


class FunctionModel(BaseModel):
    """Pydantic model representing a callable function definition.

    Validates and stores the name, natural-language description,
    typed parameter schema, and return type of a single function
    that the LLM may be asked to call.
    """
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
        str,
        dict[
            Literal["type"], Literal["string", "number", "integer", "boolean"]
        ],
    ] = Field(
        min_length=1,
        description="A mapping of parameter names to their type definitions.",
    )
    returns: dict[
        Literal["type"], Literal["string", "number", "integer", "boolean"]
    ] = Field(
        description="The return type of the function.",
    )
