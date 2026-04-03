from typing import Literal

from pydantic import BaseModel


class ArgModel(BaseModel):
    type: Literal["string", "number"]


class OutputModel(BaseModel):
    name: str
    arguments: dict[str, ArgModel]
