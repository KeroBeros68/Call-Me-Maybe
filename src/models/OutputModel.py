from pydantic import BaseModel


class OutputModel(BaseModel):
    name: str
    parameters: dict[str, str | float | int | bool]
