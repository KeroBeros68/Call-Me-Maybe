from pydantic import BaseModel


class OutputModel(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | float | int | bool]
