from pydantic import BaseModel


class OutputModel(BaseModel):
    """Pydantic model representing the structured output of one LLM call.

    Stores the original user prompt, the name of the function the model
    decided to call, and the dictionary of argument values it produced.
    """
    prompt: str
    name: str
    parameters: dict[str, str | float | int | bool]
