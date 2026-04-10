from typing import Self

from pydantic import BaseModel, Field, model_validator


class PromptModel(BaseModel):
    """Pydantic model representing a single user prompt.

    Stores the natural-language instruction text and automatically
    escapes backslashes and double-quotes so the prompt can be safely
    embedded inside a JSON string.
    """
    prompt: str = Field(
        description="A natural language instruction to be "
        "executed by the model.",
    )

    @model_validator(mode="after")
    def replace_bad_char(self) -> Self:
        """Escape characters that would break JSON embedding.

        Replaces bare backslashes with ``\\\\`` and bare double-quotes
        with ``\\"`` so the prompt can be embedded verbatim inside a
        JSON string value.

        Returns:
            Self: The current instance with the sanitised prompt.
        """
        if "\\" in self.prompt:
            self.prompt = self.prompt.replace('\\', '\\\\')
        if '"' in self.prompt:
            self.prompt = self.prompt.replace('"', '\\"')
        return self
