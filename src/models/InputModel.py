from pydantic import BaseModel, Field


class PromptModel(BaseModel):
    prompt: str = Field(
        description="A natural language instruction to be "
        "executed by the model.",
    )
