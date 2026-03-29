from pydantic import BaseModel, Field


class PromptModel(BaseModel):
    prompt: str = Field(
        min_length=1,
        description="A natural language instruction to be "
        "executed by the model.",
    )


class InputModel(BaseModel):
    input_list: list[PromptModel] = Field(
        min_length=1,
        description="A non-empty list of prompts to be "
        "processed sequentially.",
    )
