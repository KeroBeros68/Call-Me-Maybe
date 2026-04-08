from pydantic import BaseModel, Field, model_validator


class PromptModel(BaseModel):
    prompt: str = Field(
        description="A natural language instruction to be "
        "executed by the model.",
    )

    @model_validator(mode="after")
    def replace_bad_char(self):
        if "\\" in self.prompt:
            self.prompt = self.prompt.replace('\\', '\\\\')
        if '"' in self.prompt:
            self.prompt = self.prompt.replace('"', '\\"')
        return self
