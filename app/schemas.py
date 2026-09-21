
from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role: str
    content: str

    model_config = ConfigDict(extra="allow")

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool | None = False

    model_config = ConfigDict(extra="allow")

class ModelCard(BaseModel):
    id: str

class ModelListResponse(BaseModel):
    data: list[ModelCard]
