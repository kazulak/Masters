from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from shared.config import embedding_dim
from shared.text import deterministic_embedding

app = FastAPI(title="Book AI Library - LLM Service", version="0.1.0")


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1)


class EmbedResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    embedding: list[float]
    model_version: str
    dimensions: int


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    context: dict = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    text: str
    provider: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "llm-service"}


@app.post("/v1/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    dimensions = embedding_dim()
    return EmbedResponse(
        embedding=deterministic_embedding(request.text, dimensions),
        model_version=f"local-hash-{dimensions}",
        dimensions=dimensions,
    )


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    title = request.context.get("title", "this book")
    reason = request.context.get("reason", "it matches your reading history")
    return GenerateResponse(text=f"Recommended {title} because {reason}.", provider="local-template")
