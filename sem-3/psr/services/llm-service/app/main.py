from __future__ import annotations

from fastapi import FastAPI
import requests
from pydantic import BaseModel, ConfigDict, Field

from shared import config
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


def _ollama_embed(text: str) -> EmbedResponse:
    response = requests.post(
        f"{config.env('OLLAMA_BASE_URL', config.OLLAMA_BASE_URL)}/api/embed",
        json={"model": config.env("OLLAMA_EMBED_MODEL", config.OLLAMA_EMBED_MODEL), "input": text},
        timeout=float(config.env("OLLAMA_TIMEOUT_SECONDS", str(config.OLLAMA_TIMEOUT_SECONDS))),
    )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings") or []
    if not embeddings:
        raise RuntimeError("Ollama returned no embeddings")
    embedding = embeddings[0]
    return EmbedResponse(
        embedding=embedding,
        model_version=f"ollama:{config.env('OLLAMA_EMBED_MODEL', config.OLLAMA_EMBED_MODEL)}:{len(embedding)}",
        dimensions=len(embedding),
    )


def _deterministic_embed(text: str) -> EmbedResponse:
    dimensions = config.embedding_dim()
    return EmbedResponse(
        embedding=deterministic_embedding(text, dimensions),
        model_version=f"local-hash-{dimensions}",
        dimensions=dimensions,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "llm-service", "provider": config.env("LLM_PROVIDER", config.LLM_PROVIDER)}


@app.post("/v1/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    provider = config.env("LLM_PROVIDER", config.LLM_PROVIDER)
    if provider == "ollama":
        return _ollama_embed(request.text)
    if provider == "ollama-with-fallback":
        try:
            return _ollama_embed(request.text)
        except Exception:
            return _deterministic_embed(request.text)
    return _deterministic_embed(request.text)


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    provider = config.env("LLM_PROVIDER", config.LLM_PROVIDER)
    if provider.startswith("ollama"):
        generate_model = config.env("OLLAMA_GENERATE_MODEL", config.OLLAMA_GENERATE_MODEL)
        response = requests.post(
            f"{config.env('OLLAMA_BASE_URL', config.OLLAMA_BASE_URL)}/api/generate",
            json={"model": generate_model, "prompt": request.prompt, "stream": False},
            timeout=float(config.env("OLLAMA_TIMEOUT_SECONDS", str(config.OLLAMA_TIMEOUT_SECONDS))),
        )
        response.raise_for_status()
        payload = response.json()
        return GenerateResponse(text=payload.get("response", "").strip(), provider=f"ollama:{generate_model}")

    title = request.context.get("title", "this book")
    reason = request.context.get("reason", "it matches your reading history")
    return GenerateResponse(text=f"Recommended {title} because {reason}.", provider="local-template")
