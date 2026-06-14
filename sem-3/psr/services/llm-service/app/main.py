from __future__ import annotations

from fastapi import FastAPI
import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    prompt: str | None = Field(default=None, min_length=1)
    text: str | None = Field(default=None, min_length=1)
    context: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_prompt(self) -> "GenerateRequest":
        if not self.prompt and not self.text:
            raise ValueError("prompt or text is required")
        return self

    def prompt_text(self) -> str:
        return self.prompt or self.text or ""


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


def _azure_openai_headers() -> dict[str, str]:
    key = config.env("AZURE_OPENAI_API_KEY", config.AZURE_OPENAI_API_KEY)
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is required for LLM_PROVIDER=azure-openai")
    return {"api-key": key, "Content-Type": "application/json"}


def _azure_openai_url(deployment: str, operation: str) -> str:
    endpoint = config.env("AZURE_OPENAI_ENDPOINT", config.AZURE_OPENAI_ENDPOINT).rstrip("/")
    api_version = config.env("AZURE_OPENAI_API_VERSION", config.AZURE_OPENAI_API_VERSION)
    if not endpoint or not deployment:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and deployment env vars are required for Azure OpenAI")
    return f"{endpoint}/openai/deployments/{deployment}/{operation}?api-version={api_version}"


def _azure_openai_embed(text: str) -> EmbedResponse:
    deployment = config.env("AZURE_OPENAI_EMBED_DEPLOYMENT", config.AZURE_OPENAI_EMBED_DEPLOYMENT)
    response = requests.post(
        _azure_openai_url(deployment, "embeddings"),
        headers=_azure_openai_headers(),
        json={"input": text},
        timeout=float(config.env("AZURE_OPENAI_TIMEOUT_SECONDS", "30")),
    )
    response.raise_for_status()
    payload = response.json()
    embedding = payload["data"][0]["embedding"]
    return EmbedResponse(
        embedding=embedding,
        model_version=f"azure-openai:{deployment}:{len(embedding)}",
        dimensions=len(embedding),
    )


def _azure_openai_generate(prompt: str) -> GenerateResponse:
    deployment = config.env("AZURE_OPENAI_CHAT_DEPLOYMENT", config.AZURE_OPENAI_CHAT_DEPLOYMENT)
    response = requests.post(
        _azure_openai_url(deployment, "chat/completions"),
        headers=_azure_openai_headers(),
        json={"messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
        timeout=float(config.env("AZURE_OPENAI_TIMEOUT_SECONDS", "30")),
    )
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"].strip()
    return GenerateResponse(text=text, provider=f"azure-openai:{deployment}")


@app.get("/")
def root() -> dict:
    return {
        "service": "llm-service",
        "provider": config.env("LLM_PROVIDER", config.LLM_PROVIDER),
        "docs": "/docs",
        "prompt_examples": {
            "GET": "/v1/generate?prompt=Recommend%20a%20science%20fiction%20book",
            "POST": {"url": "/v1/generate", "json": {"prompt": "Recommend a science fiction book"}},
        },
        "embedding_examples": {
            "GET": "/v1/embed?text=Dune",
            "POST": {"url": "/v1/embed", "json": {"text": "Dune"}},
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "llm-service", "provider": config.env("LLM_PROVIDER", config.LLM_PROVIDER)}


@app.get("/v1/models")
def models() -> dict[str, str]:
    return {
        "provider": config.env("LLM_PROVIDER", config.LLM_PROVIDER),
        "ollama_base_url": config.env("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL),
        "ollama_embed_model": config.env("OLLAMA_EMBED_MODEL", config.OLLAMA_EMBED_MODEL),
        "ollama_generate_model": config.env("OLLAMA_GENERATE_MODEL", config.OLLAMA_GENERATE_MODEL),
        "azure_openai_embed_deployment": config.env("AZURE_OPENAI_EMBED_DEPLOYMENT", config.AZURE_OPENAI_EMBED_DEPLOYMENT),
        "azure_openai_chat_deployment": config.env("AZURE_OPENAI_CHAT_DEPLOYMENT", config.AZURE_OPENAI_CHAT_DEPLOYMENT),
    }


@app.post("/v1/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    provider = config.env("LLM_PROVIDER", config.LLM_PROVIDER)
    if provider == "azure-openai":
        return _azure_openai_embed(request.text)
    if provider == "ollama":
        return _ollama_embed(request.text)
    if provider == "ollama-with-fallback":
        try:
            return _ollama_embed(request.text)
        except Exception:
            return _deterministic_embed(request.text)
    return _deterministic_embed(request.text)


@app.get("/v1/embed", response_model=EmbedResponse)
def embed_get(text: str) -> EmbedResponse:
    return embed(EmbedRequest(text=text))


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    provider = config.env("LLM_PROVIDER", config.LLM_PROVIDER)
    prompt = request.prompt_text()
    if provider == "azure-openai":
        return _azure_openai_generate(prompt)
    if provider.startswith("ollama"):
        generate_model = config.env("OLLAMA_GENERATE_MODEL", config.OLLAMA_GENERATE_MODEL)
        response = requests.post(
            f"{config.env('OLLAMA_BASE_URL', config.OLLAMA_BASE_URL)}/api/generate",
            json={"model": generate_model, "prompt": prompt, "stream": False},
            timeout=float(config.env("OLLAMA_TIMEOUT_SECONDS", str(config.OLLAMA_TIMEOUT_SECONDS))),
        )
        response.raise_for_status()
        payload = response.json()
        return GenerateResponse(text=payload.get("response", "").strip(), provider=f"ollama:{generate_model}")

    title = request.context.get("title", "this book")
    reason = request.context.get("reason", "it matches your reading history")
    return GenerateResponse(text=f"Recommended {title} because {reason}.", provider="local-template")


@app.get("/v1/generate", response_model=GenerateResponse)
def generate_get(prompt: str) -> GenerateResponse:
    return generate(GenerateRequest(prompt=prompt))
