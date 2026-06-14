from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared import config
from shared.text import deterministic_embedding

app = FastAPI(title="Book AI Library - LLM Service", version="0.1.0")
logger = logging.getLogger("llm-service")


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


def _ollama_timeout() -> float:
    return float(config.env("OLLAMA_TIMEOUT_SECONDS", str(config.OLLAMA_TIMEOUT_SECONDS)))


def _ollama_num_predict() -> int:
    return int(config.env("OLLAMA_NUM_PREDICT", str(config.OLLAMA_NUM_PREDICT)))


def _ollama_think() -> bool:
    return config.env("OLLAMA_THINK", config.OLLAMA_THINK).lower() in {"1", "true", "yes", "on"}


def _raise_upstream_error(exc: Exception, provider: str, operation: str, timeout_seconds: float | None = None) -> None:
    if isinstance(exc, requests.Timeout):
        raise HTTPException(
            status_code=504,
            detail=f"{provider} {operation} timed out after {(timeout_seconds or 0):.0f} seconds",
        ) from exc
    if isinstance(exc, requests.RequestException):
        raise HTTPException(status_code=502, detail=f"{provider} {operation} failed: {exc}") from exc
    raise HTTPException(status_code=502, detail=f"{provider} {operation} failed: {exc}") from exc


def _ollama_embed(text: str) -> EmbedResponse:
    try:
        response = requests.post(
            f"{config.env('OLLAMA_BASE_URL', config.OLLAMA_BASE_URL)}/api/embed",
            json={"model": config.env("OLLAMA_EMBED_MODEL", config.OLLAMA_EMBED_MODEL), "input": text},
            timeout=_ollama_timeout(),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        _raise_upstream_error(exc, "Ollama", "embedding", _ollama_timeout())
    embeddings = payload.get("embeddings") or []
    if not embeddings:
        raise HTTPException(status_code=502, detail="Ollama embedding failed: no embeddings returned")
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
    timeout_seconds = float(config.env("AZURE_OPENAI_TIMEOUT_SECONDS", "30"))
    try:
        response = requests.post(
            _azure_openai_url(deployment, "embeddings"),
            headers=_azure_openai_headers(),
            json={"input": text},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        _raise_upstream_error(exc, "Azure OpenAI", "embedding", timeout_seconds)
    embedding = payload["data"][0]["embedding"]
    return EmbedResponse(
        embedding=embedding,
        model_version=f"azure-openai:{deployment}:{len(embedding)}",
        dimensions=len(embedding),
    )


def _azure_openai_generate(prompt: str) -> GenerateResponse:
    deployment = config.env("AZURE_OPENAI_CHAT_DEPLOYMENT", config.AZURE_OPENAI_CHAT_DEPLOYMENT)
    timeout_seconds = float(config.env("AZURE_OPENAI_TIMEOUT_SECONDS", "30"))
    try:
        response = requests.post(
            _azure_openai_url(deployment, "chat/completions"),
            headers=_azure_openai_headers(),
            json={"messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        _raise_upstream_error(exc, "Azure OpenAI", "generation", timeout_seconds)
    text = payload["choices"][0]["message"]["content"].strip()
    return GenerateResponse(text=text, provider=f"azure-openai:{deployment}")


def _ollama_generate(prompt: str) -> GenerateResponse:
    generate_model = config.env("OLLAMA_GENERATE_MODEL", config.OLLAMA_GENERATE_MODEL)
    timeout_seconds = _ollama_timeout()
    try:
        response = requests.post(
            f"{config.env('OLLAMA_BASE_URL', config.OLLAMA_BASE_URL)}/api/generate",
            json={
                "model": generate_model,
                "prompt": prompt,
                "stream": False,
                "think": _ollama_think(),
                "options": {"num_predict": _ollama_num_predict()},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        _raise_upstream_error(exc, "Ollama", "generation", timeout_seconds)
    text = payload.get("response", "").strip()
    if not text:
        _raise_upstream_error(RuntimeError("empty response returned"), "Ollama", "generation", timeout_seconds)
    return GenerateResponse(text=text, provider=f"ollama:{generate_model}")


def _deterministic_generate(request: GenerateRequest) -> GenerateResponse:
    title = request.context.get("title", "this book")
    reason = request.context.get("reason", "it matches your reading history")
    return GenerateResponse(text=f"Recommended {title} because {reason}.", provider="local-template")


def _fallback_reason(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc)


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
        "ollama_num_predict": str(config.env("OLLAMA_NUM_PREDICT", str(config.OLLAMA_NUM_PREDICT))),
        "ollama_timeout_seconds": str(config.env("OLLAMA_TIMEOUT_SECONDS", str(config.OLLAMA_TIMEOUT_SECONDS))),
        "ollama_think": str(config.env("OLLAMA_THINK", config.OLLAMA_THINK)),
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
        except Exception as exc:
            logger.warning("Ollama embedding fallback activated: %s", _fallback_reason(exc))
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
    if provider == "ollama":
        return _ollama_generate(prompt)
    if provider == "ollama-with-fallback":
        try:
            return _ollama_generate(prompt)
        except Exception as exc:
            logger.warning("Ollama generation fallback activated: %s", _fallback_reason(exc))
            return _deterministic_generate(request)
    return _deterministic_generate(request)


@app.get("/v1/generate", response_model=GenerateResponse)
def generate_get(prompt: str) -> GenerateResponse:
    return generate(GenerateRequest(prompt=prompt))
