from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_ollama_runs_internal_only_and_pull_uses_shell_entrypoint():
    data = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    ollama = data["services"]["ollama"]
    pull = data["services"]["ollama-pull"]

    assert "ports" not in ollama
    assert ollama["volumes"] == ["ollama-models:/root/.ollama"]
    assert pull["entrypoint"] == ["/bin/sh", "-lc"]
    assert pull["command"] == ["ollama pull $$OLLAMA_EMBED_MODEL && ollama pull $$OLLAMA_GENERATE_MODEL"]


def test_ollama_defaults_to_gemma4_edge_generation_model():
    data = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    assert data["services"]["llm-service"]["environment"]["LLM_PROVIDER"] == "${LLM_PROVIDER:-ollama-with-fallback}"
    assert data["services"]["llm-service"]["environment"]["OLLAMA_GENERATE_MODEL"] == "${OLLAMA_GENERATE_MODEL:-gemma4:e2b}"
    assert data["services"]["llm-service"]["environment"]["OLLAMA_THINK"] == "${OLLAMA_THINK:-false}"
