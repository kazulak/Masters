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


def test_standalone_llm_compose_exposes_only_llm_service():
    data = yaml.safe_load((ROOT / "docker-compose.llm.yml").read_text(encoding="utf-8"))

    ollama = data["services"]["ollama"]
    llm_service = data["services"]["llm-service"]

    assert "ports" not in ollama
    assert ollama["volumes"] == ["ollama-models:/root/.ollama"]
    assert llm_service["ports"] == ["${LLM_SERVICE_PORT:-8005}:8000"]
    assert llm_service["depends_on"]["ollama-pull"]["condition"] == "service_completed_successfully"


def test_ollama_defaults_to_gemma4_edge_generation_model():
    data = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    standalone = yaml.safe_load((ROOT / "docker-compose.llm.yml").read_text(encoding="utf-8"))

    assert data["services"]["llm-service"]["environment"]["OLLAMA_GENERATE_MODEL"] == "${OLLAMA_GENERATE_MODEL:-gemma4:e2b}"
    assert standalone["services"]["llm-service"]["environment"]["OLLAMA_GENERATE_MODEL"] == "${OLLAMA_GENERATE_MODEL:-gemma4:e2b}"


def test_amd_overrides_use_rocm_ollama_and_gpu_devices():
    for filename in ("docker-compose.amd.yml", "docker-compose.llm.amd.yml"):
        data = yaml.safe_load((ROOT / filename).read_text(encoding="utf-8"))
        for service_name in ("ollama", "ollama-pull"):
            service = data["services"][service_name]
            assert service["image"] == "ollama/ollama:rocm"
            assert service["devices"] == ["/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri"]
