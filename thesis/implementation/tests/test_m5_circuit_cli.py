from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from quantum_bench.bench.m5_circuit_commands import (
    _physical_factory,
    baseline_paths,
    parse_rank_paths,
    report,
)


def test_rank_paths_require_exact_device_names_and_unique_values() -> None:
    assert parse_rank_paths("/dev/dpu_rank0,/dev/dpu_rank12") == [
        "/dev/dpu_rank0",
        "/dev/dpu_rank12",
    ]
    with pytest.raises(ValueError, match="explicit"):
        parse_rank_paths("/dev/dpu_rank0-extra")
    with pytest.raises(ValueError, match="unique"):
        parse_rank_paths("/dev/dpu_rank0,/dev/dpu_rank0")


def test_rank_path_environment_fallback_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UPMEM_HW_RANK_PATHS", raising=False)
    monkeypatch.setenv("UPMEM_HW_RANK_PATH", "/dev/dpu_rank3")
    assert parse_rank_paths() == ["/dev/dpu_rank3"]


def test_physical_factory_selects_explicit_tasklet_binaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = tmp_path / "native/upmem/simplepim/upmem_sdk_execution_plan/bin"
    native.mkdir(parents=True)
    binaries = (
        native / "host_upmem_execution_plan_v4_t4",
        native / "dpu_gemm_tile_v4_t4",
        native / "dpu_simplepim_management_init",
    )
    for binary in binaries:
        binary.write_bytes(b"binary")
    os.chmod(binaries[0], 0o755)

    module = importlib.import_module(
        "quantum_bench.targets.upmem.m5_whole_circuit_engine"
    )
    captured: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(module, "M5WholeCircuitEngine", FakeEngine)
    factory = _physical_factory(tmp_path)
    engine = factory(
        topology=SimpleNamespace(
            backend="upmem",
            device_ids=("dpu:0", "dpu:1"),
            tasklets_per_device=4,
        ),
        engine_variant={
            "id": "upmem_test",
            "topology": {"rank_paths": ["/dev/dpu_rank0"]},
        },
        timeout_s=42.0,
    )

    assert isinstance(engine, FakeEngine)
    assert captured["host_binary"] == binaries[0]
    assert captured["dpu_binary"] == binaries[1]
    assert captured["initialization_binary"] == binaries[2]
    assert captured["rank_paths"] == ("/dev/dpu_rank0",)
    assert captured["dpu_count"] == 2
    assert captured["tasklets_per_dpu"] == 4
    assert captured["timeout_s"] == 42.0
    assert str(captured["session_root"]).startswith(
        str(tmp_path / "build/m5_circuit_sessions/upmem_test-")
    )


def test_report_baselines_are_repeatable_or_environment_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert baseline_paths(["/tmp/a", "/tmp/b"]) == (Path("/tmp/a"), Path("/tmp/b"))
    monkeypatch.setenv("M5_CIRCUIT_BASELINES", "/tmp/c,/tmp/d")
    assert baseline_paths() == (Path("/tmp/c"), Path("/tmp/d"))


def test_report_merges_external_records_without_rerunning(tmp_path: Path) -> None:
    row = {
        "case_id": "bv-4",
        "family": "BV",
        "qubits": 4,
        "engine_id": "cpu_numpy",
        "path_variant_id": "greedy",
        "numeric_policy": "float32",
        "repeat_id": 0,
        "timing_scope": "whole_circuit_steady_state_v1",
        "status": "completed",
        "validation_status": "passed",
        "scientific_validation_status": "passed",
        "exact_once": True,
        "no_fallback_used": True,
        "executor_config_hash": "cpu-v1",
        "timing_s": 1.0,
        "circuit_semantics_hash": "circuit",
        "tensor_network_hash": "network",
        "contraction_plan_hash": "plan",
    }
    main = tmp_path / "main.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    main.write_text(json.dumps(row) + "\n", encoding="utf-8")
    baseline.write_text(
        json.dumps({**row, "engine_id": "quest_cpu_full_state"}) + "\n",
        encoding="utf-8",
    )
    result = report(main, tmp_path / "report", baselines=(baseline,))
    runtime = (Path(result["report_dir"]) / "tables/runtime_by_qubits.csv").read_text()
    assert runtime.count("bv-4") == 2


def test_cli_physical_execution_fails_closed_with_exit_two() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "UPMEM_ALLOW_PHYSICAL_HARDWARE",
            "UPMEM_HW_RANK_PATH",
            "UPMEM_HW_RANK_PATHS",
        }
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "quantum_bench.bench",
            "m5-circuit-study",
            "--suite",
            "configs/suites/m5_circuit_smoke.yml",
            "--execute",
        ],
        cwd=Path(__file__).parents[1],
        env={**env, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE=1" in result.stderr


@pytest.mark.parametrize("variable", ["DPU_BACKEND", "UPMEM_EXECUTION_MODE"])
def test_cli_physical_execution_rejects_forbidden_backend_environment(
    variable: str,
) -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"DPU_BACKEND", "UPMEM_EXECUTION_MODE"}
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "quantum_bench.bench",
            "m5-circuit-study",
            "--suite",
            "configs/suites/m5_circuit_smoke.yml",
            "--execute",
            "--rank-paths",
            "/dev/dpu_rank0",
        ],
        cwd=Path(__file__).parents[1],
        env={
            **env,
            "PYTHONPATH": "src",
            "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
            variable: "",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert variable in result.stderr
