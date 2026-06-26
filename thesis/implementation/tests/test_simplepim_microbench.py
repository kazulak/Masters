from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantum_bench.core.records import to_jsonable
from quantum_bench.bench.simplepim_microbench import run_simplepim_microbench
from quantum_bench.targets.upmem import (
    SIMPLEPIM_DENSE_MICROBENCH_SCHEMA_VERSION,
    SimplePimDenseMicrobenchInput,
    make_simplepim_dense_microbench_input,
    prepare_simplepim_dense_microbench,
)
from quantum_bench.targets.upmem.simplepim import SimplePimProbeResult


def _probe(status: str, available: bool = False, reason: str | None = None) -> SimplePimProbeResult:
    return SimplePimProbeResult(
        simplepim_available=available,
        simplepim_probe_status=status,  # type: ignore[arg-type]
        simplepim_version=None,
        simplepim_home="/tmp/simplepim" if status == "configured_but_unverified" else None,
        simplepim_bin="/tmp/simplepim/bin/simplepim" if available else None,
        simplepim_library_path=None,
        simplepim_command_path="/tmp/simplepim/bin/simplepim" if available else None,
        skip_reason=reason,
        metadata={"external_command_executed": False},
    )


def _small_input() -> SimplePimDenseMicrobenchInput:
    return make_simplepim_dense_microbench_input(8, 8, 8, seed=123)


def test_unavailable_probe_returns_skipped_without_external_execution() -> None:
    result = prepare_simplepim_dense_microbench(
        _small_input(),
        probe=_probe("unavailable", reason="SimplePIM missing in test"),
    )
    payload = result.to_json_dict()

    assert result.status == "skipped"
    assert result.skip_reason == "SimplePIM missing in test"
    assert result.external_command_executed is False
    assert result.execution_implemented is False
    assert result.schema_version == SIMPLEPIM_DENSE_MICROBENCH_SCHEMA_VERSION
    assert payload["execution_implemented"] is False
    assert payload["external_command_executed"] is False


def test_configured_home_only_returns_configured_but_unverified() -> None:
    result = prepare_simplepim_dense_microbench(
        _small_input(),
        probe=_probe("configured_but_unverified", reason="home configured only"),
    )

    assert result.status == "configured_but_unverified"
    assert result.skip_reason == "home configured only"
    assert result.execution_implemented is False


def test_fake_bin_returns_ready_for_small_dry_run() -> None:
    result = prepare_simplepim_dense_microbench(_small_input(), probe=_probe("available", available=True))
    payload = result.to_json_dict()

    assert result.status == "ready"
    assert result.skip_reason is None
    assert result.error is None
    assert result.execution_implemented is False
    assert result.external_command_executed is False
    assert result.kernel_time_s is None
    assert result.host_aggregation_time_s is None
    assert result.input_shapes == ((8, 8), (8, 8))
    assert result.output_shape == (8, 8)
    assert result.host_to_dpu_bytes >= 0
    assert result.dpu_to_host_bytes >= 0
    assert result.mram_to_wram_bytes >= 0
    assert result.tile_plan["total_tile_count"] == 1
    assert result.upmem_task_estimate["estimated_tile_count"] == 1
    assert result.conversion_records["left"]["route_dtype"] == "int8"
    assert result.conversion_records["right"]["route_dtype"] == "int8"
    assert result.validation_metrics["max_abs_error"] >= 0.0
    assert payload["status"] == "ready"
    json.dumps(payload)


def test_large_tiling_status_takes_priority_over_missing_simplepim() -> None:
    result = prepare_simplepim_dense_microbench(
        make_simplepim_dense_microbench_input(256, 256, 256),
        probe=_probe("unavailable", reason="SimplePIM missing in test"),
    )

    assert result.status == "not_implemented"
    assert result.skip_reason == "requires_executable_tiling_not_implemented"
    assert result.tile_plan["requires_tiling"] is True
    assert result.tile_plan["total_tile_count"] > 1


def test_execute_request_is_not_implemented_and_never_executed() -> None:
    result = prepare_simplepim_dense_microbench(_small_input(), probe=_probe("available", available=True), execute=True)

    assert result.status == "not_implemented"
    assert result.skip_reason == "simplepim_execution_not_implemented"
    assert result.external_command_executed is False
    assert result.execution_implemented is False


def test_invalid_input_returns_failed_result() -> None:
    result = prepare_simplepim_dense_microbench(
        SimplePimDenseMicrobenchInput(gemm_m=0, gemm_k=8, gemm_n=8),
        probe=_probe("available", available=True),
    )

    assert result.status == "failed"
    assert result.error
    assert result.execution_implemented is False
    assert result.external_command_executed is False


def test_make_input_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="GEMM dimensions"):
        make_simplepim_dense_microbench_input(0, 8, 8)


def test_executed_status_is_not_emitted_in_this_wave() -> None:
    cases = [
        prepare_simplepim_dense_microbench(_small_input(), probe=_probe("unavailable")),
        prepare_simplepim_dense_microbench(_small_input(), probe=_probe("configured_but_unverified")),
        prepare_simplepim_dense_microbench(_small_input(), probe=_probe("available", available=True)),
        prepare_simplepim_dense_microbench(make_simplepim_dense_microbench_input(256, 256, 256), probe=_probe("available", available=True)),
    ]

    assert all(result.status != "executed" for result in cases)
    assert all(result.execution_implemented is False for result in cases)


def test_wrapper_writes_only_microbench_artifacts(tmp_path: Path) -> None:
    run_dir, artifact_path, status = run_simplepim_microbench(
        tmp_path,
        gemm_m=8,
        gemm_k=8,
        gemm_n=8,
        seed=42,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert run_dir.exists()
    assert artifact_path == run_dir / "simplepim_microbench.json"
    assert status == payload["status"]
    assert (run_dir / "environment.json").exists()
    assert payload["execution_implemented"] is False
    assert payload["external_command_executed"] is False
    assert sorted(path.name for path in run_dir.iterdir()) == ["environment.json", "simplepim_microbench.json"]
    assert not list((run_dir / "raw").glob("*.jsonl"))
    assert not list((run_dir / "cases").glob("*/route_decisions.jsonl"))
    assert not (run_dir / "summary.json").exists()
    assert not list((run_dir / "plots").glob("*.png"))
    assert not list((run_dir / "config").glob("*"))
    json.dumps(to_jsonable(payload))
