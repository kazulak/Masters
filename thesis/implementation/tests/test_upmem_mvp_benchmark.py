from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from quantum_bench.bench.result_artifacts import compare_results, load_result_records
from quantum_bench.bench.upmem_mvp_benchmark import run_upmem_mvp_benchmark, validate_options
from quantum_bench.targets.upmem.generic_bridge import (
    GENERIC_BRIDGE_ID,
    GENERIC_BRIDGE_SCHEMA_VERSION,
    GENERIC_LOOP_BACKEND_ID,
    GenericBridgeBlob,
    GenericBridgeExecutionResult,
    GenericBridgeOutputManifest,
)


def _write_suite(path: Path) -> None:
    path.write_text(
        """
schema_version: 2
suite_id: upmem_mvp_test
defaults:
  planner:
    engine: opt_einsum
    optimize: greedy
workloads:
  - id: bell_2q
    circuit: {kind: builtin, name: bell_2q}
  - id: qrng_3q
    circuit: {kind: builtin, name: QRNG, n_qubits: 3}
routes:
  - id: cpu_tn_einsum_exact
    role: reference
    required: true
validation:
  tolerances:
    max_abs_error: 1.0e-9
    l2_error: 1.0e-8
    max_rel_error: 1.0e-8
    norm_drift: 1.0e-8
    min_fidelity: 0.999999999
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _fake_generic_execute_from_expected(input_manifest_path: Path, backend: str = GENERIC_LOOP_BACKEND_ID, *, execute_external: bool = False, env=None):
    bridge_dir = input_manifest_path.parent
    payload = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    reference = np.load(bridge_dir / payload["expected_quantized_reference_output"]["relative_path"], allow_pickle=False)
    return _fake_generic_result(input_manifest_path, reference)


def _fake_generic_result(input_manifest_path: Path, output: np.ndarray):
    bridge_dir = input_manifest_path.parent
    output_path = bridge_dir / "outputs" / "output.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output, allow_pickle=False)
    blob = GenericBridgeBlob(
        relative_path="outputs/output.npy",
        dtype=str(np.asarray(output).dtype),
        shape=tuple(int(dim) for dim in np.asarray(output).shape),
        representation="upmem_sdk_generic_loop_output",
        nbytes=int(np.asarray(output).nbytes),
        role="output",
    )
    manifest = GenericBridgeOutputManifest(
        schema_version=GENERIC_BRIDGE_SCHEMA_VERSION,
        bridge_id=GENERIC_BRIDGE_ID,
        manifest_kind="generic_contraction_bridge_output",
        backend=GENERIC_LOOP_BACKEND_ID,
        status="upmem_sdk_simulator_generic_loop_executed",
        input_manifest=input_manifest_path.name,
        route_id="generic_loop_fallback",
        task_id="task",
        output_blob=blob,
        validation_metrics={"passed": True, "max_abs_error": 0.0},
        compute_time_s=0.001,
        write_time_s=0.0,
        total_time_s=0.001,
        external_command_executed=True,
        execution_implemented=True,
        metadata={
            "target": "simulator",
            "simplepim_api_used": False,
            "native_sdk_control_path": True,
            "simulator_kernel_executed": True,
            "hardware_kernel_executed": False,
        },
    )
    return GenericBridgeExecutionResult(
        schema_version=GENERIC_BRIDGE_SCHEMA_VERSION,
        bridge_id=GENERIC_BRIDGE_ID,
        execution_status="upmem_sdk_simulator_generic_loop_executed",
        backend_id=GENERIC_LOOP_BACKEND_ID,
        backend_identity=None,
        reason="upmem_sdk_simulator_generic_loop_executed",
        error=None,
        error_type=None,
        input_manifest_path="input_manifest.json",
        output_manifest_path="output_manifest.json",
        output_blob_path="outputs/output.npy",
        output_manifest=manifest,
        invocation_metadata={"external_command_executed": True},
        external_command_executed=True,
        execution_implemented=True,
        metadata=manifest.metadata,
    )


def test_upmem_mvp_benchmark_runs_suite_and_compare_results(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    _write_suite(suite_path)
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)
    monkeypatch.setattr(
        "quantum_bench.targets.upmem.taskgraph_runtime.dense_bridge_backend_manifest_eligibility",
        lambda preparation, backend: (False, "forced_dense_reject"),
    )

    result = run_upmem_mvp_benchmark(
        tmp_path,
        suite_path=suite_path,
        policies=("generic-only", "dense-then-generic"),
        quantization_modes=("per_task_input_quantize",),
        execute_external=True,
    )

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert payload["schema_version"] == "upmem_mvp_benchmark_v1"
    assert payload["root_summary_emits_upmem_normalized_records"] is False
    assert payload["metadata"]["cpu_reference_used_to_feed_runtime_tensors"] is False
    assert len(payload["cpu_reference_records"]) == 2
    assert len(payload["upmem_rows"]) == 4
    assert all(record["contraction_execution_target"] == "cpu" for record in payload["cpu_reference_records"])
    assert all(row["contraction_execution_target"] == "upmem" for row in payload["upmem_rows"])
    assert all(row["upmem_execution_mode"] == "sdk_simulator" for row in payload["upmem_rows"])
    assert all(row["whole_network_quantized_at_initialization"] is False for row in payload["upmem_rows"])
    assert all(row["cpu_fallback_used"] is False for row in payload["upmem_rows"])
    assert all(not Path(row["upmem_runtime_summary_artifact"]).is_absolute() for row in payload["upmem_rows"])

    with (result.run_dir / "upmem_mvp_benchmark_results.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert set(row["policy"] for row in rows) == {"generic-only", "dense-then-generic"}
    assert all(int(row["generic_loop_fallback_count"]) > 0 for row in rows)

    records = load_result_records([result.run_dir])
    upmem_records = [record for record in records if record["execution_target"] == "upmem"]
    cpu_records = [record for record in records if record["execution_target"] == "cpu"]
    assert len(upmem_records) == 4
    assert len(cpu_records) == 2
    assert all(record["hardware_speedup"] == "not_applicable" for record in upmem_records)
    assert all(record["contraction_execution_target"] == "upmem" for record in upmem_records)
    assert all(record["upmem_execution_mode"] == "sdk_simulator" for record in upmem_records)
    assert all(record["contraction_execution_target"] == "cpu" for record in cpu_records)

    comparison = compare_results([result.run_dir], tmp_path / "comparison")
    assert comparison.record_count == 6


def test_upmem_mvp_benchmark_records_unsupported_quantization(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    _write_suite(suite_path)
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)

    result = run_upmem_mvp_benchmark(
        tmp_path,
        suite_path=suite_path,
        policies=("generic-only",),
        quantization_modes=("none",),
        execute_external=True,
    )

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert payload["unsupported_count"] == 2
    assert all(row["status"] == "unsupported" for row in payload["upmem_rows"])
    assert all(row["reason"] == "unsupported_quantization_mode:none" for row in payload["upmem_rows"])
    assert (result.run_dir / "unsupported_reasons.csv").exists()


def test_upmem_mvp_benchmark_records_task_cap_without_crashing(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    _write_suite(suite_path)
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)

    result = run_upmem_mvp_benchmark(
        tmp_path,
        suite_path=suite_path,
        policies=("generic-only",),
        quantization_modes=("per_task_input_quantize",),
        execute_external=True,
        max_taskgraph_tasks=0,
    )

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert payload["unsupported_count"] == 2
    assert all(row["reason"] == "taskgraph_task_cap_exceeded" for row in payload["upmem_rows"])


def test_upmem_mvp_benchmark_requires_external_execution() -> None:
    try:
        validate_options(
            policies=("generic-only",),
            quantization_modes=("per_task_input_quantize",),
            execute_external=False,
            max_taskgraph_tasks=1,
        )
    except ValueError as exc:
        assert "--execute-external" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("upmem-mvp-benchmark should require --execute-external")
