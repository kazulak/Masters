from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from quantum_bench.bench import __main__ as bench_main
from quantum_bench.bench.pim_bridge_eval import run_pim_bridge_eval, validate_cli_options
import quantum_bench.bench.pim_bridge_eval as pim_eval_module
from quantum_bench.core.jsonio import write_json
from quantum_bench.targets.upmem import (
    DENSE_BRIDGE_ID,
    DENSE_BRIDGE_SCHEMA_VERSION,
    DenseBridgeBlob,
    DenseBridgeExecutionResult,
    DenseBridgeOutputManifest,
    dense_bridge_manifest_eligibility,
    read_dense_bridge_input_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_payload(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "pim_bridge_eval.json").read_text(encoding="utf-8"))


def _fake_success_result(input_manifest_path: Path, *, status: str = "upmem_sdk_simulator_executed") -> DenseBridgeExecutionResult:
    bridge_dir = input_manifest_path.parent
    manifest = read_dense_bridge_input_manifest(input_manifest_path)
    expected = np.load(bridge_dir / manifest.expected_output.relative_path, allow_pickle=False)
    output_path = bridge_dir / "outputs" / "fake_upmem_output.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, expected, allow_pickle=False)
    output_manifest = DenseBridgeOutputManifest(
        schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
        bridge_id=DENSE_BRIDGE_ID,
        manifest_kind="dense_bridge_output",
        backend="upmem_sdk_simulator_dense",
        status=status,  # type: ignore[arg-type]
        input_manifest="input_manifest.json",
        route_id=manifest.route_id,
        task_id=manifest.task_id,
        output_blob=DenseBridgeBlob(
            relative_path="outputs/fake_upmem_output.npy",
            dtype=str(expected.dtype),
            shape=tuple(int(dim) for dim in expected.shape),
            representation="dequantized_output",
            nbytes=int(expected.nbytes),
            labels=manifest.output_labels,
            role="unit_test_output",
        ),
        accumulator_blob=None,
        validation_metrics={
            "reference_kind": "expected_dequantized_output_vs_upmem_sdk_simulator_dense",
            "passed": True,
            "max_abs_error": 0.0,
            "l2_error": 0.0,
            "relative_l2_error": 0.0,
        },
        compute_time_s=0.001,
        write_time_s=0.001,
        total_time_s=0.01,
        external_command_executed=True,
        execution_implemented=True,
        metadata={
            "backend_family": "upmem_sdk",
            "target": "simulator",
            "simplepim_api_used": False,
            "upmem_dpu_program_executed": True,
            "simulator_kernel_executed": True,
            "hardware_kernel_executed": False,
            "build_time_s": 0.002,
            "runner_total_time_s": 0.01,
            "simulator_run_time_s": 0.001,
            "kernel_invocation_count": 4,
        },
    )
    write_json(bridge_dir / "output_manifest.json", output_manifest)
    return DenseBridgeExecutionResult(
        schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
        bridge_id=DENSE_BRIDGE_ID,
        execution_status=status,  # type: ignore[arg-type]
        backend_id="upmem_sdk_simulator_dense",
        backend_identity=None,
        reason=status,
        error=None,
        error_type=None,
        input_manifest_path="input_manifest.json",
        output_manifest_path="output_manifest.json",
        output_blob_path="outputs/fake_upmem_output.npy",
        output_manifest=output_manifest,
        invocation_metadata={"backend_id": "upmem_sdk_simulator_dense"},
        external_command_executed=True,
        execution_implemented=True,
        metadata=dict(output_manifest.metadata),
    )


def _fake_failed_result(input_manifest_path: Path) -> DenseBridgeExecutionResult:
    bridge_dir = input_manifest_path.parent
    manifest = read_dense_bridge_input_manifest(input_manifest_path)
    output_manifest = DenseBridgeOutputManifest(
        schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
        bridge_id=DENSE_BRIDGE_ID,
        manifest_kind="dense_bridge_output",
        backend="upmem_sdk_simulator_dense",
        status="failed",
        input_manifest="input_manifest.json",
        route_id=manifest.route_id,
        task_id=manifest.task_id,
        output_blob=None,
        accumulator_blob=None,
        validation_metrics={},
        compute_time_s=0.0,
        write_time_s=0.0,
        total_time_s=0.001,
        external_command_executed=True,
        execution_implemented=True,
        error="runner failed in unit test",
        metadata={"reason": "runner_execution_failed", "simulator_kernel_executed": False, "hardware_kernel_executed": False},
    )
    write_json(bridge_dir / "output_manifest.json", output_manifest)
    return DenseBridgeExecutionResult(
        schema_version=DENSE_BRIDGE_SCHEMA_VERSION,
        bridge_id=DENSE_BRIDGE_ID,
        execution_status="failed",
        backend_id="upmem_sdk_simulator_dense",
        backend_identity=None,
        reason="runner_execution_failed",
        error="runner failed in unit test",
        error_type="runner_execution_failed",
        input_manifest_path="input_manifest.json",
        output_manifest_path="output_manifest.json",
        output_blob_path=None,
        output_manifest=output_manifest,
        invocation_metadata={"backend_id": "upmem_sdk_simulator_dense"},
        external_command_executed=True,
        execution_implemented=True,
        metadata=dict(output_manifest.metadata),
    )


def test_pim_bridge_eval_dry_run_case_writes_artifacts_without_backend_execution(tmp_path: Path, monkeypatch) -> None:
    def forbidden_execute(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry-run pim-bridge-eval must not execute dense bridge backend")

    monkeypatch.setattr(pim_eval_module, "capture_environment", lambda root_dir: {})
    monkeypatch.setattr(pim_eval_module, "execute_dense_bridge", forbidden_execute)

    run_dir = run_pim_bridge_eval(tmp_path, case="bell_2q", n_qubits=2, dry_run=True, env={}, output_plots=False)
    payload = _load_payload(run_dir)
    rows = payload["rows"]
    encoded = json.dumps(payload)

    assert payload["schema_version"] == "pim_bridge_eval_v1"
    assert payload["dry_run"] is True
    assert payload["metadata"]["suite_routes_ignored"] is True
    assert rows
    assert all(row["backend_status"] is None for row in rows)
    assert any(row["readiness_status"] == "dry_run_ready" for row in rows)
    assert not list(run_dir.glob("cases/*/dense_bridge/**/input_manifest.json"))
    assert "prepared_operands" not in encoded
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded
    assert str(tmp_path) not in encoded
    assert (run_dir / "pim_bridge_eval.csv").exists()
    assert (run_dir / "pim_bridge_eval_cases.csv").exists()
    assert (run_dir / "pim_bridge_eval_summary.md").exists()


def test_pim_bridge_eval_suite_routes_are_not_iterated(tmp_path: Path, monkeypatch) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: route_ignore_test
defaults:
  planner: {engine: opt_einsum, optimize: greedy}
workloads:
  - id: bell_2q
    circuit: {kind: builtin, name: bell_2q}
routes:
  - id: cpu_tn_einsum_exact
  - id: made_up_route_that_must_not_run
validation: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(pim_eval_module, "capture_environment", lambda root_dir: {})

    run_dir = run_pim_bridge_eval(tmp_path, suite_path=suite_path, dry_run=True, env={}, output_plots=False)
    payload = _load_payload(run_dir)
    case_summary = payload["case_summaries"][0]

    assert payload["metadata"]["normal_suite_routes_executed"] is False
    assert payload["metadata"]["suite_routes_ignored"] is True
    assert len(payload["case_summaries"]) == 1
    assert len(payload["rows"]) == case_summary["analyzed_task_count"]


def test_pim_bridge_eval_cli_validation_rules() -> None:
    with pytest.raises(ValueError, match="bell_2q"):
        validate_cli_options(
            suite_path=None,
            case="bell_2q",
            n_qubits=3,
            backend="upmem_sdk_simulator_dense",
            execute_external=False,
            dry_run=False,
            max_tasks_per_case=1,
            max_executed_tasks_per_case=0,
            task_selection="eligible-only",
            timeout_seconds=60.0,
        )
    with pytest.raises(ValueError, match="requires --n-qubits"):
        validate_cli_options(
            suite_path=None,
            case="QRNG",
            n_qubits=None,
            backend="upmem_sdk_simulator_dense",
            execute_external=False,
            dry_run=False,
            max_tasks_per_case=1,
            max_executed_tasks_per_case=0,
            task_selection="eligible-only",
            timeout_seconds=60.0,
        )
    with pytest.raises(ValueError, match="dry-run"):
        validate_cli_options(
            suite_path=None,
            case="bell_2q",
            n_qubits=2,
            backend="upmem_sdk_simulator_dense",
            execute_external=True,
            dry_run=True,
            max_tasks_per_case=1,
            max_executed_tasks_per_case=0,
            task_selection="eligible-only",
            timeout_seconds=60.0,
        )


def test_pim_bridge_eval_backend_success_is_recorded_and_aggregated(tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []

    def fake_execute(input_manifest_path: Path, **kwargs: object) -> DenseBridgeExecutionResult:
        calls.append(input_manifest_path)
        return _fake_success_result(input_manifest_path)

    monkeypatch.setattr(pim_eval_module, "capture_environment", lambda root_dir: {})
    monkeypatch.setattr(pim_eval_module, "execute_dense_bridge", fake_execute)

    run_dir = run_pim_bridge_eval(
        tmp_path,
        case="bell_2q",
        n_qubits=2,
        execute_external=True,
        max_executed_tasks_per_case=1,
        env={},
        output_plots=False,
    )
    payload = _load_payload(run_dir)
    case_summary = payload["case_summaries"][0]
    executed_rows = [row for row in payload["rows"] if row["backend_status"] == "upmem_sdk_simulator_executed"]

    assert len(calls) == 1
    assert case_summary["attempted_task_count"] == 1
    assert case_summary["executed_task_count"] == 1
    assert case_summary["validated_task_count"] == 1
    assert executed_rows
    assert executed_rows[0]["validation_status"] == "passed"
    assert executed_rows[0]["simulator_kernel_executed"] is True
    assert executed_rows[0]["hardware_kernel_executed"] is False
    assert executed_rows[0]["build_time_s"] == 0.002
    assert not Path(executed_rows[0]["bridge_artifact_path"]).is_absolute()
    assert (run_dir / executed_rows[0]["bridge_artifact_path"]).exists()


def test_pim_bridge_eval_execution_cap_counts_failed_attempts(tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []

    def fake_execute(input_manifest_path: Path, **kwargs: object) -> DenseBridgeExecutionResult:
        calls.append(input_manifest_path)
        return _fake_failed_result(input_manifest_path)

    monkeypatch.setattr(pim_eval_module, "capture_environment", lambda root_dir: {})
    monkeypatch.setattr(pim_eval_module, "execute_dense_bridge", fake_execute)

    run_dir = run_pim_bridge_eval(
        tmp_path,
        case="bell_2q",
        n_qubits=2,
        execute_external=True,
        max_executed_tasks_per_case=1,
        env={},
        output_plots=False,
    )
    payload = _load_payload(run_dir)
    case_summary = payload["case_summaries"][0]

    assert len(calls) == 1
    assert case_summary["attempted_task_count"] == 1
    assert case_summary["failed_task_count"] == 1
    assert any(row["blocker_reason"] == "execution_cap_reached" for row in payload["rows"])


def test_pim_bridge_eval_csv_uses_json_strings_for_nested_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pim_eval_module, "capture_environment", lambda root_dir: {})
    run_dir = run_pim_bridge_eval(tmp_path, case="bell_2q", n_qubits=2, dry_run=True, env={}, output_plots=False)

    with (run_dir / "pim_bridge_eval.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (run_dir / "pim_bridge_eval_cases.csv").open(encoding="utf-8", newline="") as handle:
        case_rows = list(csv.DictReader(handle))

    assert rows
    assert isinstance(json.loads(rows[0]["input_tensor_ids"]), list)
    assert isinstance(json.loads(rows[0]["circuit_parameters"]), dict)
    assert isinstance(json.loads(case_rows[0]["blocker_counts"]), dict)


def test_shared_dense_bridge_eligibility_helper_rejects_missing_preparation() -> None:
    assert dense_bridge_manifest_eligibility(None) == (False, "dense_preparation_missing")


def test_pim_bridge_eval_cli_dispatch(monkeypatch, capsys, tmp_path: Path) -> None:
    called: dict[str, object] = {}

    def fake_run_pim_bridge_eval(root_dir: Path, **kwargs: object) -> Path:
        called["root_dir"] = root_dir
        called.update(kwargs)
        run_dir = tmp_path / "runs" / "fake_pim_bridge_eval"
        run_dir.mkdir(parents=True)
        return run_dir

    monkeypatch.setattr(pim_eval_module, "run_pim_bridge_eval", fake_run_pim_bridge_eval)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m quantum_bench.bench",
            "pim-bridge-eval",
            "--case",
            "bell_2q",
            "--n-qubits",
            "2",
            "--dry-run",
            "--no-output-plots",
        ],
    )

    assert bench_main.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert called["case"] == "bell_2q"
    assert called["n_qubits"] == 2
    assert called["dry_run"] is True
    assert called["output_plots"] is False
    assert payload["status"] == "completed"
    assert payload["run_dir"].endswith("fake_pim_bridge_eval")
