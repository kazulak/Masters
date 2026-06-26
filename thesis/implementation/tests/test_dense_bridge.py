from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantum_bench.core.records import to_jsonable
from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import ContractionTask, TensorSpec, TensorValue
from quantum_bench.routing import DenseTaskPreparationInput, prepare_dense_task
from quantum_bench.targets.upmem import (
    DENSE_BRIDGE_ID,
    DENSE_BRIDGE_SCHEMA_VERSION,
    SimplePimProbeResult,
    annotate_task_graph_with_upmem_estimates,
    dense_bridge_backend_registry,
    execute_dense_bridge,
    read_dense_bridge_input_manifest,
    read_dense_bridge_output_manifest,
    run_mock_dense_bridge,
    write_dense_bridge_input_manifest,
)
import quantum_bench.targets.upmem.dense_bridge as dense_bridge_module
from quantum_bench.tn import build_tensor_network, plan_task_graph

ROOT = Path(__file__).resolve().parents[1]


def _available_probe() -> SimplePimProbeResult:
    return SimplePimProbeResult(
        simplepim_available=True,
        simplepim_probe_status="available",
        simplepim_bin="/tmp/simplepim",
        simplepim_command_path="/tmp/simplepim",
        metadata={"external_command_executed": False, "source": "unit_test"},
    )


def _unavailable_probe() -> SimplePimProbeResult:
    return SimplePimProbeResult(
        simplepim_available=False,
        simplepim_probe_status="unavailable",
        skip_reason="SimplePIM is not configured in unit tests",
        metadata={"external_command_executed": False, "source": "unit_test"},
    )


def _real_preparation(probe: SimplePimProbeResult):
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    tensors = {tensor.spec.id: tensor for tensor in network.tensors}
    for task in graph.tasks:
        if all(tensor_id in tensors for tensor_id in task.input_tensor_ids):
            return prepare_dense_task(
                DenseTaskPreparationInput(
                    task=task,
                    left_tensor=tensors[task.input_tensor_ids[0]],
                    right_tensor=tensors[task.input_tensor_ids[1]],
                    simplepim_probe=probe,
                )
            )
    raise AssertionError("bell_2q did not produce a directly preparable task")


def _dense_task(
    task_id: str,
    gemm_m: int,
    gemm_k: int,
    gemm_n: int,
    output_labels: tuple[int, ...] = (0, 2),
    output_shape: tuple[int, ...] | None = None,
    index_expression: str = "ab,bc->ac",
) -> ContractionTask:
    return ContractionTask(
        id=task_id,
        input_tensor_ids=(f"{task_id}_left", f"{task_id}_right"),
        output_tensor_id=f"{task_id}_out",
        dependencies=(),
        index_expression=index_expression,
        input_shapes=((gemm_m, gemm_k), (gemm_k, gemm_n)),
        output_shape=output_shape or (gemm_m, gemm_n),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=output_labels,
        gemm_m=gemm_m,
        gemm_k=gemm_k,
        gemm_n=gemm_n,
        structure="dense",
        estimated_flops=8 * gemm_m * gemm_k * gemm_n,
        estimated_bytes=(gemm_m * gemm_k + gemm_k * gemm_n + gemm_m * gemm_n) * 16,
    )


def _task_tensors(task: ContractionTask) -> tuple[TensorValue, TensorValue]:
    left_size = int(np.prod(task.input_shapes[0]))
    right_size = int(np.prod(task.input_shapes[1]))
    left = (np.arange(left_size, dtype=np.float64).reshape(task.input_shapes[0]) / 17.0).astype(np.complex128)
    right = (np.arange(right_size, dtype=np.float64).reshape(task.input_shapes[1]) / 23.0).astype(np.complex128)
    return (
        TensorValue(TensorSpec(task.input_tensor_ids[0], task.left_labels, task.input_shapes[0], "dense"), left),
        TensorValue(TensorSpec(task.input_tensor_ids[1], task.right_labels, task.input_shapes[1], "dense"), right),
    )


def _synthetic_preparation(task: ContractionTask, probe: SimplePimProbeResult):
    left, right = _task_tensors(task)
    return prepare_dense_task(DenseTaskPreparationInput(task=task, left_tensor=left, right_tensor=right, simplepim_probe=probe))


def _stub_path() -> Path:
    return ROOT / "native" / "upmem" / "simplepim" / "simplepim_dense_stub.py"


def test_input_manifest_is_json_serializable_without_raw_arrays(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    manifest = write_dense_bridge_input_manifest(preparation, tmp_path)
    payload = json.loads((tmp_path / "input_manifest.json").read_text(encoding="utf-8"))

    assert manifest.schema_version == DENSE_BRIDGE_SCHEMA_VERSION
    assert payload["manifest_kind"] == "dense_bridge_input"
    assert payload["operands"]["left"]["relative_path"] == "operands/left_quantized.npy"
    assert payload["operands"]["right"]["relative_path"] == "operands/right_quantized.npy"
    assert payload["expected_output"]["relative_path"] == "references/expected_dequantized_output.npy"
    assert "prepared_operands" not in payload
    assert "left_matrix" not in payload
    assert "right_matrix" not in payload
    json.dumps(payload)


def test_operand_and_expected_blobs_round_trip(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    manifest = read_dense_bridge_input_manifest(tmp_path / "input_manifest.json")

    assert preparation.prepared_operands is not None
    left = np.load(tmp_path / manifest.operands["left"]["relative_path"], allow_pickle=False)
    right = np.load(tmp_path / manifest.operands["right"]["relative_path"], allow_pickle=False)
    expected = np.load(tmp_path / manifest.expected_output.relative_path, allow_pickle=False)

    np.testing.assert_array_equal(left, preparation.prepared_operands.left_quantized)
    np.testing.assert_array_equal(right, preparation.prepared_operands.right_quantized)
    np.testing.assert_allclose(expected, preparation.prepared_operands.dequantized_output)
    assert manifest.expected_output.shape == tuple(int(dim) for dim in expected.shape)
    assert manifest.expected_output.labels == preparation.output_labels


def test_mock_bridge_writes_valid_output_manifest(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = run_mock_dense_bridge(tmp_path / "input_manifest.json")
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert result.status == "mock_executed"
    assert result.external_command_executed is False
    assert result.execution_implemented is False
    assert output_manifest.status == "mock_executed"
    assert output_manifest.backend == "mock_numpy_dequantized"
    assert output_manifest.external_command_executed is False
    assert output_manifest.execution_implemented is False
    assert output_manifest.output_blob is not None
    assert output_manifest.output_blob.relative_path == "outputs/mock_dequantized_output.npy"
    assert output_manifest.validation_metrics["max_abs_error"] < 1.0e-12


def test_output_shape_and_label_order_metadata_are_preserved(tmp_path: Path) -> None:
    task = _dense_task(
        "reordered",
        2,
        3,
        4,
        output_labels=(2, 0),
        output_shape=(4, 2),
        index_expression="ab,bc->ca",
    )
    preparation = _synthetic_preparation(task, _available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = run_mock_dense_bridge(tmp_path / "input_manifest.json")
    output_manifest = result.output_manifest

    assert output_manifest is not None
    assert output_manifest.output_blob is not None
    assert output_manifest.output_blob.shape == (4, 2)
    assert output_manifest.output_blob.labels == (2, 0)
    assert preparation.prepared_operands is not None
    output = np.load(tmp_path / output_manifest.output_blob.relative_path, allow_pickle=False)
    np.testing.assert_allclose(output, preparation.prepared_operands.dequantized_output)


def test_unavailable_simplepim_is_nonfatal_for_mock_bridge(tmp_path: Path) -> None:
    preparation = _real_preparation(_unavailable_probe())
    assert preparation.status == "simplepim_unavailable"

    manifest = write_dense_bridge_input_manifest(preparation, tmp_path)
    result = run_mock_dense_bridge(tmp_path / "input_manifest.json")

    assert manifest.preparation_status == "simplepim_unavailable"
    assert manifest.simplepim_probe["simplepim_probe_status"] == "unavailable"
    assert result.status == "mock_executed"
    assert result.error is None


def test_tiled_preparation_is_rejected_before_bridge_input(tmp_path: Path) -> None:
    preparation = _synthetic_preparation(_dense_task("large", 256, 256, 256), _available_probe())
    assert preparation.status == "requires_executable_tiling_not_implemented"

    with pytest.raises(ValueError, match="executable tiling is not implemented"):
        write_dense_bridge_input_manifest(preparation, tmp_path)

    assert not (tmp_path / "input_manifest.json").exists()


def test_backend_registry_identities_are_json_safe() -> None:
    registry = dense_bridge_backend_registry()

    assert set(registry) == {"mock_numpy_dequantized", "simplepim_external", "simplepim_external_stub"}
    assert registry["mock_numpy_dequantized"].implemented is True
    assert registry["mock_numpy_dequantized"].external_command_capable is False
    assert registry["simplepim_external"].implemented is False
    assert registry["simplepim_external"].external_command_capable is True
    assert registry["simplepim_external_stub"].implemented is False
    assert registry["simplepim_external_stub"].external_command_capable is True
    json.dumps(to_jsonable(registry))


def test_generic_executor_runs_mock_backend(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(tmp_path / "input_manifest.json", backend="mock_numpy_dequantized")

    assert result.execution_status == "mock_executed"
    assert result.backend_id == "mock_numpy_dequantized"
    assert result.reason is None
    assert result.error is None
    assert result.external_command_executed is False
    assert result.execution_implemented is False
    assert result.output_manifest_path == "output_manifest.json"
    assert result.output_blob_path == "outputs/mock_dequantized_output.npy"
    assert (tmp_path / "output_manifest.json").exists()
    assert (tmp_path / "outputs" / "mock_dequantized_output.npy").exists()
    json.dumps(result.to_json_dict())


def test_simplepim_backend_without_config_is_skipped_and_writes_manifest(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(
        tmp_path / "input_manifest.json",
        backend="simplepim_external",
        execute_external=False,
        env={},
    )
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert result.execution_status == "skipped"
    assert result.reason == "simplepim_unavailable"
    assert result.error is None
    assert result.external_command_executed is False
    assert result.execution_implemented is False
    assert output_manifest.status == "skipped"
    assert output_manifest.output_blob is None
    assert output_manifest.metadata["reason"] == "simplepim_unavailable"
    assert (tmp_path / "output_manifest.json").exists()


def test_simplepim_backend_configured_but_external_disabled_records_invocation_only(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(
        tmp_path / "input_manifest.json",
        backend="simplepim_external",
        execute_external=False,
        env={"SIMPLEPIM_BIN": "/opt/simplepim/bin/simplepim"},
    )

    assert result.execution_status == "not_implemented"
    assert result.reason == "simplepim_external_execution_disabled"
    assert result.invocation_metadata["command_path"] == "/opt/simplepim/bin/simplepim"
    assert result.invocation_metadata["working_directory"] == "."
    assert result.invocation_metadata["input_manifest_path"] == "input_manifest.json"
    assert result.invocation_metadata["expected_output_manifest_path"] == "output_manifest.json"
    assert result.invocation_metadata["expected_output_blob_path"] == "outputs/simplepim_output.npy"
    assert result.external_command_executed is False
    assert result.execution_implemented is False


def test_simplepim_execute_external_true_is_not_implemented_without_subprocess(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(
        tmp_path / "input_manifest.json",
        backend="simplepim_external",
        execute_external=True,
        env={"SIMPLEPIM_BIN": "/opt/simplepim/bin/simplepim"},
    )

    assert result.execution_status == "not_implemented"
    assert result.reason == "simplepim_external_execution_not_implemented"
    assert result.external_command_executed is False
    assert result.execution_implemented is False


def test_mock_and_simplepim_external_backends_do_not_call_subprocess(tmp_path: Path, monkeypatch) -> None:
    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("subprocess must not be called by mock or simplepim_external backends")

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", forbidden_subprocess)
    mock_dir = tmp_path / "mock"
    simplepim_dir = tmp_path / "simplepim"
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, mock_dir)
    write_dense_bridge_input_manifest(preparation, simplepim_dir)

    mock_result = execute_dense_bridge(mock_dir / "input_manifest.json", backend="mock_numpy_dequantized")
    simplepim_result = execute_dense_bridge(
        simplepim_dir / "input_manifest.json",
        backend="simplepim_external",
        execute_external=True,
        env={"SIMPLEPIM_BIN": "/opt/simplepim/bin/simplepim"},
    )

    assert mock_result.execution_status == "mock_executed"
    assert simplepim_result.execution_status == "not_implemented"
    assert simplepim_result.reason == "simplepim_external_execution_not_implemented"


def test_direct_simplepim_external_stub_writes_valid_nonexecuting_output_manifest(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(_stub_path()),
            "--input-manifest",
            str(tmp_path / "input_manifest.json"),
            "--output-manifest",
            str(tmp_path / "output_manifest.json"),
            "--backend-id",
            "simplepim_external_stub",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert completed.returncode == 0
    assert output_manifest.manifest_kind == "dense_bridge_output"
    assert output_manifest.backend == "simplepim_external_stub"
    assert output_manifest.status == "stub_executed"
    assert output_manifest.output_blob is None
    assert output_manifest.external_command_executed is True
    assert output_manifest.execution_implemented is False
    assert output_manifest.validation_metrics == {
        "status": "not_applicable",
        "reason": "stub_writes_no_output_blob",
    }
    assert output_manifest.metadata["reason"] == "external_stub_contract_executed"
    assert output_manifest.metadata["native_kernel_executed"] is False


def test_direct_simplepim_external_stub_handles_malformed_input_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "input_manifest.json"
    output_path = tmp_path / "output_manifest.json"
    input_path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(_stub_path()),
            "--input-manifest",
            str(input_path),
            "--output-manifest",
            str(output_path),
            "--backend-id",
            "simplepim_external_stub",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output_manifest = read_dense_bridge_output_manifest(output_path)

    assert completed.returncode == 1
    assert output_manifest.status == "failed"
    assert output_manifest.external_command_executed is True
    assert output_manifest.execution_implemented is False
    assert output_manifest.output_blob is None
    assert output_manifest.metadata["error_type"] == "simplepim_external_stub_failed"
    assert output_manifest.metadata["native_kernel_executed"] is False


def test_simplepim_external_stub_disabled_does_not_call_subprocess(tmp_path: Path, monkeypatch) -> None:
    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("disabled stub backend must not call subprocess")

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", forbidden_subprocess)
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(
        tmp_path / "input_manifest.json",
        backend="simplepim_external_stub",
        execute_external=False,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert result.execution_status == "not_implemented"
    assert result.reason == "simplepim_external_stub_execution_disabled"
    assert result.external_command_executed is False
    assert output_manifest.status == "not_implemented"
    assert output_manifest.external_command_executed is False


def test_simplepim_external_stub_missing_config_does_not_call_subprocess(tmp_path: Path, monkeypatch) -> None:
    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("unconfigured stub backend must not call subprocess")

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", forbidden_subprocess)
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(
        tmp_path / "input_manifest.json",
        backend="simplepim_external_stub",
        execute_external=True,
        env={},
    )
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert result.execution_status == "skipped"
    assert result.reason == "simplepim_external_stub_unavailable"
    assert result.external_command_executed is False
    assert output_manifest.status == "skipped"
    assert output_manifest.external_command_executed is False


def test_simplepim_external_stub_adapter_invokes_configured_relative_path(tmp_path: Path, monkeypatch) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    monkeypatch.chdir(ROOT)
    result = execute_dense_bridge(
        tmp_path / "input_manifest.json",
        backend="simplepim_external_stub",
        execute_external=True,
        env={"SIMPLEPIM_STUB_BIN": "native/upmem/simplepim/simplepim_dense_stub.py"},
    )
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")
    encoded_result = json.dumps(result.to_json_dict())

    assert result.execution_status == "stub_executed"
    assert result.reason == "external_stub_contract_executed"
    assert result.external_command_executed is True
    assert result.execution_implemented is False
    assert result.output_blob_path is None
    assert result.metadata["native_kernel_executed"] is False
    assert output_manifest.status == "stub_executed"
    assert output_manifest.output_blob is None
    assert output_manifest.validation_metrics["status"] == "not_applicable"
    assert not (tmp_path / "outputs" / "simplepim_output.npy").exists()
    assert str(tmp_path) not in encoded_result


def test_simplepim_external_stub_nonzero_exit_is_failed(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=7, stdout="stub stdout", stderr="stub stderr")

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", fake_run)
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(
        tmp_path / "input_manifest.json",
        backend="simplepim_external_stub",
        execute_external=True,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert result.execution_status == "failed"
    assert result.reason == "simplepim_external_stub_failed"
    assert result.error_type == "simplepim_external_stub_failed"
    assert result.external_command_executed is True
    assert output_manifest.status == "failed"
    assert output_manifest.metadata["returncode"] == 7
    assert output_manifest.metadata["stdout_snippet"] == "stub stdout"
    assert output_manifest.metadata["stderr_snippet"] == "stub stderr"
    assert output_manifest.metadata["native_kernel_executed"] is False


def test_simplepim_external_stub_missing_output_manifest_is_failed(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", fake_run)
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(
        tmp_path / "input_manifest.json",
        backend="simplepim_external_stub",
        execute_external=True,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert result.execution_status == "failed"
    assert result.reason == "simplepim_external_stub_output_manifest_invalid"
    assert result.error_type == "simplepim_external_stub_output_manifest_invalid"
    assert output_manifest.status == "failed"
    assert output_manifest.metadata["error_type"] == "simplepim_external_stub_output_manifest_invalid"


def test_simplepim_external_stub_failed_output_manifest_is_not_success(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        output_manifest = tmp_path / "output_manifest.json"
        payload = {
            "schema_version": DENSE_BRIDGE_SCHEMA_VERSION,
            "bridge_id": DENSE_BRIDGE_ID,
            "manifest_kind": "dense_bridge_output",
            "backend": "simplepim_external_stub",
            "status": "failed",
            "input_manifest": "input_manifest.json",
            "route_id": "dense_gemm",
            "task_id": "task_0",
            "output_blob": None,
            "accumulator_blob": None,
            "validation_metrics": {},
            "compute_time_s": 0.0,
            "write_time_s": 0.0,
            "total_time_s": 0.0,
            "external_command_executed": True,
            "execution_implemented": False,
            "error": "stub reported failure",
            "metadata": {
                "reason": "stub_failed_in_test",
                "error_type": "stub_failed_in_test",
                "native_kernel_executed": False,
            },
        }
        output_manifest.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", fake_run)
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(
        tmp_path / "input_manifest.json",
        backend="simplepim_external_stub",
        execute_external=True,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )

    assert result.execution_status == "failed"
    assert result.reason == "stub_failed_in_test"
    assert result.error_type == "stub_failed_in_test"
    assert result.error == "stub reported failure"
    assert result.external_command_executed is True


def test_malformed_manifest_fails_before_simplepim_stub_launch(tmp_path: Path, monkeypatch) -> None:
    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("malformed bridge input must fail before launching the stub")

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", forbidden_subprocess)
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    manifest_path = tmp_path / "input_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["operands"]["left"]["shape"] = [999]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = execute_dense_bridge(
        manifest_path,
        backend="simplepim_external_stub",
        execute_external=True,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )

    assert result.execution_status == "failed"
    assert result.reason == "invalid_bridge_input_manifest"
    assert result.error_type == "invalid_bridge_input_manifest"


def test_unknown_backend_is_unsupported_and_json_safe(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(tmp_path / "input_manifest.json", backend="missing_backend")
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert result.execution_status == "unsupported"
    assert result.reason == "unsupported_backend"
    assert result.error_type == "unsupported_backend"
    assert output_manifest.status == "unsupported"
    assert output_manifest.metadata["error_type"] == "unsupported_backend"
    json.dumps(result.to_json_dict())


def test_malformed_manifest_fails_before_simplepim_backend_status(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    manifest_path = tmp_path / "input_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["operands"]["left"]["shape"] = [999]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = execute_dense_bridge(
        manifest_path,
        backend="simplepim_external",
        execute_external=False,
        env={},
    )
    output_manifest = read_dense_bridge_output_manifest(tmp_path / "output_manifest.json")

    assert result.execution_status == "failed"
    assert result.reason == "invalid_bridge_input_manifest"
    assert result.error_type == "invalid_bridge_input_manifest"
    assert output_manifest.status == "failed"
    assert output_manifest.metadata["error_type"] == "invalid_bridge_input_manifest"


def test_execution_result_and_manifests_do_not_leak_bridge_directory_paths(tmp_path: Path) -> None:
    preparation = _real_preparation(_available_probe())
    write_dense_bridge_input_manifest(preparation, tmp_path)
    result = execute_dense_bridge(tmp_path / "input_manifest.json", backend="mock_numpy_dequantized")
    encoded_result = json.dumps(result.to_json_dict())
    encoded_manifest = (tmp_path / "output_manifest.json").read_text(encoding="utf-8")

    assert str(tmp_path) not in encoded_result
    assert str(tmp_path) not in encoded_manifest
    assert "prepared_operands" not in encoded_result
    assert "left_matrix" not in encoded_result
    assert "right_matrix" not in encoded_result


def test_simplepim_external_stub_is_the_only_registered_external_subprocess_backend() -> None:
    registry = dense_bridge_backend_registry()

    assert registry["mock_numpy_dequantized"].external_command_capable is False
    assert registry["simplepim_external"].external_command_capable is True
    assert registry["simplepim_external"].implemented is False
    assert registry["simplepim_external_stub"].external_command_capable is True
    assert registry["simplepim_external_stub"].implemented is False
