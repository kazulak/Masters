from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import ContractionTask, TensorSpec, TensorValue
from quantum_bench.routing import DenseTaskPreparationInput, prepare_dense_task
from quantum_bench.targets.upmem import (
    DENSE_BRIDGE_SCHEMA_VERSION,
    SimplePimProbeResult,
    annotate_task_graph_with_upmem_estimates,
    read_dense_bridge_input_manifest,
    read_dense_bridge_output_manifest,
    run_mock_dense_bridge,
    write_dense_bridge_input_manifest,
)
from quantum_bench.tn import build_tensor_network, plan_task_graph


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
