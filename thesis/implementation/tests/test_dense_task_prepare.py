from __future__ import annotations

import json

import numpy as np

from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import ContractionTask, TensorSpec, TensorValue
from quantum_bench.routing import (
    DENSE_TASK_PREPARATION_SCHEMA_VERSION,
    DenseTaskPreparationInput,
    prepare_dense_task,
)
from quantum_bench.targets.upmem import SimplePimProbeResult, annotate_task_graph_with_upmem_estimates
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


def _real_task_preparation(probe: SimplePimProbeResult) -> DenseTaskPreparationInput:
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    graph, _ = annotate_task_graph_with_upmem_estimates(graph)
    tensors = {tensor.spec.id: tensor for tensor in network.tensors}
    for task in graph.tasks:
        if all(tensor_id in tensors for tensor_id in task.input_tensor_ids):
            return DenseTaskPreparationInput(
                task=task,
                left_tensor=tensors[task.input_tensor_ids[0]],
                right_tensor=tensors[task.input_tensor_ids[1]],
                simplepim_probe=probe,
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


def test_real_task_missing_simplepim_is_unavailable_not_failed() -> None:
    result = prepare_dense_task(_real_task_preparation(_unavailable_probe()))

    assert result.status == "simplepim_unavailable"
    assert result.reason == "SimplePIM is not configured in unit tests"
    assert result.error is None
    assert result.external_command_executed is False
    assert result.execution_implemented is False
    assert result.tile_plan is not None
    assert result.upmem_task_estimate is not None
    assert result.validation_metrics is not None
    assert result.left_conversion is not None
    assert result.right_conversion is not None


def test_real_task_fake_available_probe_can_prepare_small_task() -> None:
    result = prepare_dense_task(_real_task_preparation(_available_probe()))

    assert result.status == "prepared"
    assert result.schema_version == DENSE_TASK_PREPARATION_SCHEMA_VERSION
    assert result.route_id == "dense_gemm"
    assert result.left_matrix_shape == (result.gemm_m, result.gemm_k)
    assert result.right_matrix_shape == (result.gemm_k, result.gemm_n)
    assert result.prepared_operands is not None
    assert result.metadata["prepared_means"].startswith("host-side dense-route preparation")
    assert result.external_command_executed is False
    assert result.execution_implemented is False


def test_label_order_restoration_matches_task_output_labels() -> None:
    task = _dense_task(
        "reordered",
        2,
        3,
        4,
        output_labels=(2, 0),
        output_shape=(4, 2),
        index_expression="ab,bc->ca",
    )
    left, right = _task_tensors(task)
    result = prepare_dense_task(
        DenseTaskPreparationInput(task=task, left_tensor=left, right_tensor=right, simplepim_probe=_available_probe())
    )

    assert result.status == "prepared"
    assert result.gemm_output_labels == (0, 2)
    assert result.output_labels == (2, 0)
    assert result.validation_metrics is not None
    assert result.validation_metrics.passed_reference_shape_check is True
    assert result.validation_metrics.reference_check_max_abs_error < 1.0e-12
    assert result.prepared_operands is not None
    assert result.prepared_operands.reference_output.shape == (4, 2)
    assert result.prepared_operands.dequantized_output.shape == (4, 2)


def test_complex_tensors_use_split_real_imag_fixed_point_conversion() -> None:
    result = prepare_dense_task(_real_task_preparation(_available_probe()))

    assert result.left_conversion is not None
    assert result.right_conversion is not None
    assert result.left_conversion.representation == "split_complex_real_imag"
    assert result.right_conversion.representation == "split_complex_real_imag"
    assert result.left_conversion.metadata["complex_split_axis"] == -1
    assert result.right_conversion.metadata["complex_split_axis"] == -1


def test_large_tiled_task_is_blocked_before_prepared_status() -> None:
    task = _dense_task("large", 256, 256, 256)
    left, right = _task_tensors(task)
    result = prepare_dense_task(
        DenseTaskPreparationInput(task=task, left_tensor=left, right_tensor=right, simplepim_probe=_available_probe())
    )

    assert result.status == "requires_executable_tiling_not_implemented"
    assert result.execution_implemented is False
    assert result.tile_plan is not None
    assert result.tile_plan["fits_wram"] is True
    assert result.tile_plan["requires_tiling"] is True
    assert result.tile_plan["tiling_implemented"] is False
    assert result.upmem_task_estimate is not None
    assert result.upmem_task_estimate["estimated_tile_count"] > 1


def test_mismatched_tensor_binding_is_unsupported_shape() -> None:
    task = _dense_task("small", 4, 4, 4)
    left, right = _task_tensors(task)
    bad_left = TensorValue(TensorSpec("wrong_left", left.spec.labels, left.spec.shape, "dense"), left.array)
    result = prepare_dense_task(
        DenseTaskPreparationInput(task=task, left_tensor=bad_left, right_tensor=right, simplepim_probe=_available_probe())
    )

    assert result.status == "unsupported_shape"
    assert result.reason == "tensor_id_mismatch"
    assert result.prepared_operands is None


def test_to_json_dict_omits_raw_arrays() -> None:
    result = prepare_dense_task(_real_task_preparation(_available_probe()))
    payload = result.to_json_dict()
    json.dumps(payload)

    assert "prepared_operands" not in payload
    assert "left_matrix" not in payload
    assert "right_matrix" not in payload
    assert "left_quantized" not in payload
    assert "right_quantized" not in payload
    assert payload["execution_implemented"] is False
    assert payload["external_command_executed"] is False
