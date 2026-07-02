from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantum_bench.bench.generic_task_bridge import run_generic_task_bridge
from quantum_bench.bench.upmem_taskgraph_runtime import run_upmem_taskgraph_runtime
from quantum_bench.routing import GenericTaskPreparationInput, prepare_generic_task
from quantum_bench.targets.upmem import GENERIC_BOUNDARY_CASE_ID, build_generic_boundary_workload
from quantum_bench.targets.upmem.generic_bridge import (
    GENERIC_BRIDGE_ID,
    GENERIC_BRIDGE_SCHEMA_VERSION,
    GENERIC_LOOP_BACKEND_ID,
    GenericBridgeBlob,
    GenericBridgeExecutionResult,
    GenericBridgeOutputManifest,
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_generic_execute_from_expected(
    input_manifest_path: Path,
    backend: str = GENERIC_LOOP_BACKEND_ID,
    *,
    execute_external: bool = False,
    env=None,
):
    bridge_dir = input_manifest_path.parent
    payload = _load(input_manifest_path)
    reference = np.load(bridge_dir / payload["expected_quantized_reference_output"]["relative_path"], allow_pickle=False)
    output_path = bridge_dir / "outputs" / "output.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, reference, allow_pickle=False)
    blob = GenericBridgeBlob(
        relative_path="outputs/output.npy",
        dtype=str(reference.dtype),
        shape=tuple(int(dim) for dim in reference.shape),
        representation="upmem_sdk_generic_loop_output",
        nbytes=int(reference.nbytes),
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
        task_id=str(payload["task_id"]),
        output_blob=blob,
        validation_metrics={"passed": True, "max_abs_error": 0.0, "reference_kind": "expected_quantized_reference_output"},
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


def test_generic_boundary_workload_is_non_gemm_rank3_contraction() -> None:
    workload = build_generic_boundary_workload()
    task = workload.graph.tasks[0]

    assert workload.case_id == GENERIC_BOUNDARY_CASE_ID
    assert len(workload.graph.tasks) == 1
    assert task.index_expression == "abc,cde->abde"
    assert task.input_shapes == ((2, 3, 4), (4, 5, 2))
    assert task.output_shape == (2, 3, 5, 2)
    assert len(task.input_shapes[0]) == 3
    assert len(task.input_shapes[1]) == 3
    assert len(task.output_shape) == 4
    assert task.left_labels == (0, 1, 2)
    assert task.right_labels == (2, 3, 4)
    assert task.contracted_labels == (2,)
    assert task.output_labels == (0, 1, 3, 4)
    assert workload.manifest["workload_type"] == "generic_boundary_execution"
    assert workload.manifest["not_real_quantum_circuit"] is True


def test_generic_boundary_preparation_uses_expected_integer_axis_maps() -> None:
    workload = build_generic_boundary_workload()
    task = workload.graph.tasks[0]
    tensors = {tensor.spec.id: tensor for tensor in workload.network.tensors}
    result = prepare_generic_task(
        GenericTaskPreparationInput(
            task=task,
            left_tensor=tensors["boundary_left"],
            right_tensor=tensors["boundary_right"],
        )
    )
    payload = result.to_json_dict()
    encoded = json.dumps(payload)

    assert result.status == "prepared"
    assert payload["native_index_metadata"]["output_to_left_axes"] == [0, 1, -1, -1]
    assert payload["native_index_metadata"]["output_to_right_axes"] == [-1, -1, 1, 2]
    assert payload["native_index_metadata"]["contracted_to_left_axes"] == [2]
    assert payload["native_index_metadata"]["contracted_to_right_axes"] == [0]
    assert payload["metadata"]["validation_target"] == "expected_quantized_reference_output"
    assert "prepared_operands" not in payload
    assert "left_quantized" not in encoded
    assert "right_quantized" not in encoded


def test_generic_boundary_task_bridge_records_boundary_evidence(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("quantum_bench.bench.generic_task_bridge.execute_generic_bridge", _fake_generic_execute_from_expected)

    result = run_generic_task_bridge(
        tmp_path,
        case=GENERIC_BOUNDARY_CASE_ID,
        task_index=0,
        backend=GENERIC_LOOP_BACKEND_ID,
        execute_external=True,
    )
    summary = _load(result.summary_path)
    evidence = summary["generic_boundary_evidence"]

    assert result.status == "completed"
    assert summary["contraction_execution_target"] == "upmem"
    assert summary["upmem_execution_mode"] == "sdk_simulator"
    assert summary["execution_backend"] == "upmem_sdk"
    assert summary["cpu_fallback_used"] is False
    assert summary["dpu_program_invocations"] == 1
    assert summary["upmem_program_executed"] is True
    assert evidence["input_ranks"] == [3, 3]
    assert evidence["output_rank"] == 4
    assert evidence["native_index_metadata"]["output_to_left_axes"] == [0, 1, -1, -1]
    assert evidence["validation_target"] == "expected_quantized_reference_output"
    assert summary["bridge_validation_metrics"]["passed"] is True


def test_generic_boundary_strict_runtime_records_no_cpu_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)

    result = run_upmem_taskgraph_runtime(
        tmp_path,
        case=GENERIC_BOUNDARY_CASE_ID,
        policy="generic-only",
        quantization_mode="per_task_input_quantize",
        execute_external=True,
    )
    summary = _load(result.summary_path)
    evidence = summary["generic_boundary_evidence"]

    assert result.status == "completed"
    assert summary["cpu_fallback_used"] is False
    assert summary["dpu_program_executed_task_count"] == 1
    assert summary["dpu_program_executed_all_tasks"] is True
    assert summary["runtime_tensor_sources_all_upmem_output_blobs"] is True
    assert summary["valid_primary_upmem_codepath_result"] is True
    assert summary["kernel_family_counts"] == {"generic_loop_fallback": 1}
    assert evidence["input_ranks"] == [3, 3]
    assert evidence["output_rank"] == 4
    assert evidence["upmem_program_executed"] is True
    assert evidence["valid_primary_upmem_codepath_result"] is True
