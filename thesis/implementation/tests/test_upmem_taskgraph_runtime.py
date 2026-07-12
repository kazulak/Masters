from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantum_bench.bench.result_artifacts import compare_results, load_result_records
from quantum_bench.bench.upmem_taskgraph_runtime import run_upmem_taskgraph_runtime
from quantum_bench.core.records import CircuitSpec, ContractionTask, PathSummary, TaskGraph, TensorNetworkSpec, TensorSpec, TensorValue
from quantum_bench.targets.upmem.generic_bridge import (
    GENERIC_BRIDGE_ID,
    GENERIC_BRIDGE_SCHEMA_VERSION,
    GENERIC_LOOP_BACKEND_ID,
    GenericBridgeBlob,
    GenericBridgeExecutionResult,
    GenericBridgeOutputManifest,
)
from quantum_bench.targets.upmem.taskgraph_runtime import (
    build_generic_quantized_taskgraph_reference,
    build_generic_taskgraph_reference,
    execute_upmem_taskgraph_runtime,
)
from quantum_bench.tn.execution import execute_task_sequence_np_einsum
from quantum_bench.tn.network import TensorNetworkValue


def _simple_path_summary(task_count: int) -> PathSummary:
    return PathSummary(
        planner="test",
        optimize="greedy",
        path_length=task_count,
        largest_intermediate=1,
        naive_flops=None,
        optimized_flops=None,
        text="test path",
        planner_engine="test",
        planner_id="test",
        planner_kind="test",
        optimize_mode="greedy",
    )


def _one_task_complex_graph() -> tuple[TaskGraph, TensorNetworkValue]:
    circuit = CircuitSpec(name="complex_one_task", n_qubits=0, operations=(), source={"kind": "test"})
    left = np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0 - 1.0j]], dtype=np.complex128)
    right = np.array([[1.0, 0.0 + 1.0j], [1.0 - 1.0j, 1.0]], dtype=np.complex128)
    left_spec = TensorSpec("left", (0, 1), left.shape, "dense", dtype="complex128")
    right_spec = TensorSpec("right", (1, 2), right.shape, "dense", dtype="complex128")
    network_spec = TensorNetworkSpec(circuit, (left_spec, right_spec), (0, 2), "ab,bc->ac")
    task = ContractionTask(
        id="task_complex",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=(left.shape, right.shape),
        output_shape=(2, 2),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=2,
        gemm_k=2,
        gemm_n=2,
        structure="dense",
        estimated_flops=16,
        estimated_bytes=0,
    )
    graph = TaskGraph(network_spec, (task,), ((0, 1),), _simple_path_summary(1), planning_time_s=0.0)
    return graph, TensorNetworkValue(network_spec, [TensorValue(left_spec, left), TensorValue(right_spec, right)])


def _two_task_scalar_graph() -> tuple[TaskGraph, TensorNetworkValue]:
    circuit = CircuitSpec(name="two_task_scalar", n_qubits=0, operations=(), source={"kind": "test"})
    specs = (
        TensorSpec("a", (0,), (2,), "dense", dtype="float64"),
        TensorSpec("b", (0,), (2,), "dense", dtype="float64"),
        TensorSpec("c", (), (), "dense", dtype="float64"),
    )
    network_spec = TensorNetworkSpec(circuit, specs, (), "a,a,->")
    task0 = ContractionTask(
        id="task0",
        input_tensor_ids=("a", "b"),
        output_tensor_id="tmp",
        dependencies=(),
        index_expression="a,a->",
        input_shapes=((2,), (2,)),
        output_shape=(),
        left_labels=(0,),
        right_labels=(0,),
        contracted_labels=(0,),
        output_labels=(),
        gemm_m=1,
        gemm_k=2,
        gemm_n=1,
        structure="dense",
        estimated_flops=4,
        estimated_bytes=0,
    )
    task1 = ContractionTask(
        id="task1",
        input_tensor_ids=("tmp", "c"),
        output_tensor_id="out",
        dependencies=("task0",),
        index_expression=",->",
        input_shapes=((), ()),
        output_shape=(),
        left_labels=(),
        right_labels=(),
        contracted_labels=(),
        output_labels=(),
        gemm_m=1,
        gemm_k=1,
        gemm_n=1,
        structure="dense",
        estimated_flops=1,
        estimated_bytes=0,
    )
    graph = TaskGraph(network_spec, (task0, task1), ((0, 1), (0, 1)), _simple_path_summary(2), planning_time_s=0.0)
    tensors = [
        TensorValue(specs[0], np.array([1.0, 2.0], dtype=np.float64)),
        TensorValue(specs[1], np.array([3.0, 4.0], dtype=np.float64)),
        TensorValue(specs[2], np.array(2.0, dtype=np.float64)),
    ]
    return graph, TensorNetworkValue(network_spec, tensors)


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


def test_generic_only_runtime_consumes_upmem_output_blobs_not_cpu_reference(monkeypatch, tmp_path: Path) -> None:
    graph, network = _two_task_scalar_graph()
    reference, _ = execute_task_sequence_np_einsum(graph, network)
    seen_second_left_scale: list[float] = []
    call_count = 0

    def fake_execute(input_manifest_path: Path, backend: str = GENERIC_LOOP_BACKEND_ID, *, execute_external: bool = False, env=None):
        nonlocal call_count
        payload = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        if call_count == 0:
            output = np.array(7.0, dtype=np.float64)
        else:
            seen_second_left_scale.append(float(payload["conversion_records"]["left"]["scale"]))
            output = np.load(input_manifest_path.parent / payload["expected_quantized_reference_output"]["relative_path"], allow_pickle=False)
        call_count += 1
        return _fake_generic_result(input_manifest_path, output)

    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", fake_execute)

    result = execute_upmem_taskgraph_runtime(
        graph=graph,
        network=network,
        case_id="two_task_scalar",
        policy="generic-only",
        quantization_mode="per_task_input_quantize",
        bridge_root=tmp_path / "bridge",
        execute_external=True,
        reference_output=reference,
    )

    assert call_count == 2
    assert np.isclose(seen_second_left_scale[0], 7.0 / 127.0)
    assert result.summary["cpu_fallback_used"] is False
    assert result.summary["runtime_tensor_sources_all_upmem_output_blobs"] is True
    assert result.task_metrics[1]["cpu_reference_artifact_used_as_runtime_input"] is False
    assert result.status == "validation_failed"


def test_generic_quantized_taskgraph_reference_replays_multiple_tasks() -> None:
    graph, network = _two_task_scalar_graph()

    reference = build_generic_quantized_taskgraph_reference(graph=graph, network=network, case_id="two_task_scalar")

    assert reference.status == "completed"
    assert reference.summary["reference_kind"] == "generic_quantized_taskgraph_replay"
    assert reference.summary["completed_tasks"] == 2
    assert reference.output_labels == ()
    assert np.asarray(reference.output).shape == ()
    assert reference.task_metrics[0]["validation_target"] == "expected_quantized_reference_output"
    assert reference.task_metrics[1]["full_precision_reference_is_validation_target"] is False


def test_generic_only_runtime_validates_against_generic_quantized_reference(monkeypatch, tmp_path: Path) -> None:
    graph, network = _two_task_scalar_graph()
    reference = build_generic_quantized_taskgraph_reference(graph=graph, network=network, case_id="two_task_scalar")
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)

    result = execute_upmem_taskgraph_runtime(
        graph=graph,
        network=network,
        case_id="two_task_scalar",
        policy="generic-only",
        quantization_mode="per_task_input_quantize",
        bridge_root=tmp_path / "bridge",
        execute_external=True,
        reference_output=reference.output,
        reference_kind="generic_quantized_taskgraph_replay",
        full_precision_reference_output=execute_task_sequence_np_einsum(graph, network)[0],
    )

    assert result.status == "completed"
    assert result.summary["final_validation"]["reference_kind"] == "generic_quantized_taskgraph_replay"
    assert result.summary["generic_only_all_tasks_used_generic_backend"] is True
    assert result.summary["dpu_program_invocations"] == result.summary["total_tasks"]
    assert result.summary["runtime_tensor_sources_all_upmem_output_blobs"] is True
    assert result.summary["final_validation"]["reference_kind"] == "generic_quantized_taskgraph_replay"
    assert result.summary["final_full_precision_accuracy"]["full_precision_reference_kind"] == "cpu_exact_taskgraph_full_precision"
    assert result.summary["final_full_precision_accuracy"]["full_precision_max_abs_error"] >= 0.0


def test_generic_only_runtime_supports_float32_no_quant_mode(monkeypatch, tmp_path: Path) -> None:
    graph, network = _two_task_scalar_graph()
    reference = build_generic_taskgraph_reference(graph=graph, network=network, case_id="two_task_scalar", quantization_mode="none")
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)

    result = execute_upmem_taskgraph_runtime(
        graph=graph,
        network=network,
        case_id="two_task_scalar",
        policy="generic-only",
        quantization_mode="none",
        bridge_root=tmp_path / "bridge",
        execute_external=True,
        reference_output=reference.output,
        reference_kind="generic_float32_taskgraph_replay",
    )

    assert result.status == "completed"
    assert result.summary["quantization_mode"] == "none"
    assert result.summary["input_dtype_on_dpu"] == "float32"
    assert result.summary["accumulator_dtype_on_dpu"] == "float32"
    assert result.summary["scaling_applied"] is False
    assert result.summary["dpu_program_invocations"] == result.summary["total_tasks"]
    assert result.summary["valid_primary_upmem_codepath_result"] is True
    assert result.summary["cpu_fallback_used"] is False


def test_generic_split_complex_runtime_combines_four_real_calls(monkeypatch, tmp_path: Path) -> None:
    graph, network = _one_task_complex_graph()
    reference, _ = execute_task_sequence_np_einsum(graph, network)
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)

    result = execute_upmem_taskgraph_runtime(
        graph=graph,
        network=network,
        case_id="complex_one_task",
        policy="generic-only",
        quantization_mode="per_task_input_quantize",
        bridge_root=tmp_path / "bridge",
        execute_external=True,
        reference_output=reference,
        full_precision_reference_output=reference,
    )

    assert result.status == "completed"
    assert result.summary["whole_network_quantized_at_initialization"] is False
    assert result.summary["valid_primary_upmem_codepath_result"] is True
    metric = result.task_metrics[0]
    assert metric["complex_representation"] == "split_real_imag"
    assert metric["complex_quantization_scope"] == "per_task_operands"
    assert metric["split_complex_component_count"] == 4
    assert metric["bridge_validation_metrics"]["passed"] is True
    assert metric["full_precision_reference_kind"] == "full_precision_vs_expected_quantized_reference"
    assert metric["full_precision_max_abs_error"] >= 0.0
    assert metric["quantization_clipping_count"] == sum(
        component["quantization_clipping_count"] for component in metric["component_metrics"].values()
    )
    assert metric["left_quantization_scale"] is not None
    assert metric["right_quantization_scale"] is not None
    assert result.summary["final_full_precision_accuracy"]["full_precision_reference_kind"] == "cpu_exact_taskgraph_full_precision"
    assert result.summary["final_full_precision_accuracy"]["full_precision_max_abs_error"] >= 0.0
    assert np.allclose(result.output, reference)


def test_dense_then_generic_falls_back_to_generic_split_complex(monkeypatch, tmp_path: Path) -> None:
    graph, network = _one_task_complex_graph()
    reference, _ = execute_task_sequence_np_einsum(graph, network)
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)
    monkeypatch.setattr(
        "quantum_bench.targets.upmem.taskgraph_runtime.dense_bridge_backend_manifest_eligibility",
        lambda preparation, backend: (False, "forced_dense_reject"),
    )

    result = execute_upmem_taskgraph_runtime(
        graph=graph,
        network=network,
        case_id="complex_one_task",
        policy="dense-then-generic",
        quantization_mode="per_task_input_quantize",
        bridge_root=tmp_path / "bridge",
        execute_external=True,
        reference_output=reference,
    )

    assert result.status == "completed"
    assert result.task_metrics[0]["selected_kernel_family"] == "generic_loop_fallback"
    assert result.task_metrics[0]["dense_reject_reason"] == "forced_dense_reject"
    assert result.task_metrics[0]["complex_representation"] == "split_real_imag"


def test_run_harness_writes_compare_results_compatible_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)

    run = run_upmem_taskgraph_runtime(
        tmp_path,
        case="bell_2q",
        policy="generic-only",
        quantization_mode="per_task_input_quantize",
        execute_external=True,
    )
    records = load_result_records([run.run_dir])
    comparison = compare_results([run.run_dir], tmp_path / "comparison")

    assert run.status == "completed"
    assert run.run_dir.parent == tmp_path / "runs" / "evidence" / "bell_2q" / "upmem_generic_int8"
    assert (run.run_dir / "run_manifest.json").exists()
    assert (run.run_dir / "normalized_records.jsonl").exists()
    manifest = json.loads((run.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "evidence_run"
    assert manifest["route_label"] == "upmem_generic_int8"
    assert run.summary["contraction_execution_target"] == "upmem"
    assert run.summary["upmem_execution_mode"] == "sdk_simulator"
    assert run.summary["hardware_speedup_applicable"] is False
    assert run.summary["final_validation"]["reference_kind"] == "generic_quantized_taskgraph_replay"
    assert run.summary["reference"]["full_precision_reference_is_task_validation_target"] is False
    assert run.summary["generic_quantized_taskgraph_reference"]["status"] == "completed"
    assert run.summary["final_full_precision_accuracy"]["reference_kind"] == "cpu_exact_taskgraph_full_precision"
    assert run.summary["valid_primary_upmem_codepath_result"] is True
    task_metrics_path = run.run_dir / "cases" / run.case_id / "upmem_taskgraph_task_metrics.jsonl"
    assert task_metrics_path.exists()
    first_metric = json.loads(task_metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_metric["bridge_artifact_path"].startswith("cases/")
    assert not Path(first_metric["bridge_artifact_path"]).is_absolute()
    assert records
    assert all(record["run_id"] == run.run_dir.name for record in records)
    assert records[0]["execution_target"] == "upmem"
    assert records[0]["hardware_speedup"] == "not_applicable"
    assert records[0]["parallelism_mode"] == "sequential"
    assert records[0]["parallelism_evidence_type"] == "executed"
    assert records[0]["execution_plan_kind"] == "sequential_upmem_taskgraph"
    assert records[0]["execution_plan_executed"] is True
    assert records[0]["slicing_enabled"] is False
    assert records[0]["frontier_scheduler_enabled"] is False
    assert records[0]["modeled_parallelism_available"] is False
    assert comparison.record_count >= 1


def test_frontier_runtime_records_sdk_simulator_parallelism_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("quantum_bench.targets.upmem.taskgraph_runtime.execute_generic_bridge", _fake_generic_execute_from_expected)

    run = run_upmem_taskgraph_runtime(
        tmp_path,
        case="bell_2q",
        policy="generic-only",
        quantization_mode="per_task_input_quantize",
        execute_external=True,
        schedule_mode="frontier",
        frontier_worker_count=1,
        dpu_group_count=2,
        task_assignment_strategy="frontier_round_robin_dpu_groups",
    )
    records = load_result_records([run.run_dir])
    comparison = compare_results([run.run_dir], tmp_path / "frontier_comparison", comparison_type="parallelism_evidence")

    assert run.status == "completed"
    assert run.run_dir.parent == tmp_path / "runs" / "evidence" / "bell_2q" / "upmem_frontier_generic_int8"
    assert run.summary["route_id"] == "upmem_tn_frontier_sdk_simulator"
    assert run.summary["parallelism_mode"] == "frontier"
    assert run.summary["parallelism_evidence_type"] == "executed"
    assert run.summary["execution_plan_kind"] == "upmem_frontier_assignment_scheduler"
    assert run.summary["frontier_scheduler_enabled"] is True
    assert run.summary["frontier_worker_count"] == 1
    assert run.summary["frontier_parallel_execution"] is False
    assert run.summary["upmem_parallelism_mode"] == "frontier_multi_dpu"
    assert run.summary["upmem_parallelism_evidence_type"] == "sdk_simulator_executed"
    assert run.summary["dpu_group_count"] == 2
    assert run.summary["assigned_task_count"] == run.summary["total_tasks"]
    assert run.summary["executed_dpu_task_count"] == run.summary["executed_tasks"]
    assert run.summary["dpu_program_invocations"] == run.summary["total_tasks"]
    assert run.summary["dpu_assignment_validation_status"] == "passed"
    assert run.summary["duplicate_contraction_check"] == "passed"
    assert run.summary["missing_dependency_check"] == "passed"
    assert run.summary["dependency_violation_detected"] is False
    assert run.summary["hardware_execution"] is False
    assert run.summary["hardware_speedup_applicable"] is False
    assert run.summary["valid_primary_upmem_codepath_result"] is True

    assert records
    row = records[0]
    assert row["route_id"] == "upmem_tn_frontier_sdk_simulator"
    assert row["parallelism_mode"] == "frontier"
    assert row["parallelism_evidence_type"] == "executed"
    assert row["upmem_parallelism_evidence_type"] == "sdk_simulator_executed"
    assert row["execution_plan_executed"] is True
    assert row["frontier_scheduler_enabled"] is True
    assert row["frontier_worker_count"] == 1
    assert row["hardware_speedup_applicable"] is False

    task_metrics_path = run.run_dir / "cases" / run.case_id / "upmem_taskgraph_task_metrics.jsonl"
    task_metrics = [json.loads(line) for line in task_metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(task_metrics) == run.summary["total_tasks"]
    assert all("frontier_wave_index" in metric for metric in task_metrics)
    assert all("dpu_group_id" in metric for metric in task_metrics)

    assert (comparison.run_dir / "parallelism_mode_summary.csv").exists()
    assert (comparison.run_dir / "parallelism_capability_matrix.csv").exists()
    comparison_payload = json.loads(comparison.artifact_path.read_text(encoding="utf-8"))
    capability = {
        row["route_id"]: row
        for row in comparison_payload["parallelism_capability_matrix"]
    }
    frontier_capability = capability["upmem_tn_frontier_sdk_simulator"]
    assert frontier_capability["same_family_timing_group"] == "upmem_sdk"
    assert frontier_capability["speedup_claim_allowed"] is False
    assert frontier_capability["claim_boundary"] == "sdk_simulator_no_hardware_speedup"
