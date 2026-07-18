from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from quantum_bench.core.records import (
    CircuitSpec,
    ContractionTask,
    JsonDict,
    PathSummary,
    TaskGraph,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
)
from quantum_bench.targets.upmem.tile_plan import UPMEM_L2_MAX_HOST_BLOB_BYTES
from quantum_bench.targets.upmem.schedule import UPMEM_DENSE_ESTIMATE_KEY, estimate_dense_task
from quantum_bench.tn.task_graph import with_path_cost_summary


SYNTHETIC_PRESSURE_KIND = "synthetic_pressure"
SYNTHETIC_PRESSURE_ERROR = (
    "synthetic_pressure workloads are analysis-only; use benchmark-matrix-report "
    "or upmem-multi-dpu-assignment instead of normal circuit loading/execution"
)


def is_synthetic_pressure_case(case: dict[str, Any]) -> bool:
    circuit = case.get("circuit")
    return isinstance(circuit, dict) and circuit.get("kind") == SYNTHETIC_PRESSURE_KIND


def require_synthetic_pressure_metadata(case: dict[str, Any]) -> None:
    metadata = case.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("synthetic_pressure workloads must define metadata")
    if metadata.get("workload_type") != "synthetic_pressure":
        raise ValueError("synthetic_pressure metadata.workload_type must be synthetic_pressure")
    if metadata.get("execution_scope") != "model_only":
        raise ValueError("synthetic_pressure metadata.execution_scope must be model_only")
    if metadata.get("not_real_quantum_circuit") is not True:
        raise ValueError("synthetic_pressure metadata.not_real_quantum_circuit must be true")


def build_synthetic_pressure_task_graph(case: dict[str, Any]) -> TaskGraph:
    if not is_synthetic_pressure_case(case):
        raise ValueError("case is not a synthetic_pressure workload")
    require_synthetic_pressure_metadata(case)
    circuit_payload = dict(case["circuit"])
    name = str(circuit_payload.get("name") or case.get("case_id") or "synthetic_pressure")
    tasks = _tasks_from_payload(circuit_payload)
    source = {
        "kind": SYNTHETIC_PRESSURE_KIND,
        "name": name,
        "metadata": dict(case.get("metadata") or {}),
        "synthetic": {
            key: value
            for key, value in circuit_payload.items()
            if key not in {"kind", "name"}
        },
    }
    circuit = CircuitSpec(
        name=name,
        n_qubits=int(circuit_payload.get("n_qubits", 0) or 0),
        operations=(),
        source=source,
    )
    network = TensorNetworkSpec(
        circuit=circuit,
        tensors=(),
        output_labels=(),
        einsum_expression="synthetic_pressure_no_arrays",
    )
    graph = TaskGraph(
        network=network,
        tasks=tuple(_annotate_task(task) for task in tasks),
        path=tuple((0, 1) for _ in tasks),
        path_summary=_path_summary(name, len(tasks)),
        planning_time_s=0.0,
    )
    return with_path_cost_summary(graph)


def synthetic_pressure_manifest(graph: TaskGraph) -> JsonDict:
    source = dict(graph.network.circuit.source)
    metadata = dict(source.get("metadata") or {})
    return {
        "name": graph.network.circuit.name,
        "n_qubits": graph.network.circuit.n_qubits,
        "depth_proxy": len(graph.tasks),
        "gate_counts": {"1q": 0, "2q": 0, "total": 0},
        "gate_set": [],
        "source": source,
        "workload_kind": "synthetic_pressure_model_only",
        "workload_type": metadata.get("workload_type", "synthetic_pressure"),
        "execution_scope": metadata.get("execution_scope", "model_only"),
        "not_real_quantum_circuit": metadata.get("not_real_quantum_circuit", True),
    }


def synthetic_pressure_initial_tensors(
    graph: TaskGraph,
    *,
    max_host_blob_bytes: int = UPMEM_L2_MAX_HOST_BLOB_BYTES,
) -> dict[str, TensorValue]:
    """Materialize deterministic real-valued tensors for developer bridge tests.

    Synthetic pressure graphs intentionally carry no ndarray payloads in normal
    analysis. This helper is explicit and bounded so developer harnesses can
    prepare small L2 bring-up tasks without letting synthetic workloads enter
    normal provider execution.
    """

    specs: dict[str, TensorSpec] = {}
    produced = {task.output_tensor_id for task in graph.tasks}
    for task in graph.tasks:
        left_id, right_id = task.input_tensor_ids
        if left_id not in produced:
            _add_initial_spec(specs, left_id, task.left_labels, task.input_shapes[0])
        if right_id not in produced:
            _add_initial_spec(specs, right_id, task.right_labels, task.input_shapes[1])

    total_bytes = sum(int(np.prod(spec.shape, dtype=np.int64)) * np.dtype(np.float64).itemsize for spec in specs.values())
    if total_bytes > max_host_blob_bytes:
        raise ValueError(
            f"synthetic_pressure initial tensor payload would be {total_bytes} bytes, "
            f"exceeding max_host_blob_bytes={max_host_blob_bytes}"
        )

    tensors: dict[str, TensorValue] = {}
    for index, (tensor_id, spec) in enumerate(sorted(specs.items())):
        tensors[tensor_id] = TensorValue(spec=spec, array=_deterministic_real_array(spec.shape, seed=index))
    return tensors


def _tasks_from_payload(circuit_payload: JsonDict) -> list[ContractionTask]:
    explicit = circuit_payload.get("tasks")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("synthetic_pressure circuit.tasks must be a non-empty list")
        return [_task_from_mapping(index, item) for index, item in enumerate(explicit)]

    profile = str(circuit_payload.get("profile") or "single_gemm")
    task_count = int(circuit_payload.get("task_count", 1) or 1)
    if task_count < 1:
        raise ValueError("synthetic_pressure task_count must be >= 1")
    if profile not in {"single_gemm", "independent_gemm", "chain_gemm"}:
        raise ValueError(f"Unsupported synthetic_pressure profile: {profile}")

    gemm_m = int(circuit_payload.get("gemm_m", 8) or 8)
    gemm_k = int(circuit_payload.get("gemm_k", 8) or 8)
    gemm_n = int(circuit_payload.get("gemm_n", 8) or 8)
    if min(gemm_m, gemm_k, gemm_n) <= 0:
        raise ValueError("synthetic_pressure gemm_m/gemm_k/gemm_n must be positive")

    tasks: list[ContractionTask] = []
    for index in range(task_count):
        task_id = f"task_{index}"
        if profile == "chain_gemm" and index > 0:
            input_ids = (f"result_{index - 1}", f"synthetic_input_{index}_right")
            dependencies = (f"task_{index - 1}",)
        else:
            input_ids = (f"synthetic_input_{index}_left", f"synthetic_input_{index}_right")
            dependencies = ()
        tasks.append(
            _dense_task(
                task_id=task_id,
                gemm_m=gemm_m,
                gemm_k=gemm_k,
                gemm_n=gemm_n,
                input_tensor_ids=input_ids,
                output_tensor_id=f"result_{index}",
                dependencies=dependencies,
            )
        )
    return tasks


def _task_from_mapping(index: int, payload: Any) -> ContractionTask:
    if not isinstance(payload, dict):
        raise ValueError("synthetic_pressure task entries must be mappings")
    gemm_m = int(payload.get("gemm_m", 0) or 0)
    gemm_k = int(payload.get("gemm_k", 0) or 0)
    gemm_n = int(payload.get("gemm_n", 0) or 0)
    if min(gemm_m, gemm_k, gemm_n) <= 0:
        raise ValueError("synthetic_pressure task gemm_m/gemm_k/gemm_n must be positive")
    task_id = str(payload.get("id") or f"task_{index}")
    input_ids = payload.get("input_tensor_ids")
    if input_ids is None:
        input_tensor_ids = (f"{task_id}_left", f"{task_id}_right")
    else:
        if not isinstance(input_ids, (list, tuple)) or len(input_ids) != 2:
            raise ValueError("synthetic_pressure task input_tensor_ids must contain two IDs")
        input_tensor_ids = (str(input_ids[0]), str(input_ids[1]))
    dependencies = tuple(str(item) for item in payload.get("dependencies", ()) or ())
    return _dense_task(
        task_id=task_id,
        gemm_m=gemm_m,
        gemm_k=gemm_k,
        gemm_n=gemm_n,
        input_tensor_ids=input_tensor_ids,
        output_tensor_id=str(payload.get("output_tensor_id") or f"{task_id}_out"),
        dependencies=dependencies,
    )


def _dense_task(
    *,
    task_id: str,
    gemm_m: int,
    gemm_k: int,
    gemm_n: int,
    input_tensor_ids: tuple[str, str],
    output_tensor_id: str,
    dependencies: tuple[str, ...],
) -> ContractionTask:
    return ContractionTask(
        id=task_id,
        input_tensor_ids=input_tensor_ids,
        output_tensor_id=output_tensor_id,
        dependencies=dependencies,
        index_expression="ab,bc->ac",
        input_shapes=((gemm_m, gemm_k), (gemm_k, gemm_n)),
        output_shape=(gemm_m, gemm_n),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=gemm_m,
        gemm_k=gemm_k,
        gemm_n=gemm_n,
        structure="dense",
        estimated_flops=int(8 * gemm_m * gemm_k * gemm_n),
        estimated_bytes=int(gemm_m * gemm_k * 16 + gemm_k * gemm_n * 16 + gemm_m * gemm_n * 16),
    )


def _add_initial_spec(
    specs: dict[str, TensorSpec],
    tensor_id: str,
    labels: tuple[int, ...],
    shape: tuple[int, ...],
) -> None:
    existing = specs.get(tensor_id)
    candidate = TensorSpec(
        id=tensor_id,
        labels=tuple(int(label) for label in labels),
        shape=tuple(int(dim) for dim in shape),
        structure="dense",
        dtype="float64",
    )
    if existing is not None:
        if existing.labels != candidate.labels or existing.shape != candidate.shape:
            raise ValueError(f"Inconsistent synthetic_pressure initial tensor shape for {tensor_id}")
        return
    specs[tensor_id] = candidate


def _deterministic_real_array(shape: tuple[int, ...], *, seed: int) -> np.ndarray:
    size = int(np.prod(shape, dtype=np.int64))
    values = np.arange(size, dtype=np.float64).reshape(shape)
    return (((values + seed * 17.0) % 31.0) - 15.0) / 31.0


def _annotate_task(task: ContractionTask) -> ContractionTask:
    estimate = estimate_dense_task(task)
    return replace(
        task,
        target_estimates={
            **task.target_estimates,
            UPMEM_DENSE_ESTIMATE_KEY: estimate.as_task_estimate(),
        },
    )


def _path_summary(name: str, task_count: int) -> PathSummary:
    return PathSummary(
        planner="synthetic_pressure",
        optimize="synthetic_pressure",
        path_length=task_count,
        largest_intermediate=None,
        naive_flops=None,
        optimized_flops=None,
        text=f"synthetic pressure graph: {name}",
        planner_engine="synthetic_pressure",
        planner_id="synthetic_pressure",
        planner_kind="model_only",
        optimize_mode="model_only",
        objective="pressure_model",
        cost_basis="upmem_dense_int8_estimate",
        target_estimate_key=UPMEM_DENSE_ESTIMATE_KEY,
        options={},
    )
