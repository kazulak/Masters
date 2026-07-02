from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantum_bench.core.records import CircuitSpec, ContractionTask, JsonDict, PathSummary, TaskGraph, TensorNetworkSpec, TensorSpec, TensorValue
from quantum_bench.tn.network import TensorNetworkValue
from quantum_bench.tn.task_graph import with_path_cost_summary


GENERIC_BOUNDARY_CASE_ID = "generic_rank3_boundary"
GENERIC_BOUNDARY_EINSUM = "abc,cde->abde"


@dataclass(frozen=True)
class GenericBoundaryWorkload:
    case_id: str
    graph: TaskGraph
    network: TensorNetworkValue
    manifest: JsonDict


def is_generic_boundary_case(case: str) -> bool:
    return case == GENERIC_BOUNDARY_CASE_ID


def build_generic_boundary_workload(case: str = GENERIC_BOUNDARY_CASE_ID) -> GenericBoundaryWorkload:
    if not is_generic_boundary_case(case):
        raise ValueError(f"Unsupported generic boundary workload: {case}")

    left_shape = (2, 3, 4)
    right_shape = (4, 5, 2)
    output_shape = (2, 3, 5, 2)
    left_labels = (0, 1, 2)
    right_labels = (2, 3, 4)
    contracted_labels = (2,)
    output_labels = (0, 1, 3, 4)

    circuit = CircuitSpec(
        name=GENERIC_BOUNDARY_CASE_ID,
        n_qubits=0,
        operations=(),
        source={
            "kind": "generic_boundary_execution",
            "name": GENERIC_BOUNDARY_CASE_ID,
            "workload_type": "generic_boundary_execution",
            "developer_only": True,
            "not_real_quantum_circuit": True,
            "purpose": "non_gemm_generic_upmem_boundary",
        },
    )
    left_spec = TensorSpec("boundary_left", left_labels, left_shape, "dense", dtype="float64")
    right_spec = TensorSpec("boundary_right", right_labels, right_shape, "dense", dtype="float64")
    network_spec = TensorNetworkSpec(
        circuit=circuit,
        tensors=(left_spec, right_spec),
        output_labels=output_labels,
        einsum_expression=GENERIC_BOUNDARY_EINSUM,
    )
    task = ContractionTask(
        id="task_0",
        input_tensor_ids=(left_spec.id, right_spec.id),
        output_tensor_id="boundary_output",
        dependencies=(),
        index_expression=GENERIC_BOUNDARY_EINSUM,
        input_shapes=(left_shape, right_shape),
        output_shape=output_shape,
        left_labels=left_labels,
        right_labels=right_labels,
        contracted_labels=contracted_labels,
        output_labels=output_labels,
        gemm_m=int(np.prod((2, 3), dtype=np.int64)),
        gemm_k=4,
        gemm_n=int(np.prod((5, 2), dtype=np.int64)),
        structure="generic_boundary",
        estimated_flops=int(2 * np.prod(output_shape, dtype=np.int64) * 4),
        estimated_bytes=int((np.prod(left_shape) + np.prod(right_shape) + np.prod(output_shape)) * np.dtype(np.float64).itemsize),
    )
    graph = TaskGraph(
        network=network_spec,
        tasks=(task,),
        path=((0, 1),),
        path_summary=_path_summary(),
        planning_time_s=0.0,
    )
    graph = with_path_cost_summary(graph)
    network = TensorNetworkValue(
        spec=network_spec,
        tensors=[
            TensorValue(left_spec, _deterministic_array(left_shape, offset=0.0)),
            TensorValue(right_spec, _deterministic_array(right_shape, offset=7.0)),
        ],
    )
    return GenericBoundaryWorkload(
        case_id=GENERIC_BOUNDARY_CASE_ID,
        graph=graph,
        network=network,
        manifest=generic_boundary_manifest(graph),
    )


def generic_boundary_manifest(graph: TaskGraph) -> JsonDict:
    task = graph.tasks[0]
    return {
        "name": graph.network.circuit.name,
        "n_qubits": 0,
        "depth_proxy": 1,
        "gate_counts": {"1q": 0, "2q": 0, "total": 0},
        "gate_set": [],
        "source": dict(graph.network.circuit.source),
        "workload_kind": "generic_boundary_execution",
        "workload_type": "generic_boundary_execution",
        "developer_only": True,
        "not_real_quantum_circuit": True,
        "einsum_expression": task.index_expression,
        "input_shapes": task.input_shapes,
        "output_shape": task.output_shape,
        "input_ranks": (len(task.input_shapes[0]), len(task.input_shapes[1])),
        "output_rank": len(task.output_shape),
        "contracted_labels": task.contracted_labels,
        "output_labels": task.output_labels,
    }


def _deterministic_array(shape: tuple[int, ...], *, offset: float) -> np.ndarray:
    size = int(np.prod(shape, dtype=np.int64))
    values = np.arange(size, dtype=np.float64).reshape(shape)
    return (((values + offset) % 17.0) - 8.0) / 17.0


def _path_summary() -> PathSummary:
    return PathSummary(
        planner="generic_boundary",
        optimize="developer_boundary",
        path_length=1,
        largest_intermediate=60,
        naive_flops=None,
        optimized_flops=None,
        text="generic rank-3 x rank-3 non-GEMM boundary contraction",
        planner_engine="generic_boundary",
        planner_id="generic_boundary",
        planner_kind="developer_boundary",
        optimize_mode="developer_boundary",
        objective="generic_upmem_boundary",
        cost_basis="generic_loop_int8",
        options={"developer_only": True},
    )
