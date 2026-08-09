from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from quantum_bench.core.records import (
    CircuitSpec,
    ContractionTask,
    PathSummary,
    TaskGraph,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
)
from quantum_bench.tn.execution_bundle import with_execution_identity
from quantum_bench.tn.network import TensorNetworkValue


RANK1_DOT_LENGTH = 256
RANK1_DOT_CASE_ID = "simplepim_rank1_taskgraph_fixture"
RANK1_LEFT_TENSOR_ID = "rank1_left"
RANK1_RIGHT_TENSOR_ID = "rank1_right"
RANK1_OUTPUT_TENSOR_ID = "rank1_output"
RANK1_TASK_ID = "rank1_dot_task"
RANK1_INDEX_EXPRESSION = "a,a->"
RANK1_STRUCTURE = "rank1_dot_int8"


@dataclass(frozen=True)
class Rank1TaskGraphWorkload:
    network: TensorNetworkValue
    graph: TaskGraph
    left: np.ndarray
    right: np.ndarray
    reference_int64: int


def build_rank1_taskgraph_workload() -> Rank1TaskGraphWorkload:
    values = np.arange(RANK1_DOT_LENGTH, dtype=np.int16)
    left = ((values * 3) % 31 - 15).astype(np.int8)
    right = ((values * 5) % 29 - 14).astype(np.int8)

    circuit = CircuitSpec(
        name=RANK1_DOT_CASE_ID,
        n_qubits=0,
        operations=(),
        source={
            "kind": "taskgraph_fixture",
            "workload_type": "simplepim_rank1_dot",
            "developer_only": True,
            "not_general_quantum_evidence": True,
        },
    )
    left_spec = TensorSpec(
        id=RANK1_LEFT_TENSOR_ID,
        labels=(0,),
        shape=(RANK1_DOT_LENGTH,),
        structure="dense",
        dtype="int8",
    )
    right_spec = TensorSpec(
        id=RANK1_RIGHT_TENSOR_ID,
        labels=(0,),
        shape=(RANK1_DOT_LENGTH,),
        structure="dense",
        dtype="int8",
    )
    network_spec = TensorNetworkSpec(
        circuit=circuit,
        tensors=(left_spec, right_spec),
        output_labels=(),
        einsum_expression=RANK1_INDEX_EXPRESSION,
    )
    task = ContractionTask(
        id=RANK1_TASK_ID,
        input_tensor_ids=(RANK1_LEFT_TENSOR_ID, RANK1_RIGHT_TENSOR_ID),
        output_tensor_id=RANK1_OUTPUT_TENSOR_ID,
        dependencies=(),
        index_expression=RANK1_INDEX_EXPRESSION,
        input_shapes=((RANK1_DOT_LENGTH,), (RANK1_DOT_LENGTH,)),
        output_shape=(),
        left_labels=(0,),
        right_labels=(0,),
        contracted_labels=(0,),
        output_labels=(),
        gemm_m=1,
        gemm_k=RANK1_DOT_LENGTH,
        gemm_n=1,
        structure=RANK1_STRUCTURE,
        estimated_flops=2 * RANK1_DOT_LENGTH,
        estimated_bytes=2 * RANK1_DOT_LENGTH,
    )
    graph = with_execution_identity(
        TaskGraph(
            network=network_spec,
            tasks=(task,),
            path=((0, 1),),
            path_summary=PathSummary(
                planner="fixture",
                optimize="deterministic",
                path_length=1,
                largest_intermediate=1,
                naive_flops=float(2 * RANK1_DOT_LENGTH),
                optimized_flops=float(2 * RANK1_DOT_LENGTH),
                text="deterministic rank-1 int8 dot fixture",
                planner_engine="fixture",
                planner_id=RANK1_DOT_CASE_ID,
                planner_kind="developer_fixture",
                optimize_mode="deterministic",
                objective="rank1_dot",
                cost_basis="int8_int32",
                task_count=1,
                total_estimated_flops=2 * RANK1_DOT_LENGTH,
                peak_intermediate_bytes=8,
                max_intermediate_bytes=8,
            ),
            planning_time_s=0.0,
        )
    )
    network = TensorNetworkValue(
        spec=network_spec,
        tensors=[TensorValue(left_spec, left.copy()), TensorValue(right_spec, right.copy())],
    )
    return Rank1TaskGraphWorkload(
        network=network,
        graph=graph,
        left=left,
        right=right,
        reference_int64=int(np.sum(left.astype(np.int64) * right.astype(np.int64), dtype=np.int64)),
    )


def validate_rank1_task(workload: Rank1TaskGraphWorkload) -> None:
    graph = workload.graph
    network = workload.network
    if graph.network != network.spec:
        raise ValueError("rank1_task_network_spec_mismatch")
    if len(graph.tasks) != 1:
        raise ValueError("rank1_task_requires_exactly_one_task")
    if graph.path != ((0, 1),):
        raise ValueError("rank1_task_path_structure_mismatch")
    task = graph.tasks[0]
    if task.dependencies:
        raise ValueError("rank1_task_dependencies_not_supported")
    if task.id != RANK1_TASK_ID:
        raise ValueError("rank1_task_id_mismatch")
    if task.input_tensor_ids != (RANK1_LEFT_TENSOR_ID, RANK1_RIGHT_TENSOR_ID):
        raise ValueError("rank1_task_input_tensor_ids_mismatch")
    if task.output_tensor_id != RANK1_OUTPUT_TENSOR_ID or task.output_tensor_id in task.input_tensor_ids:
        raise ValueError("rank1_task_output_tensor_identity_mismatch")
    if task.index_expression != RANK1_INDEX_EXPRESSION:
        raise ValueError("rank1_task_index_expression_mismatch")
    if task.input_shapes != ((RANK1_DOT_LENGTH,), (RANK1_DOT_LENGTH,)) or task.output_shape != ():
        raise ValueError("rank1_task_shape_mismatch")
    if task.left_labels != (0,) or task.right_labels != (0,) or task.contracted_labels != (0,) or task.output_labels:
        raise ValueError("rank1_task_label_structure_mismatch")
    if (task.gemm_m, task.gemm_k, task.gemm_n) != (1, RANK1_DOT_LENGTH, 1):
        raise ValueError("rank1_task_gemm_shape_mismatch")
    if task.structure != RANK1_STRUCTURE:
        raise ValueError("rank1_task_structure_mismatch")
    if network.spec.output_labels != () or network.spec.einsum_expression != RANK1_INDEX_EXPRESSION:
        raise ValueError("rank1_task_network_output_identity_mismatch")
    if len(network.tensors) != 2:
        raise ValueError("rank1_task_requires_two_input_tensors")
    expected_specs = (
        (RANK1_LEFT_TENSOR_ID, workload.left),
        (RANK1_RIGHT_TENSOR_ID, workload.right),
    )
    for tensor, (expected_id, expected_array) in zip(network.tensors, expected_specs, strict=True):
        if tensor.spec.id != expected_id or tensor.spec.labels != (0,) or tensor.spec.shape != (RANK1_DOT_LENGTH,):
            raise ValueError("rank1_task_tensor_spec_mismatch")
        if tensor.spec.structure != "dense" or tensor.spec.dtype != "int8":
            raise ValueError("rank1_task_tensor_dtype_mismatch")
        array = np.asarray(tensor.array)
        if array.shape != (RANK1_DOT_LENGTH,) or array.dtype != np.dtype(np.int8) or np.iscomplexobj(array):
            raise ValueError("rank1_task_tensor_array_mismatch")
        if not np.array_equal(array, expected_array):
            raise ValueError("rank1_task_tensor_value_mismatch")
    for array in (workload.left, workload.right):
        if array.shape != (RANK1_DOT_LENGTH,) or array.dtype != np.dtype(np.int8) or np.iscomplexobj(array):
            raise ValueError("rank1_task_operand_dtype_mismatch")
    expected_reference = int(np.sum(workload.left.astype(np.int64) * workload.right.astype(np.int64), dtype=np.int64))
    if not isinstance(workload.reference_int64, Integral) or isinstance(workload.reference_int64, bool) or int(workload.reference_int64) != expected_reference:
        raise ValueError("rank1_task_reference_mismatch")
    with_execution_identity(graph)
