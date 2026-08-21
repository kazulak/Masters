"""Pure NumPy execution of a compiled contraction DAG."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping

import numpy as np

from quantum_bench.execution.contracts import (
    ExecutionPlan,
    ExecutionResult,
    RunContext,
    Target,
    TimingBreakdown,
    validate_execution_plan,
    validate_execution_result,
)
from quantum_bench.execution.numeric import contract_node, reduce_values
from quantum_bench.tn.graph import ContractNode, ContractionDAG, ReduceNode, TensorView
from quantum_bench.tn.graph import contraction_dag_hash, validate_contraction_dag
from quantum_bench.tn.network import TensorInputs, tensor_input_map, validate_dag_inputs


def run_cpu(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray] | TensorInputs,
    context: RunContext,
) -> ExecutionResult:
    """Execute a compiled CPU plan without mutating source arrays.

    The result contract carries the computed output directly.  The output is
    copied before it is returned so the local execution buffer cannot be
    mutated through a later internal reference.
    """

    tensors = _input_map(inputs)
    _validate_cpu_invocation(plan, dag, tensors, context)
    for _ in range(context.warmups):
        _execute_nodes(plan, dag, tensors)

    output: np.ndarray | None = None
    output_digest: str | None = None
    kernel_elapsed = 0.0
    for _ in range(context.repetitions):
        kernel_start = time.perf_counter()
        output = _execute_nodes(plan, dag, tensors)
        kernel_elapsed += time.perf_counter() - kernel_start
        digest = _array_hash(output)
        if output_digest is None:
            output_digest = digest
        elif digest != output_digest:
            raise RuntimeError("CPU execution produced non-deterministic output")

    if output is None or output_digest is None:  # guarded by repetitions validation
        raise RuntimeError("CPU execution did not produce an output")
    returned_output = np.array(output, copy=True)
    result = ExecutionResult(
        contraction_dag_hash=contraction_dag_hash(dag),
        target=Target.CPU,
        output=returned_output,
        executed_node_ids=tuple(plan.payload.node_order),
        timing=TimingBreakdown(
            kernel_s=kernel_elapsed,
            route_total_s=kernel_elapsed,
        ),
        output_hash=output_digest,
    )
    validate_execution_result(result)
    return result


def _execute_nodes(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    tensors: dict[str, np.ndarray],
) -> np.ndarray:
    """Execute the plan's explicit dependency order into a local tensor map."""

    node_by_id = {node.node_id: node for node in dag.nodes}
    working = dict(tensors)
    remaining_uses = _tensor_consumer_counts(dag)
    evictable_tensor_ids = {
        node.output.id for node in dag.nodes if node.output.id not in tensors
    }
    for node_id in plan.payload.node_order:  # type: ignore[union-attr]
        node = node_by_id[node_id]
        if isinstance(node, ContractNode):
            left = _resolve_view(node.left, working)
            right = _resolve_view(node.right, working)
            result = contract_node(node, left, right, plan.payload.numeric_mode)
        elif isinstance(node, ReduceNode):
            values = [_resolve_view(view, working) for view in node.inputs]
            result = reduce_values(tuple(values))
        else:  # pragma: no cover - GraphNode is closed by the graph contract
            raise TypeError(f"Unsupported DAG node: {type(node).__name__}")
        expected = node.output.shape
        if tuple(result.shape) != expected:
            raise ValueError(
                f"Node {node.node_id} produced shape {result.shape}; expected {expected}"
            )
        working[node.output.id] = result
        if (
            remaining_uses.get(node.output.id, 0) == 0
            and node.output.id != dag.output.tensor_id
        ):
            del working[node.output.id]
        for tensor_id in _node_input_ids(node):
            if tensor_id not in evictable_tensor_ids:
                continue
            remaining_uses[tensor_id] -= 1
            if (
                remaining_uses[tensor_id] == 0
                and tensor_id != dag.output.tensor_id
            ):
                del working[tensor_id]

    return _resolve_view(dag.output, working)


def _node_input_ids(node: ContractNode | ReduceNode) -> tuple[str, ...]:
    if isinstance(node, ContractNode):
        return node.left.tensor_id, node.right.tensor_id
    return tuple(view.tensor_id for view in node.inputs)


def _tensor_consumer_counts(dag: ContractionDAG) -> dict[str, int]:
    """Count node-input references for produced tensors in the DAG."""

    counts: dict[str, int] = {}
    for node in dag.nodes:
        for tensor_id in _node_input_ids(node):
            counts[tensor_id] = counts.get(tensor_id, 0) + 1
    return counts


def _validate_cpu_invocation(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    context: RunContext,
) -> None:
    validate_contraction_dag(dag)
    validate_execution_plan(plan)
    if plan.target is not Target.CPU:
        raise ValueError("run_cpu requires a CPU execution plan")
    if context.target is not Target.CPU:
        raise ValueError("run_cpu requires a CPU RunContext")
    if context.warmups < 0 or context.repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    actual_hash = contraction_dag_hash(dag)
    if plan.contraction_dag_hash != actual_hash:
        raise ValueError("execution plan hash does not match the supplied contraction DAG")
    _validate_runtime_node_order(dag, tuple(plan.payload.node_order))

    validate_dag_inputs(dag, inputs)


def _validate_runtime_node_order(dag: ContractionDAG, order: tuple[str, ...]) -> None:
    nodes = {node.node_id: node for node in dag.nodes}
    if len(order) != len(nodes) or len(set(order)) != len(order):
        raise ValueError("CPU plan node order must contain every DAG node exactly once")
    if set(order) != set(nodes):
        raise ValueError("CPU plan node order contains unknown or missing nodes")
    positions = {node_id: index for index, node_id in enumerate(order)}
    for node in dag.nodes:
        for dependency in node.dependencies:
            if positions[dependency] >= positions[node.node_id]:
                raise ValueError(
                    f"CPU plan node order violates dependency {dependency} -> {node.node_id}"
                )


def _input_map(
    inputs: Mapping[str, np.ndarray] | TensorInputs,
) -> dict[str, np.ndarray]:
    if isinstance(inputs, TensorInputs):
        if len({value.tensor_id for value in inputs.values}) != len(inputs.values):
            raise ValueError("Tensor inputs contain duplicate tensor ids")
        return tensor_input_map(inputs)
    return {tensor_id: np.asarray(array) for tensor_id, array in inputs.items()}


def _resolve_view(view: TensorView, tensors: dict[str, np.ndarray]) -> np.ndarray:
    try:
        array = tensors[view.tensor_id]
    except KeyError as exc:
        raise ValueError(f"Tensor {view.tensor_id} is not available for execution") from exc
    if not view.slice_spec:
        return array
    indices: list[slice | int] = [slice(None)] * array.ndim
    for axis, value in view.slice_spec:
        if axis < 0 or axis >= array.ndim:
            raise ValueError(f"Tensor view {view.tensor_id} has invalid fixed axis {axis}")
        indices[axis] = int(value)
    result = array[tuple(indices)]
    if tuple(result.shape) != view.shape:
        raise ValueError(
            f"Tensor view {view.tensor_id} produced shape {result.shape}; expected {view.shape}"
        )
    return result


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


__all__ = ["run_cpu"]
