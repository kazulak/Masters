"""Target-neutral CPU compilation and temporary target dispatch."""

from __future__ import annotations

from collections import defaultdict

from quantum_bench.execution.contracts import (
    CpuCompileRequest,
    CpuPlan,
    ExecutionPlan,
    Target,
    UnsupportedExecution,
    UpmemCompileRequest,
)
from quantum_bench.tn.graph import (
    ContractionDAG,
    contraction_dag_hash,
    validate_contraction_dag,
)


def compile_cpu(dag: ContractionDAG, request: CpuCompileRequest) -> ExecutionPlan:
    """Compile a DAG into a deterministic NumPy CPU execution plan."""

    if not isinstance(request, CpuCompileRequest):
        raise TypeError("compile_cpu requires a CpuCompileRequest")
    validate_contraction_dag(dag)
    actual_hash = contraction_dag_hash(dag)
    if request.contraction_dag_hash != actual_hash:
        raise ValueError(
            "CPU compile request hash does not match the supplied contraction DAG"
        )

    node_ids = tuple(node.node_id for node in dag.nodes)
    order = request.node_order or _topological_order(dag)
    _validate_node_order(dag, order)
    if set(order) != set(node_ids):
        raise ValueError("CPU node_order must contain every DAG node exactly once")

    return ExecutionPlan(
        contraction_dag_hash=actual_hash,
        target=Target.CPU,
        payload=CpuPlan(
            numeric_mode=request.numeric_mode,
            executor_id=request.executor_id,
            node_order=tuple(order),
        ),
    )


def compile_execution(
    dag: ContractionDAG,
    request: CpuCompileRequest | UpmemCompileRequest | Target,
) -> ExecutionPlan | UnsupportedExecution:
    """Compile a supported target or return an explicit capability result."""

    match request:
        case CpuCompileRequest():
            return compile_cpu(dag, request)
        case UpmemCompileRequest():
            from quantum_bench.upmem.plan import (
                compile_upmem as compile_upmem_target,
            )

            return compile_upmem_target(dag, request)
        case Target.GPU:
            return UnsupportedExecution(
                target=Target.GPU,
                capability="gpu_execution_adapter",
                reason="GPU compilation is not implemented in this slice",
            )
        case Target.UPMEM:
            return UnsupportedExecution(
                target=Target.UPMEM,
                capability="upmem_execution_adapter",
                reason="UPMEM compilation requires an explicit backend request",
            )
        case _:
            raise TypeError(
                f"Unsupported compilation request: {type(request).__name__}"
            )


def _topological_order(dag: ContractionDAG) -> tuple[str, ...]:
    nodes = {node.node_id: node for node in dag.nodes}
    dependents: dict[str, list[str]] = defaultdict(list)
    remaining = {node_id: len(node.dependencies) for node_id, node in nodes.items()}
    for node in dag.nodes:
        for dependency in node.dependencies:
            dependents[dependency].append(node.node_id)

    ready = sorted(node_id for node_id, count in remaining.items() if count == 0)
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for dependent in sorted(dependents[node_id]):
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
        ready.sort()

    if len(order) != len(nodes):
        raise ValueError("ContractionDAG cannot be topologically ordered")
    return tuple(order)


def _validate_node_order(dag: ContractionDAG, order: tuple[str, ...]) -> None:
    if len(set(order)) != len(order):
        raise ValueError("CPU node_order contains duplicate node IDs")
    positions = {node_id: index for index, node_id in enumerate(order)}
    nodes = {node.node_id: node for node in dag.nodes}
    unknown = set(order) - set(nodes)
    if unknown:
        raise ValueError(f"CPU node_order contains unknown nodes: {sorted(unknown)}")
    if set(order) != set(nodes):
        raise ValueError("CPU node_order must contain every DAG node exactly once")
    for node in dag.nodes:
        for dependency in node.dependencies:
            if positions[dependency] >= positions[node.node_id]:
                raise ValueError(
                    f"CPU node_order violates dependency {dependency} -> {node.node_id}"
                )
