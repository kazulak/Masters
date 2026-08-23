"""Target-neutral CPU compilation and temporary target dispatch."""

from __future__ import annotations

from collections import defaultdict

from quantum_bench.execution.contracts import (
    CpuCompileRequest,
    CpuPlan,
    ExecutionPlan,
    NumericMode,
    Target,
    UnsupportedExecution,
    UpmemCompileRequest,
    UpmemNodePlan,
    UpmemPlan,
    UpmemTopology,
    UpmemWorkUnit,
)
from quantum_bench.lowering import contraction_dag_hash, validate_contraction_dag
from quantum_bench.model import ContractionDAG

UPMEM_DECOMPOSITION_ID = "m5_v4_tile_decomposition"
UPMEM_PLACEMENT_ID = "m5_rank_wave_placement"
UPMEM_REDUCTION_ID = "m5_tile_host_reduction"


def _legacy_execution_constants() -> tuple[str, str, str, str, str, int, int]:
    """Load historical ABI identities only when historical compilation runs."""

    from quantum_bench.upmem.protocol import (
        INT32_MAX,
        MAX_CONTRACTED,
        NATIVE_EXECUTION_IDENTITY,
    )

    return (
        NATIVE_EXECUTION_IDENTITY["profile"],
        NATIVE_EXECUTION_IDENTITY["abi"],
        NATIVE_EXECUTION_IDENTITY["session_protocol"],
        NATIVE_EXECUTION_IDENTITY["dispatch_mode"],
        NATIVE_EXECUTION_IDENTITY["kernel_identity"],
        MAX_CONTRACTED,
        INT32_MAX // (128 * 128),
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
            return compile_upmem(dag, request)
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


def compile_upmem(
    dag: ContractionDAG, request: UpmemCompileRequest
) -> ExecutionPlan | UnsupportedExecution:
    """Compile the historical M5 contract without polluting canonical UPMEM."""

    from quantum_bench.upmem.plan import _UnsupportedUpmemNode

    if not isinstance(request, UpmemCompileRequest):
        raise TypeError("compile_upmem requires an UpmemCompileRequest")
    validate_contraction_dag(dag)
    actual_hash = contraction_dag_hash(dag)
    if request.contraction_dag_hash != actual_hash:
        raise ValueError(
            "UPMEM compile request hash does not match the supplied contraction DAG"
        )
    _validate_upmem_request(request)
    (
        profile_id,
        abi_id,
        session_id,
        dispatch_id,
        kernel_id,
        _max_contracted,
        _legacy_max_int32_safe_k,
    ) = _legacy_execution_constants()
    numeric_mode = NumericMode(request.numeric_mode)
    order = _topological_order(dag)
    nodes = {node.node_id: node for node in dag.nodes}
    node_plans: list[UpmemNodePlan] = []
    for node_id in order:
        try:
            node_plans.append(
                _compile_upmem_node(
                    nodes[node_id],
                    request,
                    numeric_mode,
                    max_contracted=_max_contracted,
                    max_int32_safe_k=_legacy_max_int32_safe_k,
                )
            )
        except _UnsupportedUpmemNode as exc:
            return UnsupportedExecution(
                target=Target.UPMEM,
                capability=exc.capability,
                reason=exc.reason,
            )
    return ExecutionPlan(
        contraction_dag_hash=actual_hash,
        target=Target.UPMEM,
        payload=UpmemPlan(
            topology=request.topology,
            numeric_mode=numeric_mode,
            kernel_id=kernel_id,
            decomposition_id=UPMEM_DECOMPOSITION_ID,
            placement_id=UPMEM_PLACEMENT_ID,
            reduction_id=UPMEM_REDUCTION_ID,
            node_plans=tuple(node_plans),
            profile_id=profile_id,
            abi_id=abi_id,
            session_id=session_id,
            dispatch_id=dispatch_id,
        ),
    )


def _validate_upmem_request(request: UpmemCompileRequest) -> None:
    topology = request.topology
    if topology.dpu_count < 1 or topology.rank_count < 1:
        raise ValueError("UPMEM topology counts must be positive")
    if topology.dpu_count % topology.rank_count:
        raise ValueError("UPMEM dpu_count must be divisible by rank_count")
    if topology.dpu_count // topology.rank_count > 64:
        raise ValueError("UPMEM supports at most 64 DPUs per rank")
    if not 1 <= topology.tasklets_per_dpu <= 24:
        raise ValueError("UPMEM tasklets_per_dpu must be in [1, 24]")
    if request.numeric_mode not in {
        NumericMode.FLOAT32_REAL,
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
    }:
        raise ValueError(f"unsupported M5 UPMEM numeric mode: {request.numeric_mode!r}")


def _compile_upmem_node(
    node: object,
    request: UpmemCompileRequest,
    numeric_mode: NumericMode,
    *,
    max_contracted: int,
    max_int32_safe_k: int,
) -> UpmemNodePlan:
    from quantum_bench.model import ContractNode, ReduceNode
    from quantum_bench.upmem.plan import (
        _UnsupportedUpmemNode,
        _validate_v4_node_geometry,
    )
    from quantum_bench.upmem.tiling import (
        order_tile_waves,
        plan_tile_shapes,
        tile_limits_for_numeric_mode,
    )

    if isinstance(node, ReduceNode):
        return UpmemNodePlan(
            node_id=node.node_id,
            node_kind="reduce",
            canonical_shape=None,
            work_units=(),
            reduction_mode="host_sum_v1",
            arithmetic_imbalance=0.0,
        )
    if not isinstance(node, ContractNode):
        raise TypeError(f"unsupported DAG node: {type(node).__name__}")
    batch, m, n, contracted_size = _validate_v4_node_geometry(node)
    if contracted_size > max_contracted:
        raise _UnsupportedUpmemNode(
            capability="upmem_max_contracted_elements",
            reason=(
                f"UPMEM node {node.node_id} contracted K {contracted_size} "
                f"exceeds v4 limit {max_contracted}"
            ),
        )
    if (
        numeric_mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
        and contracted_size > max_int32_safe_k
    ):
        raise _UnsupportedUpmemNode(
            capability="upmem_int8_int32_accumulation_bound",
            reason=(
                f"UPMEM node {node.node_id} contracted K {contracted_size} "
                f"exceeds int32 int8 safety bound {max_int32_safe_k}"
            ),
        )
    tile_numeric_mode = (
        "host_packed_int8"
        if numeric_mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
        else "float32"
    )
    limits = tile_limits_for_numeric_mode(tile_numeric_mode)
    tiles = plan_tile_shapes(batch, m, contracted_size, n, limits=limits)
    waves = order_tile_waves(tiles, request.topology.dpu_count)
    work_units = _compile_work_units(node.node_id, waves, request)
    return UpmemNodePlan(
        node_id=node.node_id,
        node_kind="contract",
        canonical_shape=(batch, m, contracted_size, n),
        work_units=work_units,
        reduction_mode=UPMEM_REDUCTION_ID,
        arithmetic_imbalance=_arithmetic_imbalance(work_units, request.topology),
    )


def _arithmetic_imbalance(
    work_units: tuple[UpmemWorkUnit, ...], topology: UpmemTopology
) -> float:
    dpus_per_rank = topology.dpu_count // topology.rank_count
    loads = [0] * topology.dpu_count
    if not work_units:
        return 0.0
    for unit in work_units:
        loads[unit.logical_rank * dpus_per_rank + unit.logical_dpu] += (
            unit.estimated_arithmetic_work
        )
    return max(loads) / (sum(loads) / len(loads))


def _compile_work_units(
    node_id: str,
    waves: tuple[tuple[object, ...], ...],
    request: UpmemCompileRequest,
) -> tuple[UpmemWorkUnit, ...]:
    dpus_per_rank = request.topology.dpu_count // request.topology.rank_count
    units: list[UpmemWorkUnit] = []
    for wave_index, wave in enumerate(waves):
        for global_slot, tile in enumerate(wave):
            units.append(
                UpmemWorkUnit(
                    node_id=node_id,
                    stable_tile_id=tile.id,
                    wave=wave_index,
                    logical_rank=global_slot // dpus_per_rank,
                    logical_dpu=global_slot % dpus_per_rank,
                    batch_start=tile.batch_index,
                    batch_size=1,
                    m_start=tile.m_start,
                    m_size=tile.m_size,
                    n_start=tile.n_start,
                    n_size=tile.n_size,
                    k_start=tile.k_start,
                    k_size=tile.k_size,
                    estimated_input_bytes=tile.left_bytes + tile.right_bytes,
                    estimated_output_bytes=tile.output_bytes,
                    aligned_mram_bytes=tile.aligned_mram_bytes,
                    estimated_arithmetic_work=tile.m_size * tile.n_size * tile.k_size,
                )
            )
    return tuple(units)


def validate_upmem_plan_for_dag(dag: ContractionDAG, plan: UpmemPlan) -> None:
    from quantum_bench.upmem.plan import _UnsupportedUpmemNode

    validate_contraction_dag(dag)
    order = _topological_order(dag)
    if tuple(item.node_id for item in plan.node_plans) != order:
        raise ValueError("UPMEM node plans do not match deterministic DAG order")
    request = UpmemCompileRequest(
        contraction_dag_hash=contraction_dag_hash(dag),
        numeric_mode=plan.numeric_mode,
        topology=plan.topology,
    )
    _, _, _, _, _, max_contracted, max_int32_safe_k = _legacy_execution_constants()
    nodes = {node.node_id: node for node in dag.nodes}
    expected: list[UpmemNodePlan] = []
    for node_id in order:
        node = nodes[node_id]
        try:
            expected.append(
                _compile_upmem_node(
                    node,
                    request,
                    plan.numeric_mode,
                    max_contracted=max_contracted,
                    max_int32_safe_k=max_int32_safe_k,
                )
            )
        except _UnsupportedUpmemNode as exc:
            raise ValueError(
                f"UPMEM node {node.node_id} is not lowerable: {exc.reason}"
            ) from exc
    if plan.node_plans != tuple(expected):
        raise ValueError("UPMEM node plans differ from pure v4 recomputation")


def validate_active_upmem_plan(plan: UpmemPlan) -> None:
    profile_id, abi_id, session_id, dispatch_id, kernel_id, _, _ = (
        _legacy_execution_constants()
    )
    expected = {
        "profile_id": profile_id,
        "abi_id": abi_id,
        "session_id": session_id,
        "dispatch_id": dispatch_id,
        "kernel_id": kernel_id,
        "decomposition_id": "m5_v4_tile_decomposition",
        "placement_id": "m5_rank_wave_placement",
        "reduction_id": "m5_tile_host_reduction",
    }
    for field, value in expected.items():
        if getattr(plan, field) != value:
            raise ValueError(
                f"unsupported active UPMEM {field}: {getattr(plan, field)!r}"
            )


__all__ = [
    "compile_cpu",
    "compile_execution",
    "compile_upmem",
    "validate_active_upmem_plan",
    "validate_upmem_plan_for_dag",
]
