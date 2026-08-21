"""Pure compilation for the active tensor-network execution slice."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from quantum_bench.execution.contracts import (
    CpuCompileRequest,
    CpuPlan,
    ExecutionPlan,
    NumericMode,
    Target,
    UnsupportedExecution,
    UpmemCompileRequest,
    UpmemNodeWorkPlan,
    UpmemPlan,
)
from quantum_bench.tn.graph import (
    ContractNode,
    ContractionDAG,
    ReduceNode,
    contraction_dag_hash,
    validate_contraction_dag,
)
from quantum_bench.targets.upmem.execution_plan_v4 import (
    MAX_CONTRACTED,
    MAX_INT32_SAFE_K,
)
from quantum_bench.targets.upmem.m5_whole_circuit_tiles import (
    TileLoweringError,
    canonical_label_geometry,
)


UPMEM_PROFILE_ID = "m5_whole_circuit_v4_v1"
UPMEM_ABI_ID = "execution_plan_v4"
UPMEM_SESSION_ID = "persistent_rank_session_v1"
UPMEM_DISPATCH_ID = "bulk_set_synchronous_v1"
UPMEM_KERNEL_ID = "dpu_gemm_tile_v4"
UPMEM_DECOMPOSITION_ID = "m5_v4_tile_decomposition"
UPMEM_PLACEMENT_ID = "m5_rank_wave_placement"
UPMEM_REDUCTION_ID = "m5_tile_host_reduction"

# The v4 ABI encodes B/M/N/K and aggregate output dimensions as uint64_t.
# ``canonical_batch_count`` is additionally limited by the Python v4 request
# builder before it serializes the native header.
_V4_UINT64_MAX = (1 << 64) - 1
_V4_MAX_BATCH_COUNT = (1 << 32) - 1


class _UnsupportedUpmemNode(Exception):
    def __init__(self, *, capability: str, reason: str) -> None:
        super().__init__(reason)
        self.capability = capability
        self.reason = reason


def compile_cpu(dag: ContractionDAG, request: CpuCompileRequest) -> ExecutionPlan:
    """Compile a DAG into a deterministic NumPy CPU execution plan.

    The compiler only records execution order and numeric policy.  It does not
    inspect tensor values, allocate buffers, or mutate the graph.
    """

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


def compile_upmem(
    dag: ContractionDAG, request: UpmemCompileRequest
) -> ExecutionPlan | UnsupportedExecution:
    """Compile a semantic DAG into the bounded M5 v4 execution contract."""

    if not isinstance(request, UpmemCompileRequest):
        raise TypeError("compile_upmem requires an UpmemCompileRequest")
    validate_contraction_dag(dag)
    actual_hash = contraction_dag_hash(dag)
    if request.contraction_dag_hash != actual_hash:
        raise ValueError(
            "UPMEM compile request hash does not match the supplied contraction DAG"
        )
    _validate_upmem_request(request)
    numeric_mode = NumericMode(request.numeric_mode)

    order = _topological_order(dag)
    work_plans_list: list[UpmemNodeWorkPlan] = []
    for node in _nodes_in_order(dag, order):
        try:
            work_plans_list.append(_compile_upmem_node(node, request, numeric_mode))
        except _UnsupportedUpmemNode as exc:
            return UnsupportedExecution(
                target=Target.UPMEM,
                capability=exc.capability,
                reason=exc.reason,
            )
    work_plans = tuple(work_plans_list)
    if request.node_work_plans and request.node_work_plans != work_plans:
        raise ValueError(
            "UPMEM node_work_plans must match compiler-generated M5 work plans"
        )

    return ExecutionPlan(
        contraction_dag_hash=actual_hash,
        target=Target.UPMEM,
        payload=UpmemPlan(
            topology=request.topology,
            numeric_mode=numeric_mode,
            kernel_id=request.kernel_id,
            decomposition_id=request.decomposition_id,
            placement_id=request.placement_id,
            reduction_id=request.reduction_id,
            node_work_plans=work_plans,
            node_order=order,
            profile_id=request.profile_id,
            abi_id=request.abi_id,
            session_id=request.session_id,
            dispatch_id=request.dispatch_id,
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
    expected = {
        "profile_id": UPMEM_PROFILE_ID,
        "abi_id": UPMEM_ABI_ID,
        "session_id": UPMEM_SESSION_ID,
        "dispatch_id": UPMEM_DISPATCH_ID,
        "kernel_id": UPMEM_KERNEL_ID,
        "decomposition_id": UPMEM_DECOMPOSITION_ID,
        "placement_id": UPMEM_PLACEMENT_ID,
        "reduction_id": UPMEM_REDUCTION_ID,
    }
    for field, value in expected.items():
        if getattr(request, field) != value:
            raise ValueError(
                f"unsupported M5 UPMEM {field}: {getattr(request, field)!r}"
            )
    if request.numeric_mode not in {
        NumericMode.FLOAT32_REAL,
        NumericMode.HOST_PACKED_INT8_PER_TASK_V1,
    }:
        raise ValueError(f"unsupported M5 UPMEM numeric mode: {request.numeric_mode!r}")


def _compile_upmem_node(
    node: ContractNode | ReduceNode,
    request: UpmemCompileRequest,
    numeric_mode: NumericMode,
) -> UpmemNodeWorkPlan:
    if isinstance(node, ReduceNode):
        return UpmemNodeWorkPlan(
            node_id=node.node_id,
            kernel_id="host_reduce_v1",
            decomposition_id="host_reduce_v1",
            placement_id="host",
            reduction_id="host_sum_v1",
            tile_count=0,
        )
    _, _, _, contracted_size = _validate_v4_node_geometry(node)
    if contracted_size > MAX_CONTRACTED:
        raise _UnsupportedUpmemNode(
            capability="upmem_max_contracted_elements",
            reason=(
                f"UPMEM node {node.node_id} contracted K {contracted_size} "
                f"exceeds v4 limit {MAX_CONTRACTED}"
            ),
        )
    if (
        numeric_mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
        and contracted_size > MAX_INT32_SAFE_K
    ):
        raise _UnsupportedUpmemNode(
            capability="upmem_int8_int32_accumulation_bound",
            reason=(
                f"UPMEM node {node.node_id} contracted K {contracted_size} "
                f"exceeds int32 int8 safety bound {MAX_INT32_SAFE_K}"
            ),
        )
    return UpmemNodeWorkPlan(
        node_id=node.node_id,
        kernel_id=UPMEM_KERNEL_ID,
        decomposition_id=UPMEM_DECOMPOSITION_ID,
        placement_id=UPMEM_PLACEMENT_ID,
        reduction_id=UPMEM_REDUCTION_ID,
        dpu_ids=tuple(range(request.topology.dpu_count)),
        # M5 tile count depends on concrete operand values and layouts.
        tile_count=None,
    )


def _validate_v4_node_geometry(node: ContractNode) -> tuple[int, int, int, int]:
    """Reject only geometry that the tiled v4 request ABI cannot encode.

    V4 receives canonical ``(B, M, K) @ (B, K, N)`` tiles. It has no raw
    tensor-rank or whole-tensor element cap: arbitrary input/output ranks and
    large logical tensors are lowered into host-side canonical views and
    bounded MRAM tiles. The capability boundary here is therefore the native
    header's integer fields, not the legacy generic-loop rank/element caps.
    """

    try:
        batch, m, n, contracted = _canonical_dimensions(node)
    except TileLoweringError as exc:
        raise _UnsupportedUpmemNode(
            capability="upmem_v4_positive_canonical_geometry",
            reason=f"UPMEM node {node.node_id} {exc}",
        ) from exc
    except OverflowError as exc:
        raise _UnsupportedUpmemNode(
            capability="upmem_v4_uint64_element_count",
            reason=f"UPMEM node {node.node_id} {exc}",
        ) from exc
    values = {
        "batch": batch,
        "M": m,
        "N": n,
        "K": contracted,
    }
    for name, value in values.items():
        if value > _V4_UINT64_MAX:
            raise _UnsupportedUpmemNode(
                capability="upmem_v4_uint64_geometry",
                reason=(
                    f"UPMEM node {node.node_id} canonical {name} {value} "
                    "exceeds the v4 uint64 ABI field"
                ),
            )
    if batch > _V4_MAX_BATCH_COUNT:
        raise _UnsupportedUpmemNode(
            capability="upmem_v4_batch_count",
            reason=(
                f"UPMEM node {node.node_id} canonical batch count {batch} "
                f"exceeds v4 request limit {_V4_MAX_BATCH_COUNT}"
            ),
        )
    try:
        _bounded_product((batch, m, contracted), label="left operand")
        _bounded_product((batch, contracted, n), label="right operand")
        _bounded_product((batch, m, n), label="output")
    except OverflowError as exc:
        raise _UnsupportedUpmemNode(
            capability="upmem_v4_uint64_element_count",
            reason=f"UPMEM node {node.node_id} {exc}",
        ) from exc
    return batch, m, n, contracted


def _canonical_dimensions(node: ContractNode) -> tuple[int, int, int, int]:
    """Return the native M5 ``B, M, N, K`` dimensions from semantic labels.

    The shared helper intentionally excludes unilateral reductions from K,
    matching the native tile canonicalizer that pre-sums them before GEMM.
    """

    batch, m, contracted, n = canonical_label_geometry(
        node.left.labels,
        node.left.shape,
        node.right.labels,
        node.right.shape,
        node.output_labels,
    )
    return batch, m, n, contracted


def _bounded_product(values: Iterable[int], *, label: str) -> int:
    result = 1
    for raw_value in values:
        value = int(raw_value)
        if value < 1:
            raise OverflowError(f"{label} has a non-positive dimension {value}")
        if result > _V4_UINT64_MAX // value:
            raise OverflowError(f"{label} element count exceeds the v4 uint64 limit")
        result *= value
    return result


def _nodes_in_order(
    dag: ContractionDAG, order: tuple[str, ...]
) -> tuple[ContractNode | ReduceNode, ...]:
    nodes = {node.node_id: node for node in dag.nodes}
    return tuple(nodes[node_id] for node_id in order)


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


__all__ = ["compile_cpu", "compile_execution", "compile_upmem"]
