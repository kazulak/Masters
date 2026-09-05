"""Pure static scheduling for dependency-ready UPMEM DAG waves.

The scheduler only rearranges the work units already present in an
``UpmemPlan``.  It does not execute work, add an intermediate representation,
or change the logical contraction graph.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from quantum_bench.lowering import contraction_dag_hash, validate_contraction_dag
from quantum_bench.model import ContractNode, ContractionDAG, GraphNode, ReduceNode
from quantum_bench.upmem.plan import UpmemPlan, UpmemStage, UpmemWorkUnit
from quantum_bench.upmem.protocol import MAX_DPUS, MAX_TASKLETS


_REAL_PRODUCT_COUNT = 4
_SUPPORTED_KERNEL_POLICY = "dpu_real_tile_v4_wram_panel_v1"


def schedule_dag_waves(
    dag: ContractionDAG, plan: UpmemPlan
) -> tuple[UpmemStage, ...]:
    """Return a deterministic dependency-ready schedule for ``dag``.

    A contract cohort is one synchronous host-visible batch.  A cohort keeps
    its disjoint DPU groups while its nodes' work units are emitted in local
    subwaves.  Host reductions are explicit zero-work stages and complete
    before a dependent contract can enter a later cohort.

    This function is intentionally software-only.  Returning a schedule does
    not make the current runtime launch multiple nodes concurrently.
    """

    nodes, units_by_node = _validate_inputs(dag, plan)
    critical_path = _remaining_critical_path(nodes, units_by_node)
    dpu_count = plan.topology.dpu_count

    completed: set[str] = set()
    result: list[UpmemStage] = []
    cohort_index = 0

    while len(completed) < len(nodes):
        ready_reductions = tuple(
            sorted(
                node_id
                for node_id, node in nodes.items()
                if isinstance(node, ReduceNode)
                and node_id not in completed
                and _dependencies_complete(node, completed)
            )
        )
        if ready_reductions:
            for node_id in ready_reductions:
                result.append(
                    UpmemStage(
                        stage_id=(
                            f"dag_cohort:{cohort_index}:host_reduce:{node_id}"
                        ),
                        kind="host_reduce",
                        node_ids=(node_id,),
                        work_units=(),
                    )
                )
                completed.add(node_id)
            continue

        ready_contracts = tuple(
            sorted(
                (
                    node_id
                    for node_id, node in nodes.items()
                    if isinstance(node, ContractNode)
                    and node_id not in completed
                    and _dependencies_complete(node, completed)
                ),
                key=lambda node_id: (-critical_path[node_id], node_id),
            )
        )
        if not ready_contracts:
            raise ValueError("DAG scheduling could not make dependency progress")

        selected = ready_contracts[:dpu_count]
        allocations = _allocate_dpus(selected, units_by_node, dpu_count)
        result.append(
            _contract_cohort(
                cohort_index,
                selected,
                units_by_node,
                allocations,
                dpu_count,
            )
        )
        completed.update(selected)
        cohort_index += 1

    _validate_schedule_coverage(result, nodes, units_by_node)
    return tuple(result)


def _validate_inputs(
    dag: ContractionDAG, plan: UpmemPlan
) -> tuple[dict[str, GraphNode], dict[str, tuple[UpmemWorkUnit, ...]]]:
    if not isinstance(dag, ContractionDAG):
        raise TypeError("schedule_dag_waves requires a ContractionDAG")
    if not isinstance(plan, UpmemPlan):
        raise TypeError("schedule_dag_waves requires an UpmemPlan")
    if any(not isinstance(node, (ContractNode, ReduceNode)) for node in dag.nodes):
        raise TypeError("ContractionDAG nodes must be ContractNode or ReduceNode records")

    validate_contraction_dag(dag)
    expected_plan_id = contraction_dag_hash(dag)
    if plan.logical_plan_id != expected_plan_id:
        raise ValueError(
            "UPMEM plan logical_plan_id does not identify the supplied ContractionDAG"
        )

    topology = plan.topology
    if topology.dpu_count < 1 or topology.dpu_count > MAX_DPUS:
        raise ValueError(
            f"DAG wave scheduling supports [1, {MAX_DPUS}] DPUs per rank"
        )
    if topology.tasklets_per_dpu < 1 or topology.tasklets_per_dpu > MAX_TASKLETS:
        raise ValueError(
            f"DAG wave scheduling supports [1, {MAX_TASKLETS}] tasklets per DPU"
        )
    if topology.rank_count != 1:
        raise ValueError("DAG wave scheduling currently requires exactly one rank")
    if plan.kernel_policy != _SUPPORTED_KERNEL_POLICY:
        raise ValueError(
            f"unsupported kernel policy for DAG wave scheduling: {plan.kernel_policy!r}"
        )

    nodes = {node.node_id: node for node in dag.nodes}
    seen_stage_ids: set[str] = set()
    seen_plan_node_ids: set[str] = set()
    seen_tile_ids: set[str] = set()
    units_by_node: dict[str, list[UpmemWorkUnit]] = {
        node_id: [] for node_id in nodes
    }
    stage_kind_by_node: dict[str, str] = {}

    for stage in plan.stages:
        if stage.stage_id in seen_stage_ids:
            raise ValueError(f"UPMEM plan contains duplicate stage ID {stage.stage_id!r}")
        seen_stage_ids.add(stage.stage_id)

        if stage.kind not in {"contract_batch", "host_reduce"}:
            raise ValueError(f"unsupported UPMEM stage kind {stage.kind!r}")
        for node_id in stage.node_ids:
            if node_id in seen_plan_node_ids:
                raise ValueError(f"UPMEM plan contains duplicate node ID {node_id!r}")
            if node_id not in nodes:
                raise ValueError(f"UPMEM plan references unknown DAG node {node_id!r}")
            seen_plan_node_ids.add(node_id)
            stage_kind_by_node[node_id] = stage.kind
            expected_type = (
                ContractNode if stage.kind == "contract_batch" else ReduceNode
            )
            if not isinstance(nodes[node_id], expected_type):
                raise ValueError(
                    f"UPMEM stage kind {stage.kind!r} does not match DAG node {node_id!r}"
                )

        if stage.kind == "host_reduce":
            if stage.work_units:
                raise ValueError("host_reduce stages cannot contain work units")
            continue

        for unit in stage.work_units:
            if unit.node_id not in nodes:
                raise ValueError(
                    f"UPMEM work unit references unknown DAG node {unit.node_id!r}"
                )
            if not isinstance(nodes[unit.node_id], ContractNode):
                raise ValueError(
                    f"UPMEM work unit references non-contract node {unit.node_id!r}"
                )
            if unit.node_id not in stage.node_ids:
                raise ValueError(
                    f"UPMEM work unit {unit.stable_tile_id!r} is outside its stage"
                )
            if unit.stable_tile_id in seen_tile_ids:
                raise ValueError(
                    f"UPMEM plan contains duplicate stable tile ID {unit.stable_tile_id!r}"
                )
            seen_tile_ids.add(unit.stable_tile_id)
            if unit.logical_rank != 0:
                raise ValueError("DAG wave scheduling requires all work on rank zero")
            if not 0 <= unit.logical_dpu < topology.dpu_count:
                raise ValueError(
                    f"UPMEM work unit {unit.stable_tile_id!r} has an invalid logical DPU"
                )
            units_by_node[unit.node_id].append(unit)

    if seen_plan_node_ids != set(nodes):
        missing = sorted(set(nodes) - seen_plan_node_ids)
        extra = sorted(seen_plan_node_ids - set(nodes))
        raise ValueError(
            f"UPMEM plan node coverage differs from DAG; missing={missing}, extra={extra}"
        )

    for node_id, node in nodes.items():
        stage_kind = stage_kind_by_node[node_id]
        if isinstance(node, ContractNode) and (
            stage_kind != "contract_batch" or not units_by_node[node_id]
        ):
            raise ValueError(
                f"contract node {node_id!r} must have a nonempty contract work-unit set"
            )
        if isinstance(node, ReduceNode) and stage_kind != "host_reduce":
            raise ValueError(f"reduce node {node_id!r} must have a host_reduce stage")

    return nodes, {
        node_id: tuple(units) for node_id, units in units_by_node.items()
    }


def _dependencies_complete(node: GraphNode, completed: set[str]) -> bool:
    return all(dependency in completed for dependency in node.dependencies)


def _remaining_critical_path(
    nodes: dict[str, GraphNode],
    units_by_node: dict[str, tuple[UpmemWorkUnit, ...]],
) -> dict[str, int]:
    dependents: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    remaining_dependencies = {
        node_id: len(node.dependencies) for node_id, node in nodes.items()
    }
    for node in nodes.values():
        for dependency in node.dependencies:
            dependents[dependency].append(node.node_id)

    ready = sorted(
        node_id
        for node_id, dependency_count in remaining_dependencies.items()
        if dependency_count == 0
    )
    topological_order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        topological_order.append(node_id)
        for dependent in sorted(dependents[node_id]):
            remaining_dependencies[dependent] -= 1
            if remaining_dependencies[dependent] == 0:
                ready.append(dependent)
        ready.sort()
    if len(topological_order) != len(nodes):
        raise ValueError("DAG dependencies contain a cycle")

    critical_path: dict[str, int] = {}
    for node_id in reversed(topological_order):
        own_work = _node_real_mac_count(units_by_node[node_id])
        downstream = max(
            (critical_path[dependent] for dependent in dependents[node_id]),
            default=0,
        )
        critical_path[node_id] = own_work + downstream
    return critical_path


def _node_real_mac_count(units: tuple[UpmemWorkUnit, ...]) -> int:
    return _REAL_PRODUCT_COUNT * sum(
        unit.estimated_arithmetic_work for unit in units
    )


def _allocate_dpus(
    selected: tuple[str, ...],
    units_by_node: dict[str, tuple[UpmemWorkUnit, ...]],
    dpu_count: int,
) -> dict[str, int]:
    allocations = {node_id: 1 for node_id in selected}
    node_work = {
        node_id: _node_real_mac_count(units_by_node[node_id])
        for node_id in selected
    }
    useful_group_caps = {
        node_id: max(
            len(wave_units)
            for _, wave_units in _original_wave_groups(units_by_node[node_id])
        )
        for node_id in selected
    }
    for _ in range(dpu_count - len(selected)):
        candidates = tuple(
            node_id
            for node_id in selected
            if allocations[node_id] < useful_group_caps[node_id]
        )
        if not candidates:
            break
        winner = min(
            candidates,
            key=lambda node_id: (
                -Fraction(node_work[node_id], allocations[node_id]),
                node_id,
            ),
        )
        allocations[winner] += 1
    return allocations


def _contract_cohort(
    cohort_index: int,
    selected: tuple[str, ...],
    units_by_node: dict[str, tuple[UpmemWorkUnit, ...]],
    allocations: dict[str, int],
    dpu_count: int,
) -> UpmemStage:
    groups: dict[str, tuple[int, ...]] = {}
    next_dpu = 0
    for node_id in selected:
        group_size = allocations[node_id]
        groups[node_id] = tuple(range(next_dpu, next_dpu + group_size))
        next_dpu += group_size

    scheduled_units: list[UpmemWorkUnit] = []
    for node_id in selected:
        group = groups[node_id]
        preserve_original_dpu = len(selected) == 1 and len(group) == dpu_count
        scheduled_units.extend(
            _schedule_node_units(
                units_by_node[node_id],
                group,
                preserve_original_dpu=preserve_original_dpu,
            )
        )

    return UpmemStage(
        stage_id=f"dag_cohort:{cohort_index}:contract_batch",
        kind="contract_batch",
        node_ids=selected,
        work_units=tuple(scheduled_units),
    )


def _original_wave_groups(
    units: tuple[UpmemWorkUnit, ...],
) -> tuple[tuple[int, tuple[UpmemWorkUnit, ...]], ...]:
    by_wave: dict[int, list[UpmemWorkUnit]] = {}
    for unit in units:
        by_wave.setdefault(unit.wave, []).append(unit)
    return tuple(
        (wave, tuple(by_wave[wave])) for wave in sorted(by_wave)
    )


def _schedule_node_units(
    units: tuple[UpmemWorkUnit, ...],
    group: tuple[int, ...],
    *,
    preserve_original_dpu: bool,
) -> tuple[UpmemWorkUnit, ...]:
    """Retile original waves without joining adjacent K-chunk boundaries."""

    transformed_by_tile: dict[str, UpmemWorkUnit] = {}
    next_wave = 0
    for _, original_wave_units in _original_wave_groups(units):
        for start in range(0, len(original_wave_units), len(group)):
            chunk = original_wave_units[start : start + len(group)]
            for local_offset, unit in enumerate(chunk):
                logical_dpu = (
                    unit.logical_dpu
                    if preserve_original_dpu
                    else group[local_offset]
                )
                transformed_by_tile[unit.stable_tile_id] = replace(
                    unit,
                    wave=next_wave,
                    logical_rank=0,
                    logical_dpu=logical_dpu,
                )
            next_wave += 1

    return tuple(transformed_by_tile[unit.stable_tile_id] for unit in units)


def _validate_schedule_coverage(
    stages: list[UpmemStage],
    nodes: dict[str, GraphNode],
    units_by_node: dict[str, tuple[UpmemWorkUnit, ...]],
) -> None:
    scheduled_nodes = tuple(
        node_id for stage in stages for node_id in stage.node_ids
    )
    if len(scheduled_nodes) != len(set(scheduled_nodes)) or set(scheduled_nodes) != set(
        nodes
    ):
        raise ValueError("scheduled node coverage differs from the DAG")

    expected_tile_ids = {
        unit.stable_tile_id
        for units in units_by_node.values()
        for unit in units
    }
    scheduled_units = tuple(
        unit for stage in stages for unit in stage.work_units
    )
    scheduled_tile_ids = tuple(unit.stable_tile_id for unit in scheduled_units)
    if len(scheduled_tile_ids) != len(set(scheduled_tile_ids)) or set(
        scheduled_tile_ids
    ) != expected_tile_ids:
        raise ValueError("scheduled work-unit coverage differs from the plan")

    for node_id, original_units in units_by_node.items():
        original_ids = tuple(unit.stable_tile_id for unit in original_units)
        scheduled_ids = tuple(
            unit.stable_tile_id
            for unit in scheduled_units
            if unit.node_id == node_id
        )
        if scheduled_ids != original_ids:
            raise ValueError(
                f"scheduled work-unit order changed for DAG node {node_id!r}"
            )


__all__ = ["schedule_dag_waves"]
