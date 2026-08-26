"""UPMEM-specific lowering for the active tensor-network execution slice."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from quantum_bench.lowering import contraction_dag_hash, validate_contraction_dag
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode
from quantum_bench.numerics import INT8_QUANTIZED_MAX_ABS, NumericPolicy
from quantum_bench.upmem.protocol import (
    INT32_MAX,
    INT8_MAX_PRODUCT,
    MAX_CONTRACTED,
)
from quantum_bench.results import UnsupportedExecution
from quantum_bench.upmem.tiling import (
    M5Tile,
    M5TileLimits,
    TileLoweringError,
    canonical_label_geometry,
    order_tile_waves,
    plan_tile_shapes,
    tile_limits_for_numeric_mode,
)


# The v4 ABI encodes B/M/N/K and aggregate output dimensions as uint64_t.
# ``canonical_batch_count`` is additionally limited by the Python v4 request
# builder before it serializes the native header.
_V4_UINT64_MAX = (1 << 64) - 1
_V4_MAX_BATCH_COUNT = (1 << 32) - 1
PLAN_SCHEMA_VERSION = 1
_INT8_MAX = INT8_QUANTIZED_MAX_ABS
_INT8_PRODUCT = INT8_MAX_PRODUCT
_INT64_MAX = (1 << 63) - 1
_FINAL_NUMERIC_POLICIES = {
    "split_complex_float32_v1",
    "split_complex_int8_shared_scale_v1",
}


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")


def _require_int(name: str, value: object, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemTopology:
    dpu_count: int
    tasklets_per_dpu: int
    rank_count: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("dpu_count", self.dpu_count),
            ("tasklets_per_dpu", self.tasklets_per_dpu),
            ("rank_count", self.rank_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemWorkUnit:
    node_id: str
    stable_tile_id: str
    wave: int
    logical_rank: int
    logical_dpu: int
    batch_start: int
    batch_size: int
    m_start: int
    m_size: int
    n_start: int
    n_size: int
    k_start: int
    k_size: int
    estimated_input_bytes: int
    estimated_output_bytes: int
    aligned_mram_bytes: int
    estimated_arithmetic_work: int

    def __post_init__(self) -> None:
        _require_nonempty_string("node_id", self.node_id)
        _require_nonempty_string("stable_tile_id", self.stable_tile_id)
        for name in (
            "wave",
            "logical_rank",
            "logical_dpu",
            "batch_start",
            "m_start",
            "n_start",
            "k_start",
            "estimated_input_bytes",
            "estimated_output_bytes",
            "aligned_mram_bytes",
        ):
            _require_int(name, getattr(self, name))
        for name in (
            "batch_size",
            "m_size",
            "n_size",
            "k_size",
            "estimated_arithmetic_work",
        ):
            _require_int(name, getattr(self, name), positive=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemStage:
    stage_id: str
    kind: Literal["contract_batch", "host_reduce"]
    node_ids: tuple[str, ...]
    work_units: tuple[UpmemWorkUnit, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string("stage_id", self.stage_id)
        if self.kind not in {"contract_batch", "host_reduce"}:
            raise ValueError(f"unsupported stage kind: {self.kind!r}")
        _require_tuple("node_ids", self.node_ids)
        _require_tuple("work_units", self.work_units)
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("stage node_ids must be unique and nonempty")
        for node_id in self.node_ids:
            _require_nonempty_string("stage node_id", node_id)
        for unit in self.work_units:
            if not isinstance(unit, UpmemWorkUnit):
                raise TypeError("stage work_units must contain UpmemWorkUnit records")
            if unit.node_id not in self.node_ids:
                raise ValueError("work unit node_id is not declared by its stage")
        if self.kind == "host_reduce":
            if len(self.node_ids) != 1 or self.work_units:
                raise ValueError("host_reduce stages have one node and no work units")
        elif not self.work_units:
            raise ValueError("contract_batch stages require work units")


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemPlan:
    logical_plan_id: str
    numeric_policy: NumericPolicy
    topology: UpmemTopology
    stages: tuple[UpmemStage, ...]
    intermediate_policy: Literal["host_roundtrip_v1"] = "host_roundtrip_v1"
    kernel_policy: str = "dpu_real_tile_v4_wram_panel_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.logical_plan_id, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.logical_plan_id
        ):
            raise ValueError("logical_plan_id must be a lowercase SHA-256 hex digest")
        if self.numeric_policy not in _FINAL_NUMERIC_POLICIES:
            raise ValueError(
                f"unsupported final numeric policy: {self.numeric_policy!r}"
            )
        if not isinstance(self.topology, UpmemTopology):
            raise TypeError("topology must be the final UpmemTopology record")
        _require_tuple("stages", self.stages)
        for stage in self.stages:
            if not isinstance(stage, UpmemStage):
                raise TypeError("stages must contain UpmemStage records")
        stage_ids = tuple(stage.stage_id for stage in self.stages)
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("plan stage IDs must be unique")
        node_ids = tuple(node_id for stage in self.stages for node_id in stage.node_ids)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("plan node IDs must be unique")
        tile_ids = tuple(
            unit.stable_tile_id for stage in self.stages for unit in stage.work_units
        )
        if len(set(tile_ids)) != len(tile_ids):
            raise ValueError("plan stable tile IDs must be globally unique")
        if self.intermediate_policy != "host_roundtrip_v1":
            raise ValueError("unsupported intermediate policy")
        _require_nonempty_string("kernel_policy", self.kernel_policy)


@dataclass(frozen=True, slots=True, kw_only=True)
class UpmemResources:
    session_root: str
    host_binary: str
    dpu_binary: str
    initialization_binary: str
    rank_paths: tuple[str, ...] = ()
    session_opener: Callable[..., object] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "session_root",
            "host_binary",
            "dpu_binary",
            "initialization_binary",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.rank_paths, tuple):
            raise TypeError("rank_paths must be a tuple")
        if any(not isinstance(path, str) or not path for path in self.rank_paths):
            raise ValueError("rank_paths must contain nonempty strings")
        if self.session_opener is not None and not callable(self.session_opener):
            raise TypeError("session_opener must be callable or None")


class _UnsupportedUpmemNode(Exception):
    def __init__(self, *, capability: str, reason: str) -> None:
        super().__init__(reason)
        self.capability = capability
        self.reason = reason


def plan_upmem(
    dag: ContractionDAG,
    *,
    numeric_policy: NumericPolicy,
    topology: UpmemTopology,
) -> UpmemPlan:
    """Create the pure T4C physical plan for an already lowered DAG."""

    validate_contraction_dag(dag)
    return _build_upmem_plan(dag, numeric_policy=numeric_policy, topology=topology)


def validate_upmem_plan(dag: ContractionDAG, plan: UpmemPlan) -> None:
    """Require that ``plan`` is the deterministic pure mapping for ``dag``."""

    if not isinstance(plan, UpmemPlan):
        raise TypeError("validate_upmem_plan requires the final UpmemPlan record")
    validate_contraction_dag(dag)
    expected = _build_upmem_plan(
        dag,
        numeric_policy=plan.numeric_policy,
        topology=plan.topology,
    )
    if plan != expected:
        raise ValueError("UPMEM physical plan differs from pure recomputation")


def physical_plan_id(plan: UpmemPlan) -> str:
    """Hash only the ordered, executable-independent physical plan fields."""

    if not isinstance(plan, UpmemPlan):
        raise TypeError("physical_plan_id requires the final UpmemPlan record")
    payload = {
        "PLAN_SCHEMA_VERSION": PLAN_SCHEMA_VERSION,
        "logical_plan_id": plan.logical_plan_id,
        "numeric_policy": plan.numeric_policy,
        "topology": {
            "dpu_count": plan.topology.dpu_count,
            "tasklets_per_dpu": plan.topology.tasklets_per_dpu,
            "rank_count": plan.topology.rank_count,
        },
        "stages": [
            {
                "stage_id": stage.stage_id,
                "kind": stage.kind,
                "node_ids": list(stage.node_ids),
                "work_units": [_work_unit_payload(unit) for unit in stage.work_units],
            }
            for stage in plan.stages
        ],
        "intermediate_policy": plan.intermediate_policy,
        "kernel_policy": plan.kernel_policy,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def collection_resource_admission(
    plan: UpmemPlan,
) -> dict[str, int | float | str | bool | None]:
    """Derive deterministic tasklet/DPU scaling facts from a physical plan.

    Correctness permits idle tasklets for small setup work. Scaling admission is
    instead tied to the one arithmetic-dominant wave that will represent the
    selected workload, while utilization accounts for every contract wave.
    """

    if not isinstance(plan, UpmemPlan):
        raise TypeError("collection_resource_admission requires the final UpmemPlan")
    topology = plan.topology
    waves: list[tuple[str, int, tuple[UpmemWorkUnit, ...], int, int]] = []
    for stage in plan.stages:
        if stage.kind != "contract_batch":
            continue
        for wave in sorted({unit.wave for unit in stage.work_units}):
            units = tuple(unit for unit in stage.work_units if unit.wave == wave)
            arithmetic_work = sum(unit.estimated_arithmetic_work for unit in units)
            useful_slots = len(
                {
                    (unit.logical_rank, unit.logical_dpu)
                    for unit in units
                    if unit.estimated_arithmetic_work > 0
                }
            )
            waves.append((stage.stage_id, wave, units, arithmetic_work, useful_slots))

    total_work = sum(arithmetic_work for _, _, _, arithmetic_work, _ in waves)
    dominant: tuple[str, int, tuple[UpmemWorkUnit, ...], int, int] | None = (
        min(waves, key=lambda item: (-item[3], item[0], item[1]), default=None)
    )
    if dominant is None:
        dominant_stage_id: str | None = None
        dominant_wave: int | None = None
        dominant_units: tuple[UpmemWorkUnit, ...] = ()
        dominant_work = dominant_slots = 0
    else:
        dominant_stage_id, dominant_wave, dominant_units, dominant_work, dominant_slots = dominant

    tasklet_rows_ok = bool(dominant_units) and all(
        unit.m_size >= topology.tasklets_per_dpu for unit in dominant_units
    )
    fully_populated_wave_count = sum(
        useful_slots == topology.dpu_count
        for _, _, _, _, useful_slots in waves
    )
    weighted_dpu_utilization = (
        sum(
            arithmetic_work * useful_slots / topology.dpu_count
            for _, _, _, arithmetic_work, useful_slots in waves
        )
        / total_work
        if total_work
        else 0.0
    )
    weighted_tasklet_utilization = (
        sum(
            unit.estimated_arithmetic_work
            * min(unit.m_size, topology.tasklets_per_dpu)
            / topology.tasklets_per_dpu
            for _, _, units, _, _ in waves
            for unit in units
        )
        / total_work
        if total_work
        else 0.0
    )
    dpu_wave_ok = topology.dpu_count == 1 or dominant_slots == topology.dpu_count
    return {
        "tasklet_row_sufficiency_passed": tasklet_rows_ok,
        "dominant_work_stage_id": dominant_stage_id,
        "dominant_work_wave": dominant_wave,
        "dominant_work_wave_arithmetic_work": dominant_work,
        "dominant_work_wave_populated_dpu_slots": dominant_slots,
        "dominant_work_wave_tasklet_row_sufficiency_passed": tasklet_rows_ok,
        "dominant_work_wave_allocated_dpu_slots": topology.dpu_count,
        "dominant_work_wave_utilization": dominant_slots / topology.dpu_count,
        "arithmetic_weighted_dpu_slot_utilization": weighted_dpu_utilization,
        "arithmetic_weighted_tasklet_utilization": weighted_tasklet_utilization,
        "total_wave_count": len(waves),
        "dominant_wave_dpu_slots": dominant_slots,
        "dominant_wave_useful_slots": dominant_slots,
        "dominant_wave_allocated_slots": topology.dpu_count,
        "dominant_wave_utilization": dominant_slots / topology.dpu_count,
        "fully_populated_wave_count": fully_populated_wave_count,
        "collection_resource_admission_passed": tasklet_rows_ok and dpu_wave_ok,
    }


def _build_upmem_plan(
    dag: ContractionDAG,
    *,
    numeric_policy: NumericPolicy,
    topology: UpmemTopology,
) -> UpmemPlan:
    if not any(isinstance(node, ContractNode) for node in dag.nodes):
        raise UnsupportedExecution(
            "mapping",
            "UPMEM physical mapping requires at least one ContractNode with kernel work",
            "upmem_no_contract_work",
        )
    _validate_final_topology(topology)
    tile_numeric_mode = _tile_numeric_mode(numeric_policy)
    limits = tile_limits_for_numeric_mode(tile_numeric_mode)
    order = _topological_order(dag)
    nodes = {node.node_id: node for node in dag.nodes}
    slice_groups = _slice_batch_groups(
        dag,
        numeric_policy=numeric_policy,
        topology=topology,
        limits=limits,
    )
    grouped_nodes = {
        node_id for node_ids in slice_groups.values() for node_id in node_ids
    }
    stages: list[UpmemStage] = []
    for node_id in order:
        if node_id in grouped_nodes:
            continue
        node = nodes[node_id]
        if isinstance(node, ReduceNode):
            group = slice_groups.get(node.node_id)
            if group is not None:
                grouped_units: list[UpmemWorkUnit] = []
                for branch_id in group:
                    grouped_units.extend(
                        _map_contract_node(
                            nodes[branch_id],
                            numeric_policy=numeric_policy,
                            topology=topology,
                            limits=limits,
                        )
                    )
                stages.append(
                    UpmemStage(
                        stage_id=f"contract_batch:{node.node_id}",
                        kind="contract_batch",
                        node_ids=group,
                        work_units=tuple(
                            sorted(grouped_units, key=_work_unit_sort_key)
                        ),
                    )
                )
            stages.append(
                UpmemStage(
                    stage_id=f"host_reduce:{node.node_id}",
                    kind="host_reduce",
                    node_ids=(node.node_id,),
                    work_units=(),
                )
            )
            continue
        stages.append(
            UpmemStage(
                stage_id=f"contract_batch:{node.node_id}",
                kind="contract_batch",
                node_ids=(node.node_id,),
                work_units=_map_contract_node(
                    node,
                    numeric_policy=numeric_policy,
                    topology=topology,
                    limits=limits,
                ),
            )
        )
    plan = UpmemPlan(
        logical_plan_id=contraction_dag_hash(dag),
        numeric_policy=numeric_policy,
        topology=topology,
        stages=tuple(stages),
    )
    _validate_final_stage_dependencies(dag, plan)
    return plan


def _map_contract_node(
    node: ContractNode,
    *,
    numeric_policy: NumericPolicy,
    topology: UpmemTopology,
    limits: M5TileLimits,
) -> tuple[UpmemWorkUnit, ...]:
    try:
        batch, m, n, contracted_size = _validate_v4_node_geometry(node)
        if contracted_size > MAX_CONTRACTED:
            raise _UnsupportedUpmemNode(
                capability="upmem_max_contracted_elements",
                reason=(
                    f"UPMEM node {node.node_id} contracted K {contracted_size} "
                    f"exceeds v4 limit {MAX_CONTRACTED}"
                ),
            )
        tiles = plan_tile_shapes(batch, m, contracted_size, n, limits=limits)
        if numeric_policy == "split_complex_int8_shared_scale_v1":
            _validate_final_int8_bounds(node.node_id, contracted_size, tiles)
        waves = order_tile_waves(tiles, topology.dpu_count)
        return _final_work_units(
            node.node_id,
            waves,
            topology,
        )
    except _UnsupportedUpmemNode as exc:
        raise UnsupportedExecution("mapping", exc.reason, exc.capability) from exc
    except TileLoweringError as exc:
        raise UnsupportedExecution(
            "mapping",
            f"UPMEM node {node.node_id} is not representable: {exc}",
            "upmem_v4_geometry",
        ) from exc


def _slice_batch_groups(
    dag: ContractionDAG,
    *,
    numeric_policy: NumericPolicy,
    topology: UpmemTopology,
    limits: M5TileLimits,
) -> dict[str, tuple[str, ...]]:
    """Return deterministic, exclusively consumed logical slice groups."""

    nodes = {node.node_id: node for node in dag.nodes}
    groups: dict[str, tuple[str, ...]] = {}
    for parent in dag.nodes:
        if not isinstance(parent, ReduceNode) or not parent.reduced_labels:
            continue
        branches = tuple(sorted(parent.dependencies))
        if not branches or not all(
            isinstance(nodes[branch_id], ContractNode) for branch_id in branches
        ):
            continue
        branch_nodes = tuple(nodes[branch_id] for branch_id in branches)
        if not any(
            node.left.slice_spec or node.right.slice_spec for node in branch_nodes
        ):
            continue
        descriptors = {tensor.id: tensor for tensor in dag.tensors}
        descriptors.update({node.output.id: node.output for node in dag.nodes})
        if not all(
            _is_logical_slice_branch(node, parent.reduced_labels, descriptors)
            for node in branch_nodes
        ):
            _raise_slice_batch_incompatibility(
                parent,
                "direct branches do not all represent the requested logical slices",
            )
        for branch in branch_nodes:
            consumers = {
                consumer.node_id
                for consumer in dag.nodes
                if any(
                    view.tensor_id == branch.output.id
                    for view in (
                        (consumer.left, consumer.right)
                        if isinstance(consumer, ContractNode)
                        else consumer.inputs
                    )
                )
            }
            if dag.output.tensor_id == branch.output.id:
                consumers.add("<dag-output>")
            if consumers != {parent.node_id}:
                _raise_slice_batch_incompatibility(
                    parent,
                    f"branch {branch.node_id!r} has unrelated consumers",
                )

        keys = tuple(
            _slice_compatibility_key(
                parent,
                branch,
                numeric_policy=numeric_policy,
                topology=topology,
                limits=limits,
            )
            for branch in branch_nodes
        )
        if any(key != keys[0] for key in keys[1:]):
            _raise_slice_batch_incompatibility(
                parent,
                "logical slice branches have incompatible execution signatures",
            )
        groups[parent.node_id] = branches
    return groups


def _is_logical_slice_branch(
    node: ContractNode,
    reduced_labels: tuple[int, ...],
    descriptors: dict[str, object],
) -> bool:
    fixed_labels: set[int] = set()
    for view in (node.left, node.right):
        labels = getattr(descriptors[view.tensor_id], "labels")
        fixed_labels.update(
            labels[axis] for axis, _ in view.slice_spec if 0 <= axis < len(labels)
        )
    return fixed_labels == set(reduced_labels)


def _slice_compatibility_key(
    parent: ReduceNode,
    node: ContractNode,
    *,
    numeric_policy: NumericPolicy,
    topology: UpmemTopology,
    limits: M5TileLimits,
) -> tuple[object, ...]:
    try:
        geometry = _canonical_dimensions(node)
    except (TileLoweringError, OverflowError) as exc:
        _raise_slice_batch_incompatibility(parent, str(exc))
    limit_fields = tuple(
        getattr(limits, field)
        for field in (
            "numeric_mode",
            "max_elements",
            "max_packed_k",
            "max_mram_bytes",
            "alignment_bytes",
            "max_tile_dim",
        )
    )
    return (
        parent.node_id,
        (
            node.left.labels,
            node.left.shape,
            node.right.labels,
            node.right.shape,
            node.contracted_labels,
            node.output_labels,
        ),
        geometry,
        numeric_policy,
        limit_fields,
        (
            topology.dpu_count,
            topology.tasklets_per_dpu,
            topology.rank_count,
        ),
        (
            node.output.dtype,
            node.output.labels,
            node.output.shape,
            node.output.structure,
        ),
    )


def _raise_slice_batch_incompatibility(parent: ReduceNode, reason: str) -> None:
    raise UnsupportedExecution(
        "mapping",
        f"UPMEM logical slice group {parent.node_id!r} is incompatible: {reason}",
        "upmem_slice_batch_compatibility",
    )


def _validate_final_topology(topology: UpmemTopology) -> None:
    if not isinstance(topology, UpmemTopology):
        raise TypeError("plan_upmem requires the final UpmemTopology record")
    if topology.dpu_count < 1 or topology.rank_count < 1:
        raise UnsupportedExecution(
            "mapping", "UPMEM topology counts must be positive", "upmem_topology"
        )
    if topology.dpu_count % topology.rank_count:
        raise UnsupportedExecution(
            "mapping",
            "UPMEM dpu_count must be divisible by rank_count",
            "upmem_topology_divisibility",
        )
    if topology.dpu_count // topology.rank_count > 64:
        raise UnsupportedExecution(
            "mapping",
            "UPMEM supports at most 64 DPUs per rank",
            "upmem_dpus_per_rank",
        )
    if not 1 <= topology.tasklets_per_dpu <= 24:
        raise UnsupportedExecution(
            "mapping",
            "UPMEM tasklets_per_dpu must be in [1, 24]",
            "upmem_tasklets_per_dpu",
        )


def _tile_numeric_mode(numeric_policy: NumericPolicy) -> str:
    if numeric_policy == "split_complex_float32_v1":
        return "float32"
    if numeric_policy == "split_complex_int8_shared_scale_v1":
        return "host_packed_int8"
    raise UnsupportedExecution(
        "mapping",
        f"unsupported UPMEM numeric policy: {numeric_policy!r}",
        "upmem_numeric_policy",
    )


def _validate_final_int8_bounds(
    node_id: str, contracted_size: int, tiles: tuple[M5Tile, ...]
) -> None:
    if 2 * contracted_size * _INT8_PRODUCT > _INT64_MAX:
        raise UnsupportedExecution(
            "mapping",
            f"UPMEM node {node_id} exceeds int64 complex accumulation bound",
            "upmem_int8_int64_accumulation_bound",
        )
    for tile in tiles:
        if tile.k_size * _INT8_PRODUCT > INT32_MAX:
            raise UnsupportedExecution(
                "mapping",
                f"UPMEM tile {tile.id} exceeds int32 int8 accumulation bound",
                "upmem_int8_int32_accumulation_bound",
            )


def _final_work_units(
    node_id: str,
    waves: tuple[tuple[M5Tile, ...], ...],
    topology: UpmemTopology,
) -> tuple[UpmemWorkUnit, ...]:
    dpus_per_rank = topology.dpu_count // topology.rank_count
    units: list[UpmemWorkUnit] = []
    for wave_index, wave in enumerate(waves):
        for global_slot, tile in enumerate(wave):
            units.append(
                UpmemWorkUnit(
                    node_id=node_id,
                    stable_tile_id=f"{node_id}:{tile.id}",
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
    return tuple(sorted(units, key=_work_unit_sort_key))


def _work_unit_sort_key(unit: UpmemWorkUnit) -> tuple[object, ...]:
    return (
        unit.logical_rank,
        unit.logical_dpu,
        unit.wave,
        unit.batch_start,
        unit.m_start,
        unit.n_start,
        unit.k_start,
        unit.node_id,
        unit.stable_tile_id,
    )


def _work_unit_payload(unit: UpmemWorkUnit) -> dict[str, object]:
    return {
        "node_id": unit.node_id,
        "stable_tile_id": unit.stable_tile_id,
        "wave": unit.wave,
        "logical_rank": unit.logical_rank,
        "logical_dpu": unit.logical_dpu,
        "batch_start": unit.batch_start,
        "batch_size": unit.batch_size,
        "m_start": unit.m_start,
        "m_size": unit.m_size,
        "n_start": unit.n_start,
        "n_size": unit.n_size,
        "k_start": unit.k_start,
        "k_size": unit.k_size,
        "estimated_input_bytes": unit.estimated_input_bytes,
        "estimated_output_bytes": unit.estimated_output_bytes,
        "aligned_mram_bytes": unit.aligned_mram_bytes,
        "estimated_arithmetic_work": unit.estimated_arithmetic_work,
    }


def _validate_final_stage_dependencies(dag: ContractionDAG, plan: UpmemPlan) -> None:
    stage_index = {
        node_id: index
        for index, stage in enumerate(plan.stages)
        for node_id in stage.node_ids
    }
    for node in dag.nodes:
        current = stage_index[node.node_id]
        for dependency in node.dependencies:
            if dependency not in stage_index or stage_index[dependency] >= current:
                raise ValueError(
                    f"UPMEM stage dependency is not earlier: {node.node_id} <- {dependency}"
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


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "UpmemPlan",
    "UpmemResources",
    "UpmemStage",
    "UpmemTopology",
    "UpmemWorkUnit",
    "collection_resource_admission",
    "physical_plan_id",
    "plan_upmem",
    "validate_upmem_plan",
]
