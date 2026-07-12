from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from quantum_bench.core.records import ContractionTask, JsonDict, TaskGraph
from quantum_bench.targets.upmem.schedule import UPMEM_DENSE_ESTIMATE_KEY
from quantum_bench.targets.upmem.tile_plan import DENSE_INT8_FORMAT, plan_l2_tiled_execution


PIM_FRONTIER_ANALYSIS_SCHEMA_VERSION = "pim_frontier_analysis_v1"

MEMORY_LEVEL_L1_WRAM = "L1_WRAM"
MEMORY_LEVEL_L2_SINGLE_DPU_MRAM = "L2_SINGLE_DPU_MRAM"
MEMORY_LEVEL_L3_MULTI_DPU = "L3_MULTI_DPU"
MEMORY_LEVEL_L4_OUT_OF_SCOPE = "L4_OUT_OF_SCOPE"
MEMORY_LEVEL_NOT_DENSE_GEMM = "NOT_DENSE_GEMM"

DOMINANT_SOURCE_SERIAL = "serial"
DOMINANT_SOURCE_INTER_TASK = "inter_task"
DOMINANT_SOURCE_INTRA_TASK = "intra_task"
DOMINANT_SOURCE_HYBRID = "hybrid"


@dataclass(frozen=True)
class UpmemResourceModel:
    available_dpus: int = 64
    per_dpu_wram_bytes: int = 64 * 1024
    effective_wram_bytes: int = 60 * 1024
    per_dpu_mram_bytes: int = 64 * 1024 * 1024
    max_task_group_dpus: int = 64
    input_element_bytes: int = DENSE_INT8_FORMAT.input_element_bytes
    output_element_bytes: int = DENSE_INT8_FORMAT.output_element_bytes
    accumulator_element_bytes: int = DENSE_INT8_FORMAT.accumulator_element_bytes

    def __post_init__(self) -> None:
        fields = {
            "available_dpus": self.available_dpus,
            "per_dpu_wram_bytes": self.per_dpu_wram_bytes,
            "effective_wram_bytes": self.effective_wram_bytes,
            "per_dpu_mram_bytes": self.per_dpu_mram_bytes,
            "max_task_group_dpus": self.max_task_group_dpus,
            "input_element_bytes": self.input_element_bytes,
            "output_element_bytes": self.output_element_bytes,
            "accumulator_element_bytes": self.accumulator_element_bytes,
        }
        invalid = [name for name, value in fields.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"UPMEM resource model values must be positive: {', '.join(invalid)}")
        if self.effective_wram_bytes > self.per_dpu_wram_bytes:
            raise ValueError("effective_wram_bytes must be <= per_dpu_wram_bytes")
        if self.max_task_group_dpus > self.available_dpus:
            raise ValueError("max_task_group_dpus must be <= available_dpus")

    @property
    def aggregate_mram_bytes(self) -> int:
        return int(self.available_dpus * self.per_dpu_mram_bytes)

    def as_dict(self) -> JsonDict:
        return {
            "available_dpus": self.available_dpus,
            "per_dpu_wram_bytes": self.per_dpu_wram_bytes,
            "effective_wram_bytes": self.effective_wram_bytes,
            "per_dpu_mram_bytes": self.per_dpu_mram_bytes,
            "aggregate_mram_bytes": self.aggregate_mram_bytes,
            "max_task_group_dpus": self.max_task_group_dpus,
            "input_element_bytes": self.input_element_bytes,
            "output_element_bytes": self.output_element_bytes,
            "accumulator_element_bytes": self.accumulator_element_bytes,
            "memory_model_note": (
                "conservative: output storage and accumulator workspace are "
                "counted as separate modeled reservations"
            ),
        }


@dataclass(frozen=True)
class PimTaskMemoryAnalysis:
    task_index: int
    task_id: str
    input_tensor_ids: tuple[str, ...]
    output_tensor_id: str
    dependencies: tuple[str, ...]
    gemm_m: int
    gemm_k: int
    gemm_n: int
    structure: str
    dense_lowerable: bool
    memory_level: str
    memory_reason: str | None
    backend_supported: bool
    current_backend_executable: bool
    current_backend_reason: str | None
    estimate_reject_reason: str | None
    requires_tiling: bool
    requires_host_aggregation: bool
    tiling_implemented: bool
    a_bytes: int
    b_bytes: int
    c_output_bytes: int
    c_accumulator_bytes: int
    full_task_bytes: int
    working_set_bytes: int
    estimated_flops: int
    estimated_bytes: int
    estimated_host_to_dpu_bytes: int
    estimated_dpu_to_host_bytes: int
    estimated_mram_to_wram_bytes: int
    estimated_output_tiles: int
    estimated_k_tiles: int
    estimated_total_tiles: int
    optional_parallel_tile_count: int
    memory_capacity_min_dpus: int
    estimated_dpus_required: int
    estimated_parallel_dpus: int
    blocker_reason: str | None
    frontier_wave_index: int | None = None
    dominant_source: str | None = None

    def with_wave(self, wave_index: int, dominant_source: str) -> "PimTaskMemoryAnalysis":
        return PimTaskMemoryAnalysis(
            **{
                **self.__dict__,
                "frontier_wave_index": wave_index,
                "dominant_source": dominant_source,
            }
        )

    def as_row(self) -> JsonDict:
        return {
            "task_index": self.task_index,
            "task_id": self.task_id,
            "input_tensor_ids": self.input_tensor_ids,
            "output_tensor_id": self.output_tensor_id,
            "dependencies": self.dependencies,
            "gemm_m": self.gemm_m,
            "gemm_k": self.gemm_k,
            "gemm_n": self.gemm_n,
            "structure": self.structure,
            "dense_lowerable": self.dense_lowerable,
            "memory_level": self.memory_level,
            "memory_reason": self.memory_reason,
            "backend_supported": self.backend_supported,
            "current_backend_executable": self.current_backend_executable,
            "current_backend_reason": self.current_backend_reason,
            "estimate_reject_reason": self.estimate_reject_reason,
            "requires_tiling": self.requires_tiling,
            "requires_host_aggregation": self.requires_host_aggregation,
            "tiling_implemented": self.tiling_implemented,
            "a_bytes": self.a_bytes,
            "b_bytes": self.b_bytes,
            "c_output_bytes": self.c_output_bytes,
            "c_accumulator_bytes": self.c_accumulator_bytes,
            "full_task_bytes": self.full_task_bytes,
            "working_set_bytes": self.working_set_bytes,
            "estimated_flops": self.estimated_flops,
            "estimated_bytes": self.estimated_bytes,
            "estimated_host_to_dpu_bytes": self.estimated_host_to_dpu_bytes,
            "estimated_dpu_to_host_bytes": self.estimated_dpu_to_host_bytes,
            "estimated_mram_to_wram_bytes": self.estimated_mram_to_wram_bytes,
            "estimated_output_tiles": self.estimated_output_tiles,
            "estimated_k_tiles": self.estimated_k_tiles,
            "estimated_total_tiles": self.estimated_total_tiles,
            "optional_parallel_tile_count": self.optional_parallel_tile_count,
            "memory_capacity_min_dpus": self.memory_capacity_min_dpus,
            "estimated_dpus_required": self.estimated_dpus_required,
            "estimated_parallel_dpus": self.estimated_parallel_dpus,
            "blocker_reason": self.blocker_reason,
            "frontier_wave_index": self.frontier_wave_index,
            "dominant_source": self.dominant_source,
        }


@dataclass(frozen=True)
class PimFrontierWave:
    wave_index: int
    task_indices: tuple[int, ...]
    task_ids: tuple[str, ...]
    ready_task_count: int
    memory_level_counts: JsonDict
    dominant_source: str
    inter_task_parallelism_potential: bool
    intra_task_parallelism_potential: bool
    hybrid_parallelism_potential: bool
    schedulable_task_count: int
    scheduling_rounds: int
    assigned_dpu_slots: int
    max_group_dpus: int
    estimated_dpu_occupancy: float
    idle_dpu_fraction: float

    def as_row(self) -> JsonDict:
        return {
            "frontier_wave_index": self.wave_index,
            "task_indices": self.task_indices,
            "task_ids": self.task_ids,
            "ready_task_count": self.ready_task_count,
            "memory_level_counts": self.memory_level_counts,
            "dominant_source": self.dominant_source,
            "inter_task_parallelism_potential": self.inter_task_parallelism_potential,
            "intra_task_parallelism_potential": self.intra_task_parallelism_potential,
            "hybrid_parallelism_potential": self.hybrid_parallelism_potential,
            "schedulable_task_count": self.schedulable_task_count,
            "scheduling_rounds": self.scheduling_rounds,
            "assigned_dpu_slots": self.assigned_dpu_slots,
            "max_group_dpus": self.max_group_dpus,
            "estimated_dpu_occupancy": self.estimated_dpu_occupancy,
            "idle_dpu_fraction": self.idle_dpu_fraction,
        }


@dataclass(frozen=True)
class PimFrontierGraphAnalysis:
    resource_model: UpmemResourceModel
    tasks: tuple[PimTaskMemoryAnalysis, ...]
    waves: tuple[PimFrontierWave, ...]
    critical_path_length_tasks: int
    unresolved_dependency_count: int = 0
    unresolved_dependencies: JsonDict = field(default_factory=dict)

    def summary(self) -> JsonDict:
        memory_level_counts = Counter(task.memory_level for task in self.tasks)
        dominant_source_counts = Counter(wave.dominant_source for wave in self.waves)
        frontier_widths = [wave.ready_task_count for wave in self.waves]
        occupancies = [wave.estimated_dpu_occupancy for wave in self.waves]
        max_frontier_width = max(frontier_widths, default=0)
        mean_frontier_width = (sum(frontier_widths) / len(frontier_widths)) if frontier_widths else 0.0
        mean_dpu_occupancy = (sum(occupancies) / len(occupancies)) if occupancies else 0.0
        total_transfer = sum(
            task.estimated_host_to_dpu_bytes + task.estimated_dpu_to_host_bytes
            for task in self.tasks
        )
        max_group = max((wave.max_group_dpus for wave in self.waves), default=0)
        potential_source = (
            "task_graph_serialized_by_planner"
            if max_frontier_width <= 1 and self.tasks
            else "frontier_parallelism_available"
            if self.tasks
            else "no_tasks"
        )
        return {
            "task_count": len(self.tasks),
            "wave_count": len(self.waves),
            "critical_path_length_tasks": self.critical_path_length_tasks,
            "max_frontier_width": max_frontier_width,
            "mean_frontier_width": mean_frontier_width,
            "mean_estimated_dpu_occupancy": mean_dpu_occupancy,
            "memory_level_counts": dict(sorted(memory_level_counts.items())),
            "dominant_source_counts": dict(sorted(dominant_source_counts.items())),
            "potential_parallelism_source": potential_source,
            "total_estimated_host_to_dpu_bytes": sum(task.estimated_host_to_dpu_bytes for task in self.tasks),
            "total_estimated_dpu_to_host_bytes": sum(task.estimated_dpu_to_host_bytes for task in self.tasks),
            "total_estimated_mram_to_wram_bytes": sum(task.estimated_mram_to_wram_bytes for task in self.tasks),
            "total_estimated_transfer_bytes": total_transfer,
            "total_estimated_flops": sum(task.estimated_flops for task in self.tasks),
            "max_full_task_bytes": max((task.full_task_bytes for task in self.tasks), default=0),
            "max_working_set_bytes": max((task.working_set_bytes for task in self.tasks), default=0),
            "max_modeled_dpu_group_size": max_group,
            "l1_task_count": memory_level_counts.get(MEMORY_LEVEL_L1_WRAM, 0),
            "l2_task_count": memory_level_counts.get(MEMORY_LEVEL_L2_SINGLE_DPU_MRAM, 0),
            "l3_task_count": memory_level_counts.get(MEMORY_LEVEL_L3_MULTI_DPU, 0),
            "l4_task_count": memory_level_counts.get(MEMORY_LEVEL_L4_OUT_OF_SCOPE, 0),
            "unclassified_dense_task_count": memory_level_counts.get(MEMORY_LEVEL_NOT_DENSE_GEMM, 0),
            "unresolved_dependency_count": self.unresolved_dependency_count,
            "unresolved_dependencies": self.unresolved_dependencies,
        }


def analyze_task(task: ContractionTask, task_index: int, resource_model: UpmemResourceModel | None = None) -> PimTaskMemoryAnalysis:
    model = resource_model or UpmemResourceModel()
    estimate = _estimate(task)
    tile_plan = _tile_plan(estimate)
    dense_lowerable, memory_reason = _dense_lowerable(task)
    a_bytes = b_bytes = c_output_bytes = c_accumulator_bytes = full_task_bytes = 0
    memory_level = MEMORY_LEVEL_NOT_DENSE_GEMM
    memory_capacity_min_dpus = 0
    estimated_parallel_dpus = 0
    blocker_reason = memory_reason

    if dense_lowerable:
        a_bytes = task.gemm_m * task.gemm_k * model.input_element_bytes
        b_bytes = task.gemm_k * task.gemm_n * model.input_element_bytes
        c_output_bytes = task.gemm_m * task.gemm_n * model.output_element_bytes
        c_accumulator_bytes = task.gemm_m * task.gemm_n * model.accumulator_element_bytes
        full_task_bytes = a_bytes + b_bytes + c_output_bytes + c_accumulator_bytes
        memory_level, memory_reason = _memory_level(full_task_bytes, model)
        memory_capacity_min_dpus = (
            1
            if memory_level in {MEMORY_LEVEL_L1_WRAM, MEMORY_LEVEL_L2_SINGLE_DPU_MRAM}
            else math.ceil(full_task_bytes / model.per_dpu_mram_bytes)
            if full_task_bytes
            else 0
        )
        if memory_level == MEMORY_LEVEL_L4_OUT_OF_SCOPE:
            blocker_reason = "aggregate_mram_capacity_exceeded"
        else:
            blocker_reason = None
        estimated_parallel_dpus = _parallel_dpus(memory_level, memory_capacity_min_dpus, model)

    backend_supported = bool(estimate.get("supported", dense_lowerable))
    requires_tiling = bool(estimate.get("requires_tiling", False))
    requires_host_aggregation = bool(estimate.get("requires_host_aggregation", tile_plan.get("requires_host_aggregation", False)))
    tiling_implemented = bool(estimate.get("tiling_implemented", False))
    l2_plan = (
        plan_l2_tiled_execution(
            task.gemm_m,
            task.gemm_k,
            task.gemm_n,
            effective_wram_bytes=model.effective_wram_bytes,
            per_dpu_mram_bytes=model.per_dpu_mram_bytes,
        )
        if dense_lowerable and memory_level == MEMORY_LEVEL_L2_SINGLE_DPU_MRAM
        else None
    )
    current_backend_executable = bool(
        backend_supported
        and dense_lowerable
        and (
            (
                memory_level == MEMORY_LEVEL_L1_WRAM
                and not requires_tiling
                and not requires_host_aggregation
            )
            or (
                memory_level == MEMORY_LEVEL_L2_SINGLE_DPU_MRAM
                and l2_plan is not None
                and l2_plan.supported
            )
        )
    )
    current_backend_reason = _current_backend_reason(
        dense_lowerable=dense_lowerable,
        backend_supported=backend_supported,
        requires_tiling=requires_tiling,
        requires_host_aggregation=requires_host_aggregation,
        memory_level=memory_level,
        memory_reason=memory_reason,
        l2_executable=bool(l2_plan.supported) if l2_plan is not None else False,
        l2_reason=l2_plan.reason if l2_plan is not None else None,
    )
    estimated_total_tiles = int(estimate.get("estimated_tile_count", tile_plan.get("total_tile_count", 1 if dense_lowerable else 0)) or 0)
    estimated_output_tiles = int(tile_plan.get("tile_count_m", 1 if dense_lowerable else 0) or 0) * int(tile_plan.get("tile_count_n", 1 if dense_lowerable else 0) or 0)
    estimated_k_tiles = int(tile_plan.get("tile_count_k", 1 if dense_lowerable else 0) or 0)
    working_set_bytes = int(estimate.get("max_working_set_bytes", tile_plan.get("working_set_bytes", full_task_bytes)) or 0)
    host_to_dpu_bytes = int(estimate.get("host_to_dpu_bytes", a_bytes + b_bytes) or 0)
    dpu_to_host_bytes = int(estimate.get("dpu_to_host_bytes", c_output_bytes) or 0)
    mram_to_wram_bytes = int(estimate.get("mram_to_wram_bytes", a_bytes + b_bytes + c_output_bytes) or 0)
    return PimTaskMemoryAnalysis(
        task_index=task_index,
        task_id=task.id,
        input_tensor_ids=tuple(task.input_tensor_ids),
        output_tensor_id=task.output_tensor_id,
        dependencies=tuple(getattr(task, "dependencies", ()) or ()),
        gemm_m=int(task.gemm_m),
        gemm_k=int(task.gemm_k),
        gemm_n=int(task.gemm_n),
        structure=str(task.structure),
        dense_lowerable=dense_lowerable,
        memory_level=memory_level,
        memory_reason=memory_reason,
        backend_supported=backend_supported,
        current_backend_executable=current_backend_executable,
        current_backend_reason=current_backend_reason,
        estimate_reject_reason=estimate.get("reject_reason"),
        requires_tiling=requires_tiling,
        requires_host_aggregation=requires_host_aggregation,
        tiling_implemented=tiling_implemented,
        a_bytes=a_bytes,
        b_bytes=b_bytes,
        c_output_bytes=c_output_bytes,
        c_accumulator_bytes=c_accumulator_bytes,
        full_task_bytes=full_task_bytes,
        working_set_bytes=working_set_bytes,
        estimated_flops=int(task.estimated_flops),
        estimated_bytes=int(task.estimated_bytes),
        estimated_host_to_dpu_bytes=host_to_dpu_bytes,
        estimated_dpu_to_host_bytes=dpu_to_host_bytes,
        estimated_mram_to_wram_bytes=mram_to_wram_bytes,
        estimated_output_tiles=estimated_output_tiles,
        estimated_k_tiles=estimated_k_tiles,
        estimated_total_tiles=estimated_total_tiles,
        optional_parallel_tile_count=estimated_total_tiles,
        memory_capacity_min_dpus=memory_capacity_min_dpus,
        estimated_dpus_required=memory_capacity_min_dpus,
        estimated_parallel_dpus=estimated_parallel_dpus,
        blocker_reason=blocker_reason,
    )


def analyze_task_graph(graph: TaskGraph, resource_model: UpmemResourceModel | None = None) -> PimFrontierGraphAnalysis:
    model = resource_model or UpmemResourceModel()
    base_tasks = tuple(analyze_task(task, index, model) for index, task in enumerate(graph.tasks))
    predecessors, unresolved = _predecessors(graph)
    waves = _frontier_waves(base_tasks, predecessors, model)
    wave_by_task_id = {
        task_id: wave
        for wave in waves
        for task_id in wave.task_ids
    }
    tasks = tuple(
        task.with_wave(wave_by_task_id[task.task_id].wave_index, wave_by_task_id[task.task_id].dominant_source)
        if task.task_id in wave_by_task_id
        else task
        for task in base_tasks
    )
    return PimFrontierGraphAnalysis(
        resource_model=model,
        tasks=tasks,
        waves=waves,
        critical_path_length_tasks=_critical_path_length(graph, predecessors),
        unresolved_dependency_count=sum(len(values) for values in unresolved.values()),
        unresolved_dependencies=unresolved,
    )


def _estimate(task: ContractionTask) -> JsonDict:
    estimate = task.target_estimates.get(UPMEM_DENSE_ESTIMATE_KEY, {})
    return estimate if isinstance(estimate, dict) else {}


def _tile_plan(estimate: JsonDict) -> JsonDict:
    tile_plan = estimate.get("tile_plan", {})
    return tile_plan if isinstance(tile_plan, dict) else {}


def _dense_lowerable(task: ContractionTask) -> tuple[bool, str | None]:
    if getattr(task, "structure", None) != "dense":
        return False, "not_lowerable_to_dense_gemm"
    if int(getattr(task, "gemm_m", 0) or 0) <= 0 or int(getattr(task, "gemm_k", 0) or 0) <= 0 or int(getattr(task, "gemm_n", 0) or 0) <= 0:
        return False, "missing_gemm_dimensions"
    return True, None


def _memory_level(full_task_bytes: int, model: UpmemResourceModel) -> tuple[str, str | None]:
    if full_task_bytes <= model.effective_wram_bytes:
        return MEMORY_LEVEL_L1_WRAM, None
    if full_task_bytes <= model.per_dpu_mram_bytes:
        return MEMORY_LEVEL_L2_SINGLE_DPU_MRAM, "requires_wram_tiling_model"
    if full_task_bytes <= model.aggregate_mram_bytes:
        return MEMORY_LEVEL_L3_MULTI_DPU, "requires_multi_dpu_distribution_model"
    return MEMORY_LEVEL_L4_OUT_OF_SCOPE, "aggregate_mram_capacity_exceeded"


def _parallel_dpus(memory_level: str, capacity_min_dpus: int, model: UpmemResourceModel) -> int:
    if memory_level in {MEMORY_LEVEL_L1_WRAM, MEMORY_LEVEL_L2_SINGLE_DPU_MRAM}:
        return 1
    if memory_level == MEMORY_LEVEL_L3_MULTI_DPU:
        return max(1, min(capacity_min_dpus, model.available_dpus, model.max_task_group_dpus))
    return 0


def _current_backend_reason(
    *,
    dense_lowerable: bool,
    backend_supported: bool,
    requires_tiling: bool,
    requires_host_aggregation: bool,
    memory_level: str,
    memory_reason: str | None,
    l2_executable: bool,
    l2_reason: str | None,
) -> str | None:
    if not dense_lowerable:
        return memory_reason
    if not backend_supported:
        return "unsupported_dense_estimate"
    if memory_level == MEMORY_LEVEL_L2_SINGLE_DPU_MRAM:
        return None if l2_executable else (l2_reason or "l2_tiled_backend_not_executable")
    if requires_tiling:
        return "requires_tiling_not_executable_backend"
    if requires_host_aggregation:
        return "requires_host_aggregation_not_executable_backend"
    if memory_level != MEMORY_LEVEL_L1_WRAM:
        return memory_reason or "not_current_backend_l1_task"
    return None


def _predecessors(graph: TaskGraph) -> tuple[dict[str, set[str]], JsonDict]:
    task_by_id = {task.id: task for task in graph.tasks}
    producer_by_tensor = {task.output_tensor_id: task.id for task in graph.tasks}
    predecessors: dict[str, set[str]] = {task.id: set() for task in graph.tasks}
    unresolved: dict[str, list[str]] = {}
    for task in graph.tasks:
        for dep in tuple(getattr(task, "dependencies", ()) or ()):
            if dep in task_by_id:
                predecessors[task.id].add(dep)
            else:
                unresolved.setdefault(task.id, []).append(str(dep))
        for tensor_id in task.input_tensor_ids:
            producer = producer_by_tensor.get(tensor_id)
            if producer and producer != task.id:
                predecessors[task.id].add(producer)
    return predecessors, unresolved


def _frontier_waves(
    tasks: tuple[PimTaskMemoryAnalysis, ...],
    predecessors: dict[str, set[str]],
    model: UpmemResourceModel,
) -> tuple[PimFrontierWave, ...]:
    task_by_id = {task.task_id: task for task in tasks}
    remaining = {task_id: set(deps) for task_id, deps in predecessors.items()}
    completed: set[str] = set()
    waves: list[PimFrontierWave] = []

    while remaining:
        ready_ids = sorted(
            [task_id for task_id, deps in remaining.items() if deps <= completed],
            key=lambda task_id: task_by_id[task_id].task_index,
        )
        if not ready_ids:
            raise ValueError("task_graph_dependency_cycle_or_unresolved_dependency")
        wave_tasks = tuple(task_by_id[task_id] for task_id in ready_ids)
        waves.append(_make_wave(len(waves), wave_tasks, model))
        completed.update(ready_ids)
        for task_id in ready_ids:
            remaining.pop(task_id, None)
    return tuple(waves)


def _make_wave(wave_index: int, tasks: tuple[PimTaskMemoryAnalysis, ...], model: UpmemResourceModel) -> PimFrontierWave:
    group_sizes = [_group_size(task, model) for task in tasks if _group_size(task, model) > 0]
    scheduling_rounds = _scheduling_rounds(group_sizes, model.available_dpus)
    assigned_slots = sum(group_sizes)
    capacity_slots = scheduling_rounds * model.available_dpus
    occupancy = (assigned_slots / capacity_slots) if capacity_slots else 0.0
    occupancy = max(0.0, min(1.0, occupancy))
    ready_count = len(tasks)
    intra = any(task.memory_level == MEMORY_LEVEL_L3_MULTI_DPU or task.optional_parallel_tile_count > 1 for task in tasks)
    inter = ready_count > 1
    dominant = (
        DOMINANT_SOURCE_HYBRID
        if inter and intra
        else DOMINANT_SOURCE_INTER_TASK
        if inter
        else DOMINANT_SOURCE_INTRA_TASK
        if intra
        else DOMINANT_SOURCE_SERIAL
    )
    return PimFrontierWave(
        wave_index=wave_index,
        task_indices=tuple(task.task_index for task in tasks),
        task_ids=tuple(task.task_id for task in tasks),
        ready_task_count=ready_count,
        memory_level_counts=dict(sorted(Counter(task.memory_level for task in tasks).items())),
        dominant_source=dominant,
        inter_task_parallelism_potential=inter,
        intra_task_parallelism_potential=intra,
        hybrid_parallelism_potential=inter and intra,
        schedulable_task_count=len(group_sizes),
        scheduling_rounds=scheduling_rounds,
        assigned_dpu_slots=assigned_slots,
        max_group_dpus=max(group_sizes, default=0),
        estimated_dpu_occupancy=occupancy,
        idle_dpu_fraction=1.0 - occupancy,
    )


def _group_size(task: PimTaskMemoryAnalysis, model: UpmemResourceModel) -> int:
    if task.memory_level in {MEMORY_LEVEL_L1_WRAM, MEMORY_LEVEL_L2_SINGLE_DPU_MRAM}:
        return 1
    if task.memory_level == MEMORY_LEVEL_L3_MULTI_DPU:
        return max(1, min(task.memory_capacity_min_dpus, model.available_dpus, model.max_task_group_dpus))
    return 0


def _scheduling_rounds(group_sizes: list[int], available_dpus: int) -> int:
    if not group_sizes:
        return 0
    rounds: list[int] = []
    for group_size in sorted(group_sizes, reverse=True):
        size = min(group_size, available_dpus)
        for index, used in enumerate(rounds):
            if used + size <= available_dpus:
                rounds[index] += size
                break
        else:
            rounds.append(size)
    return len(rounds)


def _critical_path_length(graph: TaskGraph, predecessors: dict[str, set[str]]) -> int:
    task_ids = [task.id for task in graph.tasks]
    depths: dict[str, int] = {}
    for task_id in task_ids:
        deps = [dep for dep in predecessors.get(task_id, set()) if dep in depths]
        depths[task_id] = 1 + max((depths[dep] for dep in deps), default=0)
    return max(depths.values(), default=0)
