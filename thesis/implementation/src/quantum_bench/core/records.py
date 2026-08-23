from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quantum_bench.model import (
    CircuitOperation as CircuitOperation,
    CircuitSpec as CircuitSpec,
    TensorNetwork as _TensorNetwork,
    TensorSpec as TensorSpec,
)


JsonDict = dict[str, Any]
TIMING_SCHEMA_VERSION = 2


@dataclass
class TensorValue:
    spec: TensorSpec
    array: Any


TensorNetworkSpec = _TensorNetwork


@dataclass(frozen=True)
class ContractionTask:
    id: str
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    dependencies: tuple[str, ...]
    index_expression: str
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    output_shape: tuple[int, ...]
    left_labels: tuple[int, ...]
    right_labels: tuple[int, ...]
    contracted_labels: tuple[int, ...]
    output_labels: tuple[int, ...]
    gemm_m: int
    gemm_k: int
    gemm_n: int
    structure: str
    estimated_flops: int
    estimated_bytes: int
    target_estimates: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskExecutionMetric:
    task_id: str
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    output_shape: tuple[int, ...]
    contracted_labels: tuple[int, ...]
    estimated_flops: int
    estimated_bytes: int
    execution_time_s: float
    intermediate_tensor_bytes: int
    target_estimates: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PathSummary:
    planner: str
    optimize: str
    path_length: int
    largest_intermediate: int | None
    naive_flops: float | None
    optimized_flops: float | None
    text: str
    planner_engine: str = ""
    planner_id: str = ""
    planner_kind: str = ""
    optimize_mode: str = ""
    objective: str = ""
    cost_basis: str = ""
    target_estimate_key: str | None = None
    options: JsonDict = field(default_factory=dict)
    planner_metadata: JsonDict = field(default_factory=dict)
    task_count: int = 0
    total_estimated_flops: int = 0
    peak_intermediate_bytes: int = 0
    max_intermediate_bytes: int = 0
    total_host_to_dpu_bytes: int = 0
    total_dpu_to_host_bytes: int = 0
    total_mram_to_wram_bytes: int = 0
    unsupported_task_count: int = 0
    tiling_required_task_count: int = 0
    missing_target_estimate_count: int = 0
    estimated_total_tile_count: int = 0
    estimated_max_parallel_tiles: int = 0


@dataclass(frozen=True)
class TaskGraph:
    network: TensorNetworkSpec
    tasks: tuple[ContractionTask, ...]
    path: tuple[tuple[int, ...], ...]
    path_summary: PathSummary
    planning_time_s: float
    circuit_semantics_hash: str = ""
    tensor_network_hash: str = ""
    contraction_plan_hash: str = ""


@dataclass(frozen=True)
class RouteProbe:
    route: str
    available: bool
    reason: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class RouteIdentity:
    route_id: str
    display_name: str
    role: str
    simulation_method: str
    kernel_family: str
    hardware_target: str
    execution_mode: str
    output_contract: str
    validation_mode: str


@dataclass(frozen=True)
class RouteCapabilities:
    identity: RouteIdentity
    supported_workload_families: tuple[str, ...] = ()
    supports_warmups: bool = True
    supports_repeats: bool = True
    can_return_output: bool = False
    can_measure_energy: bool = False
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class RouteOutput:
    contract: str
    array: Any | None = None
    artifact_path: Path | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationPolicy:
    mode: str
    reference_route: str | None
    tolerances: JsonDict
    reason: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    route: str
    backend_family: str
    status: str
    skip_reason: str | None
    estimated_flops: int
    estimated_bytes: int
    estimated_peak_memory: int | None
    planner_summary: str
    tile_shape: JsonDict | None = None
    wram_fit: bool | None = None
    notes: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class RouteEstimate:
    route: str
    estimated_flops: int
    estimated_bytes: int
    estimated_peak_memory: int | None
    notes: tuple[str, ...] = ()
    tile_shape: JsonDict | None = None
    wram_fit: bool | None = None
    metadata: JsonDict = field(default_factory=dict)


class TimingScope(str, Enum):
    """Ownership boundary for an ExecutionProfile timing value."""

    CASE_TENSOR_REFERENCE = "case_tensor_reference_once_per_case_planner"
    CASE_STATEVECTOR_ADAPTATION = "case_statevector_adaptation_once_per_case_planner"
    ROUTE_HOST_WALL = "route_host_wall_prepare_through_execute"
    VALIDATION = "validation_only"
    ROUTE_TOTAL = "route_attempt_prepare_through_execute_and_validation"


@dataclass(frozen=True)
class TimingContract:
    """Scopes for profile timings emitted by the suite runner.

    ``route_total_s`` is the route-attempt wall time and excludes case-scoped
    reference work. ``total_s`` is retained as an exact compatibility alias.
    """

    route_host_wall_scope: TimingScope = TimingScope.ROUTE_HOST_WALL
    validation_scope: TimingScope = TimingScope.VALIDATION
    route_total_scope: TimingScope = TimingScope.ROUTE_TOTAL
    total_s_alias_of: str = "route_total_s"


@dataclass(frozen=True)
class ExecutionProfile:
    """Versioned route timing breakdown.

    The first ten fields retain their legacy order for positional callers.
    Case-reference timings live in ``cases/<case_id>/reference_<id>.json``.
    """

    generate_s: float = 0.0
    planning_s: float = 0.0
    lowering_s: float = 0.0
    prepare_s: float = 0.0
    h2d_s: float = 0.0
    kernel_s: float = 0.0
    d2h_s: float = 0.0
    reduction_s: float = 0.0
    validation_s: float = 0.0
    total_s: float = 0.0
    route_host_wall_s: float = 0.0
    route_total_s: float = 0.0
    timing_schema_version: int = TIMING_SCHEMA_VERSION
    timing_contract: TimingContract = field(default_factory=TimingContract)


@dataclass
class RouteResult:
    route: str
    backend_family: str
    status: str
    output: RouteOutput
    profile: ExecutionProfile
    energy_joules: float | None
    energy_source: str
    error: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    max_abs_error: float
    l2_error: float
    max_rel_error: float
    norm_drift: float
    fidelity: float
    reference_norm: float
    actual_norm: float
    tolerance: JsonDict


@dataclass(frozen=True)
class BenchmarkCaseResult:
    run_id: str
    suite_id: str
    case_id: str
    repeat_id: int
    route: str
    role: str
    simulation_method: str
    kernel_family: str
    hardware_target: str
    execution_mode: str
    output_contract: str
    validation_mode: str
    backend_family: str
    status: str
    skip_reason: str | None
    n_qubits: int
    depth: int
    circuit_family: str
    gate_set: tuple[str, ...]
    planner: str
    path_summary: str
    flops: int
    bytes: int
    timings: JsonDict
    total_time_s: float
    energy_joules: float | None
    energy_source: str
    validation: JsonDict | None
    error: str | None
    route_metadata: JsonDict = field(default_factory=dict)
    reference_id: str | None = None
    reference_artifact: str | None = None
    reference_artifact_sha256: str | None = None
    reference_component: str | None = None
    timing_schema_version: int | None = None
    timing_scope: str | None = None


@dataclass(frozen=True)
class BenchmarkContext:
    root_dir: Path
    run_dir: Path
    suite: JsonDict
    case: JsonDict
    route_config: JsonDict
    repeat_id: int
    tolerances: JsonDict
    timeout_s: float | None
    memory_guard_gib: float | None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value
