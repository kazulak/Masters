"""Small, deterministic UPMEM-aware contraction-path scoring primitives.

The module deliberately contains no path-search framework and no execution
side effects.  A caller supplies complete paths, their lowered physical plans,
and measured runtime rows.  The functions here only canonicalize, describe,
normalize, rank, and fit those finite candidate sets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Literal

import numpy as np

from quantum_bench.lowering import validate_contraction_dag
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode
from quantum_bench.upmem.plan import UpmemPlan
from quantum_bench.upmem.runtime import _wram_panel_operation_facts


COST_MODEL_ID = "upmem_slr_cost_v1"
FEATURE_NAMES = (
    "B_host_dpu",
    "B_mram_wram",
    "I_dpu",
    "N_sync",
    "E_num",
    "P_wram",
)
GROUP_FEATURE_NAMES = ("movement", "compute", "coordination")
_WEIGHT_FIELDS = (
    "host_dpu",
    "mram_wram",
    "dpu_work",
    "sync",
    "numeric",
    "wram",
)
_EPSILONS = MappingProxyType({name: 1.0 for name in FEATURE_NAMES})
_VALID_SPLITS = frozenset({"train", "validation", "test"})
_VALID_MEASUREMENT_STATUSES = frozenset({"success", "failed", "unsupported"})
_PATH_HASH_PREFIX = b"quantum-bench-upmem-path-v1\0"


def _require_finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _require_positive_finite(name: str, value: object) -> float:
    value = _require_finite_nonnegative(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class RawFeatureVector:
    """Immutable raw SLR feature values in their natural units."""

    host_dpu_bytes: float
    mram_wram_bytes: float
    dpu_work: float
    sync_events: float
    numeric_overhead: float
    wram_pressure: float

    def __post_init__(self) -> None:
        for name, value in zip(
            FEATURE_NAMES,
            self.as_tuple(),
            strict=True,
        ):
            _require_finite_nonnegative(name, value)

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> RawFeatureVector:
        """Construct from canonical feature names or convenient field names."""

        aliases = {
            "B_host_dpu": "host_dpu_bytes",
            "B_mram_wram": "mram_wram_bytes",
            "I_dpu": "dpu_work",
            "N_sync": "sync_events",
            "E_num": "numeric_overhead",
            "P_wram": "wram_pressure",
        }
        normalized: dict[str, float] = {}
        for feature, field_name in aliases.items():
            if feature in values:
                normalized[field_name] = values[feature]
            elif field_name in values:
                normalized[field_name] = values[field_name]
            else:
                raise KeyError(feature)
        return cls(**normalized)

    def as_tuple(self) -> tuple[float, ...]:
        return (
            float(self.host_dpu_bytes),
            float(self.mram_wram_bytes),
            float(self.dpu_work),
            float(self.sync_events),
            float(self.numeric_overhead),
            float(self.wram_pressure),
        )

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.as_tuple(), strict=True))

    def __getitem__(self, feature: str) -> float:
        return self.as_mapping()[feature]


@dataclass(frozen=True, slots=True)
class NormalizedFeatureVector:
    """Greedy-relative logarithmic feature values."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple) or len(self.values) != len(FEATURE_NAMES):
            raise ValueError("normalized feature vectors must contain six values")
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("normalized feature values must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("normalized feature values must be finite")
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values, strict=True))

    def __getitem__(self, feature: str) -> float:
        return self.as_mapping()[feature]


@dataclass(frozen=True, slots=True)
class WeightVector:
    """Immutable nonnegative six-term weight vector on the unit simplex."""

    host_dpu: float
    mram_wram: float
    dpu_work: float
    sync: float
    numeric: float
    wram: float

    def __post_init__(self) -> None:
        values = self.as_tuple()
        for name, value in zip(_WEIGHT_FIELDS, values, strict=True):
            _require_finite_nonnegative(name, value)
        if not math.isclose(sum(values), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("weights must sum to one")

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, float] | Sequence[float],
        *,
        inactive: Iterable[str] = (),
    ) -> WeightVector:
        """Normalize positive weights and force named inactive terms to zero."""

        if isinstance(values, Mapping):
            raw = []
            for feature, field_name in zip(
                FEATURE_NAMES,
                _WEIGHT_FIELDS,
                strict=True,
            ):
                raw.append(
                    values.get(
                        field_name,
                        values.get(feature, 0.0),
                    )
                )
        else:
            raw = list(values)
        if len(raw) != len(FEATURE_NAMES):
            raise ValueError("weights must contain six values")

        inactive_names = set(inactive)
        unknown = inactive_names.difference(FEATURE_NAMES).difference(_WEIGHT_FIELDS)
        if unknown:
            raise KeyError(f"unknown inactive feature(s): {sorted(unknown)!r}")
        normalized = [
            _require_finite_nonnegative(name, value)
            for name, value in zip(_WEIGHT_FIELDS, raw, strict=True)
        ]
        for index, (feature, field_name) in enumerate(
            zip(FEATURE_NAMES, _WEIGHT_FIELDS, strict=True)
        ):
            if feature in inactive_names or field_name in inactive_names:
                normalized[index] = 0.0
        total = sum(normalized)
        if total <= 0.0:
            raise ValueError("at least one active weight must be positive")
        normalized = [value / total for value in normalized]
        return cls(*normalized)

    @classmethod
    def equal(cls) -> WeightVector:
        return cls.from_values((1.0,) * len(FEATURE_NAMES))

    def as_tuple(self) -> tuple[float, ...]:
        return (
            float(self.host_dpu),
            float(self.mram_wram),
            float(self.dpu_work),
            float(self.sync),
            float(self.numeric),
            float(self.wram),
        )

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.as_tuple(), strict=True))

    def __getitem__(self, feature: str) -> float:
        if feature in FEATURE_NAMES:
            return self.as_mapping()[feature]
        return dict(zip(_WEIGHT_FIELDS, self.as_tuple(), strict=True))[feature]


@dataclass(frozen=True, slots=True)
class FeatureDependency:
    """Machine-readable ownership notes for the six conceptual terms."""

    feature: str
    source: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    independently_identifiable: bool


FEATURE_DEPENDENCY_METADATA = (
    FeatureDependency(
        feature="B_host_dpu",
        source="UpmemWorkUnit estimated input/output bytes",
        includes=("four split-complex transfer lanes", "H2D", "D2H"),
        excludes=("DPU-local MRAM-WRAM traffic", "numeric overhead"),
        independently_identifiable=True,
    ),
    FeatureDependency(
        feature="B_mram_wram",
        source="existing WRAM-panel aligned-span estimate",
        includes=("A reads", "B reads", "partial-C reads", "C writes"),
        excludes=("host-DPU bytes", "arithmetic work"),
        independently_identifiable=True,
    ),
    FeatureDependency(
        feature="I_dpu",
        source="existing WRAM-panel exact real-MAC fact",
        includes=("four real products for split-complex float32",),
        excludes=("movement bytes", "coordination events"),
        independently_identifiable=True,
    ),
    FeatureDependency(
        feature="N_sync",
        source="explicit non-overlapping plan coordination event counts",
        includes=(
            "one event per wave (also the packed-operation alias)",
            "DPU launches",
            "host reductions",
            "tasklet barriers",
        ),
        excludes=(
            "packed-operation count as a second wave event",
            "arithmetic work",
            "transfer bytes",
        ),
        independently_identifiable=True,
    ),
    FeatureDependency(
        feature="E_num",
        source="numeric-policy-specific additional overhead",
        includes=(),
        excludes=("float32 lane movement", "four-real-product arithmetic"),
        independently_identifiable=False,
    ),
    FeatureDependency(
        feature="P_wram",
        source="modeled WRAM buffer allocation facts",
        includes=("static WRAM allocation pressure",),
        excludes=("MRAM-WRAM traffic", "hard feasibility admission"),
        independently_identifiable=False,
    ),
)


def feature_dependency_metadata() -> tuple[FeatureDependency, ...]:
    return FEATURE_DEPENDENCY_METADATA


@dataclass(frozen=True, slots=True)
class PlanFeatureFacts:
    """Raw features plus auditable physical-plan counters."""

    raw: RawFeatureVector
    h2d_bytes: int
    d2h_bytes: int
    mram_wram_bytes: int
    real_mac_count: int
    work_unit_count: int
    stage_count: int
    contract_stage_count: int
    host_reduce_count: int
    wave_count: int
    packed_operation_count: int
    dpu_launch_count: int
    barrier_events: int
    partial_wave_count: int
    wram_buffer_bytes: int
    tasklet_utilization: float
    dpu_utilization: float

    def as_mapping(self) -> dict[str, object]:
        return {
            **self.raw.as_mapping(),
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "mram_wram_bytes": self.mram_wram_bytes,
            "real_mac_count": self.real_mac_count,
            "work_unit_count": self.work_unit_count,
            "stage_count": self.stage_count,
            "contract_stage_count": self.contract_stage_count,
            "host_reduce_count": self.host_reduce_count,
            "wave_count": self.wave_count,
            "packed_operation_count": self.packed_operation_count,
            "dpu_launch_count": self.dpu_launch_count,
            "barrier_events": self.barrier_events,
            "partial_wave_count": self.partial_wave_count,
            "wram_buffer_bytes": self.wram_buffer_bytes,
            "tasklet_utilization": self.tasklet_utilization,
            "dpu_utilization": self.dpu_utilization,
        }


def extract_plan_features(plan: UpmemPlan) -> PlanFeatureFacts:
    """Derive path/topology features from the existing immutable UPMEM plan."""

    if not isinstance(plan, UpmemPlan):
        raise TypeError("extract_plan_features requires an UpmemPlan")
    if plan.schedule_policy != "serial_nodes_v1":
        raise ValueError("upmem_slr_cost_v1 requires serial_nodes_v1; wave cost extraction is not qualified")

    h2d_bytes = 0
    d2h_bytes = 0
    mram_wram_bytes = 0
    real_mac_count = 0
    work_unit_count = 0
    host_reduce_count = 0
    wave_count = 0
    packed_operation_count = 0
    dpu_launch_count = 0
    barrier_events = 0
    partial_wave_count = 0
    wram_buffer_bytes = 0
    contract_stage_count = 0

    for stage in plan.stages:
        if stage.kind == "host_reduce":
            host_reduce_count += 1
            continue
        contract_stage_count += 1
        units = tuple(stage.work_units)
        work_unit_count += len(units)
        h2d_bytes += 4 * sum(unit.estimated_input_bytes for unit in units)
        d2h_bytes += 4 * sum(unit.estimated_output_bytes for unit in units)
        stage_waves = tuple(sorted({unit.wave for unit in units}))
        wave_count += len(stage_waves)
        packed_operation_count += len(stage_waves)
        dpu_launch_count += 4 * len(stage_waves)
        for wave in stage_waves:
            wave_units = tuple(unit for unit in units if unit.wave == wave)
            useful_slots = {
                (unit.logical_rank, unit.logical_dpu)
                for unit in wave_units
                if unit.estimated_arithmetic_work > 0
            }
            if len(useful_slots) < plan.topology.dpu_count:
                partial_wave_count += 1
        facts = _wram_panel_operation_facts(
            units,
            numeric_policy=plan.numeric_policy,
            tasklets_per_dpu=plan.topology.tasklets_per_dpu,
        )
        mram_wram_bytes += int(facts["mram_aligned_transfer_bytes_estimate"])
        real_mac_count += int(facts["real_mac_count_exact"])
        barrier_events += int(facts["barrier_events_exact"])
        wram_buffer_bytes = max(
            wram_buffer_bytes,
            int(facts["wram_kernel_buffers_allocated_bytes_exact"]),
        )

    from quantum_bench.upmem.plan import collection_resource_admission

    admission = collection_resource_admission(plan)
    # ``packed_operation_count`` and ``wave_count`` are aliases for this plan;
    # count the coordination event once through ``wave_count``.
    sync_events = wave_count + dpu_launch_count + host_reduce_count + barrier_events
    raw = RawFeatureVector(
        host_dpu_bytes=h2d_bytes + d2h_bytes,
        mram_wram_bytes=mram_wram_bytes,
        dpu_work=real_mac_count,
        sync_events=sync_events,
        numeric_overhead=0.0,
        wram_pressure=float(wram_buffer_bytes),
    )
    return PlanFeatureFacts(
        raw=raw,
        h2d_bytes=h2d_bytes,
        d2h_bytes=d2h_bytes,
        mram_wram_bytes=mram_wram_bytes,
        real_mac_count=real_mac_count,
        work_unit_count=work_unit_count,
        stage_count=len(plan.stages),
        contract_stage_count=contract_stage_count,
        host_reduce_count=host_reduce_count,
        wave_count=wave_count,
        packed_operation_count=packed_operation_count,
        dpu_launch_count=dpu_launch_count,
        barrier_events=barrier_events,
        partial_wave_count=partial_wave_count,
        wram_buffer_bytes=wram_buffer_bytes,
        tasklet_utilization=float(admission["arithmetic_weighted_tasklet_utilization"] or 0.0),
        dpu_utilization=float(admission["arithmetic_weighted_dpu_slot_utilization"] or 0.0),
    )


@dataclass(frozen=True, slots=True)
class ConventionalPathFeatures:
    """Path metrics used for conventional selector comparisons."""

    flops: float
    macs: float
    peak_intermediate_elements: float
    peak_intermediate_bytes: float
    total_intermediate_writes: float
    maximum_intermediate_rank: int
    contraction_count: int

    def __post_init__(self) -> None:
        for name in (
            "flops",
            "macs",
            "peak_intermediate_elements",
            "peak_intermediate_bytes",
            "total_intermediate_writes",
        ):
            _require_finite_nonnegative(name, getattr(self, name))
        if self.maximum_intermediate_rank < 0 or self.contraction_count < 0:
            raise ValueError("path counts and ranks must be nonnegative")

    def as_mapping(self) -> dict[str, float | int]:
        return {
            "flops": self.flops,
            "macs": self.macs,
            "peak_intermediate_elements": self.peak_intermediate_elements,
            "peak_intermediate_bytes": self.peak_intermediate_bytes,
            "total_intermediate_writes": self.total_intermediate_writes,
            "maximum_intermediate_rank": self.maximum_intermediate_rank,
            "contraction_count": self.contraction_count,
        }


def _shape_size(shape: Sequence[int]) -> int:
    result = 1
    for size in shape:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("tensor dimensions must be positive integers")
        result *= size
    return result


def extract_conventional_features(dag: ContractionDAG) -> ConventionalPathFeatures:
    """Compute path metrics from the same lowered DAG used for UPMEM."""

    if not isinstance(dag, ContractionDAG):
        raise TypeError("extract_conventional_features requires a ContractionDAG")
    validate_contraction_dag(dag)
    macs = 0
    intermediate_sizes: list[int] = []
    maximum_rank = 0
    contraction_count = 0
    for node in dag.nodes:
        output_elements = _shape_size(node.output.shape)
        if node.output.id != dag.output.tensor_id:
            intermediate_sizes.append(output_elements)
            maximum_rank = max(maximum_rank, len(node.output.labels))
        if isinstance(node, ContractNode):
            contraction_count += 1
            contracted_size = 1
            for label in node.contracted_labels:
                if label in node.left.labels:
                    dimension = node.left.shape[node.left.labels.index(label)]
                else:
                    dimension = node.right.shape[node.right.labels.index(label)]
                contracted_size *= dimension
            macs += output_elements * contracted_size
        elif not isinstance(node, ReduceNode):  # pragma: no cover
            raise TypeError(f"unsupported DAG node: {type(node).__name__}")
    peak_elements = max(intermediate_sizes, default=0)
    return ConventionalPathFeatures(
        flops=float(2 * macs),
        macs=float(macs),
        peak_intermediate_elements=float(peak_elements),
        peak_intermediate_bytes=float(16 * peak_elements),
        total_intermediate_writes=float(sum(intermediate_sizes)),
        maximum_intermediate_rank=maximum_rank,
        contraction_count=contraction_count,
    )


def canonicalize_path(path: Iterable[Iterable[int]]) -> tuple[tuple[int, int], ...]:
    """Return a compact active-list path representation with basic validation."""

    canonical: list[tuple[int, int]] = []
    for step in path:
        values = tuple(step)
        if len(values) != 2:
            raise ValueError("contraction paths must contain binary pairs")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("contraction path indices must be integers")
        if values[0] < 0 or values[1] < 0 or values[0] == values[1]:
            raise ValueError("contraction path indices must be distinct and nonnegative")
        # Pair order is not semantic; the sequence of contraction steps is.
        canonical.append(tuple(sorted(values)))
    return tuple(canonical)


def path_id(path: Iterable[Iterable[int]], *, circuit_id: str = "") -> str:
    """Hash a canonical complete path with a circuit-scoped namespace."""

    canonical = canonicalize_path(path)
    payload = json.dumps(
        {"circuit_id": circuit_id, "path": canonical},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(_PATH_HASH_PREFIX + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PathCandidate:
    """One complete path and its topology-specific physical feature vectors."""

    path_id: str
    conventional: ConventionalPathFeatures
    features_by_topology: tuple[tuple[str, RawFeatureVector], ...]
    feasible_topologies: tuple[str, ...] | None = None
    is_greedy: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_string("path_id", self.path_id)
        if not isinstance(self.conventional, ConventionalPathFeatures):
            raise TypeError("conventional must be ConventionalPathFeatures")
        if not isinstance(self.features_by_topology, tuple):
            raise TypeError("features_by_topology must be a tuple")
        topology_ids = tuple(topology for topology, _ in self.features_by_topology)
        if len(set(topology_ids)) != len(topology_ids):
            raise ValueError("candidate topology feature IDs must be unique")
        for topology, features in self.features_by_topology:
            _require_nonempty_string("topology", topology)
            if not isinstance(features, RawFeatureVector):
                raise TypeError("topology features must be RawFeatureVector records")
        if self.feasible_topologies is not None and not isinstance(
            self.feasible_topologies, tuple
        ):
            raise TypeError("feasible_topologies must be a tuple or None")
        if self.feasible_topologies is not None and not set(
            self.feasible_topologies
        ).issubset(topology_ids):
            raise ValueError("feasible topology must have a feature vector")

    @classmethod
    def synthetic(
        cls,
        path_id: str,
        raw_features: RawFeatureVector,
        *,
        topology: str = "default",
        flops: float = 1.0,
        peak_intermediate: float = 1.0,
        intermediate_writes: float = 1.0,
        is_greedy: bool = False,
        feasible: bool = True,
    ) -> PathCandidate:
        conventional = ConventionalPathFeatures(
            flops=flops,
            macs=flops / 2.0,
            peak_intermediate_elements=peak_intermediate,
            peak_intermediate_bytes=peak_intermediate * 16.0,
            total_intermediate_writes=intermediate_writes,
            maximum_intermediate_rank=1,
            contraction_count=1,
        )
        return cls(
            path_id=path_id,
            conventional=conventional,
            features_by_topology=((topology, raw_features),),
            feasible_topologies=(topology,) if feasible else (),
            is_greedy=is_greedy,
        )

    def raw_for(self, topology: str) -> RawFeatureVector:
        for topology_id, features in self.features_by_topology:
            if topology_id == topology:
                return features
        raise KeyError(f"candidate {self.path_id!r} has no topology {topology!r}")

    def feasible_for(self, topology: str) -> bool:
        if not any(topology_id == topology for topology_id, _ in self.features_by_topology):
            return False
        return self.feasible_topologies is None or topology in self.feasible_topologies


@dataclass(frozen=True, slots=True)
class FeatureModelDecision:
    """Frozen decision between identifiable six-term and grouped scoring."""

    mode: Literal["six_term", "grouped"]
    active_features: tuple[str, ...]
    zero_range_features: tuple[str, ...]
    correlated_pairs: tuple[tuple[str, str], ...]
    matrix_rank: int
    rank_tolerance: float
    reason: str

    def __post_init__(self) -> None:
        valid_names = set(FEATURE_NAMES) | set(GROUP_FEATURE_NAMES)
        if self.mode not in {"six_term", "grouped"}:
            raise ValueError("unsupported feature model mode")
        if any(name not in valid_names for name in self.active_features):
            raise ValueError("unknown active feature")

    def project(self, normalized: NormalizedFeatureVector) -> tuple[float, ...]:
        values = normalized.values
        if self.mode == "six_term":
            return tuple(values[FEATURE_NAMES.index(name)] for name in self.active_features)
        grouped = {
            "movement": (values[0] + values[1]) / 2.0,
            "compute": values[2],
            "coordination": values[3],
        }
        return tuple(grouped[name] for name in self.active_features)

    def project_raw(self, raw: RawFeatureVector) -> tuple[float, ...]:
        """Project natural-unit features without changing their raw sums."""

        if not isinstance(raw, RawFeatureVector):
            raise TypeError("project_raw requires a RawFeatureVector")
        values = raw.as_tuple()
        if self.mode == "six_term":
            return tuple(values[FEATURE_NAMES.index(name)] for name in self.active_features)
        grouped = {
            "movement": raw.host_dpu_bytes + raw.mram_wram_bytes,
            "compute": raw.dpu_work,
            "coordination": raw.sync_events,
        }
        return tuple(grouped[name] for name in self.active_features)

    def score(
        self,
        normalized: NormalizedFeatureVector,
        weights: WeightVector,
    ) -> float:
        if self.mode == "six_term":
            return float(sum(value * weight for value, weight in zip(
                normalized.values,
                weights.as_tuple(),
                strict=True,
            )))
        movement = (normalized.values[0] + normalized.values[1]) / 2.0
        return float(
            movement * (weights.host_dpu + weights.mram_wram)
            + normalized.values[2] * weights.dpu_work
            + normalized.values[3] * weights.sync
        )


SIX_TERM_FEATURE_MODEL = FeatureModelDecision(
    mode="six_term",
    active_features=FEATURE_NAMES,
    zero_range_features=(),
    correlated_pairs=(),
    matrix_rank=len(FEATURE_NAMES),
    rank_tolerance=0.0,
    reason="explicit six-term model",
)
GROUPED_FEATURE_MODEL = FeatureModelDecision(
    mode="grouped",
    active_features=GROUP_FEATURE_NAMES,
    zero_range_features=(),
    correlated_pairs=(),
    matrix_rank=len(GROUP_FEATURE_NAMES),
    rank_tolerance=0.0,
    reason="explicit grouped movement/compute/coordination model",
)


def explicit_feature_model(
    mode: Literal["six_term", "grouped"],
) -> FeatureModelDecision:
    """Return a deterministic explicit six-term or grouped model.

    The current float32 policy has no independently identifiable ``E_num``
    term, so the grouped model has exactly three terms.  ``P_wram`` remains a
    feasibility constraint rather than a scored term.
    """

    if mode == "six_term":
        return SIX_TERM_FEATURE_MODEL
    if mode != "grouped":
        raise ValueError(f"unsupported explicit feature model mode: {mode!r}")
    return GROUPED_FEATURE_MODEL


def normalize_features(
    raw: RawFeatureVector,
    greedy: RawFeatureVector,
    *,
    epsilons: Mapping[str, float] = _EPSILONS,
) -> NormalizedFeatureVector:
    """Normalize each raw term relative to its same-cell greedy reference."""

    values = []
    for feature, candidate_value, reference_value in zip(
        FEATURE_NAMES,
        raw.as_tuple(),
        greedy.as_tuple(),
        strict=True,
    ):
        epsilon = _require_positive_finite(
            f"epsilon for {feature}",
            epsilons[feature],
        )
        values.append(math.log((candidate_value + epsilon) / (reference_value + epsilon)))
    return NormalizedFeatureVector(tuple(values))


def choose_feature_model(
    vectors: Sequence[NormalizedFeatureVector],
    *,
    correlation_threshold: float = 0.98,
) -> FeatureModelDecision:
    """Freeze an identifiability decision from normalized feature vectors."""

    if not vectors:
        raise ValueError("at least one normalized vector is required")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")
    matrix = np.asarray([vector.values for vector in vectors], dtype=np.float64)
    ranges = np.ptp(matrix, axis=0)
    active_indices = tuple(index for index, value in enumerate(ranges) if value > 0.0)
    zero_range = tuple(
        feature for index, feature in enumerate(FEATURE_NAMES) if index not in active_indices
    )
    if not active_indices:
        return FeatureModelDecision(
            mode="six_term",
            active_features=(),
            zero_range_features=zero_range,
            correlated_pairs=(),
            matrix_rank=0,
            rank_tolerance=0.0,
            reason="all normalized terms are constant",
        )

    active_matrix = matrix[:, active_indices]
    singular_values = np.linalg.svd(active_matrix, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = (
        max(active_matrix.shape) * np.finfo(np.float64).eps * largest
        if largest
        else 0.0
    )
    rank = int(np.linalg.matrix_rank(active_matrix, tol=tolerance))
    correlations: list[tuple[str, str]] = []
    for left_index, left in enumerate(active_indices):
        for right in active_indices[left_index + 1 :]:
            coefficient = float(np.corrcoef(matrix[:, left], matrix[:, right])[0, 1])
            if math.isfinite(coefficient) and abs(coefficient) >= correlation_threshold:
                correlations.append((FEATURE_NAMES[left], FEATURE_NAMES[right]))

    if rank == len(active_indices) and not correlations:
        return FeatureModelDecision(
            mode="six_term",
            active_features=tuple(FEATURE_NAMES[index] for index in active_indices),
            zero_range_features=zero_range,
            correlated_pairs=(),
            matrix_rank=rank,
            rank_tolerance=tolerance,
            reason="active feature matrix is identifiable",
        )

    grouped_values = np.column_stack(
        (
            (matrix[:, 0] + matrix[:, 1]) / 2.0,
            matrix[:, 2],
            matrix[:, 3],
        )
    )
    grouped_names = tuple(
        name
        for index, name in enumerate(GROUP_FEATURE_NAMES)
        if float(np.ptp(grouped_values[:, index])) > 0.0
    )
    return FeatureModelDecision(
        mode="grouped",
        active_features=grouped_names,
        zero_range_features=zero_range,
        correlated_pairs=tuple(correlations),
        matrix_rank=rank,
        rank_tolerance=tolerance,
        reason=(
            "independent six-term fit rejected by rank deficiency or high correlation; "
            "using movement/compute/coordination projection"
        ),
    )


def _weights_from_model_values(
    values: Sequence[float],
    model: FeatureModelDecision,
) -> WeightVector:
    if not model.active_features:
        return WeightVector.from_values((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    if len(values) != len(model.active_features):
        raise ValueError("weight count does not match the feature model")
    if model.mode == "six_term":
        expanded = dict.fromkeys(FEATURE_NAMES, 0.0)
        expanded.update(zip(model.active_features, values, strict=True))
        return WeightVector.from_values(expanded)

    grouped = dict(zip(model.active_features, values, strict=True))
    expanded = dict.fromkeys(FEATURE_NAMES, 0.0)
    movement = float(grouped.get("movement", 0.0))
    expanded["B_host_dpu"] = movement / 2.0
    expanded["B_mram_wram"] = movement / 2.0
    expanded["I_dpu"] = float(grouped.get("compute", 0.0))
    expanded["N_sync"] = float(grouped.get("coordination", 0.0))
    return WeightVector.from_values(expanded)


def equal_model_weights(model: FeatureModelDecision) -> WeightVector:
    """Return equal weights over the model's active terms."""

    if not model.active_features:
        return _weights_from_model_values((), model)
    return _weights_from_model_values(
        (1.0 / len(model.active_features),) * len(model.active_features),
        model,
    )


def score_normalized(
    normalized: NormalizedFeatureVector,
    weights: WeightVector,
    *,
    model: FeatureModelDecision | None = None,
) -> float:
    return (model or SIX_TERM_FEATURE_MODEL).score(normalized, weights)


def score_features(
    raw: RawFeatureVector,
    greedy: RawFeatureVector,
    weights: WeightVector,
    *,
    model: FeatureModelDecision | None = None,
) -> float:
    return score_normalized(
        normalize_features(raw, greedy),
        weights,
        model=model,
    )


def _greedy_candidate(
    candidates: Sequence[PathCandidate],
    greedy_path_id: str | None,
) -> PathCandidate:
    if greedy_path_id is not None:
        matches = [candidate for candidate in candidates if candidate.path_id == greedy_path_id]
    else:
        matches = [candidate for candidate in candidates if candidate.is_greedy]
    if len(matches) != 1:
        raise ValueError("exactly one greedy candidate is required")
    return matches[0]


def select_best_candidate(
    candidates: Sequence[PathCandidate],
    topology: str,
    weights: WeightVector,
    *,
    model: FeatureModelDecision | None = None,
    greedy_path_id: str | None = None,
) -> PathCandidate:
    """Select a feasible candidate by lower score, then full path ID."""

    feasible = tuple(candidate for candidate in candidates if candidate.feasible_for(topology))
    if not feasible:
        raise ValueError(f"no feasible candidate for topology {topology!r}")
    greedy = _greedy_candidate(feasible, greedy_path_id)
    scored = [
        (
            score_features(
                candidate.raw_for(topology),
                greedy.raw_for(topology),
                weights,
                model=model,
            ),
            candidate.path_id,
            candidate,
        )
        for candidate in feasible
    ]
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def select_calibration_candidates(
    candidates: Sequence[PathCandidate],
    topology: str,
    *,
    limit: int = 6,
    model: FeatureModelDecision | None = None,
    greedy_path_id: str | None = None,
) -> tuple[PathCandidate, ...]:
    """Select deterministic conventional and feature-diverse calibration paths."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("calibration limit must be a positive integer")
    unique: dict[str, PathCandidate] = {}
    for candidate in candidates:
        if candidate.path_id in unique and unique[candidate.path_id] != candidate:
            raise ValueError(f"duplicate path ID with different candidate: {candidate.path_id}")
        unique[candidate.path_id] = candidate
    feasible = tuple(
        sorted(
            (candidate for candidate in unique.values() if candidate.feasible_for(topology)),
            key=lambda candidate: candidate.path_id,
        )
    )
    if not feasible:
        raise ValueError(f"no feasible candidate for topology {topology!r}")
    greedy = _greedy_candidate(feasible, greedy_path_id)
    normalized = {
        candidate.path_id: normalize_features(
            candidate.raw_for(topology),
            greedy.raw_for(topology),
        )
        for candidate in feasible
    }
    chosen_model = model or choose_feature_model(tuple(normalized.values()))
    equal_weights = equal_model_weights(chosen_model)
    selected: list[PathCandidate] = []
    selected_ids: set[str] = set()

    def add(candidate: PathCandidate) -> None:
        if candidate.path_id not in selected_ids and len(selected) < limit:
            selected.append(candidate)
            selected_ids.add(candidate.path_id)

    add(greedy)
    add(min(feasible, key=lambda item: (item.conventional.flops, item.path_id)))
    add(
        min(
            feasible,
            key=lambda item: (
                item.conventional.peak_intermediate_elements,
                item.path_id,
            ),
        )
    )
    add(
        min(
            feasible,
            key=lambda item: (
                item.conventional.total_intermediate_writes,
                item.path_id,
            ),
        )
    )
    add(
        select_best_candidate(
            feasible,
            topology,
            equal_weights,
            model=chosen_model,
            greedy_path_id=greedy.path_id,
        )
    )

    while len(selected) < min(limit, len(feasible)):
        remaining = [candidate for candidate in feasible if candidate.path_id not in selected_ids]
        if not remaining:
            break

        def distance(candidate: PathCandidate) -> tuple[float, str]:
            vector = np.asarray(chosen_model.project(normalized[candidate.path_id]))
            distances = [
                float(
                    np.linalg.norm(
                        vector
                        - np.asarray(chosen_model.project(normalized[chosen.path_id]))
                    )
                )
                for chosen in selected
            ]
            return (min(distances, default=0.0), candidate.path_id)

        add(max(remaining, key=distance))
    return tuple(selected)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingCell:
    cell_id: str
    topology: str
    candidates: tuple[PathCandidate, ...]
    greedy_path_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string("cell_id", self.cell_id)
        _require_nonempty_string("topology", self.topology)
        if not isinstance(self.candidates, tuple) or not self.candidates:
            raise ValueError("training cells require a nonempty candidate tuple")
        if len({candidate.path_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("training cell candidate IDs must be unique")
        _greedy_candidate(self.candidates, self.greedy_path_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeMeasurement:
    cell_id: str
    candidate_id: str
    runtime_s: float
    split: str = "train"
    source_sha: str | None = None
    timing_scope: str | None = None
    status: str = "success"
    observation_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string("cell_id", self.cell_id)
        _require_nonempty_string("candidate_id", self.candidate_id)
        _require_positive_finite("runtime_s", self.runtime_s)
        if self.split not in _VALID_SPLITS:
            raise ValueError(f"unsupported data split: {self.split!r}")
        if self.status not in _VALID_MEASUREMENT_STATUSES:
            raise ValueError(f"unsupported runtime measurement status: {self.status!r}")
        for name, value in (
            ("source_sha", self.source_sha),
            ("timing_scope", self.timing_scope),
            ("observation_id", self.observation_id),
        ):
            if value is not None:
                _require_nonempty_string(name, value)


MeasuredRuntime = RuntimeMeasurement


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightFitResult:
    weights: WeightVector
    model: FeatureModelDecision
    selected_path_ids: tuple[tuple[str, str], ...]
    cell_speedups: tuple[tuple[str, float], ...]
    geometric_mean_speedup: float
    minimum_cell_speedup: float
    improved_cell_count: int
    evaluated_weight_vectors: int
    seed: int
    random_sample_count: int


def geometric_mean(values: Iterable[float]) -> float:
    values = tuple(_require_positive_finite("speedup", value) for value in values)
    if not values:
        raise ValueError("geometric mean requires at least one value")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _coerce_measurement(row: RuntimeMeasurement | Mapping[str, object]) -> RuntimeMeasurement:
    if isinstance(row, RuntimeMeasurement):
        return row
    if isinstance(row, Mapping):
        return RuntimeMeasurement(
            cell_id=str(row["cell_id"]),
            candidate_id=str(row["candidate_id"]),
            runtime_s=float(row["runtime_s"]),
            split=str(row.get("split", "train")),
            source_sha=row.get("source_sha"),
            timing_scope=row.get("timing_scope"),
            status=str(row.get("status", "success")),
            observation_id=row.get("observation_id"),
        )
    raise TypeError("runtime rows must be RuntimeMeasurement records or mappings")


def _fit_objective(result: WeightFitResult) -> tuple[float, float, int, tuple[float, ...]]:
    return (
        round(result.geometric_mean_speedup, 12),
        round(result.minimum_cell_speedup, 12),
        result.improved_cell_count,
        tuple(-value for value in result.weights.as_tuple()),
    )


def fit_weights(
    cells: Sequence[TrainingCell],
    measurements: Sequence[RuntimeMeasurement | Mapping[str, object]],
    *,
    model: FeatureModelDecision | None = None,
    seed: int = 20260903,
    random_sample_count: int = 100_000,
    evaluation_callback: Callable[[WeightFitResult], None] | None = None,
) -> WeightFitResult:
    """Fit weights offline using only measured training-candidate runtimes."""

    cells = tuple(cells)
    if not cells:
        raise ValueError("at least one training cell is required")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("training cell IDs must be unique")
    if isinstance(random_sample_count, bool) or not isinstance(random_sample_count, int):
        raise TypeError("random_sample_count must be an integer")
    if random_sample_count < 1:
        raise ValueError("random_sample_count must be positive")
    rows = tuple(_coerce_measurement(row) for row in measurements)
    if not rows:
        raise ValueError("at least one measured runtime row is required")
    if any(row.split != "train" for row in rows):
        raise ValueError("weight fitting accepts training rows only")
    if any(row.status != "success" for row in rows):
        raise ValueError("runtime evidence contains failed or unsupported rows")

    for name in ("source_sha", "timing_scope"):
        values = {getattr(row, name) for row in rows}
        if len(values) > 1:
            raise ValueError(f"runtime evidence has mixed {name} values")

    cells_by_id = {cell.cell_id: cell for cell in cells}
    raw_rows: dict[tuple[str, str], list[float]] = {}
    observation_ids: dict[tuple[str, str], list[str | None]] = {}
    observed_candidates: dict[str, set[str]] = {}
    seen_rows: set[tuple[object, ...]] = set()
    for row in rows:
        cell = cells_by_id.get(row.cell_id)
        if cell is None:
            raise ValueError(f"runtime row references unknown cell {row.cell_id!r}")
        if not any(candidate.path_id == row.candidate_id for candidate in cell.candidates):
            raise ValueError(
                f"runtime row references unknown candidate {row.candidate_id!r}"
            )
        identity = (
            row.cell_id,
            row.candidate_id,
            row.observation_id
            if row.observation_id is not None
            else ("runtime_s", row.runtime_s),
        )
        if identity in seen_rows:
            raise ValueError("duplicate runtime evidence row")
        seen_rows.add(identity)
        raw_rows.setdefault((row.cell_id, row.candidate_id), []).append(row.runtime_s)
        observation_ids.setdefault((row.cell_id, row.candidate_id), []).append(
            row.observation_id
        )
        observed_candidates.setdefault(row.cell_id, set()).add(row.candidate_id)

    for cell in cells:
        infeasible = tuple(
            candidate.path_id
            for candidate in cell.candidates
            if not candidate.feasible_for(cell.topology)
        )
        if infeasible:
            raise ValueError(
                f"calibration cell {cell.cell_id!r} contains infeasible candidates: "
                f"{infeasible!r}"
            )
        expected = {candidate.path_id for candidate in cell.candidates}
        observed = observed_candidates.get(cell.cell_id, set())
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"calibration cell {cell.cell_id!r} requires the exact measured "
                f"candidate set (missing={missing!r}, extra={extra!r})"
            )
        counts = {
            candidate_id: len(raw_rows[(cell.cell_id, candidate_id)])
            for candidate_id in expected
        }
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"calibration cell {cell.cell_id!r} has incomplete timing rows: "
                f"observation counts={counts!r}"
            )
        identifiers = {
            tuple(observation_ids[(cell.cell_id, candidate_id)])
            for candidate_id in expected
        }
        if any(value is None for ids in identifiers for value in ids) and any(
            value is not None for ids in identifiers for value in ids
        ):
            raise ValueError(
                f"calibration cell {cell.cell_id!r} mixes identified and "
                "unidentified timing rows"
            )
        if all(
            value is not None for ids in identifiers for value in ids
        ) and len({frozenset(ids) for ids in identifiers}) != 1:
            raise ValueError(
                f"calibration cell {cell.cell_id!r} has incomplete observation IDs"
            )
    medians = {
        key: float(np.median(values)) for key, values in raw_rows.items()
    }

    normalized_by_cell: dict[str, dict[str, NormalizedFeatureVector]] = {}
    all_normalized: list[NormalizedFeatureVector] = []
    for cell in cells:
        greedy = _greedy_candidate(cell.candidates, cell.greedy_path_id)
        feasible = [
            candidate
            for candidate in cell.candidates
            if candidate.feasible_for(cell.topology)
        ]
        if not feasible:
            raise ValueError(f"cell {cell.cell_id!r} has no feasible candidates")
        normalized_for_cell = {
            candidate.path_id: normalize_features(
                candidate.raw_for(cell.topology),
                greedy.raw_for(cell.topology),
            )
            for candidate in feasible
        }
        normalized_by_cell[cell.cell_id] = normalized_for_cell
        all_normalized.extend(normalized_for_cell.values())
        if (cell.cell_id, greedy.path_id) not in medians:
            raise ValueError(f"greedy runtime is missing for cell {cell.cell_id!r}")

    chosen_model = model or choose_feature_model(tuple(all_normalized))
    dimension = len(chosen_model.active_features)
    rng = np.random.default_rng(seed)
    random_vectors = (
        rng.dirichlet(np.ones(dimension), size=random_sample_count)
        if dimension
        else np.empty((0, 0), dtype=np.float64)
    )
    search_vectors: list[tuple[float, ...]] = [
        tuple(float(value) for value in row) for row in random_vectors
    ]
    if dimension:
        search_vectors.append((1.0 / dimension,) * dimension)
        search_vectors.extend(
            tuple(1.0 if index == selected else 0.0 for index in range(dimension))
            for selected in range(dimension)
        )
    else:
        search_vectors.append(())

    best: WeightFitResult | None = None
    for values in search_vectors:
        weights = _weights_from_model_values(values, chosen_model)
        selected_path_ids: list[tuple[str, str]] = []
        speedups: list[tuple[str, float]] = []
        for cell in cells:
            greedy = _greedy_candidate(cell.candidates, cell.greedy_path_id)
            measured_feasible = [
                candidate
                for candidate in cell.candidates
                if candidate.feasible_for(cell.topology)
                and (cell.cell_id, candidate.path_id) in medians
            ]
            if not measured_feasible:
                raise ValueError(f"cell {cell.cell_id!r} has no measured feasible candidates")
            selected = min(
                measured_feasible,
                key=lambda candidate: (
                    chosen_model.score(
                        normalized_by_cell[cell.cell_id][candidate.path_id],
                        weights,
                    ),
                    candidate.path_id,
                ),
            )
            reference_runtime = medians[(cell.cell_id, greedy.path_id)]
            selected_runtime = medians[(cell.cell_id, selected.path_id)]
            selected_path_ids.append((cell.cell_id, selected.path_id))
            speedups.append((cell.cell_id, reference_runtime / selected_runtime))
        speedup_values = tuple(value for _, value in speedups)
        candidate_result = WeightFitResult(
            weights=weights,
            model=chosen_model,
            selected_path_ids=tuple(selected_path_ids),
            cell_speedups=tuple(speedups),
            geometric_mean_speedup=geometric_mean(speedup_values),
            minimum_cell_speedup=min(speedup_values),
            improved_cell_count=sum(value > 1.0 for value in speedup_values),
            evaluated_weight_vectors=len(search_vectors),
            seed=seed,
            random_sample_count=random_sample_count,
        )
        if evaluation_callback is not None:
            evaluation_callback(candidate_result)
        if best is None or _fit_objective(candidate_result) > _fit_objective(best):
            best = candidate_result
    assert best is not None
    return best


@dataclass(frozen=True, slots=True)
class ScoreExplanationRow:
    feature: str
    raw: float
    normalized: float
    weight: float
    contribution: float

    def as_mapping(self) -> dict[str, float | str]:
        return {
            "feature": self.feature,
            "raw": self.raw,
            "normalized": self.normalized,
            "weight": self.weight,
            "contribution": self.contribution,
        }


def explain_score(
    raw: RawFeatureVector,
    greedy: RawFeatureVector,
    weights: WeightVector,
    *,
    model: FeatureModelDecision | None = None,
) -> tuple[ScoreExplanationRow, ...]:
    """Return one transparent raw/normalized/weight/contribution row per term."""

    normalized = normalize_features(raw, greedy)
    rows = []
    for feature, raw_value, normalized_value, weight in zip(
        FEATURE_NAMES,
        raw.as_tuple(),
        normalized.values,
        weights.as_tuple(),
        strict=True,
    ):
        rows.append(
            ScoreExplanationRow(
                feature=feature,
                raw=raw_value,
                normalized=normalized_value,
                weight=weight,
                contribution=normalized_value * weight,
            )
        )
    if model is not None:
        expected = model.score(normalized, weights)
        contribution_sum = sum(row.contribution for row in rows)
        if not math.isclose(expected, contribution_sum, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("grouped score explanation is inconsistent with weights")
    return tuple(rows)


__all__ = [
    "COST_MODEL_ID",
    "FEATURE_NAMES",
    "GROUP_FEATURE_NAMES",
    "SIX_TERM_FEATURE_MODEL",
    "GROUPED_FEATURE_MODEL",
    "FEATURE_DEPENDENCY_METADATA",
    "FeatureDependency",
    "RawFeatureVector",
    "NormalizedFeatureVector",
    "WeightVector",
    "PlanFeatureFacts",
    "ConventionalPathFeatures",
    "PathCandidate",
    "FeatureModelDecision",
    "TrainingCell",
    "RuntimeMeasurement",
    "MeasuredRuntime",
    "WeightFitResult",
    "ScoreExplanationRow",
    "feature_dependency_metadata",
    "extract_plan_features",
    "extract_conventional_features",
    "canonicalize_path",
    "path_id",
    "normalize_features",
    "choose_feature_model",
    "explicit_feature_model",
    "equal_model_weights",
    "score_normalized",
    "score_features",
    "select_best_candidate",
    "select_calibration_candidates",
    "geometric_mean",
    "fit_weights",
    "explain_score",
]
