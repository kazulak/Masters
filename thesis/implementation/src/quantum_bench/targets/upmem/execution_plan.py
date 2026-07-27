"""Small, immutable execution contracts for UPMEM.

The plans describe execution choices only.  They are deliberately separate
from :class:`TaskGraph`: changing a kernel or placement choice must not change
the circuit, tensor-network, or contraction-plan identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from quantum_bench.core.records import JsonDict, TaskGraph, to_jsonable
from quantum_bench.tn.execution_bundle import canonical_hash, with_execution_identity


UPMEM_EXECUTION_PLAN_SCHEMA_VERSION = "upmem_execution_plan_v1"
UPMEM_KERNEL_PLAN_SCHEMA_VERSION = "upmem_kernel_plan_v1"
UPMEM_PLACEMENT_PLAN_SCHEMA_VERSION = "upmem_placement_plan_v1"
UPMEM_COMMUNICATION_PLAN_SCHEMA_VERSION = "upmem_communication_plan_v1"
UPMEM_NUMERIC_PLAN_SCHEMA_VERSION = "upmem_numeric_plan_v1"
UPMEM_SCHEDULE_PLAN_SCHEMA_VERSION = "upmem_schedule_plan_v1"

ValidationStatus = Literal["not_run", "passed", "failed", "blocked"]
_VALIDATION_STATUSES = frozenset({"not_run", "passed", "failed", "blocked"})


@dataclass(frozen=True)
class DpuResourceContext:
    """Requested resources plus optional verified runtime allocation facts."""

    requested_dpu_count: int = 1
    requested_tasklets_per_dpu: int = 1
    allocated_dpu_count: int | None = None
    allocated_tasklets_per_dpu: int | None = None
    allocation_status: Literal["not_run", "verified"] = "not_run"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        requested = (
            ("requested_dpu_count", self.requested_dpu_count),
            ("requested_tasklets_per_dpu", self.requested_tasklets_per_dpu),
        )
        for name, value in requested:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.allocation_status not in {"not_run", "verified"}:
            raise ValueError(f"unsupported allocation_status: {self.allocation_status!r}")

        allocated = (
            ("allocated_dpu_count", self.allocated_dpu_count),
            ("allocated_tasklets_per_dpu", self.allocated_tasklets_per_dpu),
        )
        if self.allocation_status == "not_run":
            if any(value is not None for _, value in allocated):
                raise ValueError(
                    "allocated counts must be absent when allocation_status='not_run'"
                )
            return

        for name, value in allocated:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"{name} must be a positive integer when allocation_status='verified'"
                )
        if self.allocated_dpu_count != self.requested_dpu_count:
            raise ValueError(
                "allocated_dpu_count must equal requested_dpu_count "
                "when allocation_status='verified'"
            )
        if self.allocated_tasklets_per_dpu != self.requested_tasklets_per_dpu:
            raise ValueError(
                "allocated_tasklets_per_dpu must equal requested_tasklets_per_dpu "
                "when allocation_status='verified'"
            )

@dataclass(frozen=True)
class UpmemKernelPlan:
    schema_version: str = UPMEM_KERNEL_PLAN_SCHEMA_VERSION
    provider_id: str = "upmem_sdk_simulator"
    kernel_id: str = "generic_loop_v1"
    kernel_version: str = "generic_loop_v1"
    implementation: str = "explicit_sdk"
    resident: bool = False

@dataclass(frozen=True)
class UpmemPlacementPlan:
    schema_version: str = UPMEM_PLACEMENT_PLAN_SCHEMA_VERSION
    resources: DpuResourceContext = field(default_factory=DpuResourceContext)
    assignment_policy: str = "single_dpu"
    topology: str = "one_rank_one_dpu"

    def __post_init__(self) -> None:
        if not isinstance(self.resources, DpuResourceContext):
            raise TypeError("resources must be an instance of DpuResourceContext")

@dataclass(frozen=True)
class UpmemCommunicationPlan:
    schema_version: str = UPMEM_COMMUNICATION_PLAN_SCHEMA_VERSION
    host_to_dpu: str = "explicit_sdk"
    dpu_to_host: str = "explicit_sdk"
    intermediate_transport: str = "mram_resident"
    reduction: str = "host"
    collective_provider: str | None = None

@dataclass(frozen=True)
class UpmemNumericPlan:
    schema_version: str = UPMEM_NUMERIC_PLAN_SCHEMA_VERSION
    input_dtype: str = "float32"
    accumulator_dtype: str = "float32"
    output_dtype: str = "float32"
    quantization: str = "none"
    complex_policy: str = "split_real_imag"
    full_precision_reference: str = "complex128_cpu"

@dataclass(frozen=True)
class UpmemSchedulePlan:
    schema_version: str = UPMEM_SCHEDULE_PLAN_SCHEMA_VERSION
    ordering: str = "topological_task_id"
    dependency_policy: str = "strict"
    parallelism: str = "serial"
    resident_lifetime: str = "taskgraph"

@dataclass(frozen=True)
class UpmemValidationStatuses:
    """Independent admission statuses for execution and scientific claims."""

    execution_contract_status: ValidationStatus = "not_run"
    policy_reference_status: ValidationStatus = "not_run"
    full_precision_accuracy_status: ValidationStatus = "not_run"
    scientific_validation_status: ValidationStatus = "not_run"

    def __post_init__(self) -> None:
        for name in (
            "execution_contract_status",
            "policy_reference_status",
            "full_precision_accuracy_status",
            "scientific_validation_status",
        ):
            value = getattr(self, name)
            if value not in _VALIDATION_STATUSES:
                raise ValueError(f"unsupported {name}: {value!r}")

    @classmethod
    def derive(
        cls,
        *,
        execution_contract_status: str | bool | None = "not_run",
        policy_reference_status: str | bool | None = "not_run",
        full_precision_accuracy_status: str | bool | None = "not_run",
        scientific_validation_status: str | bool | None = None,
    ) -> "UpmemValidationStatuses":
        statuses = {
            "execution_contract_status": _normalize_status(execution_contract_status),
            "policy_reference_status": _normalize_status(policy_reference_status),
            "full_precision_accuracy_status": _normalize_status(full_precision_accuracy_status),
        }
        if scientific_validation_status is None:
            values = tuple(statuses.values())
            scientific = "failed" if "failed" in values else (
                "passed" if all(value == "passed" for value in values) else "not_run"
            )
        else:
            scientific = _normalize_status(scientific_validation_status)
            if scientific == "passed" and any(value != "passed" for value in statuses.values()):
                raise ValueError(
                    "scientific_validation_status='passed' requires every validation status to be 'passed'"
                )
        return cls(**statuses, scientific_validation_status=scientific)  # type: ignore[arg-type]

    @classmethod
    def from_checks(
        cls,
        *,
        execution_contract: bool | None = None,
        policy_reference: bool | None = None,
        full_precision_accuracy: bool | None = None,
        scientific_validation: bool | None = None,
    ) -> "UpmemValidationStatuses":
        return cls.derive(
            execution_contract_status=execution_contract,
            policy_reference_status=policy_reference,
            full_precision_accuracy_status=full_precision_accuracy,
            scientific_validation_status=scientific_validation,
        )


@dataclass(frozen=True)
class UpmemExecutionPlan:
    schema_version: str = UPMEM_EXECUTION_PLAN_SCHEMA_VERSION
    kernel: UpmemKernelPlan = field(default_factory=UpmemKernelPlan)
    placement: UpmemPlacementPlan = field(default_factory=UpmemPlacementPlan)
    communication: UpmemCommunicationPlan = field(default_factory=UpmemCommunicationPlan)
    numeric: UpmemNumericPlan = field(default_factory=UpmemNumericPlan)
    schedule: UpmemSchedulePlan = field(default_factory=UpmemSchedulePlan)
    validation: UpmemValidationStatuses = field(default_factory=UpmemValidationStatuses)
    circuit_semantics_hash: str = ""
    tensor_network_hash: str = ""
    contraction_plan_hash: str = ""

    def __post_init__(self) -> None:
        expected_types = (
            ("kernel", UpmemKernelPlan),
            ("placement", UpmemPlacementPlan),
            ("communication", UpmemCommunicationPlan),
            ("numeric", UpmemNumericPlan),
            ("schedule", UpmemSchedulePlan),
            ("validation", UpmemValidationStatuses),
        )
        for name, expected_type in expected_types:
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be an instance of {expected_type.__name__}")

    @classmethod
    def for_task_graph(
        cls,
        graph: TaskGraph,
        *,
        kernel: UpmemKernelPlan | None = None,
        placement: UpmemPlacementPlan | None = None,
        communication: UpmemCommunicationPlan | None = None,
        numeric: UpmemNumericPlan | None = None,
        schedule: UpmemSchedulePlan | None = None,
        validation: UpmemValidationStatuses | None = None,
    ) -> "UpmemExecutionPlan":
        identified = with_execution_identity(graph)
        return cls(
            kernel=kernel if kernel is not None else UpmemKernelPlan(),
            placement=placement if placement is not None else UpmemPlacementPlan(),
            communication=communication if communication is not None else UpmemCommunicationPlan(),
            numeric=numeric if numeric is not None else UpmemNumericPlan(),
            schedule=schedule if schedule is not None else UpmemSchedulePlan(),
            validation=validation if validation is not None else UpmemValidationStatuses(),
            circuit_semantics_hash=identified.circuit_semantics_hash,
            tensor_network_hash=identified.tensor_network_hash,
            contraction_plan_hash=identified.contraction_plan_hash,
        )

    @property
    def execution_plan_hash(self) -> str:
        return execution_plan_hash(self)

    def as_hash_payload(self) -> JsonDict:
        return to_jsonable(
            {
                "schema_version": self.schema_version,
                "kernel": self.kernel,
                "placement": self.placement,
                "communication": self.communication,
                "numeric": self.numeric,
                "schedule": self.schedule,
                "task_graph_identity": {
                    "circuit_semantics_hash": self.circuit_semantics_hash,
                    "tensor_network_hash": self.tensor_network_hash,
                    "contraction_plan_hash": self.contraction_plan_hash,
                },
            }
        )

    def to_json_dict(self) -> JsonDict:
        return self.as_hash_payload() | {
            "validation": to_jsonable(self.validation),
            "execution_plan_hash": self.execution_plan_hash,
        }


def execution_plan_hash(plan: UpmemExecutionPlan) -> str:
    """Hash choices canonically without changing any TaskGraph identity."""

    return canonical_hash(plan.as_hash_payload())


def validate_execution_plan_graph_identity(
    plan: UpmemExecutionPlan, graph: TaskGraph
) -> None:
    """Reject stale or mismatched graph identities without mutating either object."""

    identified = with_execution_identity(graph)
    expected = {
        "circuit_semantics_hash": identified.circuit_semantics_hash,
        "tensor_network_hash": identified.tensor_network_hash,
        "contraction_plan_hash": identified.contraction_plan_hash,
    }
    for name, value in expected.items():
        if getattr(plan, name) != value:
            raise ValueError(f"UpmemExecutionPlan {name} does not match the TaskGraph")


def _normalize_status(value: str | bool | None) -> ValidationStatus:
    if value is None:
        return "not_run"
    if isinstance(value, bool):
        return "passed" if value else "failed"
    if value not in _VALIDATION_STATUSES:
        raise ValueError(f"unsupported validation status: {value!r}")
    return value  # type: ignore[return-value]


__all__ = [
    "DpuResourceContext",
    "UPMEM_COMMUNICATION_PLAN_SCHEMA_VERSION",
    "UPMEM_EXECUTION_PLAN_SCHEMA_VERSION",
    "UPMEM_KERNEL_PLAN_SCHEMA_VERSION",
    "UPMEM_NUMERIC_PLAN_SCHEMA_VERSION",
    "UPMEM_PLACEMENT_PLAN_SCHEMA_VERSION",
    "UPMEM_SCHEDULE_PLAN_SCHEMA_VERSION",
    "UpmemCommunicationPlan",
    "UpmemExecutionPlan",
    "UpmemKernelPlan",
    "UpmemNumericPlan",
    "UpmemPlacementPlan",
    "UpmemSchedulePlan",
    "UpmemValidationStatuses",
    "execution_plan_hash",
    "validate_execution_plan_graph_identity",
]
