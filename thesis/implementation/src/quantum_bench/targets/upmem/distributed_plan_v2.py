"""Canonical Python model for a distributed single contraction.

This module is deliberately separate from the resident ``upmem_execution_plan_v1``
formats.  It describes one logical contraction whose work may be split across
DPUs; it does not describe a resident package, a native ABI, or a CLI request.

Communication byte counts are logical payload bytes, excluding transport
padding and protocol headers.  Host-mediated reduction counts one complete
output buffer per participating DPU; PID-Comm allreduce counts one output
buffer as the per-participant collective payload, not aggregate wire traffic;
``none`` is always zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
from typing import Any, Mapping


DISTRIBUTED_PLAN_V2_SCHEMA_VERSION = "distributed_single_contraction_plan_v2"
OUTPUT_TILE = "output_tile"
CONTRACTED_PARTIAL_SUM = "contracted_partial_sum"

COMMUNICATION_NONE = "none"
COMMUNICATION_HOST_MEDIATED_SUM_V1 = "host_mediated_sum_v1"
COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1 = "pidcomm_allreduce_int32_v1"

DTYPE_FLOAT32 = "float32"
DTYPE_INT32 = "int32"
SYNCHRONIZATION_NONE = "none"
SYNCHRONIZATION_HOST_BARRIER_V1 = "host_barrier_v1"
SYNCHRONIZATION_PIDCOMM_COLLECTIVE_BARRIER_V1 = "pidcomm_collective_barrier_v1"

OUTPUT_OWNERSHIP_EXCLUSIVE = "exclusive"
OUTPUT_OWNERSHIP_SHARED_PARTIAL_SUM = "shared_partial_sum"
SUPPORTED_BUILDER_DPU_COUNTS = (1, 2, 4)
_DTYPE_BYTES = {DTYPE_FLOAT32: 4, DTYPE_INT32: 4}
UPXDPV2_MAGIC = b"UPXDPV2\x00"
UPXDPV2_VERSION = 2
UPXDPV2_HEADER_FORMAT = "<8s15I32s32s"
UPXDPV2_RECORD_FORMAT = "<8I"
UPXDPV2_HEADER_BYTES = struct.calcsize(UPXDPV2_HEADER_FORMAT)
UPXDPV2_RECORD_BYTES = struct.calcsize(UPXDPV2_RECORD_FORMAT)
UPXDPV2_PROVIDER_COUNT = 1
UPXDPV2_PARTITION_OUTPUT = 1
UPXDPV2_PARTITION_CONTRACTED = 2
NATIVE_M5_SUPPORTED_COMBINATIONS = frozenset(
    {
        (COMMUNICATION_NONE, OUTPUT_TILE, DTYPE_FLOAT32),
        (COMMUNICATION_HOST_MEDIATED_SUM_V1, CONTRACTED_PARTIAL_SUM, DTYPE_FLOAT32),
        (COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1, CONTRACTED_PARTIAL_SUM, DTYPE_INT32),
    }
)


@dataclass(frozen=True)
class DistributedWorkUnitV2:
    """One DPU assignment for one logical contraction."""

    logical_operation_id: str
    logical_task_id: str
    dpu_id: int
    partition_kind: str
    output_offset: int
    output_elements: int
    contracted_offset: int
    contracted_elements: int
    output_ownership: str


@dataclass(frozen=True)
class DistributedCommunicationV2:
    """Communication contract for the work-unit group.

    ``predicted_bytes`` is an exact, non-negative logical payload count.  It
    is not a count of padded DMA bytes or collective control traffic.
    """

    provider: str
    dtype: str
    participants: tuple[int, ...]
    predicted_bytes: int
    synchronization: str


@dataclass(frozen=True)
class DistributedSingleContractionPlanV2:
    """Versioned plan with an explicit dense local DPU count for one contraction."""

    contraction_plan_hash: str
    total_output_elements: int
    total_contracted_elements: int
    dpu_count: int
    work_units: tuple[DistributedWorkUnitV2, ...]
    communication: DistributedCommunicationV2
    schema_version: str = DISTRIBUTED_PLAN_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_distributed_plan_v2(self)

    @property
    def execution_plan_hash(self) -> str:
        """Hash placement and execution choices without rewriting the caller hash."""

        return hashlib.sha256(_canonical_json(_hash_payload(self))).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        payload = _hash_payload(self)
        payload["execution_plan_hash"] = self.execution_plan_hash
        return payload

    def to_json_bytes(self) -> bytes:
        """Return deterministic UTF-8 JSON for the v2 plan."""

        return _canonical_json(self.to_json_dict())

    def native_m5_capability(self) -> NativeM5Capability:
        """Describe native M5 admission without claiming v2 execution support."""

        return native_m5_capability(
            self.communication.provider,
            self.work_units[0].partition_kind,
            self.communication.dtype,
        )


@dataclass(frozen=True)
class NativeM5Capability:
    """Current native M5 contract admission, separate from execution."""

    provider: str
    partition_kind: str
    dtype: str
    supported: bool
    executable: bool
    reason: str


def native_m5_capability(
    provider: str, partition_kind: str, dtype: str
) -> NativeM5Capability:
    """Return native M5 support without claiming a v2 adapter exists."""

    combination = (provider, partition_kind, dtype)
    if combination in NATIVE_M5_SUPPORTED_COMBINATIONS:
        executable = combination in {
            (COMMUNICATION_NONE, OUTPUT_TILE, DTYPE_FLOAT32),
            (COMMUNICATION_HOST_MEDIATED_SUM_V1, CONTRACTED_PARTIAL_SUM, DTYPE_FLOAT32),
        }
        return NativeM5Capability(
            provider=provider,
            partition_kind=partition_kind,
            dtype=dtype,
            supported=True,
            executable=executable,
            reason=(
                "native M5.1 UPXDPV2 output-partition adapter is available"
                if combination == (COMMUNICATION_NONE, OUTPUT_TILE, DTYPE_FLOAT32)
                else "native M5.2 UPXDPV2 contracted-partition host reducer is available"
                if executable
                else "native M5 contract admitted; this provider is not executable"
            ),
        )
    return NativeM5Capability(
        provider=provider,
        partition_kind=partition_kind,
        dtype=dtype,
        supported=False,
        executable=False,
        reason="provider/partition/dtype combination is outside the native M5 capability",
    )


def validate_distributed_plan_v2(plan: DistributedSingleContractionPlanV2) -> None:
    """Validate all structural, partition, and communication invariants."""

    if not isinstance(plan, DistributedSingleContractionPlanV2):
        raise TypeError("plan must be a DistributedSingleContractionPlanV2")
    if plan.schema_version != DISTRIBUTED_PLAN_V2_SCHEMA_VERSION:
        raise ValueError("unsupported distributed single-contraction plan schema")
    if (
        not isinstance(plan.contraction_plan_hash, str)
        or not plan.contraction_plan_hash
    ):
        raise ValueError(
            "contraction_plan_hash must be a non-empty caller-owned string"
        )
    _positive_int("total_output_elements", plan.total_output_elements)
    _positive_int("total_contracted_elements", plan.total_contracted_elements)
    _validate_builder_dpu_count(plan.dpu_count)
    if not plan.work_units:
        raise ValueError("distributed plan requires at least one work unit")
    if len(plan.work_units) != plan.dpu_count:
        raise ValueError("dpu_count must equal the number of work units")

    dpu_ids: list[int] = []
    identities: set[tuple[str, str]] = set()
    partition_kinds: set[str] = set()
    for unit in plan.work_units:
        if not isinstance(unit, DistributedWorkUnitV2):
            raise TypeError("work_units must contain DistributedWorkUnitV2 values")
        if (
            not isinstance(unit.logical_operation_id, str)
            or not unit.logical_operation_id
        ):
            raise ValueError("logical_operation_id must be a non-empty string")
        if not isinstance(unit.logical_task_id, str) or not unit.logical_task_id:
            raise ValueError("logical_task_id must be a non-empty string")
        _non_negative_int("dpu_id", unit.dpu_id)
        if unit.dpu_id in dpu_ids:
            raise ValueError("duplicate DPU assignment")
        dpu_ids.append(unit.dpu_id)
        identities.add((unit.logical_operation_id, unit.logical_task_id))
        if unit.partition_kind not in {OUTPUT_TILE, CONTRACTED_PARTIAL_SUM}:
            raise ValueError(f"unsupported partition_kind: {unit.partition_kind!r}")
        partition_kinds.add(unit.partition_kind)
        _range_within(
            "output",
            unit.output_offset,
            unit.output_elements,
            plan.total_output_elements,
        )
        _range_within(
            "contracted",
            unit.contracted_offset,
            unit.contracted_elements,
            plan.total_contracted_elements,
        )

        if unit.partition_kind == OUTPUT_TILE:
            if unit.output_ownership != OUTPUT_OWNERSHIP_EXCLUSIVE:
                raise ValueError(
                    "output_tile work units require exclusive output ownership"
                )
            if (
                unit.contracted_offset != 0
                or unit.contracted_elements != plan.total_contracted_elements
            ):
                raise ValueError(
                    "output_tile work units must cover the full contracted range"
                )
        elif unit.output_ownership != OUTPUT_OWNERSHIP_SHARED_PARTIAL_SUM:
            raise ValueError(
                "contracted_partial_sum work units require shared_partial_sum output ownership"
            )

    if len(identities) != 1:
        raise ValueError("all work units must identify one logical operation and task")
    if len(partition_kinds) != 1:
        raise ValueError("mixed partition kinds are not allowed")
    if tuple(sorted(dpu_ids)) != tuple(range(plan.dpu_count)):
        raise ValueError("DPU IDs must be dense local IDs from 0 through dpu_count - 1")

    kind = next(iter(partition_kinds))
    if kind == OUTPUT_TILE:
        _validate_complete_ranges(
            plan.work_units,
            "output_offset",
            "output_elements",
            plan.total_output_elements,
        )
    else:
        for unit in plan.work_units:
            if (
                unit.output_offset != 0
                or unit.output_elements != plan.total_output_elements
            ):
                raise ValueError(
                    "contracted_partial_sum work units must cover the full output range"
                )
        _validate_complete_ranges(
            plan.work_units,
            "contracted_offset",
            "contracted_elements",
            plan.total_contracted_elements,
        )

    _validate_communication(plan, tuple(sorted(dpu_ids)), kind)


def build_output_tile_plan_v2(
    *,
    logical_operation_id: str,
    logical_task_id: str,
    total_output_elements: int,
    total_contracted_elements: int,
    contraction_plan_hash: str,
    dtype: str = DTYPE_FLOAT32,
    dpu_count: int = 1,
) -> DistributedSingleContractionPlanV2:
    """Build a deterministic output partition for one, two, or four DPUs."""

    _validate_builder_dpu_count(dpu_count)
    if dtype != DTYPE_FLOAT32:
        raise ValueError("output_tile v1 native capability requires real float32")
    units = tuple(
        DistributedWorkUnitV2(
            logical_operation_id=logical_operation_id,
            logical_task_id=logical_task_id,
            dpu_id=dpu_id,
            partition_kind=OUTPUT_TILE,
            output_offset=offset,
            output_elements=elements,
            contracted_offset=0,
            contracted_elements=total_contracted_elements,
            output_ownership=OUTPUT_OWNERSHIP_EXCLUSIVE,
        )
        for dpu_id, (offset, elements) in enumerate(
            _partition_ranges(total_output_elements, dpu_count)
        )
    )
    return DistributedSingleContractionPlanV2(
        contraction_plan_hash=contraction_plan_hash,
        total_output_elements=total_output_elements,
        total_contracted_elements=total_contracted_elements,
        dpu_count=dpu_count,
        work_units=units,
        communication=DistributedCommunicationV2(
            provider=COMMUNICATION_NONE,
            dtype=dtype,
            participants=(),
            predicted_bytes=0,
            synchronization=SYNCHRONIZATION_NONE,
        ),
    )


def build_contracted_partial_sum_plan_v2(
    *,
    logical_operation_id: str,
    logical_task_id: str,
    total_output_elements: int,
    total_contracted_elements: int,
    contraction_plan_hash: str,
    dtype: str = DTYPE_FLOAT32,
    dpu_count: int = 1,
    communication_provider: str = COMMUNICATION_HOST_MEDIATED_SUM_V1,
) -> DistributedSingleContractionPlanV2:
    """Build a deterministic contracted-index partition with an explicit reducer."""

    _validate_builder_dpu_count(dpu_count)
    if communication_provider == COMMUNICATION_NONE and dpu_count > 1:
        raise ValueError(
            "multi-DPU contracted_partial_sum requires an explicit reducer"
        )
    if (
        communication_provider == COMMUNICATION_HOST_MEDIATED_SUM_V1
        and dtype != DTYPE_FLOAT32
    ):
        raise ValueError("host-mediated partial reduction requires float32")
    if (
        communication_provider == COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1
        and dtype != DTYPE_INT32
    ):
        raise ValueError("PID-Comm allreduce v1 requires int32")
    if communication_provider == COMMUNICATION_NONE and dtype != DTYPE_FLOAT32:
        raise ValueError("single-DPU contracted partial sums require float32")
    units = tuple(
        DistributedWorkUnitV2(
            logical_operation_id=logical_operation_id,
            logical_task_id=logical_task_id,
            dpu_id=dpu_id,
            partition_kind=CONTRACTED_PARTIAL_SUM,
            output_offset=0,
            output_elements=total_output_elements,
            contracted_offset=offset,
            contracted_elements=elements,
            output_ownership=OUTPUT_OWNERSHIP_SHARED_PARTIAL_SUM,
        )
        for dpu_id, (offset, elements) in enumerate(
            _partition_ranges(total_contracted_elements, dpu_count)
        )
    )
    participants = (
        tuple(range(dpu_count)) if communication_provider != COMMUNICATION_NONE else ()
    )
    output_bytes = total_output_elements * _DTYPE_BYTES[dtype]
    predicted_bytes = (
        output_bytes * dpu_count
        if communication_provider == COMMUNICATION_HOST_MEDIATED_SUM_V1
        else output_bytes
        if communication_provider == COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1
        else 0
    )
    synchronization = {
        COMMUNICATION_NONE: SYNCHRONIZATION_NONE,
        COMMUNICATION_HOST_MEDIATED_SUM_V1: SYNCHRONIZATION_HOST_BARRIER_V1,
        COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1: SYNCHRONIZATION_PIDCOMM_COLLECTIVE_BARRIER_V1,
    }.get(communication_provider, SYNCHRONIZATION_NONE)
    return DistributedSingleContractionPlanV2(
        contraction_plan_hash=contraction_plan_hash,
        total_output_elements=total_output_elements,
        total_contracted_elements=total_contracted_elements,
        dpu_count=dpu_count,
        work_units=units,
        communication=DistributedCommunicationV2(
            provider=communication_provider,
            dtype=dtype,
            participants=participants,
            predicted_bytes=predicted_bytes,
            synchronization=synchronization,
        ),
    )


def serialize_distributed_plan_v2(plan: DistributedSingleContractionPlanV2) -> bytes:
    """Serialize a validated v2 plan deterministically."""

    validate_distributed_plan_v2(plan)
    return plan.to_json_bytes()


def serialize_upxdpv2_output_partition(
    plan: DistributedSingleContractionPlanV2,
    *,
    package_bytes: bytes,
    operation_bytes: bytes,
    output_slot: int,
    package_operation_index: int = 0,
    operation_id: int = 0,
    tasklets_per_dpu: int = 1,
) -> bytes:
    """Serialize the exact packed output-partition sidecar consumed by native v2."""

    validate_distributed_plan_v2(plan)
    if plan.work_units[0].partition_kind != OUTPUT_TILE:
        raise ValueError("UPXDPV2 output serializer requires output_tile")
    return serialize_upxdpv2(
        plan,
        package_bytes=package_bytes,
        operation_bytes=operation_bytes,
        output_slot=output_slot,
        package_operation_index=package_operation_index,
        operation_id=operation_id,
        tasklets_per_dpu=tasklets_per_dpu,
    )


def serialize_upxdpv2_contracted_partition(
    plan: DistributedSingleContractionPlanV2,
    *,
    package_bytes: bytes,
    operation_bytes: bytes,
    output_slot: int,
    package_operation_index: int = 0,
    operation_id: int = 0,
    tasklets_per_dpu: int = 1,
) -> bytes:
    """Serialize the exact packed contracted-partition sidecar consumed by native v2."""

    validate_distributed_plan_v2(plan)
    if plan.work_units[0].partition_kind != CONTRACTED_PARTIAL_SUM:
        raise ValueError("UPXDPV2 contracted serializer requires contracted_partial_sum")
    return serialize_upxdpv2(
        plan,
        package_bytes=package_bytes,
        operation_bytes=operation_bytes,
        output_slot=output_slot,
        package_operation_index=package_operation_index,
        operation_id=operation_id,
        tasklets_per_dpu=tasklets_per_dpu,
    )


def serialize_upxdpv2(
    plan: DistributedSingleContractionPlanV2,
    *,
    package_bytes: bytes,
    operation_bytes: bytes,
    output_slot: int,
    package_operation_index: int = 0,
    operation_id: int = 0,
    tasklets_per_dpu: int = 1,
) -> bytes:
    """Serialize a validated output or contracted partition using UPXDPV2."""

    validate_distributed_plan_v2(plan)
    capability = plan.native_m5_capability()
    if not capability.executable:
        raise ValueError("native M5.1 UPXDPV2 output partition is not executable")
    if not isinstance(package_bytes, bytes) or not package_bytes:
        raise ValueError("package_bytes must be non-empty bytes")
    if not isinstance(operation_bytes, bytes) or not operation_bytes:
        raise ValueError("operation_bytes must be non-empty bytes")
    _non_negative_int("output_slot", output_slot)
    if package_operation_index != 0 or operation_id != 0:
        raise ValueError("UPXDPV2 currently binds package operation zero")
    if tasklets_per_dpu != 1:
        raise ValueError("UPXDPV2 currently requires one tasklet per DPU")
    partition_mode = {
        OUTPUT_TILE: UPXDPV2_PARTITION_OUTPUT,
        CONTRACTED_PARTIAL_SUM: UPXDPV2_PARTITION_CONTRACTED,
    }[plan.work_units[0].partition_kind]
    header = struct.pack(
        UPXDPV2_HEADER_FORMAT,
        UPXDPV2_MAGIC,
        UPXDPV2_VERSION,
        UPXDPV2_HEADER_BYTES,
        plan.dpu_count,
        plan.dpu_count,
        tasklets_per_dpu,
        UPXDPV2_PROVIDER_COUNT,
        partition_mode,
        package_operation_index,
        operation_id,
        plan.total_output_elements,
        plan.total_contracted_elements,
        output_slot,
        UPXDPV2_RECORD_BYTES,
        0,
        0,
        hashlib.sha256(package_bytes).digest(),
        hashlib.sha256(operation_bytes).digest(),
    )
    records = b"".join(
        struct.pack(
            UPXDPV2_RECORD_FORMAT,
            package_operation_index,
            operation_id,
            partition_mode,
            unit.dpu_id,
            unit.output_offset,
            unit.output_elements,
            unit.contracted_offset,
            unit.contracted_elements,
        )
        for unit in sorted(plan.work_units, key=lambda item: item.dpu_id)
    )
    return header + records


def validate_upxdpv2_output_partition(
    sidecar_bytes: bytes,
    plan: DistributedSingleContractionPlanV2,
    *,
    package_bytes: bytes,
    operation_bytes: bytes,
    output_slot: int,
) -> dict[str, Any]:
    """Assert byte-for-byte identity with the Python-to-C UPXDPV2 bridge."""

    if plan.work_units[0].partition_kind != OUTPUT_TILE:
        raise ValueError("UPXDPV2 output validator requires output_tile")
    expected = serialize_upxdpv2_output_partition(
        plan,
        package_bytes=package_bytes,
        operation_bytes=operation_bytes,
        output_slot=output_slot,
    )
    if sidecar_bytes != expected:
        raise ValueError("UPXDPV2 sidecar bytes differ from the deterministic C contract")
    if len(sidecar_bytes) != UPXDPV2_HEADER_BYTES + plan.dpu_count * UPXDPV2_RECORD_BYTES:
        raise ValueError("UPXDPV2 sidecar length differs from header and record sizes")
    header = struct.unpack_from(UPXDPV2_HEADER_FORMAT, sidecar_bytes)
    if (
        header[0] != UPXDPV2_MAGIC
        or header[1] != UPXDPV2_VERSION
        or header[2] != UPXDPV2_HEADER_BYTES
        or header[13] != UPXDPV2_RECORD_BYTES
    ):
        raise ValueError("UPXDPV2 version or record size is invalid")
    return {
        "magic": UPXDPV2_MAGIC.rstrip(b"\x00").decode("ascii"),
        "version": UPXDPV2_VERSION,
        "header_bytes": UPXDPV2_HEADER_BYTES,
        "record_bytes": UPXDPV2_RECORD_BYTES,
        "package_sha256": hashlib.sha256(package_bytes).hexdigest(),
        "operation_sha256": hashlib.sha256(operation_bytes).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "dpu_count": plan.dpu_count,
    }


def validate_upxdpv2_contracted_partition(
    sidecar_bytes: bytes,
    plan: DistributedSingleContractionPlanV2,
    *,
    package_bytes: bytes,
    operation_bytes: bytes,
    output_slot: int,
) -> dict[str, Any]:
    """Assert byte-for-byte identity for a contracted-axis UPXDPV2 sidecar."""

    if plan.work_units[0].partition_kind != CONTRACTED_PARTIAL_SUM:
        raise ValueError("UPXDPV2 contracted validator requires contracted_partial_sum")
    expected = serialize_upxdpv2_contracted_partition(
        plan,
        package_bytes=package_bytes,
        operation_bytes=operation_bytes,
        output_slot=output_slot,
    )
    if sidecar_bytes != expected:
        raise ValueError("UPXDPV2 contracted sidecar bytes differ from the deterministic contract")
    header = struct.unpack_from(UPXDPV2_HEADER_FORMAT, sidecar_bytes)
    if header[7] != UPXDPV2_PARTITION_CONTRACTED:
        raise ValueError("UPXDPV2 contracted partition mode is invalid")
    return {
        "magic": UPXDPV2_MAGIC.rstrip(b"\x00").decode("ascii"),
        "version": UPXDPV2_VERSION,
        "header_bytes": UPXDPV2_HEADER_BYTES,
        "record_bytes": UPXDPV2_RECORD_BYTES,
        "package_sha256": hashlib.sha256(package_bytes).hexdigest(),
        "operation_sha256": hashlib.sha256(operation_bytes).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "dpu_count": plan.dpu_count,
    }


serialize_native_output_partition_v2 = serialize_upxdpv2_output_partition


def parse_distributed_plan_v2(
    value: bytes | str | Mapping[str, Any],
) -> DistributedSingleContractionPlanV2:
    """Parse and validate deterministic JSON emitted by this module."""

    if isinstance(value, Mapping):
        payload: Mapping[str, Any] = value
    else:
        try:
            decoded = value.decode("utf-8") if isinstance(value, bytes) else value
            if not isinstance(decoded, str):
                raise TypeError("plan JSON must be bytes or text")
            payload = json.loads(decoded)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid distributed plan JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("distributed plan JSON must be an object")
    if payload.get("schema_version") != DISTRIBUTED_PLAN_V2_SCHEMA_VERSION:
        raise ValueError("invalid distributed plan v2 schema")
    raw_units = payload.get("work_units")
    raw_communication = payload.get("communication")
    if not isinstance(raw_units, list) or not isinstance(raw_communication, Mapping):
        raise ValueError("distributed plan work_units or communication is invalid")
    try:
        plan = DistributedSingleContractionPlanV2(
            contraction_plan_hash=_required_text(payload, "contraction_plan_hash"),
            total_output_elements=_required_int(payload, "total_output_elements"),
            total_contracted_elements=_required_int(
                payload, "total_contracted_elements"
            ),
            dpu_count=_required_int(payload, "dpu_count"),
            work_units=tuple(DistributedWorkUnitV2(**item) for item in raw_units),
            communication=DistributedCommunicationV2(
                provider=_required_text(raw_communication, "provider"),
                dtype=_required_text(raw_communication, "dtype"),
                participants=tuple(raw_communication["participants"]),
                predicted_bytes=_required_int(raw_communication, "predicted_bytes"),
                synchronization=_required_text(raw_communication, "synchronization"),
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid distributed plan v2 fields") from exc
    if payload.get("execution_plan_hash") != plan.execution_plan_hash:
        raise ValueError("distributed execution plan hash mismatch")
    return plan


def _validate_communication(
    plan: DistributedSingleContractionPlanV2,
    dpu_ids: tuple[int, ...],
    partition_kind: str,
) -> None:
    communication = plan.communication
    if not isinstance(communication, DistributedCommunicationV2):
        raise TypeError("communication must be a DistributedCommunicationV2")
    if communication.provider not in {
        COMMUNICATION_NONE,
        COMMUNICATION_HOST_MEDIATED_SUM_V1,
        COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1,
    }:
        raise ValueError(
            f"unsupported communication provider: {communication.provider!r}"
        )
    if communication.dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported communication dtype: {communication.dtype!r}")
    _non_negative_int("predicted_bytes", communication.predicted_bytes)
    if tuple(communication.participants) != tuple(
        sorted(set(communication.participants))
    ):
        raise ValueError("communication participants must be sorted and unique")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in communication.participants
    ):
        raise ValueError("communication participants must be non-negative DPU IDs")
    if communication.provider == COMMUNICATION_NONE:
        if communication.participants or communication.predicted_bytes != 0:
            raise ValueError(
                "communication provider 'none' cannot have participants or bytes"
            )
        if communication.synchronization != SYNCHRONIZATION_NONE:
            raise ValueError(
                "communication provider 'none' requires no synchronization"
            )
        if partition_kind == OUTPUT_TILE and communication.dtype != DTYPE_FLOAT32:
            raise ValueError("output_tile v1 native capability requires real float32")
        if partition_kind == CONTRACTED_PARTIAL_SUM:
            if len(dpu_ids) != 1:
                raise ValueError(
                    "multi-DPU contracted_partial_sum requires an explicit reducer"
                )
            if communication.dtype != DTYPE_FLOAT32:
                raise ValueError("single-DPU contracted partial sums require float32")
    else:
        if partition_kind != CONTRACTED_PARTIAL_SUM:
            raise ValueError("output_tile plans cannot use a reduction provider")
        if communication.provider == COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1:
            if len(dpu_ids) < 2 or tuple(communication.participants) != dpu_ids:
                raise ValueError("PID-Comm reduction requires at least two assigned DPUs")
            if communication.dtype != DTYPE_INT32:
                raise ValueError("PID-Comm allreduce v1 requires int32")
            if (
                communication.synchronization
                != SYNCHRONIZATION_PIDCOMM_COLLECTIVE_BARRIER_V1
            ):
                raise ValueError(
                    "PID-Comm requires its collective synchronization contract"
                )
            expected_bytes = (
                plan.total_output_elements * _DTYPE_BYTES[communication.dtype]
            )
        else:
            if len(dpu_ids) < 1 or tuple(communication.participants) != dpu_ids:
                raise ValueError("reduction participants must be all assigned DPUs")
            if communication.dtype != DTYPE_FLOAT32:
                raise ValueError("host-mediated partial reduction requires float32")
            if communication.synchronization != SYNCHRONIZATION_HOST_BARRIER_V1:
                raise ValueError("host-mediated sum requires its host barrier contract")
            expected_bytes = (
                plan.total_output_elements
                * plan.dpu_count
                * _DTYPE_BYTES[communication.dtype]
            )
        if communication.predicted_bytes != expected_bytes:
            raise ValueError("communication predicted_bytes does not match the plan")
    if partition_kind == OUTPUT_TILE and communication.provider != COMMUNICATION_NONE:
        raise ValueError("output_tile plans require communication provider 'none'")


def _validate_complete_ranges(
    units: tuple[DistributedWorkUnitV2, ...],
    offset_name: str,
    elements_name: str,
    total: int,
) -> None:
    ranges = sorted(
        (getattr(unit, offset_name), getattr(unit, elements_name)) for unit in units
    )
    cursor = 0
    for offset, elements in ranges:
        if offset != cursor:
            raise ValueError(
                "partition ranges must be complete, gap-free, and non-overlapping"
            )
        cursor += elements
    if cursor != total:
        raise ValueError("partition ranges do not cover the complete range")


def _partition_ranges(total: int, dpu_count: int) -> tuple[tuple[int, int], ...]:
    _positive_int("total_elements", total)
    if dpu_count > total:
        raise ValueError("DPU count cannot create zero-sized partition ranges")
    base, remainder = divmod(total, dpu_count)
    offset = 0
    result = []
    for index in range(dpu_count):
        elements = base + (1 if index < remainder else 0)
        result.append((offset, elements))
        offset += elements
    return tuple(result)


def _validate_builder_dpu_count(dpu_count: int) -> None:
    _positive_int("dpu_count", dpu_count)
    if dpu_count not in SUPPORTED_BUILDER_DPU_COUNTS:
        raise ValueError("builders support exactly 1, 2, or 4 DPUs")


def _range_within(name: str, offset: int, elements: int, total: int) -> None:
    _non_negative_int(f"{name}_offset", offset)
    _positive_int(f"{name}_elements", elements)
    if offset + elements > total:
        raise ValueError(f"{name} range exceeds its contraction bound")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _hash_payload(plan: DistributedSingleContractionPlanV2) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "contraction_plan_hash": plan.contraction_plan_hash,
        "total_output_elements": plan.total_output_elements,
        "total_contracted_elements": plan.total_contracted_elements,
        "dpu_count": plan.dpu_count,
        "work_units": [
            _work_unit_dict(item)
            for item in sorted(plan.work_units, key=lambda item: item.dpu_id)
        ],
        "communication": _communication_dict(plan.communication),
    }


def _work_unit_dict(unit: DistributedWorkUnitV2) -> dict[str, Any]:
    return {
        "logical_operation_id": unit.logical_operation_id,
        "logical_task_id": unit.logical_task_id,
        "dpu_id": unit.dpu_id,
        "partition_kind": unit.partition_kind,
        "output_offset": unit.output_offset,
        "output_elements": unit.output_elements,
        "contracted_offset": unit.contracted_offset,
        "contracted_elements": unit.contracted_elements,
        "output_ownership": unit.output_ownership,
    }


def _communication_dict(communication: DistributedCommunicationV2) -> dict[str, Any]:
    return {
        "provider": communication.provider,
        "dtype": communication.dtype,
        "participants": list(communication.participants),
        "predicted_bytes": communication.predicted_bytes,
        "synchronization": communication.synchronization,
    }


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(key)
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(key)
    return value
