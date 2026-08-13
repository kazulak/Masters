"""Additive v3 model for one distributed real-valued tensor contraction.

The JSON representation is useful for manifests and review.  The UPXDPV3
representation is the small native sidecar: its work records intentionally
remain the v2 ``<8I`` layout so a v2-style native record reader can inspect
the assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import struct
from typing import Any, Mapping

from quantum_bench.targets.upmem.distributed_plan_v2 import (
    CONTRACTED_PARTIAL_SUM,
    OUTPUT_OWNERSHIP_EXCLUSIVE,
    OUTPUT_OWNERSHIP_SHARED_PARTIAL_SUM,
    OUTPUT_TILE,
)


DISTRIBUTED_PLAN_V3_SCHEMA_VERSION = "distributed_single_contraction_plan_v3"
UPXDPV3_MAGIC = b"UPXDPV3\x00"
UPXDPV3_VERSION = 3
UPXDPV3_HEADER_FORMAT = "<8s16I32s32s"
UPXDPV3_RECORD_FORMAT = "<8I"
UPXDPV3_HEADER_BYTES = struct.calcsize(UPXDPV3_HEADER_FORMAT)
UPXDPV3_RECORD_BYTES = struct.calcsize(UPXDPV3_RECORD_FORMAT)

DEFAULT_MAX_DPUS_PER_RANK = 64
MIN_TASKLETS_PER_DPU = 1
MAX_TASKLETS_PER_DPU = 24
DEFAULT_TASKLETS_PER_DPU = 8
UPXDPV3_PROFILE_DEFAULT = "upmem_execution_plan_v3_distributed_partition"
UPXDPV3_MAX_ELEMENTS = 65536
NUMERIC_MODE_FLOAT32 = "float32"
NUMERIC_MODE_PER_TASK_RESIDENT_REQUANTIZE = "per_task_resident_requantize"
# ``float32_real`` was used by the first Python-only draft.  Keep accepting it
# at the boundary while emitting the native contract's canonical name.
NUMERIC_MODE_FLOAT32_REAL = NUMERIC_MODE_FLOAT32
NUMERIC_MODE_REAL_FLOAT32 = NUMERIC_MODE_FLOAT32
UPXDPV3_NUMERIC_REAL_FLOAT32 = NUMERIC_MODE_FLOAT32
UPXDPV3_NUMERIC_PER_TASK_RESIDENT_REQUANTIZE = NUMERIC_MODE_PER_TASK_RESIDENT_REQUANTIZE

UPXDPV3_PARTITION_OUTPUT = 1
UPXDPV3_PARTITION_CONTRACTED = 2
UPXDPV3_PROVIDER_COUNT = 1
UPXDPV3_NUMERIC_FLOAT32 = 0
UPXDPV3_NUMERIC_INT8_REQUANTIZE = 1
# Compatibility alias for callers of the additive draft.
UPXDPV3_NUMERIC_FLOAT32_REAL = UPXDPV3_NUMERIC_FLOAT32
UPXDPV3_PROFILE_DEFAULT_CODE = 1


class UnsupportedPartitionError(ValueError):
    """A requested partition count cannot be represented for this work domain."""

    failure_stage = "partition_unsupported"

_PARTITION_TO_CODE = {
    OUTPUT_TILE: UPXDPV3_PARTITION_OUTPUT,
    CONTRACTED_PARTIAL_SUM: UPXDPV3_PARTITION_CONTRACTED,
}
_CODE_TO_PARTITION = {value: key for key, value in _PARTITION_TO_CODE.items()}
_PROFILE_TO_CODE = {UPXDPV3_PROFILE_DEFAULT: UPXDPV3_PROFILE_DEFAULT_CODE}
_CODE_TO_PROFILE = {value: key for key, value in _PROFILE_TO_CODE.items()}
_NUMERIC_TO_CODE = {
    NUMERIC_MODE_FLOAT32: UPXDPV3_NUMERIC_FLOAT32,
    NUMERIC_MODE_PER_TASK_RESIDENT_REQUANTIZE: UPXDPV3_NUMERIC_INT8_REQUANTIZE,
    "float32_real": UPXDPV3_NUMERIC_FLOAT32,
}
_CODE_TO_NUMERIC = {
    UPXDPV3_NUMERIC_FLOAT32: NUMERIC_MODE_FLOAT32,
    UPXDPV3_NUMERIC_INT8_REQUANTIZE: NUMERIC_MODE_PER_TASK_RESIDENT_REQUANTIZE,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UINT32_MAX = (1 << 32) - 1


@dataclass(frozen=True)
class DistributedWorkUnitV3:
    """One dense local-DPU assignment, with v2-compatible semantics."""

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
class DistributedSingleContractionPlanV3:
    """Validated placement for one real single contraction."""

    logical_operation_id: str
    logical_task_id: str
    package_sha256: str
    operation_sha256: str
    operation_id: int
    output_slot: int
    total_output_elements: int
    total_contracted_elements: int
    dpu_count: int
    work_units: tuple[DistributedWorkUnitV3, ...]
    tasklets_per_dpu: int = DEFAULT_TASKLETS_PER_DPU
    partition_kind: str = OUTPUT_TILE
    numeric_mode: str = NUMERIC_MODE_FLOAT32
    max_dpus_per_rank: int = DEFAULT_MAX_DPUS_PER_RANK
    profile: str = UPXDPV3_PROFILE_DEFAULT
    package_operation_index: int = 0
    schema_version: str = DISTRIBUTED_PLAN_V3_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "numeric_mode", _canonical_numeric_mode(self.numeric_mode))
        validate_distributed_plan_v3(self)

    @property
    def execution_plan_hash(self) -> str:
        return hashlib.sha256(_canonical_json(_json_payload(self))).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        payload = _json_payload(self)
        payload["execution_plan_hash"] = self.execution_plan_hash
        return payload

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_dict())


def build_output_tile_plan_v3(
    *,
    logical_operation_id: str,
    logical_task_id: str,
    total_output_elements: int,
    total_contracted_elements: int,
    package_sha256: str,
    operation_sha256: str,
    operation_id: int = 0,
    output_slot: int = 0,
    package_operation_index: int = 0,
    dpu_count: int = 1,
    tasklets_per_dpu: int = DEFAULT_TASKLETS_PER_DPU,
    numeric_mode: str = NUMERIC_MODE_FLOAT32,
    max_dpus_per_rank: int = DEFAULT_MAX_DPUS_PER_RANK,
    profile: str = UPXDPV3_PROFILE_DEFAULT,
) -> DistributedSingleContractionPlanV3:
    """Build an exclusive output partition with exact remainder handling."""

    units = tuple(
        DistributedWorkUnitV3(
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
            _partition_ranges(total_output_elements, dpu_count, aligned_output=True)
        )
    )
    return DistributedSingleContractionPlanV3(
        logical_operation_id=logical_operation_id,
        logical_task_id=logical_task_id,
        package_sha256=package_sha256,
        operation_sha256=operation_sha256,
        operation_id=operation_id,
        output_slot=output_slot,
        total_output_elements=total_output_elements,
        total_contracted_elements=total_contracted_elements,
        dpu_count=dpu_count,
        work_units=units,
        tasklets_per_dpu=tasklets_per_dpu,
        partition_kind=OUTPUT_TILE,
        numeric_mode=numeric_mode,
        max_dpus_per_rank=max_dpus_per_rank,
        profile=profile,
        package_operation_index=package_operation_index,
    )


def build_contracted_partial_sum_plan_v3(
    *,
    logical_operation_id: str,
    logical_task_id: str,
    total_output_elements: int,
    total_contracted_elements: int,
    package_sha256: str,
    operation_sha256: str,
    operation_id: int = 0,
    output_slot: int = 0,
    package_operation_index: int = 0,
    dpu_count: int = 1,
    tasklets_per_dpu: int = DEFAULT_TASKLETS_PER_DPU,
    numeric_mode: str = NUMERIC_MODE_FLOAT32,
    max_dpus_per_rank: int = DEFAULT_MAX_DPUS_PER_RANK,
    profile: str = UPXDPV3_PROFILE_DEFAULT,
) -> DistributedSingleContractionPlanV3:
    """Build a shared-output contracted-axis partition."""

    units = tuple(
        DistributedWorkUnitV3(
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
    return DistributedSingleContractionPlanV3(
        logical_operation_id=logical_operation_id,
        logical_task_id=logical_task_id,
        package_sha256=package_sha256,
        operation_sha256=operation_sha256,
        operation_id=operation_id,
        output_slot=output_slot,
        total_output_elements=total_output_elements,
        total_contracted_elements=total_contracted_elements,
        dpu_count=dpu_count,
        work_units=units,
        tasklets_per_dpu=tasklets_per_dpu,
        partition_kind=CONTRACTED_PARTIAL_SUM,
        numeric_mode=numeric_mode,
        max_dpus_per_rank=max_dpus_per_rank,
        profile=profile,
        package_operation_index=package_operation_index,
    )


def validate_distributed_plan_v3(plan: DistributedSingleContractionPlanV3) -> None:
    if not isinstance(plan, DistributedSingleContractionPlanV3):
        raise TypeError("plan must be a DistributedSingleContractionPlanV3")
    if plan.schema_version != DISTRIBUTED_PLAN_V3_SCHEMA_VERSION:
        raise ValueError("unsupported distributed single-contraction plan schema")
    for name in ("package_sha256", "operation_sha256"):
        value = getattr(plan, name)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    _uint32("operation_id", plan.operation_id)
    if plan.operation_id != 0:
        raise ValueError("v3 one-operation plans require operation_id=0")
    if plan.package_operation_index != 0:
        raise ValueError("v3 one-operation plans require package_operation_index=0")
    _uint32("output_slot", plan.output_slot)
    _positive("total_output_elements", plan.total_output_elements)
    _positive("total_contracted_elements", plan.total_contracted_elements)
    for name in ("total_output_elements", "total_contracted_elements"):
        if getattr(plan, name) > UPXDPV3_MAX_ELEMENTS:
            raise ValueError(
                f"{name} exceeds the native v3 profile cap of {UPXDPV3_MAX_ELEMENTS} elements"
            )
    _positive("dpu_count", plan.dpu_count)
    _positive("max_dpus_per_rank", plan.max_dpus_per_rank)
    if plan.dpu_count > DEFAULT_MAX_DPUS_PER_RANK:
        raise ValueError("dpu_count exceeds the absolute 64-DPU v3 cap")
    if plan.max_dpus_per_rank > DEFAULT_MAX_DPUS_PER_RANK:
        raise ValueError("max_dpus_per_rank exceeds the absolute 64-DPU v3 cap")
    if plan.dpu_count > plan.max_dpus_per_rank:
        raise ValueError("dpu_count exceeds max_dpus_per_rank profile cap")
    if not isinstance(plan.tasklets_per_dpu, int) or isinstance(plan.tasklets_per_dpu, bool) or plan.tasklets_per_dpu < MIN_TASKLETS_PER_DPU or plan.tasklets_per_dpu > MAX_TASKLETS_PER_DPU:
        raise ValueError("tasklets_per_dpu must be an integer in the range 1..24")
    if plan.numeric_mode not in _NUMERIC_TO_CODE:
        raise ValueError(
            "unsupported numeric_mode; expected 'float32' or "
            "'per_task_resident_requantize'"
        )
    if plan.partition_kind not in _PARTITION_TO_CODE:
        raise ValueError(f"unsupported partition_kind: {plan.partition_kind!r}")
    if plan.profile not in _PROFILE_TO_CODE:
        raise ValueError(f"unsupported UPXDPV3 profile: {plan.profile!r}")
    if plan.max_dpus_per_rank > _UINT32_MAX:
        raise ValueError("max_dpus_per_rank does not fit the UPXDPV3 header")
    for name in ("logical_operation_id", "logical_task_id"):
        if not isinstance(getattr(plan, name), str) or not getattr(plan, name):
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(plan.package_operation_index, int) or isinstance(plan.package_operation_index, bool) or plan.package_operation_index < 0 or plan.package_operation_index > _UINT32_MAX:
        raise ValueError("package_operation_index must be a uint32")
    if len(plan.work_units) != plan.dpu_count:
        raise ValueError("dpu_count must equal the number of work units")

    dpu_ids: list[int] = []
    identities: set[tuple[str, str]] = set()
    for index, unit in enumerate(plan.work_units):
        if not isinstance(unit, DistributedWorkUnitV3):
            raise TypeError("work_units must contain DistributedWorkUnitV3 values")
        if not unit.logical_operation_id or not isinstance(unit.logical_operation_id, str) or not unit.logical_task_id or not isinstance(unit.logical_task_id, str):
            raise ValueError("work-unit logical IDs must be non-empty strings")
        _uint32("dpu_id", unit.dpu_id)
        if unit.dpu_id != index:
            raise ValueError("work units must be in native dense DPU record order")
        if unit.dpu_id in dpu_ids:
            raise ValueError("duplicate DPU assignment")
        dpu_ids.append(unit.dpu_id)
        identities.add((unit.logical_operation_id, unit.logical_task_id))
        if unit.partition_kind != plan.partition_kind:
            raise ValueError("mixed partition kinds are not allowed")
        if unit.partition_kind == OUTPUT_TILE:
            if unit.output_ownership != OUTPUT_OWNERSHIP_EXCLUSIVE or unit.contracted_offset != 0 or unit.contracted_elements != plan.total_contracted_elements:
                raise ValueError("output_tile work units must exclusively cover the full contracted range")
        elif unit.output_ownership != OUTPUT_OWNERSHIP_SHARED_PARTIAL_SUM or unit.output_offset != 0 or unit.output_elements != plan.total_output_elements:
            raise ValueError("contracted_partial_sum work units must share the full output range")
        _range("output", unit.output_offset, unit.output_elements, plan.total_output_elements)
        _range("contracted", unit.contracted_offset, unit.contracted_elements, plan.total_contracted_elements)
        if plan.partition_kind == OUTPUT_TILE:
            if unit.output_offset % 2:
                raise ValueError("output partition offsets must be aligned to two elements")
            if index + 1 < plan.dpu_count and unit.output_elements % 2:
                raise ValueError("non-final output partition ranges must be even")
    if len(identities) != 1 or next(iter(identities)) != (plan.logical_operation_id, plan.logical_task_id):
        raise ValueError("all work units must identify the plan's one logical operation and task")
    if tuple(sorted(dpu_ids)) != tuple(range(plan.dpu_count)):
        raise ValueError("DPU IDs must be dense local IDs from 0 through dpu_count - 1")
    if plan.partition_kind == OUTPUT_TILE:
        _validate_complete_ranges(plan.work_units, "output_offset", "output_elements", plan.total_output_elements)
    else:
        _validate_complete_ranges(plan.work_units, "contracted_offset", "contracted_elements", plan.total_contracted_elements)


def serialize_distributed_plan_v3(plan: DistributedSingleContractionPlanV3) -> bytes:
    validate_distributed_plan_v3(plan)
    return plan.to_json_bytes()


def parse_distributed_plan_v3(value: bytes | str | Mapping[str, Any]) -> DistributedSingleContractionPlanV3:
    if isinstance(value, Mapping):
        payload: Mapping[str, Any] = value
    else:
        try:
            decoded = value.decode("utf-8") if isinstance(value, bytes) else value
            payload = json.loads(decoded)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid distributed plan v3 JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("distributed plan v3 JSON must be an object")
    if payload.get("schema_version") != DISTRIBUTED_PLAN_V3_SCHEMA_VERSION:
        raise ValueError("invalid distributed plan v3 schema")
    try:
        units = tuple(DistributedWorkUnitV3(**item) for item in payload["work_units"])
        plan = DistributedSingleContractionPlanV3(
            logical_operation_id=payload["logical_operation_id"],
            logical_task_id=payload["logical_task_id"],
            package_sha256=payload["package_sha256"],
            operation_sha256=payload["operation_sha256"],
            operation_id=payload["operation_id"],
            output_slot=payload["output_slot"],
            total_output_elements=payload["total_output_elements"],
            total_contracted_elements=payload["total_contracted_elements"],
            dpu_count=payload["dpu_count"],
            work_units=units,
            tasklets_per_dpu=payload.get("tasklets_per_dpu", DEFAULT_TASKLETS_PER_DPU),
            partition_kind=payload.get("partition_kind", OUTPUT_TILE),
            numeric_mode=payload.get("numeric_mode", NUMERIC_MODE_FLOAT32_REAL),
            max_dpus_per_rank=payload.get("max_dpus_per_rank", DEFAULT_MAX_DPUS_PER_RANK),
            profile=payload.get("profile", UPXDPV3_PROFILE_DEFAULT),
            package_operation_index=payload.get("package_operation_index", 0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid distributed plan v3 fields") from exc
    if payload.get("execution_plan_hash") != plan.execution_plan_hash:
        raise ValueError("distributed execution plan hash mismatch")
    return plan


def serialize_upxdpv3(
    plan: DistributedSingleContractionPlanV3,
    *,
    package_bytes: bytes,
    operation_bytes: bytes,
    operation_id: int | None = None,
    output_slot: int | None = None,
    tasklets_per_dpu: int | None = None,
) -> bytes:
    """Serialize the fixed UPXDPV3 sidecar and verify its content bindings."""

    validate_distributed_plan_v3(plan)
    if operation_id is not None and operation_id != plan.operation_id:
        raise ValueError("UPXDPV3 operation_id does not match the plan")
    if output_slot is not None and output_slot != plan.output_slot:
        raise ValueError("UPXDPV3 output_slot does not match the plan")
    if tasklets_per_dpu is not None and tasklets_per_dpu != plan.tasklets_per_dpu:
        raise ValueError("UPXDPV3 tasklets_per_dpu does not match the plan")
    package_digest = _digest_bytes("package_bytes", package_bytes)
    operation_digest = _digest_bytes("operation_bytes", operation_bytes)
    if package_digest != plan.package_sha256 or operation_digest != plan.operation_sha256:
        raise ValueError("UPXDPV3 package or operation SHA256 binding mismatch")
    partition_code = _PARTITION_TO_CODE[plan.partition_kind]
    header = struct.pack(
        UPXDPV3_HEADER_FORMAT,
        UPXDPV3_MAGIC,
        UPXDPV3_VERSION,
        UPXDPV3_HEADER_BYTES,
        len(plan.work_units),
        plan.dpu_count,
        plan.tasklets_per_dpu,
        UPXDPV3_PROVIDER_COUNT,
        partition_code,
        _NUMERIC_TO_CODE[plan.numeric_mode],
        plan.package_operation_index,
        plan.operation_id,
        plan.total_output_elements,
        plan.total_contracted_elements,
        plan.output_slot,
        UPXDPV3_RECORD_BYTES,
        0,
        0,
        hashlib.sha256(package_bytes).digest(),
        hashlib.sha256(operation_bytes).digest(),
    )
    records = b"".join(
        struct.pack(
            UPXDPV3_RECORD_FORMAT,
            plan.package_operation_index,
            plan.operation_id,
            partition_code,
            unit.dpu_id,
            unit.output_offset,
            unit.output_elements,
            unit.contracted_offset,
            unit.contracted_elements,
        )
        for unit in plan.work_units
    )
    return header + records


def serialize_upxdpv3_output_partition(plan: DistributedSingleContractionPlanV3, *, package_bytes: bytes, operation_bytes: bytes) -> bytes:
    if plan.partition_kind != OUTPUT_TILE:
        raise ValueError("UPXDPV3 output serializer requires output_tile")
    return serialize_upxdpv3(plan, package_bytes=package_bytes, operation_bytes=operation_bytes)


def serialize_upxdpv3_contracted_partition(plan: DistributedSingleContractionPlanV3, *, package_bytes: bytes, operation_bytes: bytes) -> bytes:
    if plan.partition_kind != CONTRACTED_PARTIAL_SUM:
        raise ValueError("UPXDPV3 contracted serializer requires contracted_partial_sum")
    return serialize_upxdpv3(plan, package_bytes=package_bytes, operation_bytes=operation_bytes)


def load_upxdpv3(
    sidecar_bytes: bytes,
    *,
    package_bytes: bytes | None = None,
    operation_bytes: bytes | None = None,
    logical_operation_id: str | None = None,
    logical_task_id: str | None = None,
    max_dpus_per_rank: int = DEFAULT_MAX_DPUS_PER_RANK,
) -> DistributedSingleContractionPlanV3:
    """Load and validate a UPXDPV3 sidecar, including optional content hashes."""

    if not isinstance(sidecar_bytes, bytes) or len(sidecar_bytes) < UPXDPV3_HEADER_BYTES:
        raise ValueError("UPXDPV3 sidecar is shorter than its header")
    try:
        header = struct.unpack_from(UPXDPV3_HEADER_FORMAT, sidecar_bytes)
    except struct.error as exc:
        raise ValueError("invalid UPXDPV3 header") from exc
    (magic, version, header_bytes, record_count, dpu_count, tasklets, provider_count, partition_code, numeric_code, package_index, operation_id, output_elements, contracted_elements, output_slot, record_bytes, reserved0, reserved1, package_digest, operation_digest) = header
    if magic != UPXDPV3_MAGIC or version != UPXDPV3_VERSION or header_bytes != UPXDPV3_HEADER_BYTES or record_bytes != UPXDPV3_RECORD_BYTES or reserved0 != 0 or reserved1 != 0 or provider_count != UPXDPV3_PROVIDER_COUNT:
        raise ValueError("invalid UPXDPV3 header")
    partition = _CODE_TO_PARTITION.get(partition_code)
    numeric_mode = _CODE_TO_NUMERIC.get(numeric_code)
    if partition is None or numeric_mode is None:
        raise ValueError("unsupported UPXDPV3 profile, partition, or numeric mode")
    expected_length = header_bytes + record_count * record_bytes
    if len(sidecar_bytes) != expected_length or record_count != dpu_count:
        raise ValueError("UPXDPV3 sidecar length or record count mismatch")
    package_sha256 = package_digest.hex()
    operation_sha256 = operation_digest.hex()
    if package_bytes is not None and _digest_bytes("package_bytes", package_bytes) != package_sha256:
        raise ValueError("UPXDPV3 package SHA256 mismatch")
    if operation_bytes is not None and _digest_bytes("operation_bytes", operation_bytes) != operation_sha256:
        raise ValueError("UPXDPV3 operation SHA256 mismatch")
    units: list[DistributedWorkUnitV3] = []
    for index in range(record_count):
        fields = struct.unpack_from(UPXDPV3_RECORD_FORMAT, sidecar_bytes, header_bytes + index * record_bytes)
        record_package_index, record_operation_id, record_partition, dpu_id, output_offset, output_count, contracted_offset, contracted_count = fields
        if record_package_index != package_index or record_operation_id != operation_id or record_partition != partition_code or dpu_id != index:
            raise ValueError("UPXDPV3 work record binding mismatch")
        units.append(DistributedWorkUnitV3(
            logical_operation_id=logical_operation_id or f"operation_{operation_id}",
            logical_task_id=logical_task_id or f"operation_{operation_id}",
            dpu_id=dpu_id,
            partition_kind=partition,
            output_offset=output_offset,
            output_elements=output_count,
            contracted_offset=contracted_offset,
            contracted_elements=contracted_count,
            output_ownership=OUTPUT_OWNERSHIP_EXCLUSIVE if partition == OUTPUT_TILE else OUTPUT_OWNERSHIP_SHARED_PARTIAL_SUM,
        ))
    return DistributedSingleContractionPlanV3(
        logical_operation_id=logical_operation_id or f"operation_{operation_id}",
        logical_task_id=logical_task_id or f"operation_{operation_id}",
        package_sha256=package_sha256,
        operation_sha256=operation_sha256,
        operation_id=operation_id,
        output_slot=output_slot,
        total_output_elements=output_elements,
        total_contracted_elements=contracted_elements,
        dpu_count=dpu_count,
        work_units=tuple(units),
        tasklets_per_dpu=tasklets,
        partition_kind=partition,
        numeric_mode=numeric_mode,
        max_dpus_per_rank=max_dpus_per_rank,
        profile=UPXDPV3_PROFILE_DEFAULT,
        package_operation_index=package_index,
    )


parse_upxdpv3 = load_upxdpv3
load_distributed_plan_v3 = parse_distributed_plan_v3
load_native_distributed_plan_v3 = load_upxdpv3
serialize_native_distributed_plan_v3 = serialize_upxdpv3
serialize_native_output_partition_v3 = serialize_upxdpv3_output_partition


def validate_upxdpv3(sidecar_bytes: bytes, plan: DistributedSingleContractionPlanV3, *, package_bytes: bytes, operation_bytes: bytes) -> dict[str, Any]:
    expected = serialize_upxdpv3(plan, package_bytes=package_bytes, operation_bytes=operation_bytes)
    if sidecar_bytes != expected:
        raise ValueError("UPXDPV3 sidecar differs from the deterministic plan")
    loaded = load_upxdpv3(sidecar_bytes, package_bytes=package_bytes, operation_bytes=operation_bytes, logical_operation_id=plan.logical_operation_id, logical_task_id=plan.logical_task_id)
    if loaded != plan:
        raise ValueError("UPXDPV3 sidecar does not reconstruct the supplied plan")
    return {
        "magic": UPXDPV3_MAGIC.rstrip(b"\x00").decode("ascii"),
        "version": UPXDPV3_VERSION,
        "header_bytes": UPXDPV3_HEADER_BYTES,
        "record_bytes": UPXDPV3_RECORD_BYTES,
        "record_count": plan.dpu_count,
        "dpu_count": plan.dpu_count,
        "tasklets_per_dpu": plan.tasklets_per_dpu,
        "package_sha256": plan.package_sha256,
        "operation_sha256": plan.operation_sha256,
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
    }


def _partition_ranges(
    total: int,
    count: int,
    *,
    aligned_output: bool = False,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise ValueError("partition total must be a positive integer")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("partition count must be a positive integer")
    if count > total:
        raise UnsupportedPartitionError(
            "unsupported preparation: partition count "
            f"{count} cannot exceed work element count {total}"
        )
    if not aligned_output:
        base, remainder = divmod(total, count)
        ranges: list[tuple[int, int]] = []
        offset = 0
        for index in range(count):
            elements = base + (1 if index < remainder else 0)
            ranges.append((offset, elements))
            offset += elements
        return tuple(ranges)

    if count == 1:
        return ((0, total),)
    minimum_non_final_total = 2 * (count - 1)
    maximum_non_final_total = 2 * ((total - 1) // 2)
    if maximum_non_final_total < minimum_non_final_total:
        raise UnsupportedPartitionError(
            "unsupported preparation: total output cannot give every requested "
            "DPU positive aligned work"
        )
    non_final_count = count - 1
    # Keep the final range close to the same size while making every preceding
    # range even.  The final range may be odd because it is the only range
    # whose end does not need to be an aligned start for another DPU.
    target_numerator = total * (count - 1)
    target_denominator = 2 * count
    non_final_total = 2 * ((target_numerator + count) // target_denominator)
    non_final_total = min(
        maximum_non_final_total,
        max(minimum_non_final_total, non_final_total),
    )
    even_base, even_remainder = divmod(non_final_total // 2, non_final_count)
    ranges: list[tuple[int, int]] = []
    offset = 0
    for index in range(non_final_count):
        elements = 2 * (even_base + (1 if index < even_remainder else 0))
        ranges.append((offset, elements))
        offset += elements
    ranges.append((offset, total - offset))
    return tuple(ranges)


def _canonical_numeric_mode(value: str) -> str:
    try:
        code = _NUMERIC_TO_CODE[value]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "unsupported numeric_mode; expected 'float32' or "
            "'per_task_resident_requantize'"
        ) from exc
    return _CODE_TO_NUMERIC[code]


def _validate_complete_ranges(units: tuple[DistributedWorkUnitV3, ...], offset_name: str, size_name: str, total: int) -> None:
    ranges = sorted((getattr(unit, offset_name), getattr(unit, size_name)) for unit in units)
    cursor = 0
    for offset, size in ranges:
        if offset != cursor or size <= 0:
            raise ValueError("work units must provide contiguous exact-once coverage")
        cursor += size
    if cursor != total:
        raise ValueError("work units must provide contiguous exact-once coverage")


def _range(name: str, offset: int, elements: int, total: int) -> None:
    _uint32(f"{name}_offset", offset)
    _uint32(f"{name}_elements", elements)
    if elements <= 0 or offset + elements > total:
        raise ValueError(f"{name} range is outside the plan")


def _positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    _uint32(name, value)


def _uint32(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > _UINT32_MAX:
        raise ValueError(f"{name} must be a uint32")


def _digest_bytes(name: str, value: bytes) -> str:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{name} must be non-empty bytes")
    return hashlib.sha256(value).hexdigest()


def _json_payload(plan: DistributedSingleContractionPlanV3) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "logical_operation_id": plan.logical_operation_id,
        "logical_task_id": plan.logical_task_id,
        "package_sha256": plan.package_sha256,
        "operation_sha256": plan.operation_sha256,
        "package_operation_index": plan.package_operation_index,
        "operation_id": plan.operation_id,
        "output_slot": plan.output_slot,
        "total_output_elements": plan.total_output_elements,
        "total_contracted_elements": plan.total_contracted_elements,
        "dpu_count": plan.dpu_count,
        "tasklets_per_dpu": plan.tasklets_per_dpu,
        "max_dpus_per_rank": plan.max_dpus_per_rank,
        "profile": plan.profile,
        "partition_kind": plan.partition_kind,
        "numeric_mode": plan.numeric_mode,
        "work_units": [
            {
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
            for unit in sorted(plan.work_units, key=lambda item: item.dpu_id)
        ],
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
