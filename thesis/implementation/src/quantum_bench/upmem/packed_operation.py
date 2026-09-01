"""Private packed transport for one or more ABI-v4 request artifacts.

The logical request is still the existing ABI-v4 manifest, sidecar and DPU
payload.  This module only replaces the transient per-request files used to
carry those bytes to the persistent native host.  It deliberately does not
define a second tensor-network or physical-plan representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
import time
from typing import Iterable, Mapping

from quantum_bench.upmem.protocol import (
    FLAG_ZERO_WORK,
    MAX_REQUEST_BYTES,
    V4Header,
    V4Profile,
    V4WorkUnit,
    V4WorkUnitRecord,
    _digest_bytes,
    _digest_hex,
    _payload_bytes,
    _record_abi_fields,
    _safe_relative,
    _validate_output_overlaps,
    _validate_record_storage,
    _validate_shape,
    _validate_u64,
    _validate_work_geometry,
    _validated_record_template,
)


PACKED_OPERATION_TRANSPORT = "packed_operation_v1"
PACKED_OPERATION_MAGIC = b"UPOENV2\0"
PACKED_OPERATION_VERSION = 2
PACKED_OPERATION_HEADER_BYTES = 96
PACKED_OPERATION_DESCRIPTOR_BYTES = 200
PACKED_OPERATION_FLAGS = 0

_OPERATION_HEADER_FORMAT = "<8s6I4Q32s"
_OPERATION_DESCRIPTOR_PREFIX_FORMAT = "<7Q2I Q"
_OPERATION_DESCRIPTOR_BYTES_BEFORE_DIGEST = 168
assert struct.calcsize(_OPERATION_HEADER_FORMAT) == PACKED_OPERATION_HEADER_BYTES
assert struct.calcsize(_OPERATION_DESCRIPTOR_PREFIX_FORMAT) == 72


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pad_payload(payload: bytes, expected: int) -> bytes:
    if len(payload) == expected:
        return payload
    if len(payload) > expected or any(payload[expected:]):
        raise ValueError("v4 payload has invalid padding")
    return payload + b"\0" * (expected - len(payload))


@dataclass(frozen=True)
class PackedV4Request:
    """An in-memory, byte-equivalent ABI-v4 request."""

    root: Path
    request_dir: Path
    header: V4Header
    work_units: tuple[V4WorkUnitRecord, ...]
    task_contract_sha256: str
    manifest_bytes: bytes
    sidecar_bytes: bytes
    payload_bytes: bytes
    manifest_sha256: str
    sidecar_sha256: str
    payload_sha256: str
    output_paths: tuple[Path, ...]
    payload_record_staging_s: float
    manifest_sidecar_staging_s: float
    payload_materialization_s: float
    payload_file_write_s: float
    payload_hashing_s: float
    payload_record_construction_s: float
    payload_record_count: int
    payload_files_created: int
    payload_bytes_staged: int
    payload_bytes_hashed: int

    @property
    def request_sequence(self) -> int:
        return self.header.request_sequence

    @property
    def request_output_elements(self) -> int:
        return self.header.request_output_elements

    @property
    def global_output_elements(self) -> int:
        return self.header.global_output_elements


@dataclass(frozen=True)
class PackedOperation:
    """One variable-length operation envelope and its request descriptors."""

    root: Path
    path: Path
    requests: tuple[PackedV4Request, ...]
    operation_sequence: int
    data: bytes
    sha256: str
    descriptor_count: int
    descriptor_bytes: int
    payload_bytes: int


def build_packed_v4_request(
    root: Path,
    *,
    profile: V4Profile,
    canonical_batch_count: int,
    canonical_m: int,
    canonical_n: int,
    canonical_k: int,
    work_units: Iterable[V4WorkUnit],
    task_contract_sha256: str | bytes | bytearray,
    request_sequence: int,
    record_templates: Mapping[int, tuple[int, ...]] | None = None,
) -> PackedV4Request:
    """Build an ABI-v4 request without creating request/payload files."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("v4 packed request root must be an existing directory")
    for name, value in (
        ("canonical_batch_count", canonical_batch_count),
        ("canonical_m", canonical_m),
        ("canonical_n", canonical_n),
        ("canonical_k", canonical_k),
        ("request_sequence", request_sequence),
    ):
        _validate_u64(name, value)
    if not all((canonical_batch_count, canonical_m, canonical_n, canonical_k)):
        raise ValueError("v4 canonical dimensions must be positive")
    if canonical_batch_count > 0xFFFFFFFF:
        raise ValueError("v4 canonical batch count exceeds native bounds")
    contract_digest = _digest_bytes(task_contract_sha256, field="task_contract_sha256")
    if not any(contract_digest):
        raise ValueError("task_contract_sha256 cannot be all zero")
    supplied = list(work_units)
    if not supplied:
        raise ValueError("v4 packed request needs at least one work unit")
    by_id: dict[int, V4WorkUnit] = {}
    for unit in supplied:
        if unit.local_dpu_id in by_id:
            raise ValueError("duplicate v4 local DPU ID")
        if not 0 <= unit.local_dpu_id < profile.dpu_count:
            raise ValueError("v4 local DPU ID is outside the profile")
        _validate_work_geometry(
            unit,
            batch_count=canonical_batch_count,
            m=canonical_m,
            n=canonical_n,
            k=canonical_k,
            mode=profile.numeric_mode_code,
        )
        by_id[unit.local_dpu_id] = unit
    active = [unit for unit in supplied if not (unit.flags & FLAG_ZERO_WORK)]
    if not active:
        raise ValueError("v4 packed request cannot contain only zero-work records")
    if record_templates is not None and not isinstance(record_templates, Mapping):
        raise TypeError("record_templates must be a mapping when supplied")
    _validate_output_overlaps(active)
    request_output_elements = sum(unit.m_elements * unit.n_elements for unit in active)
    global_output_elements = canonical_batch_count * canonical_m * canonical_n
    if not 0 < request_output_elements <= global_output_elements:
        raise ValueError("v4 request output coverage is invalid")

    request_dir = root / "requests" / f"{request_sequence:016d}"
    records: list[V4WorkUnitRecord] = []
    output_paths: list[Path] = []
    payload_parts: list[bytes] = []
    payload_materialization_s = 0.0
    payload_hashing_s = 0.0
    payload_record_construction_s = 0.0
    payload_record_count = 0
    payload_bytes_staged = 0
    payload_bytes_hashed = 0
    for dpu_id in range(profile.dpu_count):
        materialization_started = time.perf_counter()
        unit = by_id.get(
            dpu_id,
            V4WorkUnit(
                local_dpu_id=dpu_id,
                tile_id=(1 << 63) + dpu_id,
                batch_index=0,
                m_offset=0,
                n_offset=0,
                k_offset=0,
                m_elements=0,
                n_elements=0,
                k_elements=0,
                flags=FLAG_ZERO_WORK,
            ),
        )
        template = record_templates.get(dpu_id) if record_templates is not None else None
        if template is not None:
            record_fields = _validated_record_template(template, unit, profile=profile)
        else:
            record_fields = _record_abi_fields(
                unit,
                profile=profile,
                canonical_batch_count=canonical_batch_count,
                canonical_m=canonical_m,
                canonical_k=canonical_k,
                canonical_n=canonical_n,
                validate_payload=False,
                validate_geometry=False,
            )
        a_bytes, b_bytes, _c_bytes, _a_offset, _b_offset, _c_offset = record_fields[10:]
        if unit.flags & FLAG_ZERO_WORK:
            a_payload = b_payload = b""
        else:
            a_payload = _pad_payload(_payload_bytes(unit.a_payload), a_bytes)
            b_payload = _pad_payload(_payload_bytes(unit.b_payload), b_bytes)
        payload_materialization_s += time.perf_counter() - materialization_started
        payload_parts.extend((a_payload, b_payload))
        payload_bytes_staged += len(a_payload) + len(b_payload)
        hashing_started = time.perf_counter()
        a_sha256 = _sha256(a_payload)
        b_sha256 = _sha256(b_payload)
        payload_hashing_s += time.perf_counter() - hashing_started
        payload_bytes_hashed += len(a_payload) + len(b_payload)
        record_construction_started = time.perf_counter()
        a_path = _safe_relative(
            (request_dir / "payloads" / f"dpu_{dpu_id:03d}_a.bin").relative_to(root).as_posix()
        )
        b_path = _safe_relative(
            (request_dir / "payloads" / f"dpu_{dpu_id:03d}_b.bin").relative_to(root).as_posix()
        )
        c_path = _safe_relative(
            (request_dir / "outputs" / f"dpu_{dpu_id:03d}_c.bin").relative_to(root).as_posix()
        )
        output_paths.append(root / c_path)
        records.append(
            V4WorkUnitRecord(
                *record_fields,
                a_path=a_path,
                b_path=b_path,
                c_path=c_path,
                a_sha256=a_sha256,
                b_sha256=b_sha256,
            )
        )
        payload_record_construction_s += time.perf_counter() - record_construction_started
        payload_record_count += 1

    payload_bytes = b"".join(payload_parts)
    manifest_lines = [f"sidecar {_safe_relative((request_dir / 'sidecar.bin').relative_to(root).as_posix())}"]
    for record in records:
        manifest_lines.append(
            " ".join(
                (
                    "dpu",
                    str(record.local_dpu_id),
                    str(record.tile_id),
                    record.a_path,
                    record.b_path,
                    record.c_path,
                    record.a_sha256,
                    record.b_sha256,
                )
            )
        )
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    manifest_sha256 = _digest_hex(manifest_bytes)
    header = V4Header(
        canonical_batch_count=canonical_batch_count,
        canonical_m=canonical_m,
        canonical_n=canonical_n,
        canonical_k=canonical_k,
        global_output_elements=global_output_elements,
        request_output_elements=request_output_elements,
        request_sequence=request_sequence,
        task_contract_sha256=contract_digest,
        request_sha256=bytes.fromhex(manifest_sha256),
        work_unit_count=profile.dpu_count,
        dpu_count=profile.dpu_count,
        tasklets_per_dpu=profile.tasklets_per_dpu,
        numeric_mode=profile.numeric_mode_code,
        partition_mode=profile.partition_mode,
    )
    _validate_record_storage(records, profile=profile, canonical_k=canonical_k)
    _validate_shape(header, profile)
    sidecar_bytes = header.pack() + b"".join(record.pack() for record in records)
    if len(sidecar_bytes) > MAX_REQUEST_BYTES:
        raise ValueError("v4 sidecar exceeds native request limit")
    manifest_sidecar_started = time.perf_counter()
    sidecar_sha256 = _digest_hex(sidecar_bytes)
    manifest_sidecar_staging_s = time.perf_counter() - manifest_sidecar_started
    return PackedV4Request(
        root=root,
        request_dir=request_dir,
        header=header,
        work_units=tuple(records),
        task_contract_sha256=contract_digest.hex(),
        manifest_bytes=manifest_bytes,
        sidecar_bytes=sidecar_bytes,
        payload_bytes=payload_bytes,
        manifest_sha256=manifest_sha256,
        sidecar_sha256=sidecar_sha256,
        payload_sha256=_digest_hex(payload_bytes),
        output_paths=tuple(output_paths),
        payload_record_staging_s=float(
            payload_materialization_s
            + payload_hashing_s
            + payload_record_construction_s
            + manifest_sidecar_staging_s
        ),
        manifest_sidecar_staging_s=float(manifest_sidecar_staging_s),
        payload_materialization_s=float(payload_materialization_s),
        payload_file_write_s=0.0,
        payload_hashing_s=float(payload_hashing_s),
        payload_record_construction_s=float(payload_record_construction_s),
        payload_record_count=payload_record_count,
        payload_files_created=0,
        payload_bytes_staged=payload_bytes_staged,
        payload_bytes_hashed=payload_bytes_hashed,
    )


def _descriptor_without_digest(request: PackedV4Request, *, body_offset: int) -> bytes:
    manifest_offset = body_offset
    sidecar_offset = manifest_offset + len(request.manifest_bytes)
    payload_offset = sidecar_offset + len(request.sidecar_bytes)
    prefix = struct.pack(
        "<7Q2I Q",
        request.request_sequence,
        manifest_offset,
        len(request.manifest_bytes),
        sidecar_offset,
        len(request.sidecar_bytes),
        payload_offset,
        len(request.payload_bytes),
        len(request.work_units),
        0,
        request.request_output_elements,
    )
    return prefix + bytes.fromhex(request.manifest_sha256) + bytes.fromhex(
        request.sidecar_sha256
    ) + bytes.fromhex(request.payload_sha256)


def pack_operation(
    root: Path,
    *,
    requests: Iterable[PackedV4Request],
    operation_sequence: int,
    filename: str = "packed/operation.bin",
) -> PackedOperation:
    """Pack a variable-length operation using explicit little-endian fields."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("packed operation root must be an existing directory")
    _validate_u64("operation_sequence", operation_sequence)
    ordered = tuple(requests)
    if not ordered:
        raise ValueError("packed operation needs at least one request")
    if any(request.root != root for request in ordered):
        raise ValueError("packed operation requests must share the session root")
    sequences = tuple(request.request_sequence for request in ordered)
    if sequences[0] != operation_sequence or any(
        right <= left for left, right in zip(sequences, sequences[1:])
    ):
        raise ValueError(
            "packed operation request sequences must increase from operation_sequence"
        )
    count = len(ordered)
    if count > 0xFFFFFFFF:
        raise ValueError("packed operation has too many request descriptors")
    safe_filename = _safe_relative(filename)
    descriptors_offset = PACKED_OPERATION_HEADER_BYTES
    body_offset = descriptors_offset + count * PACKED_OPERATION_DESCRIPTOR_BYTES
    descriptors: list[bytes] = []
    body = bytearray()
    current = body_offset
    for request in ordered:
        descriptor_prefix = _descriptor_without_digest(request, body_offset=current)
        descriptor_digest = hashlib.sha256(descriptor_prefix + bytes(32)).digest()
        descriptor = descriptor_prefix + descriptor_digest
        if len(descriptor) != PACKED_OPERATION_DESCRIPTOR_BYTES:
            raise AssertionError("packed operation descriptor size drift")
        descriptors.append(descriptor)
        body.extend(request.manifest_bytes)
        body.extend(request.sidecar_bytes)
        body.extend(request.payload_bytes)
        current += len(request.manifest_bytes) + len(request.sidecar_bytes) + len(request.payload_bytes)
    total_bytes = body_offset + len(body)
    header = struct.pack(
        _OPERATION_HEADER_FORMAT,
        PACKED_OPERATION_MAGIC,
        PACKED_OPERATION_VERSION,
        PACKED_OPERATION_HEADER_BYTES,
        count,
        PACKED_OPERATION_DESCRIPTOR_BYTES,
        PACKED_OPERATION_FLAGS,
        0,
        descriptors_offset,
        body_offset,
        total_bytes,
        operation_sequence,
        bytes(32),
    )
    unsigned = header + b"".join(descriptors) + bytes(body)
    envelope_digest = hashlib.sha256(unsigned).digest()
    data = (
        unsigned[:64]
        + envelope_digest
        + unsigned[96:]
    )
    if len(data) != total_bytes:
        raise AssertionError("packed operation total size drift")
    path = root / safe_filename
    return PackedOperation(
        root=root,
        path=path,
        requests=ordered,
        operation_sequence=operation_sequence,
        data=data,
        sha256=_sha256(data),
        descriptor_count=count,
        descriptor_bytes=PACKED_OPERATION_DESCRIPTOR_BYTES,
        payload_bytes=sum(len(request.payload_bytes) for request in ordered),
    )


__all__ = [
    "PACKED_OPERATION_DESCRIPTOR_BYTES",
    "PACKED_OPERATION_HEADER_BYTES",
    "PACKED_OPERATION_MAGIC",
    "PACKED_OPERATION_TRANSPORT",
    "PackedOperation",
    "PackedV4Request",
    "build_packed_v4_request",
    "pack_operation",
]
