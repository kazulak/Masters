"""Low-level Python ABI and client for the additive UPMEM v4 tile session.

This module deliberately contains no tensor-network or benchmark policy.  It
only builds the v4 binary request accepted by ``host_v4_session`` and speaks
that native process' small line protocol.  Higher layers own tiling decisions,
K-chunk assembly, quantization scales, and graph scheduling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import queue
import re
import struct
import subprocess
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


MAGIC = b"UPXDPV4\0"
VERSION = 4
PROFILE = "upmem_execution_plan_v4_tile_session"
RESPONSE_SCHEMA = "upmem_execution_plan_native_v4"

MAX_DPUS = 64
MAX_TASKLETS = 24
MRAM_POOL_BYTES = 512 * 1024
MRAM_ALIGNMENT = 8
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_CONTRACTED = 65536
INT32_MAX = 2**31 - 1
INT8_MAX_PRODUCT = 128 * 128
MAX_INT32_SAFE_K = INT32_MAX // INT8_MAX_PRODUCT

NUMERIC_FLOAT32 = "float32"
NUMERIC_HOST_PACKED_INT8 = "host_packed_int8"
NUMERIC_MODE_FLOAT32 = 0
NUMERIC_MODE_HOST_PACKED_INT8 = 1
PARTITION_OUTPUT_TILE = 1
FLAG_ZERO_WORK = 0x00000001

CONTROL_MAGIC = 0x34564354
COMPLETION_MAGIC = 0x34564350
STATUS_PENDING = 0
STATUS_COMPLETED = 1
STATUS_FAILED = 2

HEADER_FORMAT = "<8s10I7Q32s32s"
WORK_UNIT_FORMAT = "<2I5Q9I"
CONTROL_FORMAT = "<18I"
COMPLETION_FORMAT = "<4I3Q"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)
WORK_UNIT_BYTES = struct.calcsize(WORK_UNIT_FORMAT)
CONTROL_BYTES = struct.calcsize(CONTROL_FORMAT)
COMPLETION_BYTES = struct.calcsize(COMPLETION_FORMAT)
assert HEADER_BYTES == 168
assert WORK_UNIT_BYTES == 84
assert CONTROL_BYTES == 72
assert COMPLETION_BYTES == 40

_RANK_PATH = re.compile(r"^/dev/dpu_rank[0-9]+$")
_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class V4Error(RuntimeError):
    """Base error carrying a machine-readable failure stage."""

    def __init__(self, failure_stage: str, message: str) -> None:
        super().__init__(f"{failure_stage}: {message}")
        self.failure_stage = failure_stage


class V4ProtocolError(V4Error):
    """Raised when the native process violates the v4 protocol."""


def _digest_bytes(value: str | bytes | bytearray, *, field: str) -> bytes:
    if isinstance(value, str):
        if not _HEX_SHA256.fullmatch(value):
            raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
        return bytes.fromhex(value)
    result = bytes(value)
    if len(result) != 32:
        raise ValueError(f"{field} must contain exactly 32 digest bytes")
    return result


def _digest_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned8(value: int) -> int:
    if value < 0:
        raise ValueError("byte count cannot be negative")
    return (value + 7) & ~7


def _payload_bytes(value: bytes | bytearray | memoryview | Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    tobytes = getattr(value, "tobytes", None)
    if callable(tobytes):
        try:
            return bytes(tobytes(order="C"))
        except TypeError:
            return bytes(tobytes())
    raise TypeError("payload must be bytes-like or expose tobytes()")


def _safe_relative(value: str | Path) -> str:
    text = str(value).replace(os.sep, "/")
    if not text or text.startswith(("/", "\\")) or "\\" in text:
        raise ValueError(f"unsafe v4 relative path: {value!s}")
    parsed = PurePosixPath(text)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"unsafe v4 relative path: {value!s}")
    return parsed.as_posix()


def _relative_to(root: Path, path: Path, *, must_exist: bool = False) -> str:
    candidate = path.resolve(strict=must_exist)
    try:
        relative = candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside v4 session root: {path}") from exc
    return _safe_relative(relative.as_posix())


def _validate_u32(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 0xFFFFFFFF
    ):
        raise ValueError(f"{name} must be an unsigned 32-bit integer")


def _validate_u64(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError(f"{name} must be an unsigned 64-bit integer")


def _numeric_code(value: str | int) -> int:
    if value in (NUMERIC_FLOAT32, NUMERIC_MODE_FLOAT32):
        return NUMERIC_MODE_FLOAT32
    if value in (NUMERIC_HOST_PACKED_INT8, NUMERIC_MODE_HOST_PACKED_INT8):
        return NUMERIC_MODE_HOST_PACKED_INT8
    raise ValueError(f"unsupported v4 numeric mode: {value!r}")


@dataclass(frozen=True)
class V4Profile:
    """Bounded physical v4 session configuration."""

    dpu_count: int
    tasklets_per_dpu: int = 1
    numeric_mode: str | int = NUMERIC_FLOAT32
    rank_path: str | None = None
    timeout_s: float = 60.0
    max_stdout_bytes: int = 256 * 1024
    max_stderr_bytes: int = 256 * 1024
    mram_pool_bytes: int = MRAM_POOL_BYTES
    partition_mode: int = PARTITION_OUTPUT_TILE

    def __post_init__(self) -> None:
        if not 1 <= self.dpu_count <= MAX_DPUS:
            raise ValueError("v4 dpu_count must be in [1, 64]")
        if not 1 <= self.tasklets_per_dpu <= MAX_TASKLETS:
            raise ValueError("v4 tasklets_per_dpu must be in [1, 24]")
        _numeric_code(self.numeric_mode)
        if self.rank_path is not None and not _RANK_PATH.fullmatch(self.rank_path):
            raise ValueError("v4 rank_path must be an explicit /dev/dpu_rankN path")
        if self.timeout_s <= 0:
            raise ValueError("v4 timeout_s must be positive")
        if self.max_stdout_bytes <= 0 or self.max_stderr_bytes <= 0:
            raise ValueError("v4 output limits must be positive")
        if self.mram_pool_bytes != MRAM_POOL_BYTES:
            raise ValueError("v4 native ABI uses a fixed 512 KiB MRAM pool")
        if self.partition_mode != PARTITION_OUTPUT_TILE:
            raise ValueError("v4 only supports output-tile partitioning")

    @property
    def numeric_mode_code(self) -> int:
        return _numeric_code(self.numeric_mode)

    @property
    def numeric_mode_name(self) -> str:
        return (
            NUMERIC_FLOAT32
            if self.numeric_mode_code == NUMERIC_MODE_FLOAT32
            else NUMERIC_HOST_PACKED_INT8
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["numeric_mode"] = self.numeric_mode_name
        result["numeric_mode_code"] = self.numeric_mode_code
        result["profile"] = PROFILE
        return result


@dataclass(frozen=True)
class V4Header:
    canonical_batch_count: int
    canonical_m: int
    canonical_n: int
    canonical_k: int
    global_output_elements: int
    request_output_elements: int
    request_sequence: int
    task_contract_sha256: bytes
    request_sha256: bytes
    work_unit_count: int
    dpu_count: int
    tasklets_per_dpu: int
    numeric_mode: int
    partition_mode: int = PARTITION_OUTPUT_TILE
    version: int = VERSION
    header_bytes: int = HEADER_BYTES
    record_bytes: int = WORK_UNIT_BYTES
    reserved0: int = 0
    reserved1: int = 0

    def pack(self) -> bytes:
        return struct.pack(
            HEADER_FORMAT,
            MAGIC,
            self.version,
            self.header_bytes,
            self.work_unit_count,
            self.dpu_count,
            self.tasklets_per_dpu,
            self.numeric_mode,
            self.partition_mode,
            self.record_bytes,
            self.reserved0,
            self.reserved1,
            self.canonical_batch_count,
            self.canonical_m,
            self.canonical_n,
            self.canonical_k,
            self.global_output_elements,
            self.request_output_elements,
            self.request_sequence,
            self.task_contract_sha256,
            self.request_sha256,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "V4Header":
        if len(data) != HEADER_BYTES:
            raise ValueError(f"v4 header must be exactly {HEADER_BYTES} bytes")
        values = struct.unpack(HEADER_FORMAT, data)
        if values[0] != MAGIC:
            raise ValueError("invalid v4 magic")
        return cls(
            version=values[1],
            header_bytes=values[2],
            work_unit_count=values[3],
            dpu_count=values[4],
            tasklets_per_dpu=values[5],
            numeric_mode=values[6],
            partition_mode=values[7],
            record_bytes=values[8],
            reserved0=values[9],
            reserved1=values[10],
            canonical_batch_count=values[11],
            canonical_m=values[12],
            canonical_n=values[13],
            canonical_k=values[14],
            global_output_elements=values[15],
            request_output_elements=values[16],
            request_sequence=values[17],
            task_contract_sha256=values[18],
            request_sha256=values[19],
        )


@dataclass(frozen=True)
class V4WorkUnit:
    """Logical tile input.  A and B are row-major tile payloads."""

    local_dpu_id: int
    tile_id: int
    batch_index: int
    m_offset: int
    n_offset: int
    k_offset: int
    m_elements: int
    n_elements: int
    k_elements: int
    a_payload: bytes | bytearray | memoryview | Any = b""
    b_payload: bytes | bytearray | memoryview | Any = b""
    flags: int = 0


@dataclass(frozen=True)
class V4WorkUnitRecord:
    """Exact ABI work-unit fields plus staged paths."""

    local_dpu_id: int
    flags: int
    tile_id: int
    batch_index: int
    m_offset: int
    n_offset: int
    k_offset: int
    m_elements: int
    n_elements: int
    k_elements: int
    a_transfer_bytes: int
    b_transfer_bytes: int
    c_transfer_bytes: int
    a_offset_bytes: int
    b_offset_bytes: int
    c_offset_bytes: int
    a_path: str
    b_path: str
    c_path: str
    # Manifest-only commitments.  They intentionally do not change the v4 ABI.
    a_sha256: str = ""
    b_sha256: str = ""

    def pack(self) -> bytes:
        return struct.pack(
            WORK_UNIT_FORMAT,
            self.local_dpu_id,
            self.flags,
            self.tile_id,
            self.batch_index,
            self.m_offset,
            self.n_offset,
            self.k_offset,
            self.m_elements,
            self.n_elements,
            self.k_elements,
            self.a_transfer_bytes,
            self.b_transfer_bytes,
            self.c_transfer_bytes,
            self.a_offset_bytes,
            self.b_offset_bytes,
            self.c_offset_bytes,
        )

    @classmethod
    def unpack(
        cls,
        data: bytes,
        *,
        paths: tuple[str, str, str] = ("", "", ""),
        payload_sha256: tuple[str, str] = ("", ""),
    ) -> "V4WorkUnitRecord":
        if len(data) != WORK_UNIT_BYTES:
            raise ValueError(f"v4 work unit must be exactly {WORK_UNIT_BYTES} bytes")
        values = struct.unpack(WORK_UNIT_FORMAT, data)
        return cls(*values, *paths, *payload_sha256)


@dataclass(frozen=True)
class V4Completion:
    magic: int
    version: int
    status: int
    dpu_id: int
    cycles: int
    processed_elements: int
    checksum_fnv1a64: int

    def pack(self) -> bytes:
        return struct.pack(
            COMPLETION_FORMAT,
            self.magic,
            self.version,
            self.status,
            self.dpu_id,
            self.cycles,
            self.processed_elements,
            self.checksum_fnv1a64,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "V4Completion":
        if len(data) != COMPLETION_BYTES:
            raise ValueError(f"v4 completion must be exactly {COMPLETION_BYTES} bytes")
        return cls(*struct.unpack(COMPLETION_FORMAT, data))


@dataclass(frozen=True)
class V4RequestArtifact:
    root: Path
    request_dir: Path
    manifest_path: Path
    sidecar_path: Path
    header: V4Header
    work_units: tuple[V4WorkUnitRecord, ...]
    task_contract_sha256: str
    manifest_sha256: str
    sidecar_sha256: str
    output_paths: tuple[Path, ...]

    @property
    def request_sequence(self) -> int:
        return self.header.request_sequence

    @property
    def request_output_elements(self) -> int:
        return self.header.request_output_elements

    @property
    def global_output_elements(self) -> int:
        return self.header.global_output_elements

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESPONSE_SCHEMA,
            "profile": PROFILE,
            "root": str(self.root),
            "request_dir": str(self.request_dir),
            "manifest_path": str(self.manifest_path),
            "sidecar_path": str(self.sidecar_path),
            "manifest_sha256": self.manifest_sha256,
            "sidecar_sha256": self.sidecar_sha256,
            "task_contract_sha256": self.task_contract_sha256,
            "request_sequence": self.request_sequence,
            "request_output_elements": self.request_output_elements,
            "global_output_elements": self.global_output_elements,
            "work_units": [asdict(unit) for unit in self.work_units],
        }


def pack_v4_header(header: V4Header) -> bytes:
    return header.pack()


def unpack_v4_header(data: bytes) -> V4Header:
    return V4Header.unpack(data)


def pack_v4_work_unit(unit: V4WorkUnitRecord) -> bytes:
    return unit.pack()


def unpack_v4_work_unit(data: bytes) -> V4WorkUnitRecord:
    return V4WorkUnitRecord.unpack(data)


def pack_v4_completion(completion: V4Completion) -> bytes:
    return completion.pack()


def unpack_v4_completion(data: bytes) -> V4Completion:
    return V4Completion.unpack(data)


def _validate_shape(header: V4Header, profile: V4Profile) -> None:
    if (
        header.version != VERSION
        or header.header_bytes != HEADER_BYTES
        or header.record_bytes != WORK_UNIT_BYTES
    ):
        raise ValueError("v4 header ABI version or size does not match native ABI")
    if (
        header.work_unit_count != profile.dpu_count
        or header.dpu_count != profile.dpu_count
    ):
        raise ValueError("v4 header DPU count does not match profile")
    if (
        header.tasklets_per_dpu != profile.tasklets_per_dpu
        or header.numeric_mode != profile.numeric_mode_code
    ):
        raise ValueError("v4 header execution configuration does not match profile")
    if (
        header.partition_mode != PARTITION_OUTPUT_TILE
        or header.reserved0
        or header.reserved1
    ):
        raise ValueError("v4 header uses an unsupported partition or reserved field")
    for name, value in (
        ("canonical_batch_count", header.canonical_batch_count),
        ("canonical_m", header.canonical_m),
        ("canonical_n", header.canonical_n),
        ("canonical_k", header.canonical_k),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if header.canonical_k > MAX_CONTRACTED:
        raise ValueError("canonical K exceeds v4 native bound")
    expected = header.canonical_batch_count * header.canonical_m * header.canonical_n
    if expected > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("v4 output element count exceeds uint64")
    if (
        expected != header.global_output_elements
        or not 0 < header.request_output_elements <= expected
    ):
        raise ValueError("v4 output element counts are inconsistent")


def _validate_work_geometry(
    unit: V4WorkUnit, *, batch_count: int, m: int, n: int, k: int, mode: int
) -> None:
    for name, value in (
        ("local_dpu_id", unit.local_dpu_id),
        ("flags", unit.flags),
        ("m_elements", unit.m_elements),
        ("n_elements", unit.n_elements),
        ("k_elements", unit.k_elements),
    ):
        _validate_u32(name, value)
    for name, value in (
        ("tile_id", unit.tile_id),
        ("batch_index", unit.batch_index),
        ("m_offset", unit.m_offset),
        ("n_offset", unit.n_offset),
        ("k_offset", unit.k_offset),
    ):
        _validate_u64(name, value)
    if unit.flags & ~FLAG_ZERO_WORK:
        raise ValueError("v4 work unit contains unknown flags")
    if unit.batch_index >= batch_count:
        raise ValueError("v4 batch index is outside canonical batch count")
    zero = bool(unit.flags & FLAG_ZERO_WORK)
    if zero:
        if any((unit.m_elements, unit.n_elements, unit.k_elements)):
            raise ValueError("zero-work v4 unit must have zero extents")
        if _payload_bytes(unit.a_payload) or _payload_bytes(unit.b_payload):
            raise ValueError("zero-work v4 unit must not carry operands")
        return
    if not all((unit.m_elements, unit.n_elements, unit.k_elements)):
        raise ValueError("non-zero v4 work unit must have positive extents")
    if unit.m_offset + unit.m_elements > m or unit.n_offset + unit.n_elements > n:
        raise ValueError("v4 output tile is outside canonical M/N bounds")
    if unit.k_offset + unit.k_elements > k or unit.k_elements > MAX_CONTRACTED:
        raise ValueError("v4 K chunk is outside canonical K bounds")
    if unit.m_elements * unit.n_elements > 0xFFFFFFFF:
        raise ValueError("v4 output tile exceeds native uint32 element bound")
    if unit.k_elements * INT8_MAX_PRODUCT > INT32_MAX:
        raise ValueError("v4 K chunk is unsafe for int32 int8 accumulation")
    element_bytes = 4 if mode == NUMERIC_MODE_FLOAT32 else 1
    expected_a = _aligned8(unit.m_elements * unit.k_elements * element_bytes)
    expected_b = _aligned8(unit.k_elements * unit.n_elements * element_bytes)
    if len(_payload_bytes(unit.a_payload)) not in {
        unit.m_elements * unit.k_elements * element_bytes,
        expected_a,
    } or len(_payload_bytes(unit.b_payload)) not in {
        unit.k_elements * unit.n_elements * element_bytes,
        expected_b,
    }:
        raise ValueError("v4 operand payload length does not match tile geometry")


def _stage_payload(path: Path, payload: bytes, expected_bytes: int) -> None:
    if len(payload) == expected_bytes:
        padded = payload
    else:
        if len(payload) > expected_bytes or any(payload[expected_bytes:]):
            raise ValueError("v4 payload has invalid padding")
        padded = payload + b"\0" * (expected_bytes - len(payload))
    path.write_bytes(padded)


def _validate_output_overlaps(units: Sequence[V4WorkUnit]) -> None:
    active = [unit for unit in units if not (unit.flags & FLAG_ZERO_WORK)]
    for index, left in enumerate(active):
        for right in active[index + 1 :]:
            if left.batch_index != right.batch_index:
                continue
            m_overlap = (
                left.m_offset < right.m_offset + right.m_elements
                and right.m_offset < left.m_offset + left.m_elements
            )
            n_overlap = (
                left.n_offset < right.n_offset + right.n_elements
                and right.n_offset < left.n_offset + left.n_elements
            )
            if m_overlap and n_overlap:
                raise ValueError("v4 request output tiles overlap")


def build_v4_request(
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
) -> V4RequestArtifact:
    """Build one deterministic, native-compatible v4 request.

    ``work_units`` may contain only the active tiles.  Missing local DPU IDs
    are filled with zero-work records because the native parser requires a
    dense ``0..dpu_count-1`` manifest.  K chunks are separate requests and
    must be assembled by the caller; a request never claims global
    completeness.
    """

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("v4 request root must be a directory")
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
    if canonical_batch_count > 0xFFFFFFFF or canonical_k > MAX_CONTRACTED:
        raise ValueError("v4 canonical dimensions exceed native bounds")
    contract_digest = _digest_bytes(task_contract_sha256, field="task_contract_sha256")
    if not any(contract_digest):
        raise ValueError("task_contract_sha256 cannot be all zero")
    supplied = list(work_units)
    if not supplied:
        raise ValueError("v4 request needs at least one non-zero work unit")
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
        raise ValueError("v4 request cannot contain only zero-work records")
    _validate_output_overlaps(active)
    request_output_elements = sum(unit.m_elements * unit.n_elements for unit in active)
    global_output_elements = canonical_batch_count * canonical_m * canonical_n
    if not 0 < request_output_elements <= global_output_elements:
        raise ValueError("v4 request output coverage is invalid")
    request_dir = root / "requests" / f"{request_sequence:016d}"
    if request_dir.exists():
        raise ValueError(f"v4 request directory already exists: {request_dir}")
    request_dir.mkdir(parents=True)
    payload_dir = request_dir / "payloads"
    output_dir = request_dir / "outputs"
    payload_dir.mkdir()
    output_dir.mkdir()
    records: list[V4WorkUnitRecord] = []
    output_paths: list[Path] = []
    for dpu_id in range(profile.dpu_count):
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
        if unit.flags & FLAG_ZERO_WORK:
            a_bytes = b_bytes = c_bytes = 0
            a_offset = b_offset = c_offset = 0
            a_payload = b_payload = b""
        else:
            element_bytes = (
                4 if profile.numeric_mode_code == NUMERIC_MODE_FLOAT32 else 1
            )
            a_bytes = _aligned8(unit.m_elements * unit.k_elements * element_bytes)
            b_bytes = _aligned8(unit.k_elements * unit.n_elements * element_bytes)
            c_bytes = _aligned8(unit.m_elements * unit.n_elements * 4)
            a_offset = 0
            b_offset = _aligned8(a_bytes)
            c_offset = _aligned8(b_offset + b_bytes)
            if c_offset + c_bytes > profile.mram_pool_bytes:
                raise ValueError("v4 tile operands and output exceed MRAM pool")
            a_payload = _payload_bytes(unit.a_payload)
            b_payload = _payload_bytes(unit.b_payload)
        a_path = payload_dir / f"dpu_{dpu_id:03d}_a.bin"
        b_path = payload_dir / f"dpu_{dpu_id:03d}_b.bin"
        c_path = output_dir / f"dpu_{dpu_id:03d}_c.bin"
        _stage_payload(a_path, a_payload, a_bytes)
        _stage_payload(b_path, b_payload, b_bytes)
        output_paths.append(c_path)
        records.append(
            V4WorkUnitRecord(
                local_dpu_id=dpu_id,
                flags=unit.flags,
                tile_id=unit.tile_id,
                batch_index=unit.batch_index,
                m_offset=unit.m_offset,
                n_offset=unit.n_offset,
                k_offset=unit.k_offset,
                m_elements=unit.m_elements,
                n_elements=unit.n_elements,
                k_elements=unit.k_elements,
                a_transfer_bytes=a_bytes,
                b_transfer_bytes=b_bytes,
                c_transfer_bytes=c_bytes,
                a_offset_bytes=a_offset,
                b_offset_bytes=b_offset,
                c_offset_bytes=c_offset,
                a_path=_relative_to(root, a_path, must_exist=True),
                b_path=_relative_to(root, b_path, must_exist=True),
                c_path=_relative_to(root, c_path),
                a_sha256=_file_sha256(a_path),
                b_sha256=_file_sha256(b_path),
            )
        )
    _validate_record_storage(records, profile=profile, canonical_k=canonical_k)
    manifest_path = request_dir / "manifest.txt"
    sidecar_path = request_dir / "sidecar.bin"
    sidecar_rel = _relative_to(root, sidecar_path)
    manifest_lines = [f"sidecar {sidecar_rel}"]
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
    _validate_shape(header, profile)
    sidecar_bytes = header.pack() + b"".join(record.pack() for record in records)
    if len(sidecar_bytes) > MAX_REQUEST_BYTES:
        raise ValueError("v4 sidecar exceeds native request limit")
    manifest_path.write_bytes(manifest_bytes)
    sidecar_path.write_bytes(sidecar_bytes)
    return V4RequestArtifact(
        root=root,
        request_dir=request_dir,
        manifest_path=manifest_path,
        sidecar_path=sidecar_path,
        header=header,
        work_units=tuple(records),
        task_contract_sha256=contract_digest.hex(),
        manifest_sha256=manifest_sha256,
        sidecar_sha256=_file_sha256(sidecar_path),
        output_paths=tuple(output_paths),
    )


def _validate_record_storage(
    records: Sequence[V4WorkUnitRecord], *, profile: V4Profile, canonical_k: int
) -> None:
    for record in records:
        if record.flags & FLAG_ZERO_WORK:
            if any(
                (
                    record.m_elements,
                    record.n_elements,
                    record.k_elements,
                    record.a_transfer_bytes,
                    record.b_transfer_bytes,
                    record.c_transfer_bytes,
                )
            ):
                raise ValueError("zero-work v4 record contains data")
            continue
        if (
            record.k_offset + record.k_elements > canonical_k
            or record.k_elements > MAX_CONTRACTED
        ):
            raise ValueError("v4 record K range is invalid")
        element_bytes = 4 if profile.numeric_mode_code == NUMERIC_MODE_FLOAT32 else 1
        expected = (
            _aligned8(record.m_elements * record.k_elements * element_bytes),
            _aligned8(record.k_elements * record.n_elements * element_bytes),
            _aligned8(record.m_elements * record.n_elements * 4),
        )
        actual = (
            record.a_transfer_bytes,
            record.b_transfer_bytes,
            record.c_transfer_bytes,
        )
        if expected != actual:
            raise ValueError("v4 record transfer sizes do not match native ABI")
        ranges = (
            (record.a_offset_bytes, record.a_transfer_bytes),
            (record.b_offset_bytes, record.b_transfer_bytes),
            (record.c_offset_bytes, record.c_transfer_bytes),
        )
        for offset, length in ranges:
            if offset % MRAM_ALIGNMENT or offset + length > profile.mram_pool_bytes:
                raise ValueError("v4 record violates MRAM alignment or bounds")
        for index, (left_offset, left_length) in enumerate(ranges):
            for right_offset, right_length in ranges[index + 1 :]:
                if (
                    left_offset < right_offset + right_length
                    and right_offset < left_offset + left_length
                ):
                    raise ValueError("v4 record MRAM ranges overlap")


class _OutputPump:
    def __init__(
        self,
        stream: Any,
        *,
        limit: int,
        output_queue: "queue.Queue[tuple[str, str | None]]",
    ) -> None:
        self.stream = stream
        self.limit = limit
        self.output_queue = output_queue
        self.chunks: list[str] = []
        self.size = 0
        self.overflowed = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                line = self.stream.readline()
                if line in ("", b""):
                    self.output_queue.put(("eof", None))
                    return
                text = (
                    line.decode(errors="replace")
                    if isinstance(line, bytes)
                    else str(line)
                )
                encoded = len(text.encode("utf-8", errors="replace"))
                if self.size + encoded > self.limit:
                    self.overflowed = True
                    self.output_queue.put(("overflow", None))
                    return
                self.size += encoded
                self.chunks.append(text)
                self.output_queue.put(("line", text))
        except BaseException:
            self.output_queue.put(("eof", None))


@dataclass(frozen=True)
class V4Release:
    event: Mapping[str, Any]
    release_confirmed: bool
    stdout: str
    stderr: str


class V4Session:
    """Persistent one-rank physical v4 native process client."""

    def __init__(
        self,
        process: Any,
        *,
        session_root: Path,
        profile: V4Profile,
        command: tuple[str, ...],
        stdout_pump: _OutputPump,
        stderr_pump: _OutputPump,
        events: "queue.Queue[tuple[str, str | None]]",
    ) -> None:
        self.process = process
        self.session_root = session_root.resolve()
        self.profile = profile
        self.command = command
        self._stdout_pump = stdout_pump
        self._stderr_pump = stderr_pump
        self._events = events
        self._last_sequence = -1
        self._closed = False
        self._poisoned = False
        self._release: V4Release | None = None
        self.startup: Mapping[str, Any] = {}

    @classmethod
    def start(
        cls,
        command: Sequence[str | Path],
        *,
        session_root: Path,
        profile: V4Profile,
        environment: Mapping[str, str] | None = None,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> "V4Session":
        root = Path(session_root).resolve()
        if not root.is_dir():
            raise V4Error(
                "hardware_profile_violation",
                "session_root must be an existing directory",
            )
        if profile.rank_path is None or not _RANK_PATH.fullmatch(profile.rank_path):
            raise V4Error(
                "hardware_profile_violation", "an explicit /dev/dpu_rankN is required"
            )
        env = dict(os.environ if environment is None else environment)
        if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
            raise V4Error(
                "hardware_opt_in_missing", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required"
            )
        forbidden = next(
            (name for name in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE") if name in env),
            None,
        )
        if forbidden is not None:
            raise V4Error(
                "hardware_profile_violation",
                f"{forbidden} must be unset for physical v4",
            )
        argv = tuple(str(value) for value in command)
        if not argv:
            raise V4Error("hardware_profile_violation", "v4 command cannot be empty")
        try:
            process = popen_factory(
                argv,
                cwd=str(root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise V4Error("sdk_discovery_failed", str(exc)) from exc
        events: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
        stdout_pump = _OutputPump(
            process.stdout, limit=profile.max_stdout_bytes, output_queue=events
        )
        stderr_pump = _OutputPump(
            process.stderr, limit=profile.max_stderr_bytes, output_queue=queue.Queue()
        )
        stdout_pump.start()
        stderr_pump.start()
        session = cls(
            process,
            session_root=root,
            profile=profile,
            command=argv,
            stdout_pump=stdout_pump,
            stderr_pump=stderr_pump,
            events=events,
        )
        event = session._next_event(profile.timeout_s)
        if event.get("event") != "READY" or event.get("status") != "ready":
            session._poisoned = True
            session.close()
            raise V4ProtocolError(
                str(event.get("failure_stage") or "hardware_allocation_failed"),
                str(event.get("error") or "v4 native process did not become ready"),
            )
        session._validate_ready(event)
        session.startup = dict(event)
        return session

    def _stdout_text(self) -> str:
        return "".join(self._stdout_pump.chunks)

    def _stderr_text(self) -> str:
        return "".join(self._stderr_pump.chunks)

    def _next_event(self, timeout_s: float) -> Mapping[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "kernel_timeout", "timed out waiting for v4 native event"
                )
            try:
                kind, value = self._events.get(timeout=remaining)
            except queue.Empty as exc:
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "kernel_timeout", "timed out waiting for v4 native event"
                ) from exc
            if kind == "overflow":
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "protocol_output_limit", "v4 stdout exceeded configured limit"
                )
            if kind == "eof":
                self._poisoned = True
                raise V4ProtocolError(
                    "kernel_launch_failed", "v4 native stdout closed before an event"
                )
            assert value is not None
            try:
                event = json.loads(value)
            except json.JSONDecodeError as exc:
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "protocol_error", "v4 native emitted malformed JSON"
                ) from exc
            if not isinstance(event, Mapping):
                self._poisoned = True
                self._terminate()
                raise V4ProtocolError(
                    "protocol_error", "v4 native event is not an object"
                )
            return event

    def _validate_ready(self, event: Mapping[str, Any]) -> None:
        required = {
            "target_observed": "physical_hardware",
            "requested_dpu_count": self.profile.dpu_count,
            "allocated_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "hardware_allocation_verified": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "rank_path": self.profile.rank_path,
        }
        for field, expected in required.items():
            if event.get(field) != expected:
                raise V4ProtocolError(
                    "hardware_allocation_failed",
                    f"READY field {field!r} is not verified",
                )

    def _write(self, text: str) -> None:
        if self._closed or self._poisoned:
            raise V4Error("session_closed", "v4 session is closed or poisoned")
        self._send_raw(text)

    def _send_raw(self, text: str) -> None:
        if self.process.stdin is None:
            raise V4ProtocolError(
                "kernel_launch_failed", "v4 native stdin is unavailable"
            )
        try:
            self.process.stdin.write(text)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._poisoned = True
            self._terminate()
            raise V4ProtocolError(
                "kernel_launch_failed", "v4 native stdin failed"
            ) from exc

    def submit(
        self, artifact: V4RequestArtifact, *, timeout_s: float | None = None
    ) -> Mapping[str, Any]:
        if self._closed or self._poisoned:
            raise V4Error("session_closed", "v4 session is closed or poisoned")
        if artifact.root.resolve() != self.session_root:
            raise V4Error(
                "request_manifest_failed",
                "request artifact root differs from session root",
            )
        sequence = artifact.request_sequence
        if sequence <= self._last_sequence:
            raise V4Error(
                "hardware_profile_violation", "v4 request sequence must increase"
            )
        manifest_rel = _relative_to(
            self.session_root, artifact.manifest_path, must_exist=True
        )
        actual_manifest_hash = _file_sha256(artifact.manifest_path)
        if actual_manifest_hash != artifact.manifest_sha256:
            self._poisoned = True
            self._terminate()
            raise V4ProtocolError(
                "request_manifest_failed", "request manifest hash mismatch"
            )
        if _file_sha256(artifact.sidecar_path) != artifact.sidecar_sha256:
            self._poisoned = True
            self._terminate()
            raise V4ProtocolError(
                "sidecar_validation_failed", "request sidecar hash mismatch"
            )
        self._validate_artifact_payloads(artifact)
        self._last_sequence = sequence
        self._write(f"SUBMIT {manifest_rel} {artifact.manifest_sha256}\n")
        event = self._next_event(timeout_s or self.profile.timeout_s)
        if event.get("event") != "RESPONSE":
            self._poisoned = True
            self.close()
            raise V4ProtocolError(
                "protocol_error", "v4 response event has the wrong type"
            )
        try:
            self._validate_response(event, artifact)
        except V4Error:
            self._poisoned = True
            self.close()
            raise
        return event

    def _validate_artifact_payloads(self, artifact: V4RequestArtifact) -> None:
        """Reject changed staged operands before the native process can use them."""

        for record in artifact.work_units:
            for name, relative_path, expected_digest in (
                ("A", record.a_path, record.a_sha256),
                ("B", record.b_path, record.b_sha256),
            ):
                if not _HEX_SHA256.fullmatch(expected_digest):
                    self._poisoned = True
                    self._terminate()
                    raise V4ProtocolError(
                        "payload_validation_failed",
                        f"v4 {name} payload digest is malformed",
                    )
                path = self.session_root / _safe_relative(relative_path)
                try:
                    actual_digest = _file_sha256(path)
                except OSError as exc:
                    self._poisoned = True
                    self._terminate()
                    raise V4ProtocolError(
                        "payload_validation_failed",
                        f"v4 {name} payload is unavailable",
                    ) from exc
                if actual_digest != expected_digest:
                    self._poisoned = True
                    self._terminate()
                    raise V4ProtocolError(
                        "payload_validation_failed",
                        f"v4 {name} payload digest mismatch",
                    )

    def _validate_response(
        self, event: Mapping[str, Any], artifact: V4RequestArtifact
    ) -> None:
        if event.get("status") != "completed":
            raise V4ProtocolError(
                str(event.get("failure_stage") or "kernel_launch_failed"),
                str(event.get("error") or "v4 request failed"),
            )
        required = {
            "target_requested": "hardware",
            "target_observed": "physical_hardware",
            "global_completeness": False,
            "task_contract_sha256": artifact.task_contract_sha256,
            "request_sha256": artifact.manifest_sha256,
            "request_manifest_sha256": artifact.manifest_sha256,
            "sidecar_sha256": artifact.sidecar_sha256,
            "rank_path": self.profile.rank_path,
            "dispatch_mode": "bulk_set_synchronous_v1",
            "bulk_set_launch_verified": True,
            "requested_dpu_count": self.profile.dpu_count,
            "allocated_dpu_count": self.profile.dpu_count,
            "tasklets_per_dpu": self.profile.tasklets_per_dpu,
            "hardware_allocation_verified": True,
            "native_kernel_executed": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_functionality_evidence": True,
        }
        if event.get("global_output_elements") != artifact.global_output_elements:
            raise V4ProtocolError(
                "protocol_error", "v4 response global output size mismatch"
            )
        for field, expected in required.items():
            if event.get(field) != expected:
                raise V4ProtocolError(
                    "protocol_error", f"v4 response field {field!r} is not verified"
                )
        if event.get("request_sequence") != artifact.request_sequence:
            raise V4ProtocolError("protocol_error", "v4 response sequence mismatch")
        if event.get("request_output_elements") != artifact.request_output_elements:
            raise V4ProtocolError(
                "protocol_error", "v4 response output coverage mismatch"
            )
        if (
            not isinstance(event.get("per_dpu"), list)
            or len(event["per_dpu"]) != self.profile.dpu_count
        ):
            raise V4ProtocolError(
                "protocol_error", "v4 response lacks one result per requested DPU"
            )
        transfer = event.get("transfer")
        if not isinstance(transfer, Mapping):
            raise V4ProtocolError(
                "protocol_error", "v4 response transfer summary is missing"
            )
        summary_h2d = self._nonnegative_int(
            transfer.get("h2d_bytes"), "transfer.h2d_bytes"
        )
        summary_d2h = self._nonnegative_int(
            transfer.get("d2h_bytes"), "transfer.d2h_bytes"
        )
        summary_total = self._nonnegative_int(
            transfer.get("total_bytes"), "transfer.total_bytes"
        )
        if summary_total != summary_h2d + summary_d2h:
            raise V4ProtocolError(
                "protocol_error", "v4 response transfer total is inconsistent"
            )

        records_by_dpu = {record.local_dpu_id: record for record in artifact.work_units}
        seen_dpu_ids: set[int] = set()
        per_dpu_h2d = 0
        per_dpu_d2h = 0
        for result in event["per_dpu"]:
            if not isinstance(result, Mapping):
                raise V4ProtocolError(
                    "protocol_error", "v4 response per-DPU record is invalid"
                )
            dpu_id = self._nonnegative_int(result.get("dpu_id"), "per_dpu.dpu_id")
            if dpu_id in seen_dpu_ids or dpu_id not in records_by_dpu:
                raise V4ProtocolError(
                    "protocol_error", "v4 response DPU IDs are not dense and unique"
                )
            seen_dpu_ids.add(dpu_id)
            record = records_by_dpu[dpu_id]
            if result.get("tile_id") != record.tile_id:
                raise V4ProtocolError("protocol_error", "v4 response tile ID mismatch")
            if result.get("completion_status") != STATUS_COMPLETED:
                raise V4ProtocolError(
                    "protocol_error", "v4 response completion status is not completed"
                )
            expected_elements = (
                0
                if record.flags & FLAG_ZERO_WORK
                else record.m_elements * record.n_elements
            )
            if result.get("processed_elements") != expected_elements:
                raise V4ProtocolError(
                    "protocol_error", "v4 response processed element count mismatch"
                )
            expected_h2d = (
                record.a_transfer_bytes + record.b_transfer_bytes + CONTROL_BYTES
            )
            expected_d2h = record.c_transfer_bytes + COMPLETION_BYTES
            result_h2d = self._nonnegative_int(
                result.get("h2d_bytes"), "per_dpu.h2d_bytes"
            )
            result_d2h = self._nonnegative_int(
                result.get("d2h_bytes"), "per_dpu.d2h_bytes"
            )
            if result_h2d != expected_h2d or result_d2h != expected_d2h:
                raise V4ProtocolError(
                    "protocol_error", "v4 response per-DPU transfer accounting mismatch"
                )
            per_dpu_h2d += result_h2d
            per_dpu_d2h += result_d2h
        if seen_dpu_ids != set(range(self.profile.dpu_count)):
            raise V4ProtocolError(
                "protocol_error", "v4 response is missing a dense DPU result"
            )
        if (summary_h2d, summary_d2h) != (per_dpu_h2d, per_dpu_d2h):
            raise V4ProtocolError(
                "protocol_error", "v4 response aggregate transfer accounting mismatch"
            )

    @staticmethod
    def _nonnegative_int(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise V4ProtocolError(
                "protocol_error",
                f"v4 response field {field!r} is not a non-negative integer",
            )
        return value

    def close(self, *, timeout_s: float | None = None) -> V4Release:
        if self._release is not None:
            return self._release
        if self._closed:
            return V4Release({}, False, self._stdout_text(), self._stderr_text())
        alive_before_close = self.process.poll() is None
        self._closed = True
        sent_close = True
        try:
            self._send_raw("CLOSE\n")
        except V4Error:
            self._poisoned = True
            sent_close = False
        release: V4Release | None = None
        if alive_before_close and sent_close:
            try:
                event = self._next_event(timeout_s or self.profile.timeout_s)
                confirmed = (
                    event.get("event") == "RELEASE"
                    and event.get("status") == "released"
                    and event.get("release_succeeded") is True
                    and event.get("dpu_free_called_once") is True
                )
                release = V4Release(
                    event, confirmed, self._stdout_text(), self._stderr_text()
                )
            except V4Error:
                self._poisoned = True
        if release is None:
            self._terminate()
            release = V4Release({}, False, self._stdout_text(), self._stderr_text())
        self._release = release
        return release

    def _terminate(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
        except (OSError, subprocess.SubprocessError):
            pass

    def __enter__(self) -> "V4Session":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


V4HardwareSession = V4Session
build_request = build_v4_request


__all__ = [
    "COMPLETION_BYTES",
    "COMPLETION_FORMAT",
    "CONTROL_BYTES",
    "CONTROL_FORMAT",
    "FLAG_ZERO_WORK",
    "HEADER_BYTES",
    "HEADER_FORMAT",
    "MAX_CONTRACTED",
    "MAX_DPUS",
    "MAX_INT32_SAFE_K",
    "MAX_TASKLETS",
    "MRAM_ALIGNMENT",
    "MRAM_POOL_BYTES",
    "NUMERIC_FLOAT32",
    "NUMERIC_HOST_PACKED_INT8",
    "NUMERIC_MODE_FLOAT32",
    "NUMERIC_MODE_HOST_PACKED_INT8",
    "PARTITION_OUTPUT_TILE",
    "PROFILE",
    "RESPONSE_SCHEMA",
    "V4Error",
    "V4Completion",
    "V4Header",
    "V4HardwareSession",
    "V4Profile",
    "V4ProtocolError",
    "V4Release",
    "V4RequestArtifact",
    "V4Session",
    "V4WorkUnit",
    "V4WorkUnitRecord",
    "WORK_UNIT_BYTES",
    "WORK_UNIT_FORMAT",
    "build_request",
    "build_v4_request",
    "pack_v4_header",
    "pack_v4_completion",
    "pack_v4_work_unit",
    "unpack_v4_completion",
    "unpack_v4_header",
    "unpack_v4_work_unit",
]
