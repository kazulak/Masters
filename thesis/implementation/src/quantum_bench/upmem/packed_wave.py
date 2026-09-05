"""Private packed cohort/wave envelope codec.

This module owns only the byte representation and its Python-side checks.  It
does not open envelope files, hash files, schedule work, or provide a runtime
route.  The native host receives the same fixed tables and validates the
whole-file digest and executable identity at its file boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import struct
from typing import TypeAlias

from quantum_bench.upmem.wave_protocol import IDLE, CONTROL, WaveControl


MAGIC = b"UPWAVE1\0"
VERSION = 1
HEADER_BYTES = 136
OPERATION_BYTES = 112
CONTROL_BYTES = CONTROL.size
TILE_BYTES = 160
FLAGS = 0
MAX_DPUS = 64
MAX_TASKLETS = 24
MAX_K = 65536
MAX_ENVELOPE_BYTES = 512 * 1024 * 1024

HEADER_FORMAT = "<8s8I4Q32s32s"
OPERATION_FORMAT = "<32s32s4Q2d"
TILE_PREFIX_FORMAT = "<2Q"
HEADER = struct.Struct(HEADER_FORMAT)
OPERATION = struct.Struct(OPERATION_FORMAT)
TILE_PREFIX = struct.Struct(TILE_PREFIX_FORMAT)

assert HEADER.size == HEADER_BYTES
assert OPERATION.size == OPERATION_BYTES
assert TILE_PREFIX.size + CONTROL_BYTES == TILE_BYTES

_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_UINT32_MAX = (1 << 32) - 1


@dataclass(frozen=True)
class WaveOperation:
    """The minimal canonical operation identity and geometry table record."""

    node_sha256: bytes
    contract_sha256: bytes
    batch_count: int
    m: int
    n: int
    k: int
    left_scale: float
    right_scale: float


@dataclass(frozen=True)
class WaveTile:
    """One dense DPU slot and its four input-plane payloads."""

    control: WaveControl
    m_offset: int
    n_offset: int
    inputs: tuple[bytes, bytes, bytes, bytes]


WaveRecords: TypeAlias = tuple[
    tuple[WaveOperation, ...], tuple[tuple[WaveTile, ...], ...]
]


def _digest_bytes(value: object, field: str, *, nonzero: bool = True) -> bytes:
    if isinstance(value, str):
        if not _HEX_SHA256.fullmatch(value):
            raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
        result = bytes.fromhex(value)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        result = bytes(value)
    else:
        raise TypeError(f"{field} must be a 32-byte digest or SHA-256 hex string")
    if len(result) != 32:
        raise ValueError(f"{field} must contain exactly 32 digest bytes")
    if nonzero and not any(result):
        raise ValueError(f"{field} digest must be nonzero")
    return result


def _uint(value: object, bits: int, field: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    maximum = (1 << bits) - 1
    minimum = 1 if positive else 0
    if not minimum <= value <= maximum:
        qualifier = "positive " if positive else ""
        raise ValueError(f"{field} must be a {qualifier}uint{bits}")
    return value


def _scale(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a finite positive scale")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be a finite positive scale") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be a finite positive scale")
    return result


def _validate_operation(
    operation: WaveOperation, numeric_mode: int, index: int
) -> tuple[bytes, bytes, int, int, int, int, float, float]:
    if not isinstance(operation, WaveOperation):
        raise TypeError(f"operation {index} must be a WaveOperation")
    node_digest = _digest_bytes(operation.node_sha256, "node_sha256")
    contract_digest = _digest_bytes(operation.contract_sha256, "contract_sha256")
    batch_count = _uint(
        operation.batch_count, 32, f"operation {index} batch_count", positive=True
    )
    m = _uint(operation.m, 64, f"operation {index} m", positive=True)
    n = _uint(operation.n, 64, f"operation {index} n", positive=True)
    k = _uint(operation.k, 64, f"operation {index} k", positive=True)
    if k > MAX_K:
        raise ValueError(f"operation {index} k exceeds {MAX_K}")
    left_scale = _scale(operation.left_scale, f"operation {index} left_scale")
    right_scale = _scale(operation.right_scale, f"operation {index} right_scale")
    if numeric_mode == 0 and (left_scale != 1.0 or right_scale != 1.0):
        raise ValueError("float32 wave operations require unit scales")
    return (
        node_digest,
        contract_digest,
        batch_count,
        m,
        n,
        k,
        left_scale,
        right_scale,
    )


def _input_bytes(value: object, field: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{field} must be bytes-like")
    if isinstance(value, bytes):
        return value
    return bytes(value)


def _validate_inputs(
    tile: WaveTile, *, numeric_mode: int, tile_index: int
) -> tuple[bytes, bytes, bytes, bytes]:
    if type(tile.inputs) is not tuple or len(tile.inputs) != 4:
        raise ValueError(f"tile {tile_index} must have exactly four input planes")
    inputs = tuple(
        _input_bytes(value, f"tile {tile_index} input plane {plane}")
        for plane, value in enumerate(tile.inputs)
    )
    control = tile.control
    if control.flags == IDLE:
        expected_lengths = (0, 0, 0, 0)
        logical_lengths = expected_lengths
    else:
        element_bytes = 4 if numeric_mode == 0 else 1
        logical_lengths = (
            control.m * control.k * element_bytes,
            control.m * control.k * element_bytes,
            control.k * control.n * element_bytes,
            control.k * control.n * element_bytes,
        )
        expected_lengths = tuple(length for _, length in control.planes[:4])
    for plane, (payload, expected, logical) in enumerate(
        zip(inputs, expected_lengths, logical_lengths)
    ):
        if len(payload) != expected:
            raise ValueError(
                f"tile {tile_index} input plane {plane} length differs from control"
            )
        if expected == 0:
            continue
        if logical > expected:
            raise ValueError(f"tile {tile_index} input plane {plane} geometry is invalid")
        if any(payload[logical:]):
            raise ValueError(f"tile {tile_index} input plane {plane} has nonzero padding")
        if numeric_mode == 1:
            if 128 in payload[:logical]:
                raise ValueError(
                    f"tile {tile_index} input plane {plane} contains forbidden int8 -128"
                )
        else:
            for (value,) in struct.iter_unpack("<f", payload[:logical]):
                if not math.isfinite(value):
                    raise ValueError(f"tile {tile_index} input plane {plane} is nonfinite")
    return inputs  # type: ignore[return-value]


def _validate_tile_offsets(tile: WaveTile, tile_index: int) -> tuple[int, int]:
    m_offset = _uint(tile.m_offset, 64, f"tile {tile_index} m_offset")
    n_offset = _uint(tile.n_offset, 64, f"tile {tile_index} n_offset")
    return m_offset, n_offset


def _validate_cohort(
    operations: tuple[WaveOperation, ...],
    waves: tuple[tuple[WaveTile, ...], ...],
    *,
    numeric_mode: int,
    dpus: int,
    tasklets: int,
) -> tuple[
    tuple[tuple[bytes, bytes, int, int, int, int, float, float], ...],
    tuple[bytes, ...],
]:
    operation_values = tuple(
        _validate_operation(operation, numeric_mode, index)
        for index, operation in enumerate(operations)
    )
    node_digests = [values[0] for values in operation_values]
    if len(set(node_digests)) != len(node_digests):
        raise ValueError("duplicate wave operation identity")

    previous_wave_id: int | None = None
    previous_request_sequence: int | None = None
    used_operations: set[int] = set()
    owner_operations: list[int | None] = [None] * dpus
    payload_parts: list[bytes] = []
    for wave_index, wave in enumerate(waves):
        if len(wave) != dpus:
            raise ValueError(f"wave {wave_index} must contain exactly {dpus} DPU tiles")
        active = 0
        wave_id: int | None = None
        request_sequence: int | None = None
        regions: dict[int, list[tuple[int, int, int, int, int]]] = {}
        tile_ids: dict[int, set[int]] = {}
        for dpu_id, tile in enumerate(wave):
            tile_index = wave_index * dpus + dpu_id
            if not isinstance(tile, WaveTile):
                raise TypeError(f"tile {tile_index} must be a WaveTile")
            if not isinstance(tile.control, WaveControl):
                raise TypeError(f"tile {tile_index} control must be a WaveControl")
            control = tile.control
            control.validate()
            if control.dpu_id != dpu_id:
                raise ValueError(f"tile {tile_index} has the wrong DPU identity")
            if control.tasklets != tasklets:
                raise ValueError(f"tile {tile_index} tasklet count disagrees with header")
            if control.numeric_mode != numeric_mode:
                raise ValueError(f"tile {tile_index} numeric mode disagrees with header")
            _uint(control.wave_id, 64, f"tile {tile_index} wave_id")
            _uint(control.request_sequence, 64, f"tile {tile_index} request_sequence")
            if wave_id is None:
                wave_id = control.wave_id
                request_sequence = control.request_sequence
            elif (control.wave_id, control.request_sequence) != (
                wave_id,
                request_sequence,
            ):
                raise ValueError("wave controls have inconsistent identities")
            m_offset, n_offset = _validate_tile_offsets(tile, tile_index)
            payload_parts.extend(
                _validate_inputs(tile, numeric_mode=numeric_mode, tile_index=tile_index)
            )
            if control.flags == IDLE:
                if m_offset or n_offset:
                    raise ValueError("idle tile has output offsets")
                continue

            active += 1
            operation_index = control.operation_index
            if operation_index >= len(operation_values):
                raise ValueError("wave control references absent operation")
            owner = owner_operations[dpu_id]
            if owner is not None and owner != operation_index:
                raise ValueError("DPU group changes ownership inside a prepared cohort")
            owner_operations[dpu_id] = operation_index
            used_operations.add(operation_index)
            batch_count, canonical_m, canonical_n, canonical_k = (
                operation_values[operation_index][2:6]
            )
            if (
                control.batch_index >= batch_count
                or m_offset > canonical_m
                or control.m > canonical_m - m_offset
                or n_offset > canonical_n
                or control.n > canonical_n - n_offset
                or control.k_offset > canonical_k
                or control.k > canonical_k - control.k_offset
            ):
                raise ValueError("wave tile exceeds canonical operation geometry")

            operation_regions = regions.setdefault(operation_index, [])
            operation_tile_ids = tile_ids.setdefault(operation_index, set())
            if control.tile_id in operation_tile_ids:
                raise ValueError("duplicate wave tile identity")
            operation_tile_ids.add(control.tile_id)
            for (
                other_batch,
                other_m_offset,
                other_n_offset,
                other_m,
                other_n,
            ) in operation_regions:
                if other_batch != control.batch_index:
                    continue
                if (
                    other_m_offset < m_offset + control.m
                    and m_offset < other_m_offset + other_m
                    and other_n_offset < n_offset + control.n
                    and n_offset < other_n_offset + other_n
                ):
                    raise ValueError("overlapping wave outputs")
            operation_regions.append(
                (
                    control.batch_index,
                    m_offset,
                    n_offset,
                    control.m,
                    control.n,
                )
            )
        if not active:
            raise ValueError("wave has no active DPU")
        if wave_id is None or request_sequence is None:
            raise AssertionError("validated wave has no identity")
        if previous_wave_id is not None:
            assert previous_request_sequence is not None
            if wave_id <= previous_wave_id or request_sequence <= previous_request_sequence:
                raise ValueError("wave/request identities must increase")
        previous_wave_id = wave_id
        previous_request_sequence = request_sequence
    if used_operations != set(range(len(operations))):
        raise ValueError("unused wave operation")
    return operation_values, tuple(payload_parts)


def pack_wave_envelope(
    *,
    plan_sha256: object,
    dpu_binary_sha256: object,
    sequence: object,
    operations: tuple[WaveOperation, ...],
    waves: tuple[tuple[WaveTile, ...], ...],
    numeric_mode: object,
) -> bytes:
    """Pack and validate one deterministic dense cohort/wave envelope."""

    if type(waves) is not tuple or not waves or any(
        type(wave) is not tuple for wave in waves
    ):
        raise TypeError("waves must be nonempty nested wave tuples")
    operation_records = tuple(operations)
    plan_digest = _digest_bytes(plan_sha256, "plan_sha256")
    binary_digest = _digest_bytes(dpu_binary_sha256, "dpu_binary_sha256")
    envelope_sequence = _uint(sequence, 64, "sequence")
    if type(numeric_mode) is not int or numeric_mode not in (0, 1):
        raise ValueError("numeric_mode must be 0 or 1")
    if not 1 <= len(operation_records) <= MAX_DPUS:
        raise ValueError("operation_count must be in [1, 64]")
    if len(waves) > _UINT32_MAX:
        raise ValueError("wave_count exceeds uint32")
    dpus = len(waves[0])
    if not 1 <= dpus <= MAX_DPUS:
        raise ValueError("DPU count must be in [1, 64]")
    if len(operation_records) > dpus:
        raise ValueError("operation_count exceeds DPU count")
    if not isinstance(waves[0][0], WaveTile) or not isinstance(
        waves[0][0].control, WaveControl
    ):
        raise TypeError("waves must contain WaveTile controls")
    tasklets = waves[0][0].control.tasklets
    _uint(tasklets, 32, "tasklets", positive=True)
    if tasklets > MAX_TASKLETS:
        raise ValueError("tasklets must be in [1, 24]")
    operation_values, payload_parts = _validate_cohort(
        operation_records,
        waves,
        numeric_mode=numeric_mode,
        dpus=dpus,
        tasklets=tasklets,
    )
    operation_bytes = b"".join(OPERATION.pack(*values) for values in operation_values)
    tile_parts: list[bytes] = []
    for wave in waves:
        for tile in wave:
            tile_parts.append(TILE_PREFIX.pack(tile.m_offset, tile.n_offset) + tile.control.to_bytes())
    tile_bytes = b"".join(tile_parts)
    payload_bytes = sum(len(part) for part in payload_parts)
    control_count = dpus * len(waves)
    payload_offset = HEADER_BYTES + len(operation_records) * OPERATION_BYTES + control_count * TILE_BYTES
    total_bytes = payload_offset + payload_bytes
    if total_bytes > MAX_ENVELOPE_BYTES:
        raise ValueError("wave envelope exceeds the 512-MiB snapshot admission limit")
    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        dpus,
        tasklets,
        len(operation_records),
        len(waves),
        numeric_mode,
        FLAGS,
        envelope_sequence,
        control_count,
        total_bytes,
        payload_offset,
        plan_digest,
        binary_digest,
    )
    result = b"".join((header, operation_bytes, tile_bytes, *payload_parts))
    if len(result) != total_bytes:
        raise AssertionError("wave envelope size drift")
    return result


def _coerce_file(data: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("wave envelope data must be bytes-like")
    if memoryview(data).nbytes > MAX_ENVELOPE_BYTES:
        raise ValueError("wave envelope exceeds the 512-MiB snapshot admission limit")
    if isinstance(data, bytes):
        return data
    return bytes(data)


def _read_header(data: bytes) -> tuple[int, int, int, int, int, int, bytes, bytes]:
    if len(data) < HEADER_BYTES:
        raise ValueError("truncated wave envelope header")
    (
        magic,
        version,
        header_bytes,
        dpus,
        tasklets,
        operation_count,
        wave_count,
        numeric_mode,
        flags,
        sequence,
        control_count,
        total_bytes,
        payload_offset,
        plan_digest,
        binary_digest,
    ) = HEADER.unpack_from(data)
    if (
        magic != MAGIC
        or version != VERSION
        or header_bytes != HEADER_BYTES
        or not 1 <= dpus <= MAX_DPUS
        or not 1 <= tasklets <= MAX_TASKLETS
        or not 1 <= operation_count <= MAX_DPUS
        or not 1 <= wave_count <= _UINT32_MAX
        or numeric_mode not in (0, 1)
        or flags != FLAGS
    ):
        raise ValueError("invalid wave envelope header")
    if operation_count > dpus:
        raise ValueError("operation_count exceeds DPU count")
    if not any(plan_digest) or not any(binary_digest):
        raise ValueError("wave envelope digest identity must be nonzero")
    expected_controls = dpus * wave_count
    if control_count != expected_controls:
        raise ValueError("invalid dense wave control count")
    # Check the fixed table against the actual provided file before parsing any
    # count-derived records.  This keeps malformed large wave counts bounded by
    # the bytes the caller actually supplied.
    operation_end = HEADER_BYTES + operation_count * OPERATION_BYTES
    if operation_end > len(data):
        raise ValueError("truncated wave operation table")
    if expected_controls > (len(data) - operation_end) // TILE_BYTES:
        raise ValueError("truncated wave descriptor table")
    derived_payload_offset = operation_end + expected_controls * TILE_BYTES
    if payload_offset != derived_payload_offset:
        raise ValueError("noncanonical wave payload offset")
    if total_bytes != len(data):
        raise ValueError("wave envelope total_bytes does not match file length")
    if payload_offset > total_bytes:
        raise ValueError("wave envelope payload offset exceeds total length")
    return (
        dpus,
        tasklets,
        operation_count,
        wave_count,
        numeric_mode,
        sequence,
        plan_digest,
        binary_digest,
    )


def _decode_records(
    data: bytes,
) -> WaveRecords:
    (
        dpus,
        tasklets,
        operation_count,
        wave_count,
        numeric_mode,
        _sequence,
        _plan_digest,
        _binary_digest,
    ) = _read_header(data)

    operation_records: list[WaveOperation] = []
    operation_offset = HEADER_BYTES
    for index in range(operation_count):
        values = OPERATION.unpack_from(data, operation_offset + index * OPERATION_BYTES)
        operation_records.append(WaveOperation(*values))
    operations = tuple(operation_records)

    tile_offset = operation_offset + operation_count * OPERATION_BYTES
    waves: list[tuple[WaveTile, ...]] = []
    tile_records: list[tuple[WaveTile, ...]] = []
    payload_lengths = 0
    for wave_index in range(wave_count):
        wave_tiles: list[WaveTile] = []
        for dpu_id in range(dpus):
            offset = tile_offset + (wave_index * dpus + dpu_id) * TILE_BYTES
            m_offset, n_offset = TILE_PREFIX.unpack_from(data, offset)
            control = WaveControl.from_bytes(
                data[offset + TILE_PREFIX.size : offset + TILE_BYTES]
            )
            payload_lengths += sum(length for _, length in control.planes[:4])
            wave_tiles.append(WaveTile(control, m_offset, n_offset, (b"", b"", b"", b"")))
        tile_records.append(tuple(wave_tiles))

    payload_offset = HEADER_BYTES + operation_count * OPERATION_BYTES + dpus * wave_count * TILE_BYTES
    if payload_lengths > len(data) - payload_offset:
        raise ValueError("truncated wave input payload")
    cursor = payload_offset
    for wave_index, wave in enumerate(tile_records):
        decoded_tiles: list[WaveTile] = []
        for dpu_id, tile in enumerate(wave):
            inputs: list[bytes] = []
            for _, length in tile.control.planes[:4]:
                if length > len(data) - cursor:
                    raise ValueError("truncated wave input payload")
                inputs.append(data[cursor : cursor + length])
                cursor += length
            decoded_tiles.append(
                WaveTile(tile.control, tile.m_offset, tile.n_offset, tuple(inputs))  # type: ignore[arg-type]
            )
        waves.append(tuple(decoded_tiles))
    normalized_waves = tuple(waves)
    _validate_cohort(
        operations,
        normalized_waves,
        numeric_mode=numeric_mode,
        dpus=dpus,
        tasklets=tasklets,
    )
    if cursor != len(data):
        raise ValueError("trailing wave payload bytes")
    return operations, normalized_waves


def validate_wave_envelope(
    data: bytes | bytearray | memoryview,
) -> None:
    """Validate an envelope without opening files or calculating file hashes."""

    _decode_records(_coerce_file(data))


def unpack_wave_envelope(
    data: bytes | bytearray | memoryview,
) -> WaveRecords:
    """Validate and return ``(operations, wave_tiles)`` records."""

    return _decode_records(_coerce_file(data))
