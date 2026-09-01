from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import pytest

from quantum_bench.upmem.packed_operation import (
    PACKED_OPERATION_DESCRIPTOR_BYTES,
    PACKED_OPERATION_HEADER_BYTES,
    PACKED_OPERATION_MAGIC,
    PackedV4Request,
    build_packed_v4_request,
    pack_operation,
)
from quantum_bench.upmem.protocol import (
    V4Profile,
    V4WorkUnit,
    build_v4_request,
)


TASK_HASH = "ab" * 32
HEADER_FORMAT = "<8s6I4Q32s"


def _units() -> tuple[V4WorkUnit, ...]:
    left = struct.pack("<3f", 1.0, 2.0, 3.0)
    right = struct.pack("<3f", 4.0, 5.0, 6.0)
    return (
        V4WorkUnit(
            local_dpu_id=0,
            tile_id=11,
            batch_index=0,
            m_offset=0,
            n_offset=0,
            k_offset=0,
            m_elements=1,
            n_elements=1,
            k_elements=3,
            a_payload=left,
            b_payload=right,
        ),
        V4WorkUnit(
            local_dpu_id=1,
            tile_id=12,
            batch_index=0,
            m_offset=0,
            n_offset=1,
            k_offset=0,
            m_elements=1,
            n_elements=1,
            k_elements=3,
            a_payload=left,
            b_payload=right,
        ),
    )


def _packed_request(root: Path, sequence: int) -> PackedV4Request:
    root.mkdir(parents=True, exist_ok=True)
    return build_packed_v4_request(
        root,
        profile=V4Profile(dpu_count=2),
        canonical_batch_count=1,
        canonical_m=1,
        canonical_n=2,
        canonical_k=3,
        work_units=_units(),
        task_contract_sha256=TASK_HASH,
        request_sequence=sequence,
    )


def test_packed_request_matches_directory_request_bytes(tmp_path: Path) -> None:
    profile = V4Profile(dpu_count=2)
    directory = build_v4_request(
        tmp_path / "directory",
        profile=profile,
        canonical_batch_count=1,
        canonical_m=1,
        canonical_n=2,
        canonical_k=3,
        work_units=_units(),
        task_contract_sha256=TASK_HASH,
        request_sequence=7,
    )
    packed = _packed_request(tmp_path / "packed", 7)

    assert packed.manifest_bytes == directory.manifest_path.read_bytes()
    assert packed.sidecar_bytes == directory.sidecar_path.read_bytes()
    assert packed.work_units == directory.work_units
    assert packed.manifest_sha256 == directory.manifest_sha256
    assert packed.sidecar_sha256 == directory.sidecar_sha256
    assert packed.payload_bytes == b"".join(
        (directory.root / record.a_path).read_bytes()
        + (directory.root / record.b_path).read_bytes()
        for record in directory.work_units
    )


def test_packed_operation_has_variable_descriptor_count_and_verified_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "session"
    root.mkdir()
    requests = tuple(_packed_request(root, index) for index in range(65))
    operation = pack_operation(root, requests=requests, operation_sequence=0)

    header = struct.unpack(
        HEADER_FORMAT, operation.data[:PACKED_OPERATION_HEADER_BYTES]
    )
    assert header[:7] == (PACKED_OPERATION_MAGIC, 2, 96, 65, 200, 0, 0)
    assert header[7] == PACKED_OPERATION_HEADER_BYTES
    assert header[8] == 96 + 65 * PACKED_OPERATION_DESCRIPTOR_BYTES
    assert header[9] == len(operation.data)
    assert operation.sha256 == hashlib.sha256(operation.data).hexdigest()

    unsigned = bytearray(operation.data)
    expected_digest = bytes(unsigned[64:96])
    unsigned[64:96] = b"\0" * 32
    assert hashlib.sha256(unsigned).digest() == expected_digest


def test_packed_operation_requires_ordered_sequences(tmp_path: Path) -> None:
    root = tmp_path / "session"
    root.mkdir()
    first = _packed_request(root, 3)
    second = _packed_request(root, 4)
    with pytest.raises(ValueError, match="sequences"):
        pack_operation(root, requests=(second, first), operation_sequence=4)


def test_packed_operation_rejects_paths_outside_session_root(tmp_path: Path) -> None:
    root = tmp_path / "session"
    root.mkdir()
    request = _packed_request(root, 1)
    with pytest.raises(ValueError, match="unsafe"):
        pack_operation(root, requests=(request,), operation_sequence=1, filename="../operation.bin")
