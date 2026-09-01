#!/usr/bin/env python3
"""Host-only feasibility probe for the packed operation boundary.

UPOENV1 is retained as the original compatibility probe.  UPOENV2 is the
variable-count operation envelope used by the replay.  Neither path opens a
rank, imports an SDK, or calls the production runtime.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import random
import struct
import subprocess
import sys
import tempfile
import time
from typing import Sequence

from quantum_bench.circuits import builtin_circuit
from quantum_bench.lowering import build_contraction_dag, lower_tensor_network
from quantum_bench.model import ContractNode, make_simulation_job
from quantum_bench.planning import plan_opt_einsum
from quantum_bench.upmem.plan import UpmemPlan, UpmemTopology, plan_upmem
from quantum_bench.upmem.protocol import (
    HEADER_BYTES,
    NUMERIC_FLOAT32,
    NUMERIC_MODE_FLOAT32,
    PARTITION_OUTPUT_TILE,
    VERSION,
    V4Header,
    V4Profile,
    V4RequestArtifact,
    V4WorkUnit,
    WORK_UNIT_BYTES,
    build_v4_request,
)
from quantum_bench.upmem.tiling import canonical_label_geometry


UPOENV1_MAGIC = b"UPOENV1\0"
UPOENV2_MAGIC = b"UPOENV2\0"
ENVELOPE_MAGIC = UPOENV1_MAGIC
ENVELOPE_VERSION = 1
UPOENV2_VERSION = 2
ENVELOPE_HEADER_FORMAT = "<8s6I4Q32s"
DESCRIPTOR_FORMAT = "<5Q2IQ32s32s32s"
UPOENV2_HEADER_FORMAT = "<8s6I4Q32s"
UPOENV2_DESCRIPTOR_FORMAT = "<7Q2IQ32s32s32s32s"
ENVELOPE_HEADER_BYTES = 96
UPOENV2_HEADER_BYTES = 96
DESCRIPTOR_BYTES = 152
UPOENV2_DESCRIPTOR_BYTES = 200
MAX_REQUESTS = 64
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
TASK_CONTRACT_SHA256 = "42" * 32
SUMMARY_SCHEMA = "upoenv1_probe_summary_v1"
UPOENV2_SUMMARY_SCHEMA = "upoenv2_probe_summary_v1"
CELL_DEFINITIONS = (
    ("quantization_stress_18q_l2", 1, 141, 888),
    ("quantization_stress_18q_l2", 4, 141, 636),
    ("hs_18q_d1", 1, 53, 224),
    ("hs_18q_d1", 4, 53, 212),
    ("ghz_chain_18q", 1, 35, 1484),
    ("ghz_chain_18q", 4, 35, 464),
)
REAL_CASE_IDS = (
    "quantization_stress_18q_l2",
    "hs_18q_d1",
    "ghz_chain_18q",
)
TASKLETS_PER_DPU = 8
COMPLEX_LANE_COUNT = 4
REPLAY_SCHEMA = "host_boundary_real_corpus_replay_v1"
BENCHMARK_SCHEMA = "host_request_boundary_feasibility_v2"
RAW_SCHEMA = "host_boundary_benchmark_raw_v1"
PHASE0_SCHEMA = "host_boundary_phase0_decision_v1"
RAW_RANDOM_SEED = 20260901

_ENVELOPE_DIGEST_OFFSET = 64
_DESCRIPTOR_DIGEST_OFFSET = 120
_UPOENV2_DESCRIPTOR_DIGEST_OFFSET = 168
_IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PackedDescriptor:
    """The descriptor values committed into one UPOENV1 descriptor."""

    request_sequence: int
    request_offset: int
    request_bytes: int
    payload_offset: int
    payload_bytes: int
    work_unit_count: int
    request_output_elements: int
    request_sha256: str
    payload_sha256: str
    descriptor_sha256: str


@dataclass(frozen=True)
class PackedDescriptorV2:
    """The descriptor values committed into one UPOENV2 descriptor."""

    request_sequence: int
    manifest_offset: int
    manifest_bytes: int
    sidecar_offset: int
    sidecar_bytes: int
    payload_offset: int
    payload_bytes: int
    work_unit_count: int
    request_output_elements: int
    manifest_sha256: str
    sidecar_sha256: str
    payload_sha256: str
    descriptor_sha256: str


@dataclass(frozen=True)
class _PreparedRequest:
    """Prepared bytes used by either envelope arm."""

    request_sequence: int
    sidecar: bytes
    payload: bytes
    work_unit_count: int
    request_output_elements: int
    manifest: bytes = b""


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _require_count(value: int, *, cap: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("descriptor count must be a positive integer")
    if value > UINT32_MAX:
        raise ValueError("descriptor count exceeds uint32")
    if cap is not None and value > cap:
        raise ValueError(f"descriptor count exceeds {cap}")


def _checked_add(left: int, right: int, field: str) -> int:
    result = left + right
    if result > UINT64_MAX:
        raise ValueError(f"{field} exceeds uint64")
    return result


def _checked_mul(left: int, right: int, field: str) -> int:
    result = left * right
    if result > UINT64_MAX:
        raise ValueError(f"{field} exceeds uint64")
    return result


def _descriptor_prefix(
    *,
    request_sequence: int,
    request_offset: int,
    request_bytes: int,
    payload_offset: int,
    payload_bytes: int,
    work_unit_count: int,
    request_output_elements: int,
    request_sha256: bytes,
    payload_sha256: bytes,
) -> bytes:
    return struct_pack_descriptor(
        request_sequence,
        request_offset,
        request_bytes,
        payload_offset,
        payload_bytes,
        work_unit_count,
        0,
        request_output_elements,
        request_sha256,
        payload_sha256,
        b"\0" * 32,
    )


def struct_pack_header(
    request_count: int,
    descriptors_offset: int,
    body_offset: int,
    total_bytes: int,
    operation_sequence: int,
    digest: bytes,
) -> bytes:
    """Pack the fixed UPOENV1 header without native alignment."""

    return struct.pack(
        ENVELOPE_HEADER_FORMAT,
        ENVELOPE_MAGIC,
        ENVELOPE_VERSION,
        ENVELOPE_HEADER_BYTES,
        request_count,
        DESCRIPTOR_BYTES,
        0,
        0,
        descriptors_offset,
        body_offset,
        total_bytes,
        operation_sequence,
        digest,
    )


def struct_pack_descriptor(
    request_sequence: int,
    request_offset: int,
    request_bytes: int,
    payload_offset: int,
    payload_bytes: int,
    work_unit_count: int,
    reserved: int,
    request_output_elements: int,
    request_sha256: bytes,
    payload_sha256: bytes,
    descriptor_sha256: bytes,
) -> bytes:
    return struct.pack(
        DESCRIPTOR_FORMAT,
        request_sequence,
        request_offset,
        request_bytes,
        payload_offset,
        payload_bytes,
        work_unit_count,
        reserved,
        request_output_elements,
        request_sha256,
        payload_sha256,
        descriptor_sha256,
    )


def _fixture_payloads(sequence: int, dpu_id: int) -> tuple[bytes, bytes]:
    a_payload = bytes(
        (16 + sequence * 7 + dpu_id * 3 + index) & 0xFF for index in range(16)
    )
    b_payload = bytes(
        (64 + sequence * 11 + dpu_id * 5 + index) & 0xFF for index in range(32)
    )
    return a_payload, b_payload


def _fixture_units(sequence: int, dpu_count: int) -> tuple[V4WorkUnit, ...]:
    return tuple(
        V4WorkUnit(
            local_dpu_id=dpu_id,
            tile_id=1000 + dpu_id,
            batch_index=0,
            m_offset=dpu_id,
            n_offset=0,
            k_offset=0,
            m_elements=1,
            n_elements=2,
            k_elements=4,
            a_payload=_fixture_payloads(sequence, dpu_id)[0],
            b_payload=_fixture_payloads(sequence, dpu_id)[1],
        )
        for dpu_id in range(dpu_count)
    )


def build_synthetic_artifacts(
    root: Path, request_count: int = 2, *, dpu_count: int = 2
) -> tuple[V4RequestArtifact, ...]:
    """Create current-style ABI-v4 artifacts with deterministic host payloads."""

    _require_count(request_count)
    if not 1 <= dpu_count <= 4:
        raise ValueError("dpu_count must be in the host-only probe range [1, 4]")
    profile = V4Profile(
        dpu_count=dpu_count,
        tasklets_per_dpu=TASKLETS_PER_DPU,
        numeric_mode=NUMERIC_FLOAT32,
    )
    artifacts: list[V4RequestArtifact] = []
    for sequence in range(request_count):
        artifacts.append(
            build_v4_request(
                root / f"request_{sequence}",
                profile=profile,
                canonical_batch_count=1,
                canonical_m=dpu_count,
                canonical_n=2,
                canonical_k=4,
                work_units=_fixture_units(sequence, dpu_count),
                task_contract_sha256=TASK_CONTRACT_SHA256,
                request_sequence=sequence,
            )
        )
    return tuple(artifacts)


def _manifest_for(sequence: int, units: Sequence[V4WorkUnit]) -> bytes:
    lines = [f"sidecar requests/{sequence:016d}/sidecar.bin"]
    for dpu_id, unit in enumerate(units):
        a_payload, b_payload = unit.a_payload, unit.b_payload
        a_bytes = bytes(a_payload)
        b_bytes = bytes(b_payload)
        lines.append(
            "dpu {dpu} {tile} requests/{seq:016d}/payloads/dpu_{dpu:03d}_a.bin "
            "requests/{seq:016d}/payloads/dpu_{dpu:03d}_b.bin "
            "requests/{seq:016d}/outputs/dpu_{dpu:03d}_c.bin {a} {b}".format(
                dpu=dpu_id,
                tile=unit.tile_id,
                seq=sequence,
                a=hashlib.sha256(a_bytes).hexdigest(),
                b=hashlib.sha256(b_bytes).hexdigest(),
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _direct_prepared_request(sequence: int, dpu_count: int) -> _PreparedRequest:
    """Prepare the same sidecar and payload bytes without filesystem staging."""

    units = _fixture_units(sequence, dpu_count)
    payloads = [_fixture_payloads(sequence, dpu_id) for dpu_id in range(dpu_count)]
    manifest = _manifest_for(sequence, units)
    record_bytes = []
    for dpu_id, unit in enumerate(units):
        record_bytes.append(
            struct.pack(
                "<2I5Q9I",
                unit.local_dpu_id,
                0,
                unit.tile_id,
                unit.batch_index,
                unit.m_offset,
                unit.n_offset,
                unit.k_offset,
                unit.m_elements,
                unit.n_elements,
                unit.k_elements,
                16,
                32,
                8,
                0,
                16,
                48,
            )
        )
    sidecar = V4Header(
        canonical_batch_count=1,
        canonical_m=dpu_count,
        canonical_n=2,
        canonical_k=4,
        global_output_elements=dpu_count * 2,
        request_output_elements=dpu_count * 2,
        request_sequence=sequence,
        task_contract_sha256=bytes.fromhex(TASK_CONTRACT_SHA256),
        request_sha256=hashlib.sha256(manifest).digest(),
        work_unit_count=dpu_count,
        dpu_count=dpu_count,
        tasklets_per_dpu=TASKLETS_PER_DPU,
        numeric_mode=NUMERIC_MODE_FLOAT32,
        partition_mode=PARTITION_OUTPUT_TILE,
        version=VERSION,
        header_bytes=HEADER_BYTES,
        record_bytes=WORK_UNIT_BYTES,
    ).pack() + b"".join(record_bytes)
    payload = b"".join(value for pair in payloads for value in pair)
    return _PreparedRequest(
        request_sequence=sequence,
        sidecar=sidecar,
        payload=payload,
        work_unit_count=dpu_count,
        request_output_elements=dpu_count * 2,
        manifest=manifest,
    )


def direct_prepared_requests(
    request_count: int, dpu_count: int
) -> tuple[_PreparedRequest, ...]:
    """Prepare a variable-count sequence for the V1 or V2 packer."""

    _require_count(request_count)
    if not 1 <= dpu_count <= 4:
        raise ValueError("dpu_count must be in [1, 4]")
    return tuple(
        _direct_prepared_request(sequence, dpu_count)
        for sequence in range(request_count)
    )


def _payload_blob(artifact: V4RequestArtifact) -> bytes:
    chunks: list[bytes] = []
    for record in artifact.work_units:
        chunks.append((artifact.root / record.a_path).read_bytes())
        chunks.append((artifact.root / record.b_path).read_bytes())
    return b"".join(chunks)


def _prepared_from_artifact(artifact: V4RequestArtifact) -> _PreparedRequest:
    return _PreparedRequest(
        request_sequence=artifact.request_sequence,
        sidecar=artifact.sidecar_path.read_bytes(),
        payload=_payload_blob(artifact),
        work_unit_count=len(artifact.work_units),
        request_output_elements=artifact.request_output_elements,
        manifest=artifact.manifest_path.read_bytes(),
    )


def _descriptor_for(
    artifact: V4RequestArtifact,
    *,
    request_offset: int,
    payload_offset: int,
) -> tuple[PackedDescriptor, bytes]:
    request_bytes = artifact.sidecar_path.read_bytes()
    payload_bytes = _payload_blob(artifact)
    request_digest = _sha256(request_bytes)
    payload_digest = _sha256(payload_bytes)
    prefix = _descriptor_prefix(
        request_sequence=artifact.request_sequence,
        request_offset=request_offset,
        request_bytes=len(request_bytes),
        payload_offset=payload_offset,
        payload_bytes=len(payload_bytes),
        work_unit_count=len(artifact.work_units),
        request_output_elements=artifact.request_output_elements,
        request_sha256=request_digest,
        payload_sha256=payload_digest,
    )
    descriptor_digest = _sha256(prefix)
    descriptor_bytes = prefix[:_DESCRIPTOR_DIGEST_OFFSET] + descriptor_digest
    descriptor = PackedDescriptor(
        request_sequence=artifact.request_sequence,
        request_offset=request_offset,
        request_bytes=len(request_bytes),
        payload_offset=payload_offset,
        payload_bytes=len(payload_bytes),
        work_unit_count=len(artifact.work_units),
        request_output_elements=artifact.request_output_elements,
        request_sha256=request_digest.hex(),
        payload_sha256=payload_digest.hex(),
        descriptor_sha256=descriptor_digest.hex(),
    )
    return descriptor, descriptor_bytes


def pack_operation_envelope(artifacts: Sequence[V4RequestArtifact]) -> bytes:
    """Pack the legacy UPOENV1 envelope, retaining its fixed 64-request cap."""

    _require_count(len(artifacts), cap=MAX_REQUESTS)
    if tuple(artifact.request_sequence for artifact in artifacts) != tuple(
        range(len(artifacts))
    ):
        raise ValueError("request artifacts must be ordered from sequence zero")
    body_offset = ENVELOPE_HEADER_BYTES + len(artifacts) * DESCRIPTOR_BYTES
    body = bytearray()
    descriptors: list[bytes] = []
    for artifact in artifacts:
        request = artifact.sidecar_path.read_bytes()
        payload = _payload_blob(artifact)
        request_offset = body_offset + len(body)
        payload_offset = request_offset + len(request)
        _, descriptor_bytes = _descriptor_for(
            artifact, request_offset=request_offset, payload_offset=payload_offset
        )
        descriptors.append(descriptor_bytes)
        body.extend(request)
        body.extend(payload)
    total_bytes = body_offset + len(body)
    header = struct_pack_header(
        len(artifacts),
        ENVELOPE_HEADER_BYTES,
        body_offset,
        total_bytes,
        0,
        b"\0" * 32,
    )
    envelope = bytearray(header + b"".join(descriptors) + body)
    envelope[_ENVELOPE_DIGEST_OFFSET : _ENVELOPE_DIGEST_OFFSET + 32] = _sha256(
        bytes(envelope)
    )
    return bytes(envelope)


def pack_prepared_operation(requests: Sequence[_PreparedRequest]) -> bytes:
    """Pack prepared bytes into the legacy UPOENV1 envelope."""

    _require_count(len(requests), cap=MAX_REQUESTS)
    if tuple(request.request_sequence for request in requests) != tuple(
        range(len(requests))
    ):
        raise ValueError("prepared requests must be ordered from sequence zero")
    body_offset = ENVELOPE_HEADER_BYTES + len(requests) * DESCRIPTOR_BYTES
    body = bytearray()
    descriptors: list[bytes] = []
    for request in requests:
        request_offset = body_offset + len(body)
        payload_offset = request_offset + len(request.sidecar)
        prefix = struct_pack_descriptor(
            request.request_sequence,
            request_offset,
            len(request.sidecar),
            payload_offset,
            len(request.payload),
            request.work_unit_count,
            0,
            request.request_output_elements,
            _sha256(request.sidecar),
            _sha256(request.payload),
            b"\0" * 32,
        )
        descriptors.append(prefix[:_DESCRIPTOR_DIGEST_OFFSET] + _sha256(prefix))
        body.extend(request.sidecar)
        body.extend(request.payload)
    total_bytes = body_offset + len(body)
    header = struct_pack_header(
        len(requests),
        ENVELOPE_HEADER_BYTES,
        body_offset,
        total_bytes,
        0,
        b"\0" * 32,
    )
    envelope = bytearray(header + b"".join(descriptors) + body)
    envelope[_ENVELOPE_DIGEST_OFFSET : _ENVELOPE_DIGEST_OFFSET + 32] = _sha256(
        bytes(envelope)
    )
    return bytes(envelope)


def _v2_header(
    descriptor_count: int,
    total_bytes: int,
    operation_sequence: int,
    digest: bytes,
) -> bytes:
    _require_count(descriptor_count)
    if not 0 <= operation_sequence <= UINT64_MAX:
        raise ValueError("operation_sequence exceeds uint64")
    body_offset = _checked_add(
        UPOENV2_HEADER_BYTES,
        _checked_mul(descriptor_count, UPOENV2_DESCRIPTOR_BYTES, "descriptor table"),
        "body offset",
    )
    if total_bytes < body_offset or total_bytes > UINT64_MAX:
        raise ValueError("total_bytes is outside the UPOENV2 body")
    return struct.pack(
        UPOENV2_HEADER_FORMAT,
        UPOENV2_MAGIC,
        UPOENV2_VERSION,
        UPOENV2_HEADER_BYTES,
        descriptor_count,
        UPOENV2_DESCRIPTOR_BYTES,
        0,
        0,
        UPOENV2_HEADER_BYTES,
        body_offset,
        total_bytes,
        operation_sequence,
        digest,
    )


def _v2_descriptor_prefix(
    request: _PreparedRequest,
    *,
    manifest_offset: int,
    sidecar_offset: int,
    payload_offset: int,
) -> bytes:
    return struct.pack(
        UPOENV2_DESCRIPTOR_FORMAT,
        request.request_sequence,
        manifest_offset,
        len(request.manifest),
        sidecar_offset,
        len(request.sidecar),
        payload_offset,
        len(request.payload),
        request.work_unit_count,
        0,
        request.request_output_elements,
        _sha256(request.manifest),
        _sha256(request.sidecar),
        _sha256(request.payload),
        b"\0" * 32,
    )


def pack_upoenv2_prepared_operation(
    requests: Sequence[_PreparedRequest], *, operation_sequence: int = 0
) -> bytes:
    """Pack an unlimited-by-policy sequence using the exact UPOENV2 layout."""

    _require_count(len(requests))
    if tuple(request.request_sequence for request in requests) != tuple(
        range(len(requests))
    ):
        raise ValueError("UPOENV2 requests must be ordered from sequence zero")
    body_offset = _checked_add(
        UPOENV2_HEADER_BYTES,
        _checked_mul(len(requests), UPOENV2_DESCRIPTOR_BYTES, "descriptor table"),
        "body offset",
    )
    body = bytearray()
    descriptors: list[bytes] = []
    for request in requests:
        manifest_offset = _checked_add(body_offset, len(body), "manifest offset")
        sidecar_offset = _checked_add(
            manifest_offset, len(request.manifest), "sidecar offset"
        )
        payload_offset = _checked_add(
            sidecar_offset, len(request.sidecar), "payload offset"
        )
        prefix = _v2_descriptor_prefix(
            request,
            manifest_offset=manifest_offset,
            sidecar_offset=sidecar_offset,
            payload_offset=payload_offset,
        )
        descriptors.append(
            prefix[:_UPOENV2_DESCRIPTOR_DIGEST_OFFSET]
            + _sha256(prefix)
        )
        body.extend(request.manifest)
        body.extend(request.sidecar)
        body.extend(request.payload)
    total_bytes = _checked_add(body_offset, len(body), "total bytes")
    header = _v2_header(
        len(requests), total_bytes, operation_sequence, b"\0" * 32
    )
    envelope = bytearray(header + b"".join(descriptors) + body)
    envelope[_ENVELOPE_DIGEST_OFFSET : _ENVELOPE_DIGEST_OFFSET + 32] = _sha256(
        bytes(envelope)
    )
    return bytes(envelope)


def pack_upoenv2_operation(
    artifacts: Sequence[V4RequestArtifact], *, operation_sequence: int = 0
) -> bytes:
    """Pack exact manifest, sidecar, and padded payload bytes from artifacts."""

    return pack_upoenv2_prepared_operation(
        tuple(_prepared_from_artifact(artifact) for artifact in artifacts),
        operation_sequence=operation_sequence,
    )


# Descriptive aliases make the format boundary explicit to callers.
pack_operation_envelope_v2 = pack_upoenv2_operation
pack_prepared_operation_v2 = pack_upoenv2_prepared_operation


def _summary_from_envelope_v1(envelope: bytes) -> dict[str, object]:
    header = struct.unpack(ENVELOPE_HEADER_FORMAT, envelope[:ENVELOPE_HEADER_BYTES])
    request_count = header[3]
    requests: list[dict[str, object]] = []
    for index in range(request_count):
        offset = ENVELOPE_HEADER_BYTES + index * DESCRIPTOR_BYTES
        descriptor = struct.unpack(
            DESCRIPTOR_FORMAT, envelope[offset : offset + DESCRIPTOR_BYTES]
        )
        requests.append(
            {
                "descriptor_sha256": descriptor[10].hex(),
                "index": index,
                "payload_bytes": descriptor[4],
                "payload_sha256": descriptor[9].hex(),
                "request_bytes": descriptor[2],
                "request_output_elements": descriptor[7],
                "request_sequence": descriptor[0],
                "request_sha256": descriptor[8].hex(),
                "work_unit_count": descriptor[5],
            }
        )
    return {
        "envelope_sha256": envelope[
            _ENVELOPE_DIGEST_OFFSET : _ENVELOPE_DIGEST_OFFSET + 32
        ].hex(),
        "request_count": request_count,
        "requests": requests,
        "schema": SUMMARY_SCHEMA,
        "status": "accepted",
        "version": ENVELOPE_VERSION,
    }


def _summary_from_envelope_v2(envelope: bytes) -> dict[str, object]:
    if len(envelope) < UPOENV2_HEADER_BYTES:
        raise ValueError("UPOENV2 envelope is truncated")
    header = struct.unpack(
        UPOENV2_HEADER_FORMAT, envelope[:UPOENV2_HEADER_BYTES]
    )
    descriptor_count = header[3]
    table_bytes = _checked_mul(
        descriptor_count, UPOENV2_DESCRIPTOR_BYTES, "descriptor table"
    )
    body_offset = _checked_add(UPOENV2_HEADER_BYTES, table_bytes, "body offset")
    if (
        header[0] != UPOENV2_MAGIC
        or header[1] != UPOENV2_VERSION
        or header[2] != UPOENV2_HEADER_BYTES
        or descriptor_count < 1
        or header[4] != UPOENV2_DESCRIPTOR_BYTES
        or header[5] != 0
        or header[6] != 0
        or header[7] != UPOENV2_HEADER_BYTES
        or header[8] != body_offset
        or header[9] != len(envelope)
    ):
        raise ValueError("invalid UPOENV2 header")
    digest = bytearray(envelope)
    expected_envelope_digest = bytes(digest[64:96])
    digest[64:96] = b"\0" * 32
    if _sha256(bytes(digest)) != expected_envelope_digest:
        raise ValueError("UPOENV2 envelope digest mismatch")
    requests: list[dict[str, object]] = []
    cursor = body_offset
    for index in range(descriptor_count):
        offset = UPOENV2_HEADER_BYTES + index * UPOENV2_DESCRIPTOR_BYTES
        descriptor = struct.unpack(
            UPOENV2_DESCRIPTOR_FORMAT,
            envelope[offset : offset + UPOENV2_DESCRIPTOR_BYTES],
        )
        manifest_offset, manifest_bytes = descriptor[1:3]
        sidecar_offset, sidecar_bytes = descriptor[3:5]
        payload_offset, payload_bytes = descriptor[5:7]
        if descriptor[0] != index or descriptor[8] != 0:
            raise ValueError("UPOENV2 descriptor order or reserved field is invalid")
        if manifest_offset != cursor:
            raise ValueError("UPOENV2 body regions are not contiguous")
        if sidecar_offset != _checked_add(manifest_offset, manifest_bytes, "sidecar offset"):
            raise ValueError("UPOENV2 manifest and sidecar overlap")
        if payload_offset != _checked_add(sidecar_offset, sidecar_bytes, "payload offset"):
            raise ValueError("UPOENV2 sidecar and payload overlap")
        payload_end = _checked_add(payload_offset, payload_bytes, "payload end")
        if payload_end > len(envelope):
            raise ValueError("UPOENV2 payload is truncated")
        ranges = (
            (manifest_offset, manifest_bytes, descriptor[10]),
            (sidecar_offset, sidecar_bytes, descriptor[11]),
            (payload_offset, payload_bytes, descriptor[12]),
        )
        for start, length, expected in ranges:
            if start < body_offset or start + length > len(envelope):
                raise ValueError("UPOENV2 region is outside the body")
            if _sha256(envelope[start : start + length]) != expected:
                raise ValueError("UPOENV2 region digest mismatch")
        descriptor_digest = bytearray(envelope[offset : offset + UPOENV2_DESCRIPTOR_BYTES])
        expected_descriptor_digest = bytes(descriptor_digest[168:200])
        descriptor_digest[168:200] = b"\0" * 32
        if _sha256(bytes(descriptor_digest)) != expected_descriptor_digest:
            raise ValueError("UPOENV2 descriptor digest mismatch")
        requests.append(
            {
                "descriptor_sha256": descriptor[13].hex(),
                "index": index,
                "manifest_bytes": manifest_bytes,
                "manifest_sha256": descriptor[10].hex(),
                "payload_bytes": payload_bytes,
                "payload_sha256": descriptor[12].hex(),
                "request_output_elements": descriptor[9],
                "request_sequence": descriptor[0],
                "sidecar_bytes": sidecar_bytes,
                "sidecar_sha256": descriptor[11].hex(),
                "work_unit_count": descriptor[7],
            }
        )
        cursor = payload_end
    if cursor != len(envelope):
        raise ValueError("UPOENV2 body has a gap or trailing bytes")
    return {
        "descriptor_count": descriptor_count,
        "envelope_sha256": envelope[64:96].hex(),
        "operation_sequence": header[10],
        "requests": requests,
        "schema": UPOENV2_SUMMARY_SCHEMA,
        "status": "accepted",
        "version": UPOENV2_VERSION,
    }


def _summary_from_envelope(envelope: bytes) -> dict[str, object]:
    if len(envelope) < 8:
        raise ValueError("envelope is truncated")
    if envelope[:8] == UPOENV2_MAGIC:
        return _summary_from_envelope_v2(envelope)
    return _summary_from_envelope_v1(envelope)


def validate_upoenv2(envelope: bytes) -> dict[str, object]:
    """Validate UPOENV2 independently of the C probe and return its summary."""

    summary = _summary_from_envelope_v2(envelope)
    if summary["version"] != UPOENV2_VERSION:
        raise ValueError("unsupported UPOENV2 version")
    return summary


validate_operation_envelope_v2 = validate_upoenv2


def _summary_bytes(summary: dict[str, object]) -> bytes:
    return (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def compile_probe(output: Path) -> Path:
    """Compile the standalone C validator into an explicitly chosen path."""

    source = Path(__file__).with_name("upoenv_probe.c")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-o",
            str(output),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def invoke_probe(probe: Path, envelope: bytes) -> subprocess.CompletedProcess[bytes]:
    """Invoke C on bytes from memory through a temporary host-only file."""

    with tempfile.TemporaryDirectory(prefix="upoenv-input-") as directory:
        input_path = Path(directory) / "operation.upoenv"
        input_path.write_bytes(envelope)
        return subprocess.run(
            [str(probe), str(input_path)], check=False, capture_output=True
        )


def probe_envelope(envelope: bytes, probe: Path) -> dict[str, object]:
    """Require C's canonical output to equal Python's independent summary."""

    expected = _summary_from_envelope(envelope)
    result = invoke_probe(probe, envelope)
    if result.returncode != 0:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    if result.stdout != _summary_bytes(expected):
        raise ValueError("C probe output is not equivalent to the deterministic summary")
    return expected


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _raw_mad(values: Sequence[float]) -> float:
    center = _median(values)
    return _median([abs(value - center) for value in values])


def _operation_request_count(total_requests: int, operation_count: int) -> int:
    return (total_requests + operation_count - 1) // operation_count


def _source_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_IMPLEMENTATION_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def _provenance() -> dict[str, object]:
    return {
        "probe_schema": BENCHMARK_SCHEMA,
        "source_sha": _source_sha(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sdk_executed": False,
        "hardware_executed": False,
        "production_runtime_imported": False,
        "execution_scope": "host_only_file_boundary",
        "random_seed": RAW_RANDOM_SEED,
    }


def _fixture_measurement(
    *,
    case_id: str,
    dpu_count: int,
    request_count: int,
    arm: str,
    probe: Path,
    root: Path,
) -> dict[str, object]:
    if arm == "current":
        started = time.perf_counter_ns()
        with tempfile.TemporaryDirectory(
            prefix=f"upoenv-current-{case_id}-{dpu_count}-", dir=root
        ) as directory:
            artifact_started = time.perf_counter_ns()
            artifacts = build_synthetic_artifacts(
                Path(directory), request_count, dpu_count=dpu_count
            )
            artifact_s = (time.perf_counter_ns() - artifact_started) / 1e9
            scan_started = time.perf_counter_ns()
            current_bytes = sum(
                path.stat().st_size
                for artifact in artifacts
                for path in (
                    artifact.manifest_path,
                    artifact.sidecar_path,
                    *(
                        path
                        for record in artifact.work_units
                        for path in (
                            artifact.root / record.a_path,
                            artifact.root / record.b_path,
                        )
                    ),
                )
            )
            scan_s = (time.perf_counter_ns() - scan_started) / 1e9
        return {
            "phase_timings": {
                "artifact_build_s": artifact_s,
                "file_scan_s": scan_s,
                "envelope_pack_s": 0.0,
                "probe_s": 0.0,
                "total_s": (time.perf_counter_ns() - started) / 1e9,
            },
            "counts": {
                "request_count": request_count,
                "descriptor_count": request_count,
                "request_directory_count": request_count,
                "file_count": request_count * (2 * dpu_count + 2),
                "process_count": 0,
            },
            "bytes": {
                "current_file_bytes": current_bytes,
                "envelope_bytes": 0,
                "body_bytes": current_bytes,
            },
            "process_status": {
                "status": "not_run",
                "returncode": None,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
            },
        }
    started = time.perf_counter_ns()
    prepared_started = time.perf_counter_ns()
    prepared = direct_prepared_requests(request_count, dpu_count)
    prepared_s = (time.perf_counter_ns() - prepared_started) / 1e9
    pack_started = time.perf_counter_ns()
    envelope = pack_upoenv2_prepared_operation(prepared)
    pack_s = (time.perf_counter_ns() - pack_started) / 1e9
    probe_started = time.perf_counter_ns()
    result = probe_envelope(envelope, probe)
    probe_s = (time.perf_counter_ns() - probe_started) / 1e9
    return {
        "phase_timings": {
            "artifact_build_s": 0.0,
            "file_scan_s": 0.0,
            "prepared_request_s": prepared_s,
            "envelope_pack_s": pack_s,
            "probe_s": probe_s,
            "total_s": (time.perf_counter_ns() - started) / 1e9,
        },
        "counts": {
            "request_count": request_count,
            "descriptor_count": int(result["descriptor_count"]),
            "request_directory_count": 0,
            "file_count": 1,
            "process_count": 1,
        },
        "bytes": {
            "current_file_bytes": 0,
            "envelope_bytes": len(envelope),
            "body_bytes": len(envelope)
            - UPOENV2_HEADER_BYTES
            - request_count * UPOENV2_DESCRIPTOR_BYTES,
        },
        "process_status": {
            "status": "accepted",
            "returncode": 0,
            "stdout_bytes": len(_summary_bytes(result)),
            "stderr_bytes": 0,
        },
    }


def _raw_row(
    *,
    case_id: str,
    dpu_count: int,
    block: int,
    phase: str,
    arm: str,
    arm_order: tuple[str, str],
    arm_order_index: int,
    measurement: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    timings = measurement["phase_timings"]
    counts = measurement["counts"]
    byte_counts = measurement["bytes"]
    process_status = measurement["process_status"]
    return {
        "schema": RAW_SCHEMA,
        "cell": f"{case_id}/{dpu_count}dpu/t{TASKLETS_PER_DPU}",
        "case_id": case_id,
        "dpu_count": dpu_count,
        "tasklets_per_dpu": TASKLETS_PER_DPU,
        "block": block,
        "phase": phase,
        "arm": arm,
        "arm_order": list(arm_order),
        "arm_order_index": arm_order_index,
        "phase_timings": timings,
        "counts": counts,
        "bytes": byte_counts,
        "process_status": process_status,
        "environment": {
            "host": provenance["host"],
            "platform": provenance["platform"],
            "python": provenance["python"],
        },
        "provenance": provenance,
    }


RAW_CSV_FIELDS = (
    "schema",
    "cell",
    "case_id",
    "dpu_count",
    "tasklets_per_dpu",
    "block",
    "phase",
    "arm",
    "arm_order",
    "arm_order_index",
    "total_s",
    "artifact_build_s",
    "file_scan_s",
    "prepared_request_s",
    "envelope_pack_s",
    "probe_s",
    "request_count",
    "descriptor_count",
    "request_directory_count",
    "file_count",
    "process_count",
    "current_file_bytes",
    "envelope_bytes",
    "body_bytes",
    "process_status",
    "environment",
    "provenance",
)


def _raw_csv_row(row: dict[str, object]) -> dict[str, object]:
    timings = row["phase_timings"]
    counts = row["counts"]
    byte_counts = row["bytes"]
    return {
        "schema": row["schema"],
        "cell": row["cell"],
        "case_id": row["case_id"],
        "dpu_count": row["dpu_count"],
        "tasklets_per_dpu": row["tasklets_per_dpu"],
        "block": row["block"],
        "phase": row["phase"],
        "arm": row["arm"],
        "arm_order": json.dumps(row["arm_order"], separators=(",", ":")),
        "arm_order_index": row["arm_order_index"],
        "total_s": timings.get("total_s", 0.0),
        "artifact_build_s": timings.get("artifact_build_s", 0.0),
        "file_scan_s": timings.get("file_scan_s", 0.0),
        "prepared_request_s": timings.get("prepared_request_s", 0.0),
        "envelope_pack_s": timings.get("envelope_pack_s", 0.0),
        "probe_s": timings.get("probe_s", 0.0),
        "request_count": counts["request_count"],
        "descriptor_count": counts["descriptor_count"],
        "request_directory_count": counts["request_directory_count"],
        "file_count": counts["file_count"],
        "process_count": counts["process_count"],
        "current_file_bytes": byte_counts["current_file_bytes"],
        "envelope_bytes": byte_counts["envelope_bytes"],
        "body_bytes": byte_counts["body_bytes"],
        "process_status": json.dumps(row["process_status"], sort_keys=True),
        "environment": json.dumps(row["environment"], sort_keys=True),
        "provenance": json.dumps(row["provenance"], sort_keys=True),
    }


def _aggregate_cell(
    *,
    case_id: str,
    dpu_count: int,
    operation_count: int,
    total_requests: int,
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    measured = [row for row in rows if row["phase"] == "measured"]
    current = [row for row in measured if row["arm"] == "current"]
    packed = [row for row in measured if row["arm"] == "packed"]
    current_times = [float(row["phase_timings"]["total_s"]) for row in current]
    packed_times = [float(row["phase_timings"]["total_s"]) for row in packed]
    request_count = _operation_request_count(total_requests, operation_count)
    current_file_bytes = int(current[0]["bytes"]["current_file_bytes"])
    packed_envelope_bytes = int(packed[0]["bytes"]["envelope_bytes"])
    current_median = _median(current_times)
    packed_median = _median(packed_times)
    return {
        "case_id": case_id,
        "dpu_count": dpu_count,
        "tasklets_per_dpu": TASKLETS_PER_DPU,
        "contraction_operation_count": operation_count,
        "total_embedded_request_count": total_requests,
        "operation_request_count": request_count,
        "current_submit_count": request_count,
        "packed_submit_count": 1,
        "current_request_directory_count": request_count,
        "current_file_count": int(current[0]["counts"]["file_count"]),
        "packed_file_count": 1,
        "current_file_bytes": current_file_bytes,
        "packed_envelope_bytes": packed_envelope_bytes,
        "current_process_count": 0,
        "packed_process_count": 1,
        "current_median_s": current_median,
        "current_raw_mad_s": _raw_mad(current_times),
        "packed_median_s": packed_median,
        "packed_raw_mad_s": _raw_mad(packed_times),
        "boundary_speedup": current_median / max(packed_median, 1e-12),
        "packed_descriptor_count": request_count,
        "raw_iteration_count": len(rows),
    }


def _write_raw_rows(output_dir: Path, rows: Sequence[dict[str, object]]) -> None:
    with (output_dir / "host_boundary_benchmark.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    with (output_dir / "host_boundary_benchmark_raw.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_raw_csv_row(row) for row in rows)


def _write_aggregate_csv(output_dir: Path, cells: Sequence[dict[str, object]]) -> None:
    with (output_dir / "host_boundary_benchmark.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = tuple(cells[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)


def benchmark_boundary(
    output_dir: Path,
    *,
    warmups: int = 2,
    repeats: int = 10,
    prior_evidence: Path | None = None,
) -> dict[str, object]:
    """Run a deterministic blocked randomized host-only boundary benchmark."""

    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _provenance()
    raw_rows: list[dict[str, object]] = []
    aggregate_cells: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="upoenv-probe-build-") as probe_dir:
        probe_root = Path(probe_dir)
        probe = compile_probe(probe_root / "upoenv_probe")
        for cell_index, (case_id, dpu_count, operation_count, total_requests) in enumerate(
            CELL_DEFINITIONS
        ):
            request_count = _operation_request_count(total_requests, operation_count)
            cell_rows: list[dict[str, object]] = []
            randomizer = random.Random(RAW_RANDOM_SEED + cell_index)
            for block in range(warmups + repeats):
                arm_order = (
                    ("current", "packed")
                    if randomizer.randrange(2) == 0
                    else ("packed", "current")
                )
                phase = "warmup" if block < warmups else "measured"
                for order_index, arm in enumerate(arm_order):
                    measurement = _fixture_measurement(
                        case_id=case_id,
                        dpu_count=dpu_count,
                        request_count=request_count,
                        arm=arm,
                        probe=probe,
                        root=probe_root,
                    )
                    row = _raw_row(
                        case_id=case_id,
                        dpu_count=dpu_count,
                        block=block,
                        phase=phase,
                        arm=arm,
                        arm_order=arm_order,
                        arm_order_index=order_index,
                        measurement=measurement,
                        provenance=provenance,
                    )
                    raw_rows.append(row)
                    cell_rows.append(row)
            aggregate_cells.append(
                _aggregate_cell(
                    case_id=case_id,
                    dpu_count=dpu_count,
                    operation_count=operation_count,
                    total_requests=total_requests,
                    rows=cell_rows,
                )
            )
    _write_raw_rows(output_dir, raw_rows)
    _write_aggregate_csv(output_dir, aggregate_cells)
    result: dict[str, object] = {
        "analysis_version": BENCHMARK_SCHEMA,
        "warmups": warmups,
        "measurements": repeats,
        "block_count": warmups + repeats,
        "random_seed": RAW_RANDOM_SEED,
        "cells": aggregate_cells,
        "raw_jsonl": "host_boundary_benchmark.jsonl",
        "raw_csv": "host_boundary_benchmark_raw.csv",
        "equivalence": {
            "request_bytes": True,
            "payload_bytes": True,
            "request_order": True,
            "work_unit_order": True,
            "manifest_bytes": True,
            "output_summary": True,
        },
        "timing_scope": "host_only_preparation_and_transport_probe_v2",
        "execution": {
            "hardware_executed": False,
            "sdk_executed": False,
            "production_runtime_executed": False,
        },
        "provenance": provenance,
        "interpretation": (
            "The current arm materializes accepted ABI-v4 request artifacts. "
            "The packed arm constructs one variable-count UPOENV2 operation "
            "and validates it in one standalone C process."
        ),
    }
    (output_dir / "host_boundary_benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The benchmark probe is scoped to the synthetic fixture block above;
    # replay compiles its own short-lived validator so the result never holds
    # a path into a deleted temporary directory.
    replay = replay_real_corpus(output_dir)
    decision = phase0_decision(result, replay, prior_evidence=prior_evidence)
    result["real_corpus_replay"] = "host_boundary_replay.json"
    result["phase0_decision"] = "phase0_decision.json"
    (output_dir / "host_boundary_benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "phase0_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_characterization_candidates() -> dict[str, object]:
    source = _IMPLEMENTATION_ROOT / "scripts" / "characterize_circuit_resources.py"
    spec = importlib.util.spec_from_file_location("host_boundary_characterization", source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load existing characterization construction")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {candidate.candidate_id: candidate for candidate in module.CANDIDATES}


def _real_plan(case_id: str, dpu_count: int) -> tuple[object, UpmemPlan, dict[str, object]]:
    candidates = _load_characterization_candidates()
    candidate = candidates.get(case_id)
    if candidate is None:
        raise ValueError(f"characterization candidate is missing: {case_id}")
    circuit = builtin_circuit(candidate.circuit_name, dict(candidate.parameters))
    job = make_simulation_job(circuit)
    network, _ = lower_tensor_network(job)
    path, planner_provenance = plan_opt_einsum(network, optimize="greedy")
    dag = build_contraction_dag(network, path)
    topology = UpmemTopology(
        dpu_count=dpu_count, rank_count=1, tasklets_per_dpu=TASKLETS_PER_DPU
    )
    plan = plan_upmem(
        dag, numeric_policy="split_complex_float32_v1", topology=topology
    )
    return dag, plan, {
        "candidate_id": case_id,
        "planner_engine": planner_provenance["planner_engine"],
        "planner_id": planner_provenance["planner_id"],
        "planner_mode": planner_provenance["optimize_mode"],
    }


def _replay_payload(length: int, seed: int) -> bytes:
    return bytes((seed + index) % 251 for index in range(length))


def _replay_work_unit(unit: object, *, lane: int, case_index: int) -> V4WorkUnit:
    tile_id = int.from_bytes(
        hashlib.sha256(unit.stable_tile_id.encode("utf-8")).digest()[:8], "little"
    )
    a_length = unit.m_size * unit.k_size * 4
    b_length = unit.k_size * unit.n_size * 4
    seed = case_index * 41 + lane * 17 + unit.wave * 3 + unit.logical_dpu
    return V4WorkUnit(
        local_dpu_id=unit.logical_dpu,
        tile_id=tile_id,
        batch_index=unit.batch_start,
        m_offset=unit.m_start,
        n_offset=unit.n_start,
        k_offset=unit.k_start,
        m_elements=unit.m_size,
        n_elements=unit.n_size,
        k_elements=unit.k_size,
        a_payload=_replay_payload(a_length, seed),
        b_payload=_replay_payload(b_length, seed + 97),
    )


def _replay_operation(
    *,
    case_id: str,
    case_index: int,
    dpu_count: int,
    operation_index: int,
    stage: object,
    node: ContractNode,
    root: Path,
    probe: Path,
) -> dict[str, object]:
    batch, canonical_m, canonical_k, canonical_n = canonical_label_geometry(
        node.left.labels,
        node.left.shape,
        node.right.labels,
        node.right.shape,
        node.output_labels,
    )
    waves = [
        tuple(
            sorted(
                (unit for unit in stage.work_units if unit.wave == wave),
                key=lambda unit: (unit.logical_rank, unit.logical_dpu, unit.stable_tile_id),
            )
        )
        for wave in sorted({unit.wave for unit in stage.work_units})
    ]
    descriptor_count = COMPLEX_LANE_COUNT * len(waves)
    operation_root = root / case_id / str(dpu_count) / f"operation_{operation_index:04d}"
    current_started = time.perf_counter_ns()
    artifacts: list[V4RequestArtifact] = []
    request_sequence = 0
    for lane in range(COMPLEX_LANE_COUNT):
        for wave in waves:
            artifacts.append(
                build_v4_request(
                    operation_root / f"current_{request_sequence:04d}",
                    profile=V4Profile(
                        dpu_count=dpu_count,
                        tasklets_per_dpu=TASKLETS_PER_DPU,
                        numeric_mode=NUMERIC_FLOAT32,
                    ),
                    canonical_batch_count=batch,
                    canonical_m=canonical_m,
                    canonical_n=canonical_n,
                    canonical_k=canonical_k,
                    work_units=tuple(
                        _replay_work_unit(
                            unit, lane=lane, case_index=case_index
                        )
                        for unit in wave
                    ),
                    task_contract_sha256=hashlib.sha256(
                        f"host-replay:{case_id}:{operation_index}".encode("ascii")
                    ).hexdigest(),
                    request_sequence=request_sequence,
                )
            )
            request_sequence += 1
    current_s = (time.perf_counter_ns() - current_started) / 1e9
    prepared = tuple(_prepared_from_artifact(artifact) for artifact in artifacts)
    packed_started = time.perf_counter_ns()
    packed = pack_upoenv2_operation(artifacts, operation_sequence=operation_index)
    direct_packed = pack_upoenv2_prepared_operation(
        prepared, operation_sequence=operation_index
    )
    if packed != direct_packed:
        raise ValueError(
            f"UPOENV2 artifact and prepared constructions differ for {case_id}/{dpu_count}/{operation_index}"
        )
    summary = probe_envelope(packed, probe)
    packed_s = (time.perf_counter_ns() - packed_started) / 1e9
    current_manifest_bytes = sum(len(request.manifest) for request in prepared)
    current_sidecar_bytes = sum(len(request.sidecar) for request in prepared)
    current_payload_bytes = sum(len(request.payload) for request in prepared)
    current_body_bytes = current_manifest_bytes + current_sidecar_bytes + current_payload_bytes
    packed_body_bytes = len(packed) - UPOENV2_HEADER_BYTES - descriptor_count * UPOENV2_DESCRIPTOR_BYTES
    return {
        "kind": "operation",
        "case_id": case_id,
        "dpu_count": dpu_count,
        "tasklets_per_dpu": TASKLETS_PER_DPU,
        "operation_index": operation_index,
        "stage_id": stage.stage_id,
        "node_id": node.node_id,
        "canonical_batch_count": batch,
        "canonical_m": canonical_m,
        "canonical_k": canonical_k,
        "canonical_n": canonical_n,
        "output_elements": batch * canonical_m * canonical_n,
        "work_unit_count": len(stage.work_units),
        "wave_count": len(waves),
        "descriptor_count": descriptor_count,
        "current_request_count": descriptor_count,
        "packed_request_count": 1,
        "current_file_count": descriptor_count * (2 * dpu_count + 2),
        "packed_file_count": 1,
        "current_manifest_bytes": current_manifest_bytes,
        "current_sidecar_bytes": current_sidecar_bytes,
        "current_payload_bytes": current_payload_bytes,
        "current_body_bytes": current_body_bytes,
        "current_total_bytes": current_body_bytes,
        "packed_body_bytes": packed_body_bytes,
        "packed_envelope_bytes": len(packed),
        "current_construction_s": current_s,
        "packed_construction_s": packed_s,
        "projected_saving_s": current_s - packed_s,
        "semantic_equivalent": summary["status"] == "accepted",
        "envelope_sha256": summary["envelope_sha256"],
    }


def _replay_cell(
    *,
    case_id: str,
    case_index: int,
    dpu_count: int,
    root: Path,
    probe: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    dag, plan, planner = _real_plan(case_id, dpu_count)
    nodes = {node.node_id: node for node in dag.nodes if isinstance(node, ContractNode)}
    operations: list[dict[str, object]] = []
    operation_index = 0
    for stage in plan.stages:
        if stage.kind != "contract_batch":
            continue
        if len(stage.node_ids) != 1:
            raise ValueError("real replay requires one node per contract stage")
        node = nodes[stage.node_ids[0]]
        operations.append(
            _replay_operation(
                case_id=case_id,
                case_index=case_index,
                dpu_count=dpu_count,
                operation_index=operation_index,
                stage=stage,
                node=node,
                root=root,
                probe=probe,
            )
        )
        operation_index += 1
    savings = [float(row["projected_saving_s"]) for row in operations]
    current_total = sum(float(row["current_construction_s"]) for row in operations)
    packed_total = sum(float(row["packed_construction_s"]) for row in operations)
    cell = {
        "kind": "cell",
        "case_id": case_id,
        "dpu_count": dpu_count,
        "tasklets_per_dpu": TASKLETS_PER_DPU,
        "planner": planner,
        "operation_count": len(operations),
        "current_request_count": sum(int(row["current_request_count"]) for row in operations),
        "packed_request_count": len(operations),
        "current_file_count": sum(int(row["current_file_count"]) for row in operations),
        "packed_file_count": len(operations),
        "current_total_bytes": sum(int(row["current_total_bytes"]) for row in operations),
        "packed_envelope_bytes": sum(int(row["packed_envelope_bytes"]) for row in operations),
        "maximum_descriptor_count": max(int(row["descriptor_count"]) for row in operations),
        "maximum_envelope_bytes": max(int(row["packed_envelope_bytes"]) for row in operations),
        "current_construction_s": current_total,
        "packed_construction_s": packed_total,
        "projected_saving_s": current_total - packed_total,
        "projected_request_reduction": sum(
            int(row["current_request_count"]) - int(row["packed_request_count"])
            for row in operations
        ),
        "projected_file_reduction": sum(
            int(row["current_file_count"]) - int(row["packed_file_count"])
            for row in operations
        ),
        "replay_raw_mad_s": _raw_mad(savings),
        "semantic_equivalence": all(row["semantic_equivalent"] for row in operations),
    }
    return cell, operations


def _replay_csv_rows(
    cells: Sequence[dict[str, object]], operations: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in operations:
        rows.append(row)
    for row in cells:
        rows.append(row)
    return rows


REPLAY_CSV_FIELDS = (
    "kind",
    "case_id",
    "dpu_count",
    "tasklets_per_dpu",
    "operation_index",
    "stage_id",
    "node_id",
    "canonical_batch_count",
    "canonical_m",
    "canonical_k",
    "canonical_n",
    "output_elements",
    "work_unit_count",
    "wave_count",
    "descriptor_count",
    "current_request_count",
    "packed_request_count",
    "current_file_count",
    "packed_file_count",
    "current_manifest_bytes",
    "current_sidecar_bytes",
    "current_payload_bytes",
    "current_body_bytes",
    "current_total_bytes",
    "packed_body_bytes",
    "packed_envelope_bytes",
    "current_construction_s",
    "packed_construction_s",
    "projected_saving_s",
    "semantic_equivalent",
    "operation_count",
    "maximum_descriptor_count",
    "maximum_envelope_bytes",
    "projected_request_reduction",
    "projected_file_reduction",
    "replay_raw_mad_s",
    "semantic_equivalence",
)


def replay_real_corpus(
    output_dir: Path, *, probe: Path | None = None
) -> dict[str, object]:
    """Replay exact characterized plans without SDK or hardware execution."""

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if probe is None:
        temporary = tempfile.TemporaryDirectory(prefix="upoenv-replay-")
        probe = compile_probe(Path(temporary.name) / "upoenv_probe")
    cells: list[dict[str, object]] = []
    operations: list[dict[str, object]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="upoenv-real-replay-") as root:
            replay_root = Path(root)
            for case_index, case_id in enumerate(REAL_CASE_IDS):
                for dpu_count in (1, 4):
                    cell, cell_operations = _replay_cell(
                        case_id=case_id,
                        case_index=case_index,
                        dpu_count=dpu_count,
                        root=replay_root,
                        probe=probe,
                    )
                    cells.append(cell)
                    operations.extend(cell_operations)
    finally:
        if temporary is not None:
            temporary.cleanup()
    result: dict[str, object] = {
        "schema": REPLAY_SCHEMA,
        "execution": {
            "kind": "source_only_real_corpus_replay",
            "hardware_executed": False,
            "sdk_executed": False,
            "production_runtime_executed": False,
        },
        "planner": {"engine": "opt_einsum", "mode": "greedy"},
        "numeric_policy": "split_complex_float32_v1",
        "tasklets_per_dpu": TASKLETS_PER_DPU,
        "complex_lane_count": COMPLEX_LANE_COUNT,
        "cells": cells,
        "operations": operations,
        "provenance": _provenance(),
    }
    (output_dir / "host_boundary_replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = _replay_csv_rows(cells, operations)
    with (output_dir / "host_boundary_replay.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=REPLAY_CSV_FIELDS,
            extrasaction="ignore",
            restval="",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return result


def _prior_raw_mad(path: Path | None, key: tuple[str, int]) -> float | None:
    if path is None or not path.exists():
        return None
    candidates = [path]
    if path.is_dir():
        candidates = sorted(path.rglob("*.json"))
    for candidate in candidates:
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stack: list[object] = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                item_key = (item.get("case_id"), item.get("dpu_count"))
                if item_key == key:
                    for field in (
                        "session_inclusive_raw_mad_s",
                        "raw_mad_session_inclusive_s",
                    ):
                        raw = item.get(field)
                        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw >= 0:
                            return float(raw)
                    session = item.get("session_inclusive")
                    if isinstance(session, dict):
                        raw = session.get("raw_mad_s")
                        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw >= 0:
                            return float(raw)
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return None


def phase0_decision(
    benchmark: dict[str, object],
    replay: dict[str, object],
    *,
    prior_evidence: Path | None = None,
) -> dict[str, object]:
    """Apply the Phase 0 gate using only host-only and source replay evidence."""

    benchmark_cells = {
        (row["case_id"], int(row["dpu_count"])): row
        for row in benchmark["cells"]
    }
    replay_cells = {
        (row["case_id"], int(row["dpu_count"])): row
        for row in replay["cells"]
    }
    cells: list[dict[str, object]] = []
    for case_id, dpu_count, _, _ in CELL_DEFINITIONS:
        key = (case_id, dpu_count)
        replay_cell = replay_cells[key]
        benchmark_cell = benchmark_cells[key]
        replay_mad = float(replay_cell["replay_raw_mad_s"])
        prior_mad = _prior_raw_mad(prior_evidence, key)
        threshold = max(replay_mad, prior_mad if prior_mad is not None else replay_mad)
        projected_saving = float(replay_cell["projected_saving_s"])
        cells.append(
            {
                "case_id": case_id,
                "dpu_count": dpu_count,
                "current_median_s": benchmark_cell["current_median_s"],
                "packed_median_s": benchmark_cell["packed_median_s"],
                "packed_median_lower": float(benchmark_cell["packed_median_s"])
                < float(benchmark_cell["current_median_s"]),
                "projected_saving_s": projected_saving,
                "projected_saving_positive": projected_saving > 0.0,
                "replay_raw_mad_s": replay_mad,
                "prior_session_inclusive_raw_mad_s": prior_mad,
                "noise_threshold_s": threshold,
                "saving_above_noise": projected_saving > threshold,
                "semantic_equivalence": bool(replay_cell["semantic_equivalence"]),
                "maximum_descriptor_count": replay_cell["maximum_descriptor_count"],
                "maximum_envelope_bytes": replay_cell["maximum_envelope_bytes"],
            }
        )
    equivalence_passed = all(bool(cell["semantic_equivalence"]) for cell in cells)
    median_passed = all(bool(cell["packed_median_lower"]) for cell in cells)
    projected_passed = all(bool(cell["projected_saving_positive"]) for cell in cells)
    noise_passing = sum(bool(cell["saving_above_noise"]) for cell in cells)
    gate_passed = equivalence_passed and median_passed and projected_passed and noise_passing >= 5
    decision = {
        "schema": PHASE0_SCHEMA,
        "phase": "0",
        "decision": "go" if gate_passed else "no_go",
        "gate_passed": gate_passed,
        "gate": {
            "equivalence_passed": equivalence_passed,
            "packed_median_lower_in_all_six": median_passed,
            "projected_saving_positive_in_all_six": projected_passed,
            "cells_above_noise": noise_passing,
            "minimum_cells_above_noise": 5,
        },
        "cells": cells,
        "maximum_descriptor_count": max(int(cell["maximum_descriptor_count"]) for cell in cells),
        "maximum_envelope_bytes": max(int(cell["maximum_envelope_bytes"]) for cell in cells),
        "semantic_equivalence": equivalence_passed,
        "prior_evidence": str(prior_evidence) if prior_evidence is not None else None,
        "future_physical_optimized_data_used": False,
        "evidence_scope": "host_only_preparation_and_source_plan_replay",
    }
    return decision


def _write_summary(output_dir: Path, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_bytes(_summary_bytes(summary))


def _run_probe_command(output_dir: Path, request_count: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if request_count > MAX_REQUESTS:
        raise SystemExit("--requests must be in [1, 64] for UPOENV1")
    with tempfile.TemporaryDirectory(prefix="upoenv-artifacts-") as directory:
        artifacts = build_synthetic_artifacts(Path(directory), request_count)
        envelope = pack_operation_envelope(artifacts)
        (output_dir / "operation.upoenv").write_bytes(envelope)
        with tempfile.TemporaryDirectory(prefix="upoenv-probe-") as probe_dir:
            summary = probe_envelope(envelope, compile_probe(Path(probe_dir) / "probe"))
    _write_summary(output_dir, summary)
    return summary


def _run_benchmark_command(output_dir: Path, request_count: int, repeats: int) -> None:
    with tempfile.TemporaryDirectory(prefix="upoenv-artifacts-") as directory:
        artifacts = build_synthetic_artifacts(Path(directory), request_count)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            pack_operation_envelope(artifacts)
            timings.append(time.perf_counter_ns() - started)
    summary = {
        "median_pack_ns": sorted(timings)[len(timings) // 2],
        "request_count": request_count,
        "repeats": repeats,
        "schema": "upoenv1_benchmark_v1",
    }
    _write_summary(output_dir, summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--requests", type=int, default=2)
        if name == "benchmark":
            command.add_argument("--repeats", type=int, default=5)
            command.add_argument("--warmups", type=int, default=2)
    boundary = commands.add_parser("boundary-benchmark")
    boundary.add_argument("--output-dir", type=Path, required=True)
    boundary.add_argument("--repeats", type=int, default=10)
    boundary.add_argument("--warmups", type=int, default=2)
    boundary.add_argument("--prior-evidence", type=Path)
    replay = commands.add_parser("replay")
    replay.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "boundary-benchmark":
        benchmark_boundary(
            args.output_dir,
            warmups=args.warmups,
            repeats=args.repeats,
            prior_evidence=args.prior_evidence,
        )
        return 0
    if args.command == "replay":
        replay_real_corpus(args.output_dir)
        return 0
    if not 1 <= args.requests <= MAX_REQUESTS:
        raise SystemExit("--requests must be in [1, 64]")
    if args.command == "probe":
        _run_probe_command(args.output_dir, args.requests)
        return 0
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    _run_benchmark_command(args.output_dir, args.requests, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
