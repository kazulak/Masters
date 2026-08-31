#!/usr/bin/env python3
"""Build and validate a deterministic host-only UPOENV1 packed operation.

The packed body contains existing ABI-v4 request sidecars and the staged A/B
payload bytes.  This module intentionally stops at a file boundary: it never
opens a rank, loads an SDK, or reaches the production runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from typing import Sequence

from quantum_bench.upmem.protocol import (
    NUMERIC_HOST_PACKED_INT8,
    V4Profile,
    V4WorkUnit,
    V4RequestArtifact,
    build_v4_request,
)


ENVELOPE_MAGIC = b"UPOENV1\0"
ENVELOPE_VERSION = 1
ENVELOPE_HEADER_FORMAT = "<8s6I4Q32s"
DESCRIPTOR_FORMAT = "<5Q2IQ32s32s32s"
ENVELOPE_HEADER_BYTES = 96
DESCRIPTOR_BYTES = 152
MAX_REQUESTS = 64
TASK_CONTRACT_SHA256 = "42" * 32
SUMMARY_SCHEMA = "upoenv1_probe_summary_v1"

_ENVELOPE_DIGEST_OFFSET = 64
_DESCRIPTOR_DIGEST_OFFSET = 120


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


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    """Pack the fixed UPOENV1 header without relying on native alignment."""

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


def build_synthetic_artifacts(root: Path, request_count: int = 2) -> tuple[V4RequestArtifact, ...]:
    """Create current-style ABI-v4 artifacts with deterministic host payloads."""

    if not 1 <= request_count <= MAX_REQUESTS:
        raise ValueError("request_count must be in [1, 64]")
    profile = V4Profile(dpu_count=2, numeric_mode=NUMERIC_HOST_PACKED_INT8)
    artifacts: list[V4RequestArtifact] = []
    for sequence in range(request_count):
        units = []
        for dpu_id in range(profile.dpu_count):
            a_payload = bytes(
                (16 + sequence * 7 + dpu_id * 3 + index) & 0xFF for index in range(4)
            )
            b_payload = bytes(
                (64 + sequence * 11 + dpu_id * 5 + index) & 0xFF for index in range(8)
            )
            units.append(
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
                    a_payload=a_payload,
                    b_payload=b_payload,
                )
            )
        artifacts.append(
            build_v4_request(
                root / f"request_{sequence}",
                profile=profile,
                canonical_batch_count=1,
                canonical_m=2,
                canonical_n=2,
                canonical_k=4,
                work_units=units,
                task_contract_sha256=TASK_CONTRACT_SHA256,
                request_sequence=sequence,
            )
        )
    return tuple(artifacts)


def _payload_blob(artifact: V4RequestArtifact) -> bytes:
    chunks: list[bytes] = []
    for record in artifact.work_units:
        chunks.append((artifact.root / record.a_path).read_bytes())
        chunks.append((artifact.root / record.b_path).read_bytes())
    return b"".join(chunks)


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
    """Pack one deterministic operation from already-staged request artifacts."""

    if not 1 <= len(artifacts) <= MAX_REQUESTS:
        raise ValueError("an operation must contain between 1 and 64 requests")
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
    descriptor_blob = b"".join(descriptors)
    total_bytes = body_offset + len(body)
    header = struct_pack_header(
        len(artifacts),
        ENVELOPE_HEADER_BYTES,
        body_offset,
        total_bytes,
        0,
        b"\0" * 32,
    )
    envelope = bytearray(header + descriptor_blob + body)
    envelope[_ENVELOPE_DIGEST_OFFSET : _ENVELOPE_DIGEST_OFFSET + 32] = _sha256(
        bytes(envelope)
    )
    return bytes(envelope)


def _summary_from_envelope(envelope: bytes) -> dict[str, object]:
    header = struct.unpack(ENVELOPE_HEADER_FORMAT, envelope[:ENVELOPE_HEADER_BYTES])
    request_count = header[3]
    requests: list[dict[str, object]] = []
    for index in range(request_count):
        offset = ENVELOPE_HEADER_BYTES + index * DESCRIPTOR_BYTES
        descriptor = struct.unpack(DESCRIPTOR_FORMAT, envelope[offset : offset + DESCRIPTOR_BYTES])
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
        "envelope_sha256": envelope[_ENVELOPE_DIGEST_OFFSET : _ENVELOPE_DIGEST_OFFSET + 32].hex(),
        "request_count": request_count,
        "requests": requests,
        "schema": SUMMARY_SCHEMA,
        "status": "accepted",
        "version": ENVELOPE_VERSION,
    }


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
    expected_bytes = _summary_bytes(expected)
    if result.stdout != expected_bytes:
        raise ValueError("C probe output is not equivalent to the deterministic summary")
    return expected


def _write_summary(output_dir: Path, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_bytes(_summary_bytes(summary))


def _run_probe_command(output_dir: Path, request_count: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
