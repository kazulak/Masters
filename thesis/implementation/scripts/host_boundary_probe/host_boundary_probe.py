#!/usr/bin/env python3
"""Build and validate a deterministic host-only UPOENV1 packed operation.

The packed body contains existing ABI-v4 request sidecars and the staged A/B
payload bytes.  This module intentionally stops at a file boundary: it never
opens a rank, loads an SDK, or reaches the production runtime.
"""

from __future__ import annotations

import argparse
import csv
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
    NUMERIC_FLOAT32,
    NUMERIC_MODE_FLOAT32,
    V4Profile,
    V4Header,
    V4WorkUnit,
    V4RequestArtifact,
    HEADER_BYTES,
    PARTITION_OUTPUT_TILE,
    VERSION,
    WORK_UNIT_BYTES,
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
CELL_DEFINITIONS = (
    ("quantization_stress_18q_l2", 1, 141, 888),
    ("quantization_stress_18q_l2", 4, 141, 636),
    ("hs_18q_d1", 1, 53, 224),
    ("hs_18q_d1", 4, 53, 212),
    ("ghz_chain_18q", 1, 35, 1484),
    ("ghz_chain_18q", 4, 35, 464),
)

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


@dataclass(frozen=True)
class _PreparedRequest:
    """Directly prepared request bytes used by the candidate arm."""

    request_sequence: int
    sidecar: bytes
    payload: bytes
    work_unit_count: int
    request_output_elements: int


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


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

    if not 1 <= request_count <= MAX_REQUESTS:
        raise ValueError("request_count must be in [1, 64]")
    if not 1 <= dpu_count <= 4:
        raise ValueError("dpu_count must be one of the frozen one- or four-DPU cells")
    profile = V4Profile(
        dpu_count=dpu_count,
        tasklets_per_dpu=8,
        numeric_mode=NUMERIC_FLOAT32,
    )
    artifacts: list[V4RequestArtifact] = []
    for sequence in range(request_count):
        units = _fixture_units(sequence, dpu_count)
        artifacts.append(
            build_v4_request(
                root / f"request_{sequence}",
                profile=profile,
                canonical_batch_count=1,
                canonical_m=dpu_count,
                canonical_n=2,
                canonical_k=4,
                work_units=units,
                task_contract_sha256=TASK_CONTRACT_SHA256,
                request_sequence=sequence,
            )
        )
    return tuple(artifacts)


def _direct_prepared_request(sequence: int, dpu_count: int) -> _PreparedRequest:
    """Prepare the same sidecar and payload bytes without filesystem staging."""

    units = _fixture_units(sequence, dpu_count)
    payloads = [_fixture_payloads(sequence, dpu_id) for dpu_id in range(dpu_count)]
    manifest_lines = [f"sidecar requests/{sequence:016d}/sidecar.bin"]
    for dpu_id, unit in enumerate(units):
        a_payload, b_payload = payloads[dpu_id]
        a_digest = hashlib.sha256(a_payload).hexdigest()
        b_digest = hashlib.sha256(b_payload).hexdigest()
        manifest_lines.append(
            "dpu {dpu} {tile} requests/{seq:016d}/payloads/dpu_{dpu:03d}_a.bin "
            "requests/{seq:016d}/payloads/dpu_{dpu:03d}_b.bin "
            "requests/{seq:016d}/outputs/dpu_{dpu:03d}_c.bin {a} {b}".format(
                dpu=dpu_id,
                tile=unit.tile_id,
                seq=sequence,
                a=a_digest,
                b=b_digest,
            )
        )
    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    # Build numeric record bytes with the accepted ABI-v4 layout.  The
    # sidecar header and records are the same bytes emitted by build_v4_request.
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
        tasklets_per_dpu=8,
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
    )


def direct_prepared_requests(
    request_count: int, dpu_count: int
) -> tuple[_PreparedRequest, ...]:
    if not 1 <= request_count <= MAX_REQUESTS:
        raise ValueError("request_count must be in [1, 64]")
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


def pack_prepared_operation(requests: Sequence[_PreparedRequest]) -> bytes:
    """Pack prepared bytes without materializing the current request files."""

    if not 1 <= len(requests) <= MAX_REQUESTS:
        raise ValueError("an operation must contain between 1 and 64 requests")
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


def benchmark_boundary(
    output_dir: Path,
    *,
    warmups: int = 2,
    repeats: int = 10,
) -> dict[str, object]:
    """Compare current per-request staging with one packed operation envelope."""

    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="upoenv-probe-build-") as probe_dir:
        probe = compile_probe(Path(probe_dir) / "upoenv_probe")
        rows: list[dict[str, object]] = []
        for case_id, dpu_count, operation_count, total_requests in CELL_DEFINITIONS:
            request_count = _operation_request_count(total_requests, operation_count)
            equivalence_root = Path(probe_dir) / f"equivalence-{case_id}-{dpu_count}"
            artifacts = build_synthetic_artifacts(
                equivalence_root / "artifacts", request_count, dpu_count=dpu_count
            )
            prepared = direct_prepared_requests(request_count, dpu_count)
            if any(
                artifact.sidecar_path.read_bytes() != request.sidecar
                or _payload_blob(artifact) != request.payload
                for artifact, request in zip(artifacts, prepared, strict=True)
            ):
                raise ValueError(f"prepared request bytes differ for {case_id}/{dpu_count}")
            packed = pack_prepared_operation(prepared)
            if probe_envelope(packed, probe)["request_count"] != request_count:
                raise ValueError("packed probe returned the wrong request count")
            current_times: list[float] = []
            packed_times: list[float] = []
            packed_bytes = len(packed)
            current_bytes = 0
            current_files = 0
            for iteration in range(warmups + repeats):
                with tempfile.TemporaryDirectory(
                    prefix=f"upoenv-current-{case_id}-{dpu_count}-"
                ) as current_directory:
                    current_started = time.perf_counter_ns()
                    current_artifacts = build_synthetic_artifacts(
                        Path(current_directory), request_count, dpu_count=dpu_count
                    )
                    current_file_paths = [
                        path
                        for artifact in current_artifacts
                        for path in artifact.request_dir.rglob("*")
                        if path.is_file()
                    ]
                    current_files = len(current_file_paths)
                    current_bytes = sum(path.stat().st_size for path in current_file_paths)
                    current_digest = _sha256(
                        b"".join(
                            artifact.sidecar_path.read_bytes()
                            for artifact in current_artifacts
                        )
                    )
                    (Path(current_directory) / "current.digest").write_bytes(current_digest)
                    current_elapsed = (time.perf_counter_ns() - current_started) / 1e9
                packed_started = time.perf_counter_ns()
                candidate = direct_prepared_requests(request_count, dpu_count)
                candidate_envelope = pack_prepared_operation(candidate)
                if candidate_envelope != packed:
                    raise ValueError("packed envelope is not deterministic")
                probe_envelope(candidate_envelope, probe)
                packed_elapsed = (time.perf_counter_ns() - packed_started) / 1e9
                if iteration >= warmups:
                    current_times.append(current_elapsed)
                    packed_times.append(packed_elapsed)
            rows.append(
                {
                    "case_id": case_id,
                    "dpu_count": dpu_count,
                    "tasklets_per_dpu": 8,
                    "contraction_operation_count": operation_count,
                    "total_embedded_request_count": total_requests,
                    "operation_request_count": request_count,
                    "current_submit_count": request_count,
                    "packed_submit_count": 1,
                    "current_request_directory_count": request_count,
                    "current_file_count": current_files,
                    "packed_file_count": 1,
                    "current_file_bytes": current_bytes,
                    "packed_envelope_bytes": packed_bytes,
                    "current_process_count": 0,
                    "packed_process_count": 1,
                    "current_median_s": _median(current_times),
                    "current_raw_mad_s": _raw_mad(current_times),
                    "packed_median_s": _median(packed_times),
                    "packed_raw_mad_s": _raw_mad(packed_times),
                    "boundary_speedup": _median(current_times) / _median(packed_times),
                }
            )
    result: dict[str, object] = {
        "analysis_version": "host_request_boundary_feasibility_v1",
        "warmups": warmups,
        "measurements": repeats,
        "cells": rows,
        "equivalence": {
            "request_bytes": True,
            "payload_bytes": True,
            "request_order": True,
            "work_unit_order": True,
            "output_summary": True,
        },
        "timing_scope": "host_only_preparation_and_transport_probe_v1",
        "interpretation": (
            "The current arm builds one accepted ABI-v4 artifact directory per "
            "embedded request. The packed arm constructs one operation envelope "
            "directly from the same prepared bytes and validates it in one C process."
        ),
    }
    (output_dir / "host_boundary_benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "host_boundary_benchmark.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = tuple(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return result


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
            command.add_argument("--warmups", type=int, default=2)
    boundary = commands.add_parser("boundary-benchmark")
    boundary.add_argument("--output-dir", type=Path, required=True)
    boundary.add_argument("--repeats", type=int, default=10)
    boundary.add_argument("--warmups", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "boundary-benchmark":
        benchmark_boundary(args.output_dir, warmups=args.warmups, repeats=args.repeats)
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
