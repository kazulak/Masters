from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "host_boundary_probe" / "host_boundary_probe.py"
SPEC = importlib.util.spec_from_file_location("host_boundary_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _probe_path(tmp_path: Path) -> Path:
    return probe.compile_probe(tmp_path / "upoenv_probe")


def _v1_envelope(tmp_path: Path) -> bytes:
    artifacts = probe.build_synthetic_artifacts(tmp_path / "artifacts")
    return probe.pack_operation_envelope(artifacts)


def _v2_envelope(tmp_path: Path, count: int = 2) -> bytes:
    requests = probe.direct_prepared_requests(count, 2)
    return probe.pack_upoenv2_prepared_operation(requests, operation_sequence=9)


def _reseal_v2(envelope: bytearray, *, descriptors: tuple[int, ...] = ()) -> bytes:
    for index in descriptors:
        offset = probe.UPOENV2_HEADER_BYTES + index * probe.UPOENV2_DESCRIPTOR_BYTES
        envelope[offset + 168 : offset + 200] = b"\0" * 32
        envelope[offset + 168 : offset + 200] = hashlib.sha256(
            envelope[offset : offset + probe.UPOENV2_DESCRIPTOR_BYTES]
        ).digest()
    envelope[64:96] = b"\0" * 32
    envelope[64:96] = hashlib.sha256(envelope).digest()
    return bytes(envelope)


def _run(probe_path: Path, envelope: bytes) -> tuple[int, bytes, bytes]:
    result = probe.invoke_probe(probe_path, envelope)
    return result.returncode, result.stdout, result.stderr


def test_v1_is_deterministic_and_c_equivalent(tmp_path: Path) -> None:
    c_probe = _probe_path(tmp_path)
    first = _v1_envelope(tmp_path / "first")
    second = _v1_envelope(tmp_path / "second")

    assert first == second
    summary = probe.probe_envelope(first, c_probe)
    assert summary["request_count"] == 2
    assert [item["request_sequence"] for item in summary["requests"]] == [0, 1]
    assert all(item["work_unit_count"] == 2 for item in summary["requests"])


def test_v2_uses_exact_layout_and_has_no_64_descriptor_cap(tmp_path: Path) -> None:
    c_probe = _probe_path(tmp_path)
    envelope = _v2_envelope(tmp_path / "many", count=65)
    header = struct.unpack(
        probe.UPOENV2_HEADER_FORMAT, envelope[: probe.UPOENV2_HEADER_BYTES]
    )
    assert header[:7] == (probe.UPOENV2_MAGIC, 2, 96, 65, 200, 0, 0)
    assert header[7] == 96
    assert header[8] == 96 + 65 * 200
    assert header[9] == len(envelope)
    assert probe.validate_upoenv2(envelope)["descriptor_count"] == 65
    summary = probe.probe_envelope(envelope, c_probe)
    assert summary["descriptor_count"] == 65
    assert summary["operation_sequence"] == 9
    assert [row["request_sequence"] for row in summary["requests"]] == list(range(65))


@pytest.mark.parametrize(
    "case",
    (
        "truncation",
        "descriptor_count_arithmetic",
        "overflow_bounds",
        "overlap",
        "reorder",
        "digest_corruption",
        "unsupported_version",
        "unsupported_flags",
        "trailing_bytes",
    ),
)
def test_v2_c_probe_rejects_malformed_envelopes(
    tmp_path: Path, case: str
) -> None:
    c_probe = _probe_path(tmp_path)
    malformed = bytearray(_v2_envelope(tmp_path / case))
    if case == "truncation":
        malformed = malformed[:-1]
    elif case == "descriptor_count_arithmetic":
        struct.pack_into("<I", malformed, 16, (1 << 32) - 1)
        malformed = bytearray(_reseal_v2(malformed))
    elif case == "overflow_bounds":
        struct.pack_into("<Q", malformed, 96 + 8, (1 << 64) - 1)
        malformed = bytearray(_reseal_v2(malformed, descriptors=(0,)))
    elif case == "overlap":
        first_offset = struct.unpack_from("<Q", malformed, 96 + 8)[0]
        struct.pack_into("<Q", malformed, 96 + probe.UPOENV2_DESCRIPTOR_BYTES + 8, first_offset)
        malformed = bytearray(_reseal_v2(malformed, descriptors=(1,)))
    elif case == "reorder":
        first = bytes(malformed[96 : 96 + probe.UPOENV2_DESCRIPTOR_BYTES])
        second = bytes(
            malformed[
                96 + probe.UPOENV2_DESCRIPTOR_BYTES : 96 + 2 * probe.UPOENV2_DESCRIPTOR_BYTES
            ]
        )
        malformed[96 : 96 + probe.UPOENV2_DESCRIPTOR_BYTES] = second
        malformed[96 + probe.UPOENV2_DESCRIPTOR_BYTES : 96 + 2 * probe.UPOENV2_DESCRIPTOR_BYTES] = first
        malformed = bytearray(_reseal_v2(malformed))
    elif case == "digest_corruption":
        malformed[-1] ^= 1
    elif case == "unsupported_version":
        struct.pack_into("<I", malformed, 8, 7)
        malformed = bytearray(_reseal_v2(malformed))
    elif case == "unsupported_flags":
        struct.pack_into("<I", malformed, 24, 1)
        malformed = bytearray(_reseal_v2(malformed))
    else:
        struct.pack_into("<Q", malformed, 48, len(malformed) + 1)
        malformed.extend(b"trailing")
        malformed = bytearray(_reseal_v2(malformed))

    returncode, stdout, stderr = _run(c_probe, bytes(malformed))
    assert returncode != 0
    assert stdout == b""
    assert stderr.startswith(b"reject: ")


def test_v1_cli_remains_explicit_and_deterministic(tmp_path: Path) -> None:
    first_dir = tmp_path / "cli-first"
    second_dir = tmp_path / "cli-second"
    assert probe.main(["probe", "--output-dir", str(first_dir), "--requests", "2"]) == 0
    assert probe.main(["probe", "--output-dir", str(second_dir), "--requests", "2"]) == 0
    assert (first_dir / "operation.upoenv").read_bytes() == (second_dir / "operation.upoenv").read_bytes()
    assert (first_dir / "summary.json").read_bytes() == (second_dir / "summary.json").read_bytes()


def test_legacy_benchmark_is_explicit_and_does_not_run_on_import(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    assert probe.main(["benchmark", "--output-dir", str(output_dir), "--repeats", "2"]) == 0
    summary = (output_dir / "summary.json").read_text(encoding="utf-8")
    assert '"schema":"upoenv1_benchmark_v1"' in summary


def test_boundary_benchmark_writes_raw_rows_replay_and_decision(tmp_path: Path) -> None:
    output_dir = tmp_path / "boundary-benchmark"
    result = probe.benchmark_boundary(output_dir, warmups=0, repeats=1)
    assert len(result["cells"]) == 6
    raw_rows = (output_dir / "host_boundary_benchmark.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(raw_rows) == 12
    first = json.loads(raw_rows[0])
    assert {
        "cell",
        "block",
        "arm_order",
        "phase_timings",
        "counts",
        "bytes",
        "process_status",
        "environment",
        "provenance",
    } <= set(first)
    assert (output_dir / "host_boundary_benchmark_raw.csv").exists()
    replay = json.loads((output_dir / "host_boundary_replay.json").read_text(encoding="utf-8"))
    assert len(replay["cells"]) == 6
    assert replay["operations"]
    assert max(row["descriptor_count"] for row in replay["operations"]) > 64
    decision = json.loads((output_dir / "phase0_decision.json").read_text(encoding="utf-8"))
    assert decision["future_physical_optimized_data_used"] is False
    assert (output_dir / "host_boundary_benchmark.csv").exists()
    assert (output_dir / "host_boundary_replay.csv").exists()
