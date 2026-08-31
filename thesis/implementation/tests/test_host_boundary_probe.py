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


def _envelope(tmp_path: Path) -> bytes:
    artifacts = probe.build_synthetic_artifacts(tmp_path / "artifacts")
    return probe.pack_operation_envelope(artifacts)


def _reseal(envelope: bytearray) -> bytes:
    envelope[64:96] = b"\0" * 32
    envelope[64:96] = hashlib.sha256(envelope).digest()
    return bytes(envelope)


def _run(probe_path: Path, envelope: bytes) -> tuple[int, bytes, bytes]:
    result = probe.invoke_probe(probe_path, envelope)
    return result.returncode, result.stdout, result.stderr


def test_host_packed_operation_is_deterministic_and_c_equivalent(tmp_path: Path) -> None:
    c_probe = _probe_path(tmp_path)
    first = _envelope(tmp_path / "first")
    second = _envelope(tmp_path / "second")

    assert first == second
    summary = probe.probe_envelope(first, c_probe)
    repeat = probe.invoke_probe(c_probe, first)
    assert repeat.returncode == 0
    assert repeat.stdout == probe._summary_bytes(summary)
    assert summary["request_count"] == 2
    assert [item["request_sequence"] for item in summary["requests"]] == [0, 1]
    assert all(item["work_unit_count"] == 2 for item in summary["requests"])


@pytest.mark.parametrize("case", ("truncated", "overlap", "reordered", "invalid_count", "digest"))
def test_c_probe_rejects_malformed_envelopes(tmp_path: Path, case: str) -> None:
    c_probe = _probe_path(tmp_path)
    malformed = bytearray(_envelope(tmp_path / case))
    if case == "truncated":
        malformed = malformed[:-1]
    elif case == "overlap":
        first_descriptor = 96
        second_descriptor = 96 + probe.DESCRIPTOR_BYTES
        first_request_offset = struct.unpack_from("<Q", malformed, first_descriptor + 8)[0]
        struct.pack_into("<Q", malformed, second_descriptor + 8, first_request_offset)
        malformed = bytearray(_reseal(malformed))
    elif case == "reordered":
        first = bytes(malformed[96 : 96 + probe.DESCRIPTOR_BYTES])
        second = bytes(malformed[96 + probe.DESCRIPTOR_BYTES : 96 + 2 * probe.DESCRIPTOR_BYTES])
        malformed[96 : 96 + probe.DESCRIPTOR_BYTES] = second
        malformed[96 + probe.DESCRIPTOR_BYTES : 96 + 2 * probe.DESCRIPTOR_BYTES] = first
        malformed = bytearray(_reseal(malformed))
    elif case == "invalid_count":
        struct.pack_into("<I", malformed, 16, 0)
        malformed = bytearray(_reseal(malformed))
    else:
        malformed[-1] ^= 1

    returncode, stdout, stderr = _run(c_probe, bytes(malformed))
    assert returncode != 0
    assert stdout == b""
    assert stderr.startswith(b"reject: ")


def test_cli_writes_explicit_deterministic_output_directory(tmp_path: Path) -> None:
    first_dir = tmp_path / "cli-first"
    second_dir = tmp_path / "cli-second"
    assert probe.main(["probe", "--output-dir", str(first_dir), "--requests", "2"]) == 0
    assert probe.main(["probe", "--output-dir", str(second_dir), "--requests", "2"]) == 0
    assert (first_dir / "operation.upoenv").read_bytes() == (second_dir / "operation.upoenv").read_bytes()
    assert (first_dir / "summary.json").read_bytes() == (second_dir / "summary.json").read_bytes()


def test_benchmark_is_explicit_and_does_not_run_on_import(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    assert probe.main(["benchmark", "--output-dir", str(output_dir), "--repeats", "2"]) == 0
    summary = (output_dir / "summary.json").read_text(encoding="utf-8")
    assert '"schema":"upoenv1_benchmark_v1"' in summary


def test_boundary_benchmark_cli_writes_measurement_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / "boundary-benchmark"
    assert probe.main(
        [
            "boundary-benchmark",
            "--output-dir",
            str(output_dir),
            "--warmups",
            "0",
            "--repeats",
            "1",
        ]
    ) == 0
    result = json.loads(
        (output_dir / "host_boundary_benchmark.json").read_text(encoding="utf-8")
    )
    assert len(result["cells"]) == 6
    assert (output_dir / "host_boundary_benchmark.csv").exists()
