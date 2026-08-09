import json
import shutil
import subprocess
from pathlib import Path

import pytest


IMPLEMENTATION = Path(__file__).resolve().parents[1]
ROUTE = IMPLEMENTATION / "native" / "upmem" / "simplepim" / "upmem_sdk_rank1_dot_m4_2"


def _host_and_manifest():
    if shutil.which("dpu-pkg-config") is None or shutil.which("dpu-upmem-dpurte-clang") is None:
        pytest.skip("UPMEM SDK is not installed")
    subprocess.run(["make", "-C", str(ROUTE), "build"], check=True)
    staged = ROUTE / "build" / "simplepim_rank1_dot_m4_2" / "staged"
    return (
        staged / "benchmarks" / "rank1_dot" / "bin" / "rank1_dot_host",
        staged / "simplepim_stage_manifest.json",
    )


def _fnv1a64(payload):
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def _run_parser(tmp_path, payload=None):
    host, manifest = _host_and_manifest()
    operands = tmp_path / "operands.bin"
    if payload is not None:
        operands.write_bytes(payload)
    response_path = tmp_path / "response.json"
    result = subprocess.run(
        [
            str(host),
            "--mode",
            "parser",
            "--operands-file",
            str(operands),
            "--response",
            str(response_path),
            "--stage-manifest",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result, json.loads(response_path.read_text())


def test_external_operand_file_reports_exact_raw_input_hash_and_length(tmp_path):
    payload = bytes(range(256)) + bytes((255 - value for value in range(256)))
    result, response = _run_parser(tmp_path, payload)
    assert result.returncode == 0
    assert response["status"] == "prepared"
    assert response["external_operand_transport"] is True
    assert response["operand_input_length_bytes"] == 512
    assert response["operand_input_hash"] == _fnv1a64(payload)
    assert response["allocation_attempted"] is False


@pytest.mark.parametrize(
    "payload, reason",
    [
        (b"", "operand_file_short"),
        (bytes(511), "operand_file_short"),
        (bytes(512) + b"x", "operand_file_trailing_data"),
    ],
)
def test_malformed_external_operand_file_fails_before_allocation(tmp_path, payload, reason):
    result, response = _run_parser(tmp_path, payload)
    assert result.returncode != 0
    assert response["status"] == "failed"
    assert response["failure_stage"] == "input"
    assert response["reason"] == reason
    assert response["external_operand_transport"] is True
    assert response["allocation_attempted"] is False


def test_missing_external_operand_file_fails_before_allocation(tmp_path):
    result, response = _run_parser(tmp_path)
    assert result.returncode != 0
    assert response["status"] == "failed"
    assert response["failure_stage"] == "input"
    assert response["reason"] == "operand_file_open_failed"
    assert response["external_operand_transport"] is True
    assert response["allocation_attempted"] is False


def test_native_source_keeps_fixed_default_and_rejects_external_fallback():
    source = (ROUTE / "host.c").read_text()
    readme = (ROUTE / "README.md").read_text()
    hardening = (ROUTE / "simplepim_rank1_hardening.patch").read_text()
    assert "--operands-file" in source
    assert "M42_EXTERNAL_OPERAND_BYTES (2u * M42_VECTOR_LENGTH * sizeof(int8_t))" in source
    assert "operand_file_trailing_data" in source
    assert "never falls back" in readme
    assert "fill_inputs(values_a, values_b, &reference);" in source
    assert "external_operand_transport" in source
    assert 'printf("%s", table_id);' in hardening
    assert "+        printf(table_id);" not in hardening
