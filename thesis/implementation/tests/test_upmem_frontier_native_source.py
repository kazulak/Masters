from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = (
    IMPLEMENTATION_ROOT
    / "native"
    / "upmem"
    / "simplepim"
    / "upmem_sdk_generic_loop_frontier_two_dpu"
)
HOST_BINARY = NATIVE_ROOT / "bin" / "host_frontier_two_dpu"


@pytest.fixture(scope="module")
def native_host() -> Path:
    if shutil.which("dpu-pkg-config") is None or shutil.which("dpu-upmem-dpurte-clang") is None:
        pytest.skip("UPMEM SDK compiler is unavailable")
    subprocess.run(["make", "clean", "all"], cwd=NATIVE_ROOT, check=True)
    return HOST_BINARY


def test_frontier_source_serializes_a_generated_dependency_order_and_one_transfer_breakdown() -> None:
    source = (NATIVE_ROOT / "host.c").read_text(encoding="ascii")

    assert "completed_task_ids" in source
    assert "wave_dependency_order_not_intra_wave_finish_order" in source
    assert "completed_task_ids_scope" in source
    assert "completion_order_scope" not in source
    assert '"completion_order":["task_0","task_1","task_2"]' not in source
    assert '\\"wave_index\\":%d,\\"operation_index\\":%d,\\"dpu_index\\":%d' in source
    assert '\\"descriptor_package_h2d_bytes\\"' in source
    assert '\\"operation_control_h2d_bytes\\"' in source
    assert '\\"h2d_bytes\\"' in source
    assert '\\"d2h_bytes\\"' in source
    assert '\\"total_bytes\\"' in source
    assert '"sdk_visible"' not in source
    assert '\\"accounting_scope\\":\\"sdk_argument_byte_counts\\"' in source
    assert '\\"written\\":%s' in source
    assert "frontier_hash_bytes(frontier->final_output" in source


def test_native_success_schema_uses_python_required_transfer_and_completion_keys() -> None:
    source = (NATIVE_ROOT / "host.c").read_text(encoding="ascii")
    required = (
        '\\"completed_task_ids\\"',
        '\\"completed_task_ids_scope\\"',
        '\\"descriptor_package_h2d_bytes\\"',
        '\\"initial_h2d_bytes\\"',
        '\\"operation_control_h2d_bytes\\"',
        '\\"inter_wave_h2d_bytes\\"',
        '\\"inter_wave_d2h_bytes\\"',
        '\\"final_d2h_bytes\\"',
        '\\"h2d_bytes\\"',
        '\\"d2h_bytes\\"',
        '\\"total_bytes\\"',
        '\\"transfer_invariant\\"',
        '\\"accounting_scope\\"',
    )
    assert all(key in source for key in required)


def test_native_response_uses_manifest_relative_output_and_counts_control_h2d_time() -> None:
    source = (NATIVE_ROOT / "host.c").read_text(encoding="ascii")

    assert "frontier_output_reference" in source
    assert "frontier_validate_output_path" in source
    assert "realpath" in source
    assert "lstat" in source
    assert "frontier_json_string(file, final_output_reference)" in source
    assert "timing->initial_h2d_time_s + timing->descriptor_h2d_time_s + timing->control_h2d_time_s" in source
    assert "&timing.descriptor_h2d_time_s, &timing.initial_h2d_time_s" in source
    assert "*descriptor_h2d_time_s += frontier_now_s()" in source


def test_validate_only_malformed_request_is_parseable_and_never_allocates_hardware(
    native_host: Path, tmp_path: Path
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="ascii")

    completed = subprocess.run(
        [str(native_host), "--validate-frontier-package", str(request)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "invalid"
    assert payload["failure_stage"] == "manifest_parse_failed"
    assert payload["native_execution"] is False
    assert payload["allocation_attempted"] is False
    assert payload["launch_attempted"] is False
    assert payload["release_attempted"] is False
    assert payload["operation_count"] == 0
    assert payload["final_output_count"] == 0


def test_runtime_malformed_request_writes_structured_failed_response_without_sigsegv(
    native_host: Path, tmp_path: Path
) -> None:
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text("{}", encoding="ascii")

    completed = subprocess.run(
        [
            str(native_host),
            "--frontier-package",
            str(request),
            "--frontier-response",
            str(response),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.returncode >= 0
    payload = json.loads(response.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_stage"] == "manifest_parse_failed"
    assert payload["failure_context"] is None
    assert payload["final_output"] is None
    assert payload["hashes"]["manifest_fnv1a64"]
    assert payload["hashes"]["package_fnv1a64"] is None


def test_failed_response_transfer_and_output_fields_have_native_semantics(
    native_host: Path, tmp_path: Path
) -> None:
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text("{}", encoding="ascii")
    subprocess.run(
        [
            str(native_host),
            "--frontier-package",
            str(request),
            "--frontier-response",
            str(response),
        ],
        check=False,
    )
    payload = json.loads(response.read_text(encoding="utf-8"))
    transfer = payload["transfer"]
    components = (
        "descriptor_package_h2d_bytes",
        "initial_h2d_bytes",
        "operation_control_h2d_bytes",
        "inter_wave_h2d_bytes",
        "inter_wave_d2h_bytes",
        "final_d2h_bytes",
    )
    assert all(isinstance(transfer[key], int) and transfer[key] >= 0 for key in components)
    assert transfer["h2d_bytes"] == transfer["descriptor_package_h2d_bytes"] + transfer["initial_h2d_bytes"] + transfer["operation_control_h2d_bytes"] + transfer["inter_wave_h2d_bytes"]
    assert transfer["d2h_bytes"] == transfer["inter_wave_d2h_bytes"] + transfer["final_d2h_bytes"]
    assert transfer["total_bytes"] == transfer["h2d_bytes"] + transfer["d2h_bytes"]
    assert payload["actual_h2d_bytes"] == transfer["h2d_bytes"]
    assert payload["actual_d2h_bytes"] == transfer["d2h_bytes"]
    assert payload["actual_transfer_bytes"] == transfer["total_bytes"]
    assert payload["final_output"] is None
