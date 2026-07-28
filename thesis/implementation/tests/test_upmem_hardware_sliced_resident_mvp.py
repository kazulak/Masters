from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import quantum_bench.bench.upmem_hardware_sliced_resident_mvp as mvp
import quantum_bench.targets.upmem.hardware_sliced_resident_session as adapter
from quantum_bench.core.jsonio import read_jsonl, write_json
from quantum_bench.targets.upmem.hardware_session import HardwareSessionBuild
from quantum_bench.targets.upmem.hardware_sliced_resident_session import (
    SlicedResidentSessionExecution,
)
from quantum_bench.targets.upmem.hardware_taskgraph_sliced_resident import (
    _file_fingerprints,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs" / "suites" / "upmem_hardware_sliced_resident_mvp.yml"


def _fake_build(root: Path, session_root: Path, **_: object) -> HardwareSessionBuild:
    binary_dir = session_root / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)
    host = binary_dir / "host_two_dpu"
    dpu = binary_dir / "dpu_resident_two_dpu"
    host.write_bytes(b"host")
    dpu.write_bytes(b"dpu")
    return HardwareSessionBuild(
        session_root=session_root,
        source_snapshot=session_root / "src",
        build_dir=binary_dir,
        host_binary=host,
        dpu_binary=dpu,
        source_tree_hash="source-hash",
        host_binary_hash="host-hash",
        dpu_binary_hash="dpu-hash",
        build_time_s=0.125,
        build_command=("make", "all"),
        sdk_tools={"make": "fake"},
    )


def _fake_execute(build, *, manifest_paths, response_path, **_: object):
    payload = _write_completed_native_response(manifest_paths, response_path)
    return SlicedResidentSessionExecution(
        status="completed",
        failure_stage=None,
        response_path=response_path,
        response=payload,
        process_time_s=0.25,
        command=("fake-host",),
        stdout_snippet="",
        stderr_snippet="",
        timed_out=False,
        cleanup_confirmed=True,
    )


def _write_completed_native_response(manifest_paths, response_path: Path) -> dict:
    outputs = {
        "one_qubit_x_m2": np.asarray([0.0, 1.0], dtype=np.float32),
        "one_qubit_h_m2": np.asarray([2**-0.5, 2**-0.5], dtype=np.float32),
        "one_qubit_z_m2": np.asarray([1.0, 0.0], dtype=np.float32),
    }
    entries = []
    for slice_id, manifest_path in enumerate(manifest_paths):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        case_id = next(key for key in outputs if key in manifest_path.name)
        output_path = Path(payload["final_outputs"][0]["output_path"])
        if not output_path.is_absolute():
            output_path = manifest_path.parent / output_path
        (outputs[case_id] / 2).tofile(output_path)
        entries.append(
            {
                "slice_id": slice_id,
                "dpu_index": slice_id,
                "allocated": True,
                "release_confirmed": True,
                "package_transferred": True,
                "input_count": 2,
                "inputs_transferred": True,
                "partial_output_elements": 2,
                "partial_output_bytes": 8,
                "partial_output_path": str(output_path),
                "partial_output_read": True,
                "partial_output_written": True,
                "completion_confirmed": True,
                "manifest_path": str(manifest_path),
                "manifest_fnv1a64": _file_fingerprints(manifest_path)["fnv1a64"],
            }
        )
    payload = {
        "schema_version": "generic_loop_resident_two_dpu_contraction_slice_v1",
        "manifest_kind": "resident_two_slice_response",
        "status": "completed",
        "failure_stage": None,
        "cpu_fallback_used": False,
        "topology": "two_dpu_allocation",
        "hardware_execution": True,
        "native_reconstruction_performed": False,
        "reconstruction_contract": "python_sum_partials",
        "allocation": {
            "requested_dpus": 2,
            "allocated_dpus": 2,
            "profile": "backend=hw",
            "verified": True,
        },
        "launch": {
            "mode": "asynchronous",
            "async_launch_count": 1,
            "synchronize_count": 1,
            "completed": True,
        },
        "release": {"attempted": True, "confirmed": True},
        "slices": entries,
    }
    write_json(response_path, payload)
    return payload


def test_prepare_only_writes_all_plans_without_adapter_execution(
    tmp_path, monkeypatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        mvp,
        "execute_sliced_resident_hardware_session",
        lambda *args, **kwargs: calls.append(args),
    )

    result = mvp.prepare_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={}
    )

    assert result.status == "prepared"
    assert calls == []
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert len(summary["prepared_operations"]) == 12
    assert summary["dpu_allocation_attempted"] is False
    assert (
        result.plan_dir
        / "cases"
        / "one_qubit_x_m2"
        / "measured_02"
        / "slice_0_manifest.json"
    ).is_file()


def test_execution_uses_one_fake_session_and_preserves_m2_evidence(
    tmp_path, monkeypatch
) -> None:
    builds: list[object] = []

    def build(*args, **kwargs):
        value = _fake_build(*args, **kwargs)
        builds.append(value)
        return value

    monkeypatch.setattr(mvp, "build_sliced_resident_hardware_session", build)
    monkeypatch.setattr(mvp, "execute_sliced_resident_hardware_session", _fake_execute)

    result = mvp.run_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
    )

    assert result.status == "completed"
    assert len(builds) == 1
    measured = read_jsonl(result.run_dir / "normalized_records.jsonl")
    warmups = read_jsonl(result.run_dir / "warmups.jsonl")
    assert len(measured) == 9
    assert len(warmups) == 3
    assert {row["phase"] for row in measured} == {"measured"}
    assert all(row["validation_status"] == "passed" for row in measured)
    assert all(row["allocated_dpu_count"] == 2 for row in measured)
    assert all(
        row["slice_count"] == 2 and row["tasklets_per_dpu"] == 1 for row in measured
    )
    for row in measured:
        assert row["application_visible_transfer_bytes"] == (
            row["application_visible_h2d_bytes"] + row["application_visible_d2h_bytes"]
        )
        assert row["actual_transfer_bytes"] == (
            row["actual_h2d_bytes"] + row["actual_d2h_bytes"]
        )
        assert row["actual_transfer_bytes"] == row["application_visible_transfer_bytes"]
    x_row = next(row for row in measured if row["case_id"] == "one_qubit_x_m2")
    expected_hash = hashlib.sha256(
        (ROOT / "configs/circuits/upmem_m2/one_qubit_x.qasm").read_bytes()
    ).hexdigest()
    assert x_row["qasm_source_sha256"] == expected_hash
    assert (
        x_row["source_hashes"]["circuit_semantics_hash"]
        == x_row["circuit_semantics_hash"]
    )
    assert (
        result.run_dir
        / "cases"
        / "one_qubit_h_m2"
        / "measured_01"
        / "reconstructed_output.npy"
    ).is_file()


def test_execute_requires_opt_in_and_retains_failure_row(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        mvp,
        "build_sliced_resident_hardware_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    result = mvp.run_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={}
    )

    assert result.status == "failed"
    rows = read_jsonl(result.run_dir / "normalized_records.jsonl")
    assert len(rows) == 1
    assert rows[0]["failure_stage"] == "hardware_opt_in_missing"
    assert rows[0]["application_visible_transfer_bytes"] == 0
    assert rows[0]["actual_transfer_bytes"] == 0


def test_native_failure_is_preserved_for_each_measured_operation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(mvp, "build_sliced_resident_hardware_session", _fake_build)

    def failed(build, *, response_path, **kwargs):
        return SlicedResidentSessionExecution(
            status="failed",
            failure_stage="kernel_launch_failed",
            response_path=response_path,
            response={"status": "failed", "failure_stage": "kernel_launch_failed"},
            process_time_s=0.125,
            command=("fake-host", "--slice-package-0"),
            stdout_snippet="native stdout",
            stderr_snippet="native stderr",
            timed_out=False,
            cleanup_confirmed=True,
        )

    monkeypatch.setattr(mvp, "execute_sliced_resident_hardware_session", failed)
    result = mvp.run_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
    )

    rows = read_jsonl(result.run_dir / "normalized_records.jsonl")
    assert result.status == "failed"
    assert len(rows) == 9
    assert {row["failure_stage"] for row in rows} == {"kernel_launch_failed"}
    assert all(
        row["native_response_artifact"].startswith("native_session/") for row in rows
    )
    assert all(len(row["package_manifest_artifacts"]) == 2 for row in rows)
    assert all(
        row["native_session_command"] == ["fake-host", "--slice-package-0"]
        for row in rows
    )
    assert all(row["native_stdout_snippet"] == "native stdout" for row in rows)
    assert all(row["native_stderr_snippet"] == "native stderr" for row in rows)
    assert all(row["native_failure_stage"] == "kernel_launch_failed" for row in rows)
    assert all(
        row["application_visible_transfer_bytes"]
        == row["application_visible_h2d_bytes"] + row["application_visible_d2h_bytes"]
        == row["actual_transfer_bytes"]
        for row in rows
    )


def test_runner_rejects_copied_or_noncanonical_suite_paths(tmp_path) -> None:
    copied = tmp_path / SUITE.name
    copied.write_text(SUITE.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical"):
        mvp.load_m2_suite(copied)


def test_runner_uses_real_adapter_session_path_and_command_contract(
    tmp_path, monkeypatch
) -> None:
    calls: list[tuple[tuple[str, ...], Path, dict[str, str], float]] = []
    monkeypatch.setattr(mvp, "build_sliced_resident_hardware_session", _fake_build)

    def physical(command, *, cwd, env, timeout_s):
        command = tuple(command)
        calls.append((command, cwd, dict(env), timeout_s))
        _write_completed_native_response(
            (Path(command[2]), Path(command[4])), Path(command[6])
        )
        return {
            "returncode": 0,
            "elapsed_s": 0.25,
            "timed_out": False,
            "cleanup_confirmed": True,
            "stdout_snippet": "low-level stdout",
            "stderr_snippet": "",
        }

    monkeypatch.setattr(adapter, "_run_physical_command", physical)
    result = mvp.run_upmem_hardware_sliced_resident_mvp(
        tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
    )

    assert result.status == "completed"
    assert len(calls) == 12
    for command, cwd, environment, timeout_s in calls:
        session_root = cwd.parent
        assert command[1] == "--slice-package-0"
        assert command[3] == "--slice-package-1"
        assert command[5] == "--resident-response"
        assert Path(command[2]).is_relative_to(session_root)
        assert Path(command[4]).is_relative_to(session_root)
        assert Path(command[6]).is_relative_to(session_root)
        assert environment == {"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"}
        assert timeout_s == 30.0


def test_cli_and_make_targets_expose_the_internal_research_mvp() -> None:
    command = subprocess.run(
        [sys.executable, "-m", "quantum_bench.bench", "--help"],
        cwd=ROOT,
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    plan = subprocess.run(
        ["make", "-n", "upmem-hw-sliced-resident-plan"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    execute = subprocess.run(
        ["make", "-n", "upmem-hw-sliced-resident"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert command.returncode == 0
    assert "upmem-hardware-sliced-resident-mvp" in command.stdout
    assert plan.returncode == execute.returncode == 0
    assert "--prepare-only --build" in plan.stdout
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE" in execute.stdout
    assert "--execute" in execute.stdout
