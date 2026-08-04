from __future__ import annotations

import json
from pathlib import Path

import pytest

import quantum_bench.targets.upmem.hardware_frontier_session as session


def _profile(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "hardware_profile_version": session.PROFILE_ID,
        "target": "hardware",
        "backend_id": session.BACKEND_ID,
        "route_id": session.ROUTE_ID,
        "native_schema": session.NATIVE_SCHEMA,
        "requested_dpu_count": 2,
        "tasklets_per_dpu": 1,
        "numeric_mode": "none",
        "numeric_modes": ["none"],
        "synchronous_execution": True,
        "device_launch_mode": "asynchronous_dpu_set",
        "host_completion_mode": "blocking_sync",
        "timeout_s": 1.0,
        "performance_claim_applicable": False,
    }
    value.update(updates)
    return value


def test_profile_is_frozen_to_m31_identity() -> None:
    parsed = session.parse_hardware_frontier_profile(_profile())
    assert parsed.route_id == session.ROUTE_ID
    with pytest.raises(ValueError, match="backend_id"):
        session.parse_hardware_frontier_profile(_profile(backend_id="simulator"))


def test_execute_requires_opt_in_and_no_dpu_backend(tmp_path: Path) -> None:
    build = session.HardwareSessionBuild(
        tmp_path, tmp_path, tmp_path, tmp_path / "host", tmp_path / "dpu", "", "", "", 0.0, (), {}
    )
    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE"):
        session.execute_hardware_frontier_session(build, manifest_path=tmp_path / "m", response_path=tmp_path / "r", profile=_profile(), environment={})
    with pytest.raises(ValueError, match="DPU_BACKEND"):
        session.execute_hardware_frontier_session(build, manifest_path=tmp_path / "m", response_path=tmp_path / "r", profile=_profile(), environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "DPU_BACKEND": "sim"})


def test_build_command_snapshots_source_and_uses_frontier_binary_names(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "native" / "upmem" / "simplepim" / session.NATIVE_SOURCE_DIR
    source.mkdir(parents=True)
    (source / "Makefile").write_text("all:", encoding="ascii")
    resident = source.parent / "upmem_sdk_generic_loop_resident"
    resident.mkdir()
    (resident / "common.h").write_text("", encoding="ascii")
    (resident / "dpu.c").write_text("", encoding="ascii")
    (resident / "session_protocol.c").write_text("", encoding="ascii")
    (resident / "session_protocol.h").write_text("", encoding="ascii")
    monkeypatch.setattr(session, "_required_build_tools", lambda environment: ("make", {"make": "make"}))

    def fake_run(command, *, cwd, env, timeout_s):
        (cwd / "bin").mkdir()
        (cwd / "bin" / session.HOST_BINARY_NAME).write_bytes(b"host")
        (cwd / "bin" / session.DPU_BINARY_NAME).write_bytes(b"dpu")
        return {"returncode": 0, "timed_out": False, "stdout_snippet": "", "stderr_snippet": ""}

    monkeypatch.setattr(session, "_run_build_command", fake_run)
    build = session.build_hardware_frontier_session(tmp_path, tmp_path / "run", profile=_profile(), environment={})
    assert build.source_snapshot.is_dir()
    assert "FRONTIER_BARRIER_COUNT=2" in build.build_command
    assert build.host_binary.name == session.HOST_BINARY_NAME
    assert build.dpu_binary.name == session.DPU_BINARY_NAME
    assert (build.source_snapshot.parent / "upmem_sdk_generic_loop_resident").is_dir()


def test_execute_preserves_native_failure_stage(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "run"
    root.mkdir()
    host = root / "host"
    dpu = root / "dpu"
    host.write_bytes(b"host")
    dpu.write_bytes(b"dpu")
    manifest = {
        "schema_version": session.REQUEST_SCHEMA,
        "native_schema_version": session.NATIVE_SCHEMA,
        "route_id": session.ROUTE_ID,
        "backend_id": session.BACKEND_ID,
        "hardware_profile_version": session.PROFILE_ID,
        "target": "hardware",
        "session_protocol": session.REQUEST_SCHEMA,
        "requested_dpus": 2,
        "tasklets": 1,
        "tasklets_per_dpu": 1,
        "numeric_mode": "none",
        "quantization_mode": "none",
        "barrier_count": 2,
        "performance_claim_applicable": False,
        "expected_task_ids": ["task_0", "task_1", "task_2"],
        "expected_dpu_task_counts": [2, 1],
        "frontier_plan": {
            "plan_schema_version": "upmem_hardware_frontier_plan_m3_1_v1",
            "wave_count": 2,
            "waves": [
                {"wave_index": 0, "tasks": [{"task_id": "task_0", "wave_index": 0, "dpu_id": 0}, {"task_id": "task_1", "wave_index": 0, "dpu_id": 1}], "barrier_after": True},
                {"wave_index": 1, "tasks": [{"task_id": "task_2", "wave_index": 1, "dpu_id": 0}], "barrier_after": True},
            ],
            "assignments": [
                {"task_id": "task_0", "wave_index": 0, "dpu_id": 0},
                {"task_id": "task_1", "wave_index": 0, "dpu_id": 1},
                {"task_id": "task_2", "wave_index": 1, "dpu_id": 0},
            ],
            "barrier_count": 2,
            "expected_dpu_task_counts": [2, 1],
            "co_dispatch": True,
            "overlap_measured": False,
            "overlap_evidence": "co_dispatch_without_overlap_measurement",
        },
        "co_dispatch": True,
        "overlap_evidence": "co_dispatch_without_overlap_measurement",
        "overlap_measured": False,
        "frontier_task_operations": [
            {"task_id": "task_0", "wave_index": 0, "dpu_id": 0, "operation_id": 0, "component": "real", "kind": "contract", "mode": "none", "input_slot_ids": [0, 1], "output_slot_id": 2, "output_elements": 2, "dependencies": []},
            {"task_id": "task_1", "wave_index": 0, "dpu_id": 1, "operation_id": 1, "component": "real", "kind": "contract", "mode": "none", "input_slot_ids": [3, 4], "output_slot_id": 5, "output_elements": 2, "dependencies": []},
            {"task_id": "task_2", "wave_index": 1, "dpu_id": 0, "operation_id": 2, "component": "real", "kind": "contract", "mode": "none", "input_slot_ids": [2, 5], "output_slot_id": 6, "output_elements": 2, "dependencies": ["task_0", "task_1"]},
        ],
        "resident_package_binding": {"package_sha256": "0" * 64, "slot_descriptors": [], "operations": []},
        "expected_frontier_transfer": {key: 0 for key in ("descriptor_package_h2d_bytes", "initial_h2d_bytes", "operation_control_h2d_bytes", "inter_wave_d2h_bytes", "inter_wave_h2d_bytes", "final_d2h_bytes", "h2d_bytes", "d2h_bytes", "total_bytes")},
        "final_output_binding": {"component": "real", "slot_id": 6, "elements": 2, "raw_bytes": 8, "output_path": "outputs/final.bin", "hash_fnv1a64_required": True},
        "package_path": "package.bin",
        "dpu_binary": "dpu",
    }
    (root / "package.bin").write_bytes(b"package")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    response_path = root / "response.json"
    response_path.write_text(json.dumps({"status": "failed", "failure_stage": "hardware_allocation_failed", "release": {"confirmed": True}}), encoding="utf-8")
    build = session.HardwareSessionBuild(root, root, root, host, dpu, "", "", "", 0.0, (), {})
    monkeypatch.setattr(session, "validate_frontier_package_against_manifest", lambda *_args: None)
    monkeypatch.setattr(session, "_run_frontier_command", lambda *args, **kwargs: {"returncode": 1, "timed_out": False, "elapsed_s": 0.1, "stdout_snippet": "", "stderr_snippet": ""})
    result = session.execute_hardware_frontier_session(build, manifest_path=manifest_path, response_path=response_path, profile=_profile(), environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"})
    assert result.status == "failed"
    assert result.failure_stage == "hardware_allocation_failed"
    assert result.cleanup_confirmed is True


def test_validate_copied_run_uses_explicit_local_session_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied_root = tmp_path / "copied-run" / "native_session"
    output = copied_root / "outputs" / "final.bin"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"\x00" * 8)
    manifest_path = copied_root / "manifest.json"
    response_path = copied_root / "response.json"
    manifest_path.write_text(
        json.dumps({"final_output_binding": {"output_path": "outputs/final.bin", "elements": 2}}),
        encoding="utf-8",
    )
    response_path.write_text(
        json.dumps(
            {
                "final_output": {
                    "output_path": "outputs/final.bin",
                    "path": "outputs/final.bin",
                    "hash_fnv1a64": session._fnv1a64(output.read_bytes()),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(session, "_validate_manifest_identity", lambda _manifest: None)
    monkeypatch.setattr(session, "_validate_frontier_manifest", lambda *_args: None)
    monkeypatch.setattr(session, "validate_frontier_native_response", lambda *_args: None)
    monkeypatch.setattr(session, "_validate_native_hashes", lambda *_args: None)

    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["final_output"]["hash_fnv1a64"] = "0" * 16
    response_path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="FNV-1a64"):
        session.validate_hardware_frontier_session(
            manifest_path,
            response_path,
            profile=_profile(),
            session_root=copied_root,
        )
    response["final_output"]["hash_fnv1a64"] = session._fnv1a64(output.read_bytes())
    response_path.write_text(json.dumps(response), encoding="utf-8")

    result = session.validate_hardware_frontier_session(
        manifest_path,
        response_path,
        profile=_profile(),
        session_root=copied_root,
    )

    assert result["final_output"]["path"] == "outputs/final.bin"
