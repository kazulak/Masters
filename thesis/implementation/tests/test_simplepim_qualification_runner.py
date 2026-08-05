from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from quantum_bench.providers.qualification import (
    load_provider_catalog,
    parse_runner_result,
    resolve_host_cc,
)
ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "native/upmem/simplepim/simplepim_qualification_runner.py"
SPEC = importlib.util.spec_from_file_location("simplepim_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
def _payload(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])
def _physical(monkeypatch) -> None:
    for key in runner.BACKEND_SELECTOR_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
def _preflight() -> dict:
    return {
        "verified": True,
        "device_nodes": [
            {
                "path": "/dev/dpu_rank0",
                "exists": True,
                "character_device": True,
                "readable": True,
                "writable": True,
                "sysfs_path": "/sys/class/dpu_rank/dpu_rank0",
                "sysfs_exists": True,
            }
        ],
        "required_pattern": "/dev/dpu_rank*",
        "reason": "hardware_device_node_verified",
    }
def _host() -> dict:
    return {
        "schema_version": runner.HOST_SCHEMA_VERSION,
        "provider_id": runner.PROVIDER_ID,
        "probe_id": runner.PROBE_ID,
        "status": "passed",
        "backend_profile": runner.BACKEND_PROFILE,
        "requested_dpu_count": 1,
        "observed_dpu_count": 1,
        "configured_tasklets_per_dpu": 12,
        "observed_tasklets_per_dpu": None,
        "native_run_completed": True,
        "validation_performed": True,
        "host_exact_validation": True,
        "fallback": False,
        "release_status": "released",
        "logical_input_bytes": runner.LOGICAL_INPUT_BYTES,
        "logical_output_bytes": runner.LOGICAL_OUTPUT_BYTES,
        "physical_transfer_bytes_available": False,
        "physical_transfer_bytes": None,
        "timing": {"kernel_s": 0.01},
        "failure_stage": None,
        "reason": None,
    }


def _provide_dpu_compiler(tmp_path: Path, monkeypatch) -> None:
    """Provide deterministic compiler provenance for hardware-only mocks."""
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    compiler = toolchain / runner.DPU_CC
    compiler.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    compiler.chmod(0o755)
    monkeypatch.setenv("PATH", f"{toolchain}{os.pathsep}{os.environ.get('PATH', '')}")
def _fake_commands(command, cwd, env, timeout_seconds):
    assert env["DPU_BACKEND"] == "hw"
    if command[0] == "make":
        (cwd / "bin").mkdir(parents=True, exist_ok=True)
        for name in ("host", "dpu_init_binary", "dpu_zip", "dpu_map_va_funcs"):
            (cwd / "bin" / name).write_bytes(name.encode())
        return {
            "status": "passed",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "wall_s": 0.01,
        }
    a = runner._deterministic_values(0)
    b = runner._deterministic_values(1)
    Path(command[1]).write_bytes(runner._pack_uint32(a))
    Path(command[2]).write_bytes(runner._pack_uint32(b))
    Path(command[3]).write_bytes(runner._pack_uint32(tuple(x + y for x, y in zip(a, b, strict=True))))
    return {
        "status": "passed",
        "returncode": 0,
        "stdout": json.dumps(_host()),
        "stderr": "",
        "wall_s": 0.01,
    }
def test_prepare_only_writes_contract_without_preflight_or_build(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        runner,
        "_hardware_device_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("preflight")),
    )
    monkeypatch.setattr(
        runner,
        "_run_command",
        lambda *args: (_ for _ in ()).throw(AssertionError("build")),
    )
    assert runner.main(["--prepare-only", "--workdir", str(tmp_path)]) == 0
    payload = _payload(capsys)
    assert payload["status"] == "prepared"
    assert payload["reason"] == "prepare_only_no_compiler_or_hardware_invoked"
    assert not (tmp_path / "build").exists()
    payload["hardware_preflight_verified"] = True
    payload["device_evidence"] = _preflight()["device_nodes"]
    with pytest.raises(ValueError, match="must not claim"):
        runner._validate_output_schema(payload)
def test_execute_requires_opt_in_before_hardware(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", raising=False)
    monkeypatch.setattr(
        runner,
        "_hardware_device_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("preflight")),
    )
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 1
    payload = _payload(capsys)
    assert payload["failure_stage"] == "opt_in"
    assert payload["release_status"] == "not_attempted"
@pytest.mark.parametrize("selector", ["sim", "simulator", "fsim"])
def test_simulator_cli_selectors_are_rejected(selector: str, tmp_path: Path, capsys) -> None:
    assert runner.main(["--execute", "--backend", selector, "--workdir", str(tmp_path)]) == 1
    assert _payload(capsys)["failure_stage"] == "backend_selection"
def test_missing_device_fails_before_staging(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
    monkeypatch.setattr(
        runner,
        "_hardware_device_preflight",
        lambda: {"verified": False, "device_nodes": [], "reason": "missing hardware"},
    )
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 1
    payload = _payload(capsys)
    assert payload["failure_stage"] == "hardware_preflight"
    assert not (tmp_path / "build").exists()
def test_staging_is_fresh_patched_and_does_not_mutate_upstream(tmp_path: Path) -> None:
    plan = runner.qualification_plan(tmp_path)
    upstream = runner._hash_file(plan["external_root"] / runner.PATCH_TARGET)
    stale = plan["build_root"] / "stale"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")
    evidence = runner._stage_sources(plan)
    target = plan["staged_simplepim"] / runner.PATCH_TARGET
    assert evidence["applied"] is True
    assert evidence["replacement_count"] == 2
    assert runner.FIXED_UNROLL_LINE in target.read_text(encoding="utf-8")
    assert runner.BUGGY_UNROLL_LINE not in target.read_text(encoding="utf-8")
    assert runner._hash_file(plan["external_root"] / runner.PATCH_TARGET) == upstream
    assert not stale.exists()
    drifted = runner.qualification_plan(tmp_path / "drifted")
    drifted["source_fingerprint"]["patch_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed since qualification plan"):
        runner._stage_sources(drifted)


def test_staging_applies_patch_inside_parent_git_worktree(tmp_path: Path) -> None:
    parent_worktree = tmp_path / "parent-worktree"
    parent_worktree.mkdir()
    subprocess.run(["git", "init", "--quiet", str(parent_worktree)], check=True)

    plan = runner.qualification_plan(parent_worktree / "qualification")
    evidence = runner._stage_sources(plan)
    target = plan["staged_simplepim"] / runner.PATCH_TARGET

    assert evidence["applied"] is True
    assert evidence["replacement_count"] == 2
    assert target.read_text(encoding="utf-8").count(runner.FIXED_UNROLL_LINE) == 2
    assert runner.BUGGY_UNROLL_LINE not in target.read_text(encoding="utf-8")
    assert plan["command_environment"]["patch"]["GIT_CEILING_DIRECTORIES"] == str(
        plan["staged_simplepim"].parent
    )
    assert evidence["environment"] == plan["command_environment"]["patch"]

    expected_fingerprint = runner._hash_json(
        {
            "command": plan["commands"]["patch"],
            "environment": plan["command_environment"]["patch"],
        }
    )
    assert evidence["command_fingerprint"] == expected_fingerprint


def test_patch_timeout_emits_structured_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    _physical(monkeypatch)
    monkeypatch.setattr(runner, "_hardware_device_preflight", _preflight)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(
            runner.subprocess.TimeoutExpired(command, runner.PATCH_TIMEOUT_SECONDS)
        ),
    )
    output = tmp_path / "result.json"
    assert runner.main(["--execute", "--workdir", str(tmp_path), "--json-output", str(output)]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert payload == json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_stage"] == "staged_patch_timeout"
    assert payload["reason"] == "tracked_patch_application_timed_out"
    assert payload["fallback"] is payload["simulator_kernel_executed"] is False
    assert payload["release_status"] == "not_attempted"
    assert "timeout_cleanup" not in payload
    assert "Traceback" not in captured.err
def test_independent_validation_checks_both_inputs_and_output(tmp_path: Path) -> None:
    a = runner._pack_uint32(runner._deterministic_values(0))
    b = runner._pack_uint32(runner._deterministic_values(1))
    out = runner._pack_uint32(
        tuple(
            x + y
            for x, y in zip(
                runner._deterministic_values(0),
                runner._deterministic_values(1),
                strict=True,
            )
        )
    )
    for name, blob in (("a", a), ("b", b), ("result", out)):
        (tmp_path / name).write_bytes(blob)
    assert runner._validate_artifacts(tmp_path / "a", tmp_path / "b", tmp_path / "result")["passed"] is True
    (tmp_path / "b").write_bytes(b"bad")
    assert runner._validate_artifacts(tmp_path / "a", tmp_path / "b", tmp_path / "result")["passed"] is False
def test_valid_passed_payload_proves_physical_result(tmp_path: Path, monkeypatch, capsys) -> None:
    _physical(monkeypatch)
    _provide_dpu_compiler(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_hardware_device_preflight", _preflight)
    monkeypatch.setattr(runner, "_run_command", _fake_commands)
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 0
    payload = _payload(capsys)
    assert payload["status"] == "passed"
    assert payload["target"] == payload["target_observed"] == "physical_hardware"
    assert payload["requested_dpu_count"] == payload["observed_dpu_count"] == 1
    assert payload["configured_tasklets_per_dpu"] == 12
    assert payload["observed_tasklets_per_dpu"] is None
    assert payload["release_status"] == "released"
    assert payload["fallback"] is False
    assert payload["source_hash"] == payload["source_hashes"]["combined_sha256"]
    preflight = runner._base_payload(runner.qualification_plan(tmp_path / "preflight"), "prepared")
    preflight.update(commands={}, reason="prepare_only_no_compiler_or_hardware_invoked")
    provider = load_provider_catalog(ROOT / "configs/qualification/upmem_provider_m1.yml").get("simplepim")
    assert parse_runner_result(payload, provider, expected_host_cc=resolve_host_cc(), expected_preflight=preflight).status == "qualified"
    drifted_preflight = json.loads(json.dumps(preflight))
    drifted_preflight["source_hashes"]["patch_sha256"] = "0" * 64
    assert parse_runner_result(payload, provider, expected_host_cc=resolve_host_cc(), expected_preflight=drifted_preflight).status == "failed"
    false_device = json.loads(json.dumps(payload))
    false_device["device_evidence"] = [{}]
    assert parse_runner_result(false_device, provider, expected_host_cc=resolve_host_cc(), expected_preflight=preflight).status == "failed"
    with pytest.raises(ValueError, match="lacks required physical qualification evidence"):
        runner._validate_output_schema(false_device)


def test_missing_dpu_compiler_evidence_fails_closed(tmp_path: Path) -> None:
    payload = runner._base_payload(runner.qualification_plan(tmp_path), "passed")
    payload["reason"] = None
    payload["effective_compilers"]["dpu_cc"] = {
        "command": runner.DPU_CC,
        "available": False,
        "path": None,
        "sha256": None,
    }
    provider = load_provider_catalog(ROOT / "configs/qualification/upmem_provider_m1.yml").get("simplepim")
    result = parse_runner_result(payload, provider, expected_host_cc=resolve_host_cc(), expected_preflight=payload)
    assert result.status == "failed"
    assert any("compiler" in error for error in result.contract_errors)
    with pytest.raises(ValueError, match="lacks required physical qualification evidence"):
        runner._validate_output_schema(payload)
def test_minimal_passed_payload_is_rejected(tmp_path: Path) -> None:
    payload = runner._base_payload(runner.qualification_plan(tmp_path), "passed")
    payload["reason"] = None
    provider = load_provider_catalog(ROOT / "configs/qualification/upmem_provider_m1.yml").get("simplepim")
    result = parse_runner_result(payload, provider, expected_host_cc=resolve_host_cc(), expected_preflight=payload)
    assert result.status == "failed"
    with pytest.raises(ValueError, match="lacks required physical qualification evidence"):
        runner._validate_output_schema(payload)
def test_timeout_cleanup_is_attempted_but_unverified(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import subprocess,time,sys; subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); time.sleep(60)",
    ]
    result = runner._run_command(command, tmp_path, os.environ, 0.05)
    assert result["status"] == "timeout"
    cleanup = result["timeout_cleanup"]
    assert cleanup["attempted"] is True
    assert cleanup["verified"] is False
    assert cleanup["verification"] == "unavailable"
    assert cleanup["output_capture_complete"] is True
    assert "process_group_terminated" not in cleanup
def test_runner_timeout_drain_is_bounded(tmp_path: Path, monkeypatch) -> None:
    class Process:
        pid = 123
        returncode = None
        stdout = None
        stderr = None
        def __init__(self):
            self.timeouts = []
        def communicate(self, *, timeout):
            self.timeouts.append(timeout)
            raise runner.subprocess.TimeoutExpired(["native"], timeout, output=b"partial", stderr=b"error")
    process = Process()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner.os, "killpg", lambda *args: None)
    result = runner._run_command(["native"], tmp_path, os.environ, 0.1)
    assert process.timeouts == [0.1, 2, 2]
    assert result["stdout"] == "partial" and result["stderr"] == "error"
    assert result["timeout_cleanup"]["verified"] is False
    assert result["timeout_cleanup"]["output_capture_complete"] is False
def test_parser_reports_release_and_observation_contract() -> None:
    provider = load_provider_catalog(ROOT / "configs/qualification/upmem_provider_m1.yml").get("simplepim")
    payload = {
        "status": "failed",
        "requested_dpu_count": 1,
        "observed_dpu_count": 1,
        "configured_tasklets_per_dpu": 12,
        "observed_tasklets_per_dpu": None,
        "reason": "hardware unavailable",
    }
    parsed = parse_runner_result(payload, provider, expected_host_cc=resolve_host_cc())
    assert parsed.status == "failed"
    assert parsed.release_status is None
    assert parsed.observed_dpus == 1
