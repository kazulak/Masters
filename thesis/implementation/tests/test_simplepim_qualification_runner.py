from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pytest

from quantum_bench.providers.qualification import (
    load_provider_catalog,
    parse_runner_result,
    resolve_host_cc,
    simplepim_runner_schema_errors,
)


ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "native/upmem/simplepim/simplepim_qualification_runner.py"
CATALOG_PATH = ROOT / "configs/qualification/upmem_provider_m1.yml"
SPEC = importlib.util.spec_from_file_location(
    "simplepim_qualification_runner", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _payload(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _enable_physical(monkeypatch) -> None:
    for key in runner.BACKEND_SELECTOR_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")


def _verified_preflight() -> dict:
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


def _host_result(**updates: object) -> dict:
    payload = {
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
        "timing": {"kernel_s": 0.002},
        "failure_stage": None,
        "reason": None,
    }
    payload.update(updates)
    return payload


def _write_artifacts(command, *, output_mode: str = "correct") -> None:
    a_values = runner._deterministic_values(0)
    b_values = runner._deterministic_values(1)
    output_values = tuple(
        (left + right) & 0xFFFFFFFF
        for left, right in zip(a_values, b_values, strict=True)
    )
    if output_mode == "bad_input":
        a_values = (a_values[0] + 1, *a_values[1:])
    if output_mode == "wrong_value":
        output_values = (output_values[0] + 1, *output_values[1:])
    Path(command[1]).write_bytes(runner._pack_uint32(a_values))
    Path(command[2]).write_bytes(runner._pack_uint32(b_values))
    output_blob = runner._pack_uint32(output_values)
    if output_mode == "truncated":
        output_blob = output_blob[:-8]
    Path(command[3]).write_bytes(output_blob)


def _fake_build_and_host(command, cwd, env, timeout_seconds):
    assert env["DPU_BACKEND"] == "hw"
    if command[0] == "make":
        binary_dir = cwd / "bin"
        binary_dir.mkdir(parents=True, exist_ok=True)
        for name in ("host", "dpu_init_binary", "dpu_zip", "dpu_map_va_funcs"):
            (binary_dir / name).write_bytes(name.encode("ascii"))
        return {
            "status": "passed",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "wall_s": 0.01,
        }
    _write_artifacts(command)
    return {
        "status": "passed",
        "returncode": 0,
        "stdout": json.dumps(_host_result()),
        "stderr": "",
        "wall_s": 0.02,
    }


def _install_fake_dpu_compiler(tmp_path: Path, monkeypatch) -> Path:
    compiler = tmp_path / "test-toolchain/dpu-upmem-dpurte-clang"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"controlled test compiler identity\n")
    real_compiler_identity = runner._compiler_identity

    def compiler_identity(command: str, resolved_path: Path | None = None) -> dict:
        if command == runner.DPU_CC and resolved_path is None:
            return real_compiler_identity(command, compiler)
        return real_compiler_identity(command, resolved_path)

    monkeypatch.setattr(runner, "_compiler_identity", compiler_identity)
    return compiler


def _controlled_success_payload(tmp_path: Path, monkeypatch, capsys) -> dict:
    _install_fake_dpu_compiler(tmp_path, monkeypatch)
    _enable_physical(monkeypatch)
    monkeypatch.setattr(runner, "_hardware_device_preflight", _verified_preflight)
    monkeypatch.setattr(runner, "_run_command", _fake_build_and_host)
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 0
    return _payload(capsys)


def test_prepare_only_fingerprints_patch_and_commands_without_side_effects(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("prepare-only must not run commands or hardware preflight")

    monkeypatch.setattr(runner, "_run_command", unexpected)
    monkeypatch.setattr(runner, "_hardware_device_preflight", unexpected)
    output_path = tmp_path / "prepare.json"
    assert (
        runner.main(
            [
                "--prepare-only",
                "--workdir",
                str(tmp_path / "work"),
                "--json-output",
                str(output_path),
            ]
        )
        == 0
    )

    payload = _payload(capsys)
    assert payload["schema_version"] == "simplepim_provider_qualification_v1"
    assert payload["provider_id"] == "simplepim"
    assert payload["probe_id"] == "simplepim_va_map_zip_v1"
    assert payload["status"] == "prepared"
    assert payload["hardware_preflight_verified"] is False
    assert payload["native_execution"] is False
    assert payload["validation_performed"] is False
    assert payload["exact_validation"] is False
    assert payload["fallback"] is False
    assert payload["simulator_kernel_executed"] is False
    assert payload["configured_tasklets_per_dpu"] == 12
    assert payload["observed_tasklets_per_dpu"] is None
    assert payload["staged_patch"]["sha256"]
    assert payload["staged_patch"]["applied"] is False
    assert payload["source_hashes"]["patch_sha256"] == payload["staged_patch"]["sha256"]
    expected_host_cc = resolve_host_cc(os.environ)
    assert payload["effective_compilers"]["host_cc"]["path"] == expected_host_cc["path"]
    assert (
        payload["effective_compilers"]["host_cc"]["sha256"]
        == expected_host_cc["sha256"]
    )
    provider = load_provider_catalog(CATALOG_PATH).get("simplepim")
    assert (
        simplepim_runner_schema_errors(
            payload,
            provider,
            mode="prepare",
            expected_host_cc=expected_host_cc,
        )
        == ()
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / "work" / "build").exists()


def test_execute_requires_explicit_opt_in(tmp_path: Path, monkeypatch, capsys) -> None:
    for key in runner.BACKEND_SELECTOR_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", raising=False)
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 1

    payload = _payload(capsys)
    assert payload["failure_stage"] == "opt_in"
    assert payload["release_status"] == "not_attempted"
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DPU_BACKEND", "sim"),
        ("DPU_PROFILE", "simulator"),
        ("SIMPLEPIM_BACKEND", "fsim"),
        ("UPMEM_BACKEND", "casim"),
        ("UPMEM_MODE", "simulator"),
        ("UPMEM_TARGET", "fsim"),
        ("UPMEM_PROFILE", "backend=simulator"),
        ("UPMEM_PROFILE_BASE", "casim"),
    ],
)
def test_all_external_backend_selectors_are_rejected_before_preflight(
    key: str, value: str, tmp_path: Path, monkeypatch, capsys
) -> None:
    _enable_physical(monkeypatch)
    monkeypatch.setenv(key, value)

    def unexpected_preflight():
        raise AssertionError("selector rejection must precede hardware preflight")

    monkeypatch.setattr(runner, "_hardware_device_preflight", unexpected_preflight)
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 1

    payload = _payload(capsys)
    assert payload["failure_stage"] == "backend_selection"
    assert key in payload["reason"]
    assert payload["hardware_preflight_verified"] is False


@pytest.mark.parametrize("flag", ["--backend", "--target"])
@pytest.mark.parametrize("alias", ["sim", "simulator", "fsim", "casim"])
def test_cli_simulator_aliases_are_rejected(
    flag: str, alias: str, tmp_path: Path, capsys
) -> None:
    assert runner.main(["--execute", flag, alias, "--workdir", str(tmp_path)]) == 1
    assert _payload(capsys)["failure_stage"] == "backend_selection"


def test_missing_device_node_fails_before_staging(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _enable_physical(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_hardware_device_preflight",
        lambda: {
            "verified": False,
            "device_nodes": [],
            "required_pattern": "/dev/dpu_rank*",
            "reason": "no_accessible_physical_dpu_rank_device_node",
        },
    )
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 1

    payload = _payload(capsys)
    assert payload["failure_stage"] == "hardware_preflight"
    assert payload["target"] is None
    assert payload["target_observed"] is None
    assert payload["device_evidence"] == []
    assert not (tmp_path / "build").exists()


def test_command_and_makefile_enforce_twelve_tasklets(tmp_path: Path) -> None:
    plan = runner.qualification_plan(tmp_path)
    build = plan["commands"]["build"]
    makefile = (RUNNER_PATH.parent / "qualification/Makefile").read_text(
        encoding="utf-8"
    )

    assert all(not argument.startswith("NR_TASKLETS=") for argument in build)
    assert "-DNR_TASKLETS=12" in makefile
    assert "NR_TASKLETS ?=" not in makefile
    assert "ELEMENTS ?=" not in makefile
    assert f"HOST_CC={plan['host_cc']}" in build
    assert Path(plan["host_cc"]).is_absolute()
    assert plan["effective_compilers"]["host_cc"]["command"] == "gcc"
    assert plan["effective_compilers"]["host_cc"]["path"] == plan["host_cc"]
    assert "HOST_CC := gcc" in makefile
    assert "HOST_CC ?=" not in makefile
    assert "simulator" not in " ".join(build + plan["commands"]["run"]).lower()


def test_staged_patch_is_hashed_applied_twice_and_submodule_is_unchanged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = runner.qualification_plan(tmp_path)
    upstream_target = plan["external_root"] / runner.PATCH_TARGET
    upstream_before = runner._hash_file(upstream_target)
    stale_stage = plan["staged_simplepim"] / "stale-source"
    stale_input = plan["inputs_dir"] / "stale-input"
    stale_output = plan["outputs_dir"] / "stale-output"
    for stale in (stale_stage, stale_input, stale_output):
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("stale\n", encoding="utf-8")

    patch_calls: list[list[str]] = []
    real_run = runner.subprocess.run

    def tracking_run(command, **kwargs):
        patch_calls.append(list(command))
        return real_run(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", tracking_run)

    evidence = runner._stage_sources(plan)

    staged_target = plan["staged_simplepim"] / runner.PATCH_TARGET
    staged_text = staged_target.read_text(encoding="utf-8")
    patch_text = plan["patch_path"].read_text(encoding="utf-8")
    hunk_headers = [line for line in patch_text.splitlines() if line.startswith("@@ ")]
    assert len(hunk_headers) == 2
    assert all(runner._patch_hunk_context_size(header) > 1 for header in hunk_headers)
    assert runner._patch_hunks_have_nonblank_context(patch_text)
    assert not runner._patch_hunks_have_nonblank_context("@@ -1 +1 @@\n-old\n+new\n")
    assert patch_calls == [plan["commands"]["patch"]]
    assert patch_calls[0][:3] == ["git", "apply", "--no-index"]
    assert "--unidiff-zero" not in patch_calls[0]
    assert str(plan["staged_patch_path"]) in patch_calls[0]
    assert not stale_stage.exists()
    assert not stale_input.exists()
    assert not stale_output.exists()
    assert evidence["sha256"] == runner._hash_file(plan["patch_path"])
    assert evidence["staged_patch_sha256"] == evidence["sha256"]
    assert evidence["applied"] is True
    assert evidence["replacement_count"] == 2
    assert (
        evidence["staged_source_before_sha256"]
        != evidence["staged_source_after_sha256"]
    )
    assert evidence["staged_source_after_sha256"] == runner._hash_tree(
        plan["staged_simplepim"]
    )
    assert (
        evidence["source_hashes"]["combined_sha256"]
        == evidence["staged_source_after_sha256"]
    )
    assert evidence["source_hashes"]["owned_qualification_sha256"] == (
        runner._hash_tree(plan["staged_benchmark"])
    )
    assert evidence["source_hashes"]["upstream_library_sha256"] == runner._hash_tree(
        plan["staged_simplepim"] / "lib"
    )
    assert staged_text.count(runner.FIXED_UNROLL_LINE) == 2
    assert runner.BUGGY_UNROLL_LINE not in staged_text
    assert runner._hash_file(upstream_target) == upstream_before


def test_controlled_success_uses_independent_files_and_core_schema(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    payload = _controlled_success_payload(tmp_path, monkeypatch, capsys)
    assert payload["status"] == "passed"
    assert payload["target"] == payload["target_observed"] == "physical_hardware"
    assert payload["hardware_preflight_verified"] is True
    assert payload["device_evidence"][0]["path"] == "/dev/dpu_rank0"
    assert payload["observed_dpu_count"] == 1
    assert payload["configured_tasklets_per_dpu"] == 12
    assert payload["observed_tasklets_per_dpu"] is None
    assert payload["native_execution"] is True
    assert payload["validation_performed"] is True
    assert payload["exact_validation"] is True
    assert payload["release_status"] == "released"
    assert payload["failure_stage"] is None
    assert payload["reason"] is None
    assert payload["host_result"]["status"] == "passed"
    assert payload["host_result"]["release_status"] == "released"
    assert payload["host_result"]["reason"] is None
    assert payload["build"]["status"] == "passed"
    assert payload["build"]["command"] == payload["commands"]["build"]
    assert payload["execution"]["status"] == "passed"
    assert payload["execution"]["command"] == payload["commands"]["run"]
    for role in ("host_cc", "dpu_cc"):
        identity = payload["effective_compilers"][role]
        assert identity["available"] is True
        assert runner._hash_file(Path(identity["path"])) == identity["sha256"]
    assert (
        f"HOST_CC={payload['effective_compilers']['host_cc']['path']}"
        in payload["build"]["command"]
    )
    assert (
        f"DPU_CC={payload['effective_compilers']['dpu_cc']['command']}"
        in payload["build"]["command"]
    )
    assert payload["staged_patch"]["applied"] is True
    assert payload["staged_patch"]["staged_sha256"] == payload["staged_patch"]["sha256"]
    assert payload["source_hash"] == payload["source_hashes"]["combined_sha256"]
    assert (
        payload["source_hashes"]["staged_source_after_patch_sha256"]
        == payload["source_hash"]
    )
    assert payload["logical_transfer_bytes"] == {
        "h2d": 2048,
        "d2h": 1024,
        "total": 3072,
        "scope": "logical_application_payload_only",
    }
    assert payload["payload_sizes_8_byte_aligned"] is True
    assert payload["physical_transfer_bytes_available"] is False
    assert payload["physical_transfer_bytes"] is None
    assert all(payload["input_hashes"].values())
    assert payload["output_hash"]

    prepared_plan = runner.qualification_plan(tmp_path)
    canonical_preflight = runner._base_payload(prepared_plan, "prepared")
    canonical_preflight.update(
        {
            "commands": prepared_plan["commands"],
            "reason": "prepare_only_no_compiler_or_hardware_invoked",
        }
    )
    provider = load_provider_catalog(CATALOG_PATH).get("simplepim")
    qualification = parse_runner_result(
        payload,
        provider,
        expected_host_cc=resolve_host_cc(os.environ),
        expected_dpu_cc=canonical_preflight["effective_compilers"]["dpu_cc"],
        expected_preflight=canonical_preflight,
    )
    assert qualification.status == "qualified"
    assert qualification.configured_tasklets == 12
    assert qualification.observed_tasklets is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "failed"),
        ("release_status", "failed"),
        ("validation_performed", False),
        ("host_exact_validation", False),
        ("failure_stage", "contradiction"),
        ("reason", "contradiction"),
        ("configured_tasklets_per_dpu", 11),
    ],
)
def test_passed_payload_rejects_nested_host_contradictions(
    field: str,
    value: object,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = _controlled_success_payload(tmp_path, monkeypatch, capsys)
    payload["host_result"][field] = value

    with pytest.raises(ValueError, match="strict qualification"):
        runner._validate_output_schema(payload)


@pytest.mark.parametrize("missing", ["commands", "build", "execution"])
def test_passed_payload_requires_retained_command_evidence(
    missing: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = _controlled_success_payload(tmp_path, monkeypatch, capsys)
    del payload[missing]

    with pytest.raises(ValueError):
        runner._validate_output_schema(payload)


@pytest.mark.parametrize(
    ("container", "key"),
    [
        (None, "source_hash"),
        ("source_hashes", "combined_sha256"),
        ("source_hashes", "staged_source_after_patch_sha256"),
        ("staged_patch", "staged_source_after_sha256"),
    ],
)
def test_staged_source_hash_chain_rejects_any_contradiction(
    container: str | None,
    key: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = _controlled_success_payload(tmp_path, monkeypatch, capsys)
    target = payload if container is None else payload[container]
    target[key] = "0" * 64

    with pytest.raises(ValueError, match="hash chain"):
        runner._validate_output_schema(payload)


@pytest.mark.parametrize("role", ["host_cc", "dpu_cc"])
def test_compiler_hashes_are_verified_against_executed_build(
    role: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = _controlled_success_payload(tmp_path, monkeypatch, capsys)
    payload["effective_compilers"][role]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="compiler file evidence"):
        runner._validate_output_schema(payload)


@pytest.mark.parametrize(
    ("variable", "replacement"),
    [
        ("HOST_CC", "HOST_CC=/bin/false"),
        ("DPU_CC", "DPU_CC=/bin/false"),
    ],
)
def test_build_command_cannot_contradict_compiler_identities(
    variable: str,
    replacement: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = _controlled_success_payload(tmp_path, monkeypatch, capsys)
    build_command = [
        replacement if argument.startswith(f"{variable}=") else argument
        for argument in payload["commands"]["build"]
    ]
    payload["commands"]["build"] = build_command
    payload["command_fingerprint"] = runner._hash_json(payload["commands"])
    payload["build"]["command"] = build_command
    payload["build"]["command_fingerprint"] = runner._hash_json(build_command)

    with pytest.raises(ValueError, match="compiler identities"):
        runner._validate_output_schema(payload)


@pytest.mark.parametrize(
    ("output_mode", "failure_stage", "reason"),
    [
        (
            "wrong_value",
            "exact_validation",
            "independent_exact_uint32_validation_failed",
        ),
        (
            "truncated",
            "artifact_size",
            "artifact_uint32_element_count_mismatch",
        ),
        (
            "bad_input",
            "input_validation",
            "deterministic_input_validation_failed",
        ),
    ],
)
def test_independent_validation_rejects_negative_artifacts(
    output_mode: str,
    failure_stage: str,
    reason: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _enable_physical(monkeypatch)
    monkeypatch.setattr(runner, "_hardware_device_preflight", _verified_preflight)

    def fake(command, cwd, env, timeout_seconds):
        if command[0] == "make":
            return _fake_build_and_host(command, cwd, env, timeout_seconds)
        _write_artifacts(command, output_mode=output_mode)
        return {
            "status": "passed",
            "returncode": 0,
            "stdout": json.dumps(_host_result()),
            "stderr": "",
            "wall_s": 0.02,
        }

    monkeypatch.setattr(runner, "_run_command", fake)
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 1

    payload = _payload(capsys)
    assert payload["failure_stage"] == failure_stage
    assert payload["reason"] == reason
    assert payload["exact_validation"] is False
    assert payload["fallback"] is False


def test_upstream_abort_without_result_fails_with_release_unconfirmed(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _enable_physical(monkeypatch)
    monkeypatch.setattr(runner, "_hardware_device_preflight", _verified_preflight)
    stale_output = tmp_path / "build/outputs/result_u32.bin"
    stale_output.parent.mkdir(parents=True)
    stale_output.write_bytes(b"\x00" * runner.LOGICAL_OUTPUT_BYTES)

    def fake_abort(command, cwd, env, timeout_seconds):
        if command[0] == "make":
            return _fake_build_and_host(command, cwd, env, timeout_seconds)
        Path(command[1]).write_bytes(
            runner._pack_uint32(runner._deterministic_values(0))
        )
        Path(command[2]).write_bytes(
            runner._pack_uint32(runner._deterministic_values(1))
        )
        return {
            "status": "failed",
            "returncode": 134,
            "stdout": "",
            "stderr": "DPU_ASSERT terminated the upstream call",
            "wall_s": 0.02,
        }

    monkeypatch.setattr(runner, "_run_command", fake_abort)
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 1

    payload = _payload(capsys)
    assert payload["status"] == "failed"
    assert payload["failure_stage"] == "host_process"
    assert payload["reason"] == (
        "upstream_native_process_failed_without_host_result_release_unconfirmed"
    )
    assert payload["release_status"] == "unknown"
    assert payload["target"] is None
    assert payload["native_execution"] is False
    assert payload["output_hash"] is None
    assert not stale_output.exists()
    provider = load_provider_catalog(CATALOG_PATH).get("simplepim")
    qualification = parse_runner_result(
        payload,
        provider,
        expected_host_cc=resolve_host_cc(os.environ),
    )
    assert qualification.status == "failed"
    assert qualification.release_status == "unknown"


def test_timeout_never_claims_release_or_physical_target(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _enable_physical(monkeypatch)
    monkeypatch.setattr(runner, "_hardware_device_preflight", _verified_preflight)

    def fake_timeout(command, cwd, env, timeout_seconds):
        if command[0] == "make":
            return _fake_build_and_host(command, cwd, env, timeout_seconds)
        return {
            "status": "timeout",
            "returncode": -15,
            "stdout": "",
            "stderr": "",
            "wall_s": timeout_seconds,
            "timeout_cleanup": {
                "process_group": 123,
                "sigterm_sent": True,
                "sigkill_sent": False,
                "process_exited": True,
                "group_probe_performed": True,
                "leader_exited_before_group_probe": True,
                "live_members_after_sigterm": [],
                "live_members_after_cleanup": [],
                "process_group_terminated": True,
                "signal_errors": [],
            },
        }

    monkeypatch.setattr(runner, "_run_command", fake_timeout)
    assert runner.main(["--execute", "--workdir", str(tmp_path)]) == 1

    payload = _payload(capsys)
    assert payload["failure_stage"] == "host_timeout"
    assert payload["release_status"] == "unknown"
    assert payload["target"] is None
    assert payload["target_observed"] is None
    assert payload["native_execution"] is False
    assert payload["timeout_cleanup"]["process_exited"] is True
    assert payload["timeout_cleanup"]["group_probe_performed"] is True


def test_run_command_terminates_the_whole_process_group_on_timeout(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import subprocess,sys,time;"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            "time.sleep(60)"
        ),
    ]
    result = runner._run_command(command, tmp_path, os.environ, 0.05)

    assert result["status"] == "timeout"
    assert result["timeout_cleanup"]["sigterm_sent"] is True
    assert result["timeout_cleanup"]["process_exited"] is True


def test_timeout_kills_stubborn_descendant_after_session_leader_exits(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "descendant-ready"
    child_code = (
        "import signal,sys,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "open(sys.argv[1], 'w', encoding='utf-8').close()\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import os,subprocess,sys,time\n"
        f"marker = {str(marker)!r}\n"
        f"child_code = {child_code!r}\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', child_code, marker],\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        "deadline = time.time() + 5\n"
        "while not os.path.exists(marker) and time.time() < deadline:\n"
        "    time.sleep(0.01)\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )

    result = runner._run_command(
        [sys.executable, "-c", parent_code],
        tmp_path,
        os.environ,
        0.5,
    )

    assert result["status"] == "timeout"
    cleanup = result["timeout_cleanup"]
    child_pid = int(result["stdout"].strip().splitlines()[-1])
    assert cleanup["leader_exited_before_group_probe"] is True
    assert child_pid in cleanup["live_members_after_sigterm"]
    assert cleanup["sigkill_sent"] is True
    assert cleanup["process_group_terminated"] is True
    assert cleanup["live_members_after_cleanup"] == []
    time.sleep(0.05)
    assert child_pid not in runner._live_process_group_members(cleanup["process_group"])


def test_strict_schema_rejects_configured_tasklets_as_observed(tmp_path: Path) -> None:
    payload = runner._base_payload(runner.qualification_plan(tmp_path), "prepared")
    payload["observed_tasklets_per_dpu"] = 12
    with pytest.raises(ValueError, match="configured, not independently observed"):
        runner._validate_output_schema(payload)


def test_host_main_has_central_cleanup_and_single_return() -> None:
    host = (RUNNER_PATH.parent / "qualification/host.c").read_text(encoding="utf-8")
    attribution = (RUNNER_PATH.parent / "qualification/ATTRIBUTION.md").read_text(
        encoding="utf-8"
    )
    main = host[host.index("int main(") :]

    assert "cleanup:" in main
    assert main.count("return ") == 1
    assert "dpu_free(management->set)" in main
    assert "table_management_init(" not in main
    assert 'PHYSICAL_ALLOCATION_PROFILE "backend=hw"' in host
    assert "DPU_ASSERT" in attribution
    assert "release unconfirmed" in attribution
    assert "does not replace or reimplement" in attribution


def test_staged_physical_sdk_build_when_compilers_are_available(
    tmp_path: Path,
) -> None:
    required = (
        "make",
        "gcc",
        "dpu-upmem-dpurte-clang",
        "dpu-pkg-config",
    )
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        pytest.skip(f"UPMEM SDK build tools unavailable: {', '.join(missing)}")
    plan = runner.qualification_plan(tmp_path)
    evidence = runner._stage_sources(plan)

    result = runner._run_command(
        plan["commands"]["build"],
        plan["staged_benchmark"],
        runner._physical_environment(os.environ),
        30.0,
    )

    assert evidence["applied"] is True
    assert result["status"] == "passed", runner._command_summary(
        result,
        plan["commands"]["build"],
    )
    assert set(runner._binary_hashes(plan["binary_dir"])) == {
        "host",
        "dpu_init_binary",
        "dpu_zip",
        "dpu_map_va_funcs",
    }
