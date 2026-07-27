from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

import quantum_bench.bench.provider_qualification as harness
from quantum_bench.bench.provider_qualification import (
    execute_provider_qualification,
    prepare_provider_qualification,
)
from quantum_bench.providers.qualification import (
    SIMPLEPIM_PROBE_ID,
    SIMPLEPIM_RUNNER_SCHEMA_VERSION,
    load_provider_catalog,
    qualification_repository_fingerprint,
    require_qualification_repository_source,
    simplepim_runner_schema_errors,
)


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "configs/qualification/upmem_provider_m1.yml"
RUNNER_PATH = ROOT / "native/upmem/simplepim/simplepim_qualification_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "provider_qualification_contract_runner",
    RUNNER_PATH,
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
native_runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(native_runner)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tool_identity(command: str, *, search_path: str | None = None) -> dict:
    raw_path = shutil.which(command, path=search_path)
    path = Path(raw_path).resolve() if raw_path else None
    return {
        "command": command,
        "available": path is not None,
        "path": str(path) if path is not None else None,
        "sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None
        ),
    }


def _with_workspace_dpu(payload: dict, root: Path) -> dict:
    result = copy.deepcopy(payload)
    result["effective_compilers"]["dpu_cc"] = _tool_identity(
        "dpu-upmem-dpurte-clang",
        search_path=str(root / "tools"),
    )
    assert result["effective_compilers"]["dpu_cc"]["available"] is True
    return result


def _commands() -> dict:
    host_cc = _tool_identity("gcc")["path"]
    assert host_cc is not None
    return {
        "patch": ["patch", "--batch", "--input=qualification.patch"],
        "build": [
            "make",
            "-f",
            "qualification/Makefile",
            "clean",
            "all",
            f"HOST_CC={host_cc}",
            "DPU_CC=dpu-upmem-dpurte-clang",
        ],
        "run": ["qualification/bin/host", "a.bin", "b.bin", "result.bin"],
    }


def _source_hashes(*, staged: bool) -> dict:
    combined = _digest("staged-after") if staged else _digest("combined")
    return {
        "combined_sha256": combined,
        "owned_qualification_sha256": _digest("owned"),
        "upstream_library_sha256": (
            _digest("staged-library") if staged else _digest("upstream")
        ),
        "upstream_map_processing_sha256": (
            _digest("staged-map-processing") if staged else _digest("map-processing")
        ),
        "patch_sha256": _digest("patch"),
        "staged_source_before_patch_sha256": (
            _digest("staged-before") if staged else None
        ),
        "staged_source_after_patch_sha256": (
            _digest("staged-after") if staged else None
        ),
        "staged_patch_sha256": _digest("patch") if staged else None,
        "upstream_submodule": "/qualification/source/SimplePIM",
    }


def _staged_patch(*, applied: bool) -> dict:
    return {
        "path": "patches/simplepim-map-unroll-rest.patch",
        "sha256": _digest("patch"),
        "staged_sha256": _digest("patch") if applied else None,
        "applied": applied,
        "replacement_count": 2 if applied else 0,
        "command_fingerprint": _json_digest(_commands()["patch"]),
        "staged_source_before_sha256": (_digest("staged-before") if applied else None),
        "staged_source_after_sha256": (_digest("staged-after") if applied else None),
        "staged_target_sha256": (_digest("staged-map-processing") if applied else None),
    }


def _device_evidence() -> list[dict]:
    return [
        {
            "path": "/dev/dpu_rank0",
            "exists": True,
            "character_device": True,
            "readable": True,
            "writable": True,
            "sysfs_path": "/sys/class/dpu_rank/dpu_rank0",
            "sysfs_exists": True,
        }
    ]


def _host_result() -> dict:
    return {
        "schema_version": "simplepim_qualification_host_v2",
        "provider_id": "simplepim",
        "probe_id": SIMPLEPIM_PROBE_ID,
        "status": "passed",
        "backend_profile": "backend=hw",
        "requested_dpu_count": 1,
        "observed_dpu_count": 1,
        "configured_tasklets_per_dpu": 12,
        "observed_tasklets_per_dpu": None,
        "native_run_completed": True,
        "validation_performed": True,
        "host_exact_validation": True,
        "fallback": False,
        "release_status": "released",
        "logical_input_bytes": 2048,
        "logical_output_bytes": 1024,
        "physical_transfer_bytes_available": False,
        "physical_transfer_bytes": None,
        "timing": {"input_s": 0.001, "kernel_s": 0.002, "output_s": 0.001},
        "failure_stage": None,
        "reason": None,
    }


def _base_payload(*, status: str, staged: bool) -> dict:
    source_hashes = _source_hashes(staged=staged)
    return {
        "schema_version": SIMPLEPIM_RUNNER_SCHEMA_VERSION,
        "provider_id": "simplepim",
        "probe_id": SIMPLEPIM_PROBE_ID,
        "status": status,
        "target": None,
        "target_observed": None,
        "requested_dpu_count": 1,
        "observed_dpu_count": None,
        "configured_tasklets_per_dpu": 12,
        "observed_tasklets_per_dpu": None,
        "hardware_preflight_verified": False,
        "device_evidence": [],
        "native_execution": False,
        "validation_performed": False,
        "exact_validation": False,
        "fallback": False,
        "simulator_kernel_executed": False,
        "release_status": "not_attempted",
        "backend_profile": "backend=hw",
        "source_hash": source_hashes["combined_sha256"],
        "source_hashes": source_hashes,
        "command_fingerprint": _digest("runner-command"),
        "effective_compilers": {
            "host_cc": _tool_identity("gcc"),
            "dpu_cc": _tool_identity("dpu-upmem-dpurte-clang"),
        },
        "staged_patch": _staged_patch(applied=staged),
        "binary_hashes": {},
        "input_hashes": {},
        "output_hash": None,
        "logical_transfer_bytes": {
            "h2d": 2048,
            "d2h": 1024,
            "total": 3072,
            "scope": "logical_application_payload_only",
        },
        "payload_sizes_8_byte_aligned": True,
        "physical_transfer_bytes_available": False,
        "physical_transfer_bytes": None,
        "timing": {},
        "failure_stage": None,
        "reason": "not_run",
    }


def _run_git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _source_checkout(root: Path) -> tuple[Path, str]:
    source = root / "source"
    source.mkdir(parents=True)
    _run_git(source, "init", "-q")
    (source / "source.txt").write_text("pinned source\n", encoding="utf-8")
    _run_git(source, "add", "source.txt")
    _run_git(
        source,
        "-c",
        "user.name=Qualification Test",
        "-c",
        "user.email=qualification@example.invalid",
        "commit",
        "-qm",
        "pinned source",
    )
    return source, _run_git(source, "rev-parse", "HEAD")


def _prepare_payload(**updates: object) -> dict:
    payload = _base_payload(status="prepared", staged=False)
    payload["commands"] = _commands()
    payload["command_fingerprint"] = _json_digest(payload["commands"])
    payload["reason"] = "prepare_only_no_compiler_or_hardware_invoked"
    payload.update(updates)
    return payload


def _command_evidence(command: list[str]) -> dict:
    return {
        "command": command,
        "command_fingerprint": _json_digest(command),
        "status": "passed",
        "returncode": 0,
        "wall_s": 0.01,
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _success_payload(**updates: object) -> dict:
    payload = _base_payload(status="passed", staged=True)
    commands = _commands()
    payload.update(
        {
            "hardware_preflight_verified": True,
            "target": "physical_hardware",
            "target_observed": "physical_hardware",
            "observed_dpu_count": 1,
            "device_evidence": _device_evidence(),
            "native_execution": True,
            "validation_performed": True,
            "exact_validation": True,
            "release_status": "released",
            "commands": commands,
            "command_fingerprint": _json_digest(commands),
            "build": _command_evidence(commands["build"]),
            "execution": _command_evidence(commands["run"]),
            "binary_hashes": {
                "host": _digest("binary-host"),
                "dpu_init_binary": _digest("binary-init"),
                "dpu_zip": _digest("binary-zip"),
                "dpu_map_va_funcs": _digest("binary-map"),
            },
            "input_hashes": {
                "a_u32": _digest("input-a"),
                "b_u32": _digest("input-b"),
            },
            "output_hash": _digest("output"),
            "timing": {"build_s": 0.01, "host_wall_s": 0.02},
            "host_result": _host_result(),
            "reason": None,
        }
    )
    payload.update(updates)
    return payload


def _failed_payload(**updates: object) -> dict:
    payload = _base_payload(status="failed", staged=True)
    payload.update(
        {
            "hardware_preflight_verified": True,
            "device_evidence": _device_evidence(),
            "release_status": "unknown",
            "binary_hashes": {
                "host": _digest("binary-host"),
                "dpu_init_binary": _digest("binary-init"),
                "dpu_zip": _digest("binary-zip"),
                "dpu_map_va_funcs": _digest("binary-map"),
            },
            "input_hashes": {"a_u32": _digest("input-a"), "b_u32": None},
            "timing": {"build_s": 0.01, "host_wall_s": 0.02},
            "failure_stage": "host_process",
            "reason": "upstream_native_process_failed_without_host_result_"
            "release_unconfirmed",
        }
    )
    payload.update(updates)
    return payload


def _fake_runner(
    root: Path,
    *,
    prepare_payload: dict | None = None,
    execution_payload: dict | None = None,
    execution_returncode: int = 0,
    write_execution_result: bool = True,
    print_execution_payload: bool = False,
    fail_on_stale_output: bool = False,
    marker: Path | None = None,
    environment_output: Path | None = None,
    timeout_child_marker: Path | None = None,
    detach_timeout_child: bool = False,
    bind_execution_dpu_identity: bool = True,
) -> Path:
    runner = root / "fake_runner.py"
    prepare_payload = prepare_payload or _prepare_payload()
    execution_payload = execution_payload or _success_payload()
    runner.write_text(
        "import argparse, hashlib, json, os, shutil, subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"PREPARE = {prepare_payload!r}\n"
        f"EXECUTION = {execution_payload!r}\n"
        f"EXECUTION_RETURNCODE = {execution_returncode}\n"
        f"WRITE_EXECUTION = {write_execution_result!r}\n"
        f"PRINT_EXECUTION = {print_execution_payload!r}\n"
        f"FAIL_ON_STALE = {fail_on_stale_output!r}\n"
        f"MARKER = {(str(marker) if marker is not None else None)!r}\n"
        "ENVIRONMENT_OUTPUT = "
        f"{(str(environment_output) if environment_output is not None else None)!r}\n"
        "TIMEOUT_CHILD_MARKER = "
        f"{(str(timeout_child_marker) if timeout_child_marker is not None else None)!r}\n"
        f"DETACH_TIMEOUT_CHILD = {detach_timeout_child!r}\n"
        f"BIND_EXECUTION_DPU = {bind_execution_dpu_identity!r}\n"
        "def bind_dpu_identity(payload):\n"
        "    raw = shutil.which('dpu-upmem-dpurte-clang')\n"
        "    path = Path(raw).resolve() if raw else None\n"
        "    payload['effective_compilers']['dpu_cc'] = {\n"
        "        'command': 'dpu-upmem-dpurte-clang',\n"
        "        'available': path is not None,\n"
        "        'path': str(path) if path is not None else None,\n"
        "        'sha256': hashlib.sha256(path.read_bytes()).hexdigest() "
        "if path is not None else None,\n"
        "    }\n"
        "parser = argparse.ArgumentParser()\n"
        "mode = parser.add_mutually_exclusive_group(required=True)\n"
        "mode.add_argument('--prepare-only', action='store_true')\n"
        "mode.add_argument('--execute', action='store_true')\n"
        "parser.add_argument('--workdir', required=True)\n"
        "parser.add_argument('--json-output', required=True)\n"
        "args = parser.parse_args()\n"
        "output = Path(args.json_output)\n"
        "if MARKER:\n"
        "    Path(MARKER).write_text('invoked\\n', encoding='utf-8')\n"
        "if ENVIRONMENT_OUTPUT:\n"
        "    Path(ENVIRONMENT_OUTPUT).write_text("
        "os.environ.get('HOST_CC', '') + '\\n', encoding='utf-8')\n"
        "if args.prepare_only:\n"
        "    bind_dpu_identity(PREPARE)\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text(json.dumps(PREPARE, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "    raise SystemExit(0)\n"
        "if FAIL_ON_STALE and output.exists():\n"
        "    raise SystemExit(91)\n"
        "if TIMEOUT_CHILD_MARKER:\n"
        "    child_code = ("
        '"import signal,sys,time;from pathlib import Path;"'
        '"marker=Path(sys.argv[1]);"'
        '"signal.signal(signal.SIGTERM,lambda *_:(marker.write_text('
        "'terminated\\\\n',encoding='utf-8'),sys.exit(0)));\""
        "\"marker.with_suffix('.started').write_text('started\\\\n',encoding='utf-8');\""
        '"time.sleep(60)"'
        ")\n"
        "    child = subprocess.Popen([sys.executable, '-c', child_code, "
        "TIMEOUT_CHILD_MARKER], start_new_session=DETACH_TIMEOUT_CHILD)\n"
        "    Path(TIMEOUT_CHILD_MARKER + '.pid').write_text("
        "str(child.pid), encoding='utf-8')\n"
        "    time.sleep(60)\n"
        "if BIND_EXECUTION_DPU:\n"
        "    bind_dpu_identity(EXECUTION)\n"
        "if WRITE_EXECUTION:\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text(json.dumps(EXECUTION, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "if PRINT_EXECUTION:\n"
        "    print(json.dumps(EXECUTION, sort_keys=True))\n"
        "raise SystemExit(EXECUTION_RETURNCODE)\n",
        encoding="utf-8",
    )
    return runner


def _catalog(
    root: Path,
    runner: Path,
    source: Path,
    pinned_commit: str,
) -> Path:
    catalog = root / "catalog.yml"
    catalog.write_text(
        "\n".join(
            [
                "schema_version: provider_qualification_catalog_v1",
                "catalog_id: test_catalog",
                "providers:",
                "  - id: simplepim",
                "    name: SimplePIM test",
                "    status: executable",
                "    executable: true",
                f"    runner: {runner.relative_to(root).as_posix()}",
                "    runner_contract:",
                f"      schema_version: {SIMPLEPIM_RUNNER_SCHEMA_VERSION}",
                f"      probe_id: {SIMPLEPIM_PROBE_ID}",
                "    source:",
                f"      path: {source.relative_to(root).as_posix()}",
                f"      pinned_commit: '{pinned_commit}'",
                "    hardware:",
                "      requested_dpus: 1",
                "      requested_tasklets: 12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return catalog


def _workspace(
    tmp_path: Path,
    *,
    prepare_payload: dict | None = None,
    execution_payload: dict | None = None,
    execution_returncode: int = 0,
    write_execution_result: bool = True,
    print_execution_payload: bool = False,
    fail_on_stale_output: bool = False,
    pinned_commit: str | None = None,
    marker: Path | None = None,
    environment_output: Path | None = None,
    timeout_child_marker: Path | None = None,
    detach_timeout_child: bool = False,
    bind_execution_dpu_identity: bool = True,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "workspace"
    root.mkdir()
    source, actual_commit = _source_checkout(root)
    tool_dir = root / "tools"
    tool_dir.mkdir()
    dpu_compiler = tool_dir / "dpu-upmem-dpurte-clang"
    dpu_compiler.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dpu_compiler.chmod(0o755)
    prepared = prepare_payload or _prepare_payload()
    executed = execution_payload or _success_payload()
    if bind_execution_dpu_identity:
        executed["effective_compilers"]["dpu_cc"] = _tool_identity(
            "dpu-upmem-dpurte-clang",
            search_path=str(tool_dir),
        )
    runner = _fake_runner(
        root,
        prepare_payload=prepared,
        execution_payload=executed,
        execution_returncode=execution_returncode,
        write_execution_result=write_execution_result,
        print_execution_payload=print_execution_payload,
        fail_on_stale_output=fail_on_stale_output,
        marker=marker,
        environment_output=environment_output,
        timeout_child_marker=timeout_child_marker,
        detach_timeout_child=detach_timeout_child,
        bind_execution_dpu_identity=bind_execution_dpu_identity,
    )
    catalog = _catalog(root, runner, source, pinned_commit or actual_commit)
    return root, catalog, source


def _execute(root: Path, catalog: Path, **environment: str):
    search_path = os.pathsep.join((str(root / "tools"), os.environ.get("PATH", "")))
    return harness._execute_provider_qualification_for_test(
        root,
        catalog_path=catalog,
        provider_id="simplepim",
        environment={
            "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
            "PATH": search_path,
            **environment,
        },
        hook=harness._TEST_EXECUTION_HOOK,
    )


def test_prepare_uses_unique_dirs_and_preserves_raw_preflight(tmp_path: Path) -> None:
    root, catalog, _ = _workspace(tmp_path)

    first = prepare_provider_qualification(root, catalog_path=catalog)
    second = prepare_provider_qualification(root, catalog_path=catalog)

    assert first.status == second.status == "prepared"
    assert first.plan_dir.is_absolute()
    assert second.plan_dir.is_absolute()
    assert first.plan_dir != second.plan_dir
    first_plan = json.loads(first.plan_path.read_text(encoding="utf-8"))
    policy = first_plan["execution_policy"]
    assert policy["build_attempted"] is False
    assert policy["dpu_allocation_attempted"] is False
    assert policy["dpu_launch_attempted"] is False
    row = first_plan["providers"][0]
    assert row["source_fingerprint"]["commit_matches_pin"] is True
    assert row["source_fingerprint"]["clean"] is True
    assert first_plan["catalog_fingerprint"]["catalog_sha256"]
    assert first_plan["tool_fingerprints"]["python"]["sha256"]
    assert first_plan["tool_fingerprints"]["host_cc"]["configured"] == "gcc"
    assert first_plan["runner_environment"] == {"HOST_CC": "gcc"}
    assert row["runner_prepare"]["environment"] == {"HOST_CC": "gcc"}
    raw_prepare = first.plan_dir / "providers/simplepim/raw_runner_prepare.json"
    assert json.loads(raw_prepare.read_text(encoding="utf-8")) == _prepare_payload()


def test_harness_and_native_runner_share_the_same_payload_contract(
    tmp_path: Path,
) -> None:
    root, catalog_path, _ = _workspace(tmp_path)
    provider = load_provider_catalog(catalog_path).get("simplepim")
    success = _with_workspace_dpu(_success_payload(), root)
    failed = _with_workspace_dpu(_failed_payload(), root)
    cases: list[tuple[str, dict, bool]] = [
        ("prepare", _prepare_payload(), True),
        ("execute", success, True),
        ("execute", failed, True),
    ]
    invalid_payloads = [
        ("execute", {**copy.deepcopy(success), "unknown": True}),
        (
            "execute",
            {**copy.deepcopy(success), "source_hash": "not-a-sha256"},
        ),
        (
            "execute",
            {**copy.deepcopy(success), "observed_tasklets_per_dpu": 12},
        ),
    ]
    for mode, payload in invalid_payloads:
        cases.append((mode, payload, False))
    for mode, payload, expected_valid in cases:
        native_valid = True
        try:
            native_runner._validate_output_schema(copy.deepcopy(payload))
        except ValueError:
            native_valid = False
        harness_valid = not simplepim_runner_schema_errors(
            copy.deepcopy(payload),
            provider,
            mode=mode,
        )
        assert native_valid is expected_valid
        assert harness_valid is expected_valid


def test_prepare_ingestion_rejects_completed_or_unknown_evidence(
    tmp_path: Path,
) -> None:
    payload = _prepare_payload(observed_dpu_count=1)
    payload["staged_patch"]["unexpected"] = _digest("unexpected")
    root, catalog, _ = _workspace(tmp_path, prepare_payload=payload)

    result = prepare_provider_qualification(root, catalog_path=catalog)

    assert result.status == "failed"
    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))
    errors = plan["providers"][0]["runner_prepare"]["runner_contract_errors"]
    assert any("unknown fields" in error for error in errors)
    assert any("prepared observed_dpu_count" in error for error in errors)


def test_relative_root_and_catalog_paths_resolve_at_api_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = _workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    plan = prepare_provider_qualification(
        Path("workspace"), catalog_path=Path("catalog.yml")
    )

    assert plan.plan_path.is_absolute()
    assert plan.plan_path.is_relative_to(root.resolve())


def test_commit_mismatch_blocks_prepare_and_execute_without_runner(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "runner-invoked"
    root, catalog, _ = _workspace(
        tmp_path,
        pinned_commit="0" * 40,
        marker=marker,
    )

    plan = prepare_provider_qualification(root, catalog_path=catalog)

    assert plan.status == "failed"
    payload = json.loads(plan.plan_path.read_text(encoding="utf-8"))
    row = payload["providers"][0]
    assert row["preparation_status"] == "failed"
    assert "does not match pinned commit" in row["preparation_reason"]
    assert row["runner_prepare"] is None
    assert not marker.exists()
    with pytest.raises(ValueError, match="does not match pinned commit"):
        _execute(root, catalog)
    assert not (root / "runs").exists()
    assert not marker.exists()


def test_dirty_source_blocks_execution(tmp_path: Path) -> None:
    root, catalog, source = _workspace(tmp_path)
    (source / "source.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="worktree is not clean"):
        _execute(root, catalog)

    assert not (root / "runs").exists()


def test_execute_requires_opt_in(tmp_path: Path) -> None:
    root, catalog, _ = _workspace(tmp_path)

    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE=1"):
        harness._execute_provider_qualification_for_test(
            root,
            catalog_path=catalog,
            provider_id="simplepim",
            environment={},
            hook=harness._TEST_EXECUTION_HOOK,
        )
    assert not (root / "runs").exists()


def test_public_execute_rejects_custom_catalog_before_runner(tmp_path: Path) -> None:
    marker = tmp_path / "runner-invoked"
    root, catalog, _ = _workspace(tmp_path, marker=marker)

    with pytest.raises(ValueError, match="canonical qualification root"):
        execute_provider_qualification(
            root,
            catalog_path=catalog,
            provider_id="simplepim",
            environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
        )

    assert not marker.exists()
    assert not (root / "runs").exists()


def test_public_execute_treats_noncanonical_catalog_at_root_as_prepare_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "runner-invoked"
    root, catalog, _ = _workspace(tmp_path, marker=marker)
    monkeypatch.setattr(harness, "CANONICAL_ROOT", root)
    monkeypatch.setattr(harness, "CANONICAL_CATALOG_PATH", "canonical.yml")

    with pytest.raises(ValueError, match="custom provider catalogs are prepare-only"):
        execute_provider_qualification(
            root,
            catalog_path=catalog,
            provider_id="simplepim",
            environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
        )

    assert not marker.exists()
    assert not (root / "runs").exists()


def test_internal_execute_requires_unforgeable_test_hook(tmp_path: Path) -> None:
    root, catalog, _ = _workspace(tmp_path)

    with pytest.raises(ValueError, match="invalid internal"):
        harness._execute_provider_qualification_for_test(
            root,
            catalog_path=catalog,
            provider_id="simplepim",
            environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
            hook=object(),
        )


def test_qualification_repository_gate_requires_tracked_clean_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "qualification-repository"
    source = root / "qualification"
    source.mkdir(parents=True)
    tracked = source / "contract.py"
    tracked.write_text("CONTRACT = 1\n", encoding="utf-8")
    _run_git(root, "init", "-q")
    _run_git(root, "add", "qualification/contract.py")
    _run_git(
        root,
        "-c",
        "user.name=Qualification Test",
        "-c",
        "user.email=qualification@example.invalid",
        "commit",
        "-qm",
        "qualification source",
    )

    fingerprint = require_qualification_repository_source(
        root,
        ("qualification",),
    )

    assert fingerprint["all_files_tracked"] is True
    assert fingerprint["clean"] is True
    assert fingerprint["untracked_files"] == []

    tracked.write_text("CONTRACT = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be clean"):
        require_qualification_repository_source(root, ("qualification",))
    tracked.write_text("CONTRACT = 1\n", encoding="utf-8")
    (source / "untracked.py").write_text("UNTRACKED = True\n", encoding="utf-8")
    untracked = qualification_repository_fingerprint(root, ("qualification",))
    assert untracked["all_files_tracked"] is False
    with pytest.raises(ValueError, match="fully tracked"):
        require_qualification_repository_source(root, ("qualification",))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DPU_BACKEND", "simulator"),
        ("UPMEM_MODE", "simulation"),
        ("UPMEM_TARGET", "fsim"),
        ("UPMEM_PROFILE", "backend=casim"),
        ("UPMEM_PROFILE_BASE", "/profiles/functional_simulator"),
    ],
)
def test_all_simulator_selector_aliases_are_rejected(
    tmp_path: Path, key: str, value: str
) -> None:
    root, catalog, _ = _workspace(tmp_path)

    with pytest.raises(ValueError, match=key):
        _execute(root, catalog, **{key: value})

    assert not (root / "runs").exists()


def test_missing_designated_result_does_not_use_stdout_json(tmp_path: Path) -> None:
    root, catalog, _ = _workspace(
        tmp_path,
        write_execution_result=False,
        print_execution_payload=True,
    )

    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert not result.raw_result_path.exists()
    assert summary["raw_runner_result"] == {"path": None, "sha256": None}
    assert "did not create designated JSON result" in summary["failure_reason"]
    assert summary["observed_counts"] == {"dpus": None, "tasklets": None}
    assert summary["resource_release_status"] == "unconfirmed"


@pytest.mark.parametrize(
    "mutation",
    [
        "uppercase_hash",
        "unknown_binary",
        "missing_source_field",
        "unknown_patch_field",
        "incomplete_device",
        "wrong_logical_total",
        "unknown_host_field",
        "host_provenance_mismatch",
    ],
)
def test_nested_schema_mutations_never_qualify(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _success_payload()
    if mutation == "uppercase_hash":
        payload["output_hash"] = str(payload["output_hash"]).upper()
    elif mutation == "unknown_binary":
        payload["binary_hashes"]["other"] = _digest("other")
    elif mutation == "missing_source_field":
        del payload["source_hashes"]["patch_sha256"]
    elif mutation == "unknown_patch_field":
        payload["staged_patch"]["note"] = "not evidence"
    elif mutation == "incomplete_device":
        del payload["device_evidence"][0]["writable"]
    elif mutation == "wrong_logical_total":
        payload["logical_transfer_bytes"]["total"] = 2048
    elif mutation == "unknown_host_field":
        payload["host_result"]["claim"] = True
    elif mutation == "host_provenance_mismatch":
        payload["effective_compilers"]["host_cc"]["sha256"] = _digest(
            "different-host-compiler"
        )

    root, catalog, _ = _workspace(tmp_path, execution_payload=payload)
    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert summary["runner_contract_errors"]
    assert summary["resource_release_status"] == "unconfirmed"


def test_effective_dpu_compiler_must_match_outer_fingerprint(
    tmp_path: Path,
) -> None:
    root, catalog, _ = _workspace(
        tmp_path,
        bind_execution_dpu_identity=False,
    )

    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert any(
        "effective_compilers.dpu_cc" in error and "outer runner provenance" in error
        for error in summary["runner_contract_errors"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "owned_source",
        "patch_hash",
        "planned_command",
    ],
)
def test_execute_payload_must_match_designated_preflight(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _success_payload()
    if mutation == "owned_source":
        payload["source_hashes"]["owned_qualification_sha256"] = _digest(
            "other-owned-source"
        )
    elif mutation == "patch_hash":
        replacement = _digest("other-patch")
        payload["source_hashes"]["patch_sha256"] = replacement
        payload["source_hashes"]["staged_patch_sha256"] = replacement
        payload["staged_patch"]["sha256"] = replacement
        payload["staged_patch"]["staged_sha256"] = replacement
    elif mutation == "planned_command":
        payload["commands"]["run"][-1] = "other-result.bin"
        payload["command_fingerprint"] = _json_digest(payload["commands"])
        payload["execution"] = _command_evidence(payload["commands"]["run"])

    root, catalog, _ = _workspace(tmp_path, execution_payload=payload)
    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert any(
        "canonical preflight" in error for error in summary["runner_contract_errors"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "staged_before",
        "staged_after",
        "staged_patch",
        "staged_target",
    ],
)
def test_all_staged_source_hash_aliases_are_bound(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _success_payload()
    if mutation == "staged_before":
        payload["source_hashes"]["staged_source_before_patch_sha256"] = _digest(
            "other-before"
        )
    elif mutation == "staged_after":
        payload["source_hashes"]["staged_source_after_patch_sha256"] = _digest(
            "other-after"
        )
    elif mutation == "staged_patch":
        payload["source_hashes"]["staged_patch_sha256"] = _digest("other-staged-patch")
    elif mutation == "staged_target":
        payload["source_hashes"]["upstream_map_processing_sha256"] = _digest(
            "other-staged-target"
        )

    root, catalog, _ = _workspace(tmp_path, execution_payload=payload)
    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert summary["runner_contract_errors"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_build",
        "failed_build",
        "missing_execution",
        "wrong_execution_command",
        "nested_host_failed",
        "nested_native_false",
        "nested_validation_false",
        "nested_exact_validation_false",
        "nested_release_failed",
        "nested_failure_reason",
        "top_failure_stage",
        "top_reason",
    ],
)
def test_top_level_pass_requires_build_execution_and_nested_host_pass(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _success_payload()
    if mutation == "missing_build":
        del payload["build"]
    elif mutation == "failed_build":
        payload["build"]["status"] = "failed"
        payload["build"]["returncode"] = 1
    elif mutation == "missing_execution":
        del payload["execution"]
    elif mutation == "wrong_execution_command":
        payload["execution"]["command"][-1] = "other-result.bin"
        payload["execution"]["command_fingerprint"] = _json_digest(
            payload["execution"]["command"]
        )
    elif mutation == "nested_host_failed":
        payload["host_result"]["status"] = "failed"
    elif mutation == "nested_native_false":
        payload["host_result"]["native_run_completed"] = False
    elif mutation == "nested_validation_false":
        payload["host_result"]["validation_performed"] = False
    elif mutation == "nested_exact_validation_false":
        payload["host_result"]["host_exact_validation"] = False
    elif mutation == "nested_release_failed":
        payload["host_result"]["release_status"] = "failed"
    elif mutation == "nested_failure_reason":
        payload["host_result"]["failure_stage"] = "validation"
        payload["host_result"]["reason"] = "failed"
    elif mutation == "top_failure_stage":
        payload["failure_stage"] = "validation"
    elif mutation == "top_reason":
        payload["reason"] = "passed"

    root, catalog, _ = _workspace(tmp_path, execution_payload=payload)
    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert summary["runner_contract_errors"]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema_version", "wrong_schema"),
        ("provider_id", "not_simplepim"),
        ("probe_id", "wrong_probe"),
        ("hardware_preflight_verified", 1),
        ("target", "simulator"),
        ("observed_dpu_count", True),
        ("configured_tasklets_per_dpu", 11),
        ("native_execution", False),
        ("validation_performed", False),
        ("exact_validation", False),
        ("release_status", "failed"),
        ("fallback", True),
        ("simulator_kernel_executed", True),
    ],
)
def test_adversarial_runner_contract_never_qualifies(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    root, catalog, _ = _workspace(
        tmp_path,
        execution_payload=_success_payload(**{field: bad_value}),
    )

    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert summary["runner_contract_errors"]
    assert result.raw_result_path.is_file()


def test_observed_tasklets_may_be_null_but_configured_count_must_match(
    tmp_path: Path,
) -> None:
    root, catalog, _ = _workspace(tmp_path)

    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "test_passed_non_evidence"
    assert summary["configured_tasklets_per_dpu"] == 12
    assert summary["observed_counts"] == {"dpus": 1, "tasklets": None}


def test_raw_result_and_provenance_are_preserved_with_manifest(
    tmp_path: Path,
) -> None:
    payload = _success_payload()
    root, catalog, _ = _workspace(tmp_path, execution_payload=payload)

    result = _execute(root, catalog)

    raw_bytes = result.raw_result_path.read_bytes()
    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    assert all(
        path.is_absolute()
        for path in (
            result.run_dir,
            result.result_path,
            result.raw_result_path,
            result.normalized_records_path,
            result.manifest_path,
        )
    )
    assert json.loads(raw_bytes) == payload
    assert summary["raw_runner_result"] == {
        "path": "raw_runner_result.json",
        "sha256": raw_hash,
    }
    raw_preflight = result.run_dir / "raw_runner_preflight.json"
    preflight_hash = hashlib.sha256(raw_preflight.read_bytes()).hexdigest()
    assert summary["raw_runner_preflight"]["path"] == raw_preflight.name
    assert summary["raw_runner_preflight"]["sha256"] == preflight_hash
    assert summary["raw_runner_preflight"]["status"] == "passed"
    assert summary["source_fingerprint"]["commit_matches_pin"] is True
    assert summary["source_fingerprint"]["clean"] is True
    assert summary["tool_fingerprints"]["python"]["sha256"]
    assert summary["tool_fingerprints"]["host_cc"]["configured"] == "gcc"
    assert summary["tool_fingerprints"]["host_cc"]["sha256"]
    assert summary["runner"]["sha256"]
    assert summary["runner"]["environment"] == {"HOST_CC": "gcc"}
    assert summary["runner"]["recorded_command"].startswith("HOST_CC=gcc ")
    assert (
        payload["effective_compilers"]["host_cc"]["sha256"]
        == (summary["tool_fingerprints"]["host_cc"]["sha256"])
    )
    assert summary["resource_release_status"] == "test_only_unverified"
    assert summary["qualified"] is False
    assert summary["canonical_evidence"] is False
    assert summary["result_classification"] == "internal_test_non_evidence"
    assert manifest["schema_version"] == "run_manifest_v1"
    assert manifest["artifact_kind"] == "legacy_run"
    assert manifest["run_kind"] == "provider_qualification"
    assert manifest["artifact_retention"] == "compact"
    assert manifest["summary"] == "provider_qualification.json"
    assert manifest["normalized_records"] == "normalized_records.jsonl"
    assert manifest["raw_runner_result_sha256"] == raw_hash
    assert manifest["raw_runner_preflight"] == raw_preflight.name
    assert manifest["raw_runner_preflight_sha256"] == preflight_hash
    assert manifest["runner_preflight_status"] == "passed"
    assert manifest["runner_environment"] == {"HOST_CC": "gcc"}
    assert manifest["resource_release_status"] == "test_only_unverified"
    assert manifest["canonical_evidence"] is False
    assert manifest["source_fingerprint"] == summary["source_fingerprint"]
    assert manifest["tool_fingerprints"] == summary["tool_fingerprints"]


def test_configured_host_compiler_is_resolved_for_runner_and_provenance(
    tmp_path: Path,
) -> None:
    environment_output = tmp_path / "host-cc.txt"
    root, catalog, _ = _workspace(
        tmp_path,
        environment_output=environment_output,
    )

    result = _execute(root, catalog, HOST_CC="gcc")

    resolved_gcc = str(Path(shutil.which("gcc") or "").resolve())
    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "test_passed_non_evidence"
    assert environment_output.read_text(encoding="utf-8").strip() == resolved_gcc
    assert summary["runner"]["environment"] == {"HOST_CC": resolved_gcc}
    assert summary["tool_fingerprints"]["host_cc"]["path"] == resolved_gcc
    assert summary["runner"]["recorded_command"].startswith(f"HOST_CC={resolved_gcc} ")


def test_nonzero_runner_failure_preserves_designated_raw_json(tmp_path: Path) -> None:
    payload = _failed_payload()
    root, catalog, _ = _workspace(
        tmp_path,
        execution_payload=payload,
        execution_returncode=7,
    )

    result = _execute(root, catalog)

    assert result.status == "failed"
    assert json.loads(result.raw_result_path.read_text(encoding="utf-8")) == payload
    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert summary["runner"]["returncode"] == 7
    assert summary["failure_reason"] == "runner exited with return code 7"
    assert summary["resource_release_status"] == "unconfirmed"


def test_upstream_assert_style_failure_keeps_release_unconfirmed(
    tmp_path: Path,
) -> None:
    payload = _failed_payload(
        failure_stage="host_process",
        reason="upstream_native_process_failed_without_host_result_release_unconfirmed",
    )
    root, catalog, _ = _workspace(
        tmp_path,
        execution_payload=payload,
        execution_returncode=1,
    )

    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert summary["runner_status"] == "failed"
    assert summary["release_status"] == "unknown"
    assert summary["resource_release_status"] == "unconfirmed"
    assert json.loads(result.raw_result_path.read_text(encoding="utf-8")) == payload


def test_outer_runner_timeout_kills_process_group_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_marker = tmp_path / "runner-child"
    root, catalog, _ = _workspace(
        tmp_path,
        timeout_child_marker=child_marker,
        detach_timeout_child=True,
    )
    monkeypatch.setattr(harness, "EXECUTE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(harness, "PROCESS_GROUP_GRACE_SECONDS", 0.5)

    result = _execute(root, catalog)

    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    cleanup = summary["runner"]["timeout_cleanup"]
    assert result.status == "failed"
    assert not result.raw_result_path.exists()
    assert "runner timed out" in summary["failure_reason"]
    assert summary["resource_release_status"] == "unconfirmed"
    assert cleanup["sigterm_sent"] is True
    assert cleanup["group_probe_performed"] is True
    assert cleanup["live_members_after_cleanup"] == []
    assert cleanup["process_group_terminated"] is True
    child_pid = int(child_marker.with_suffix(".pid").read_text(encoding="utf-8"))
    assert child_pid in cleanup["descendant_pids_before_sigterm"]
    assert child_pid in cleanup["detached_descendant_pids_before_sigterm"]
    assert cleanup["descendant_pids_after_cleanup"] == []
    assert cleanup["descendant_tree_terminated"] is True
    assert not Path(f"/proc/{child_pid}").exists()
    deadline = time.monotonic() + 1.0
    while not child_marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_marker.read_text(encoding="utf-8") == "terminated\n"


def test_stale_designated_result_is_removed_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, catalog, _ = _workspace(tmp_path, fail_on_stale_output=True)
    fixed_run = root / "fixed-run"
    (fixed_run / "config").mkdir(parents=True)
    (fixed_run / "cases").mkdir()
    (fixed_run / "raw_runner_result.json").write_text(
        '{"status":"stale"}\n', encoding="utf-8"
    )
    monkeypatch.setattr(harness, "create_run_dir", lambda *args, **kwargs: fixed_run)

    result = _execute(root, catalog)

    assert result.status == "test_passed_non_evidence"
    raw = json.loads(result.raw_result_path.read_text(encoding="utf-8"))
    assert raw["status"] == "passed"
    assert raw["effective_compilers"]["dpu_cc"]["path"] == str(
        (root / "tools/dpu-upmem-dpurte-clang").resolve()
    )


def test_catalog_keeps_planned_providers_truthfully_blocked() -> None:
    catalog = load_provider_catalog(CATALOG)
    providers = {provider.provider_id: provider for provider in catalog.providers}

    assert providers["simplepim"].runner == (
        "native/upmem/simplepim/simplepim_qualification_runner.py"
    )
    assert providers["simplepim"].pinned_commit == (
        "1d639c53532555f01e9f71d872e7712b166d6cba"
    )
    assert "2021.3.0/1024-DPU/AVX512" in (providers["pid-comm"].reason or "")
    assert providers["pid-comm"].executable is False
    assert providers["atim"].reason == "Source is not pinned."
    assert providers["sparsep"].reason == "Source is not pinned."
