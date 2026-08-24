from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "native/upmem/pidcomm_qualification/pidcomm_qualification_runner.py"
SPEC = importlib.util.spec_from_file_location("pidcomm_qualification_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _fake_capture(command, *, env=None):
    del env
    if command == ["lscpu"]:
        return {"returncode": 0, "stdout": "Flags: avx512f avx2\n", "stderr": ""}
    if command[-2:] == ["rev-parse", "HEAD"]:
        return {"returncode": 0, "stdout": runner.PINNED_COMMIT + "\n", "stderr": ""}
    if command[-2:] == ["status", "--porcelain"]:
        return {"returncode": 0, "stdout": "", "stderr": ""}
    if command[-2:] == ["--variable=prefix", "dpu"]:
        return {"returncode": 0, "stdout": "/opt/upmem\n", "stderr": ""}
    return {"returncode": 0, "stdout": "fake version\n", "stderr": ""}


def _fake_environment() -> dict[str, str]:
    return {
        "PATH": "/fake/bin",
        runner.PHYSICAL_OPT_IN: "1",
    }


def _fake_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_capture", _fake_capture)
    monkeypatch.setattr(
        runner,
        "_resolve_tool",
        lambda name, environment: f"/fake/bin/{name}",
    )


def _valid_manifest(count: int) -> dict:
    topology = {
        2: ([2, 1, 1], "100"),
        4: ([2, 2, 1], "110"),
        64: ([8, 8, 1], "110"),
    }[count]
    return {
        "status": "passed",
        "dpu_count": count,
        "payload_bytes": 256,
        "payload_dtype": "int32",
        "operation": "sum_all_reduce",
        "topology": {
            "dimension": 3,
            "axis_lengths": topology[0],
            "communicator": topology[1],
        },
        "hardware_observed": True,
        "fallback": False,
        "pidcomm_api": "pidcomm_all_reduce",
    }


def test_prepare_records_contract_and_toolchain_without_staging_or_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _fake_tools(monkeypatch)
    monkeypatch.setattr(runner, "_version_file", lambda sdk_root: {"path": sdk_root, "text": "fake version"})
    assert runner.main(["--prepare-only", "--workdir", str(tmp_path), "--run-id", "plan-test"]) == 0
    payload = json.loads(capsys.readouterr().out)
    plan = json.loads((tmp_path / "build/pidcomm_qualification/plan-test/plan.json").read_text())

    assert payload["status"] == "prepared"
    assert plan["cpu"]["avx512f"] is True
    assert plan["toolchain"]["system_sdk_version"]["text"] == "fake version"
    assert plan["source"]["pinned_commit"] == runner.PINNED_COMMIT
    assert plan["source"]["input_hash"]
    assert plan["source"]["thesis_git"]["dirty"] is False
    assert plan["toolchain"]["linked_sdk_path"] == "/opt/upmem"
    assert plan["contract"] == {
        "candidate_dpu_counts": [2, 4, 64],
        "payload_bytes": 256,
        "payload_dtype": "int32",
        "operation": "sum_all_reduce",
    }
    assert plan["allocation_free"] is True
    assert not (tmp_path / "build/pidcomm_qualification/plan-test/staged").exists()


def test_execute_without_physical_opt_in_writes_only_preflight_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tools(monkeypatch)
    plan = runner.qualification_plan(tmp_path, run_id="optout-test", environment={})
    result = runner.execute(plan, {}, timeout_seconds=1)

    assert result["event"] == "pidcomm_sdk_compatibility_blocked"
    assert result["failure_stage"] == "preflight"
    assert result["fallback"] is False
    assert (tmp_path / "runs/pidcomm_qualification/optout-test/preflight.json").is_file()
    assert not (tmp_path / "build/pidcomm_qualification/optout-test/staged").exists()


def test_build_failure_is_structured_and_does_not_attempt_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tools(monkeypatch)
    plan = runner.qualification_plan(tmp_path, run_id="build-test", environment=_fake_environment())
    calls: list[list[str]] = []

    def fake_stage(_plan):
        return {"staged": True}

    def fake_process(command, *, cwd, env, timeout_seconds):
        del cwd, env, timeout_seconds
        calls.append(list(command))
        return {"returncode": 1, "stdout": "", "stderr": "missing dpu_alloc_comm", "timed_out": False}

    monkeypatch.setattr(runner, "_stage", fake_stage)
    monkeypatch.setattr(runner, "_run_process", fake_process)
    result = runner.execute(plan, _fake_environment(), timeout_seconds=1)

    assert result["event"] == "pidcomm_sdk_compatibility_blocked"
    assert result["failure_stage"] == "compatibility_preflight"
    assert result["candidates"] == []
    assert len(calls) == 1
    assert Path(tmp_path / "runs/pidcomm_qualification/build-test/logs/compatibility.log").is_file()


def test_smallest_passing_candidate_is_selected_from_fake_host_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tools(monkeypatch)
    plan = runner.qualification_plan(tmp_path, run_id="candidate-test", environment=_fake_environment())
    monkeypatch.setattr(runner, "_stage", lambda _plan: {"staged": True})
    calls: list[list[str]] = []

    def fake_process(command, *, cwd, env, timeout_seconds):
        del cwd, env, timeout_seconds
        calls.append(list(command))
        if command[-1] in {"compatibility", "all"}:
            return {"returncode": 0, "stdout": "build ok\n", "stderr": "", "timed_out": False}
        count = int(command[-1])
        if count == 2:
            return {"returncode": 1, "stdout": "", "stderr": "not enough hardware\n", "timed_out": False}
        return {
            "returncode": 0,
            "stdout": json.dumps(_valid_manifest(count)) + "\n",
            "stderr": "",
            "timed_out": False,
        }

    monkeypatch.setattr(runner, "_run_process", fake_process)
    result = runner.execute(plan, _fake_environment(), timeout_seconds=1)

    assert result["status"] == "qualified"
    assert result["selected_dpu_count"] == 4
    assert calls[0][-1] == "compatibility"
    assert calls[1][-1] == "all"
    assert [int(command[-1]) for command in calls[2:]] == [2, 4]
    assert result["fallback"] is False


def test_all_invalid_host_manifests_block_with_last_candidate_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tools(monkeypatch)
    plan = runner.qualification_plan(tmp_path, run_id="runtime-test", environment=_fake_environment())
    monkeypatch.setattr(runner, "_stage", lambda _plan: {"staged": True})

    def fake_process(command, *, cwd, env, timeout_seconds):
        del cwd, env, timeout_seconds
        if command[-1] in {"compatibility", "all"}:
            return {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False}
        return {"returncode": 0, "stdout": "not-json\n", "stderr": "", "timed_out": False}

    monkeypatch.setattr(runner, "_run_process", fake_process)
    result = runner.execute(plan, _fake_environment(), timeout_seconds=1)

    assert result["event"] == "pidcomm_sdk_compatibility_blocked"
    assert result["failure_stage"] == "runtime"
    assert result["log_path"].endswith("candidate-64.log")
    assert len(result["candidates"]) == 3


def test_staging_does_not_modify_external_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tools(monkeypatch)
    before = runner._sha256(runner.EXTERNAL_ROOT / "pidcomm_lib/support/commlib.c")
    plan = runner.qualification_plan(tmp_path, run_id="stage-test", environment=_fake_environment())
    runner._stage(plan)
    after = runner._sha256(runner.EXTERNAL_ROOT / "pidcomm_lib/support/commlib.c")

    assert before == after
    manifest = json.loads((Path(plan["stage_root"]) / "stage_manifest.json").read_text())
    assert manifest["bundled_sdk_staged"] is False
    assert manifest["input_hash"]
    assert manifest["staged_hashes"]
    assert (Path(plan["stage_root"]) / "include/pidcomm.h").is_file()
    assert (Path(plan["stage_root"]) / "pidcomm_commlib.c").is_file()


def test_pidcomm_source_and_build_contracts_are_isolated() -> None:
    host = (ROOT / "native/upmem/pidcomm_qualification/qualification/host.c").read_text()
    makefile = (ROOT / "native/upmem/pidcomm_qualification/qualification/Makefile").read_text()
    top_makefile = (ROOT / "Makefile").read_text()

    assert "dpu_alloc_comm" in host
    assert "pidcomm_all_reduce" in host
    assert "pidcomm_api" in host
    assert "PAYLOAD_BYTES 256u" in host
    assert "-Werror=implicit-function-declaration" in makefile
    assert "dpu-pkg-config" in makefile
    assert "LD_LIBRARY_PATH" not in makefile
    assert "pidcomm-check" in top_makefile
    assert "M4.6" not in str(ROOT / "native/upmem/pidcomm_qualification")
    assert "UPMEM_ALLOW_PHYSICAL_HARDWARE" in top_makefile


def test_run_id_must_be_a_safe_component(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tools(monkeypatch)
    with pytest.raises(ValueError, match="safe path component"):
        runner.qualification_plan(tmp_path, run_id="../escape", environment=_fake_environment())


def test_candidate_manifest_rejects_non_strict_contract_values(tmp_path: Path) -> None:
    manifest = _valid_manifest(2)
    manifest["payload_bytes"] = True
    manifest["hardware_observed"] = 1
    result = runner._candidate_result(
        2,
        {"returncode": 0, "stdout": json.dumps(manifest), "timed_out": False},
        tmp_path / "candidate.log",
        tmp_path,
    )

    assert result["host_manifest_valid"] is False
    assert any("payload_bytes" in error for error in result["validation_errors"])
    assert any("hardware_observed" in error for error in result["validation_errors"])


def test_compatibility_command_is_allocation_free_and_classifies_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tools(monkeypatch)
    plan = runner.qualification_plan(tmp_path, run_id="compatibility-test", environment=_fake_environment())
    monkeypatch.setattr(runner, "_stage", lambda _plan: {"staged": True})
    calls: list[list[str]] = []

    def fake_process(command, *, cwd, env, timeout_seconds):
        del cwd, env, timeout_seconds
        calls.append(list(command))
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": "undefined reference to dpu_alloc_comm",
            "timed_out": False,
        }

    monkeypatch.setattr(runner, "_run_process", fake_process)
    result = runner.compatibility_probe(plan, _fake_environment(), timeout_seconds=1)

    assert calls == [plan["commands"]["compatibility"]]
    assert calls[0][-1] == "compatibility"
    assert result["failure_stage"] == "compatibility_preflight"
    assert result["dpu_allocation_attempted"] is False
    assert result["dpu_launch_attempted"] is False


def test_fake_compiler_records_exact_staged_inputs_and_failure_is_preallocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tools(monkeypatch)
    plan = runner.qualification_plan(tmp_path, run_id="fake-compiler-test", environment=_fake_environment())
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    args_log = tmp_path / "compiler-args.txt"
    compiler = fake_bin / "cc"
    compiler.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > '{args_log}'\n"
        "printf '%s\\n' 'fake undefined reference to dpu_alloc_comm' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    compiler.chmod(0o755)
    pkg_config = fake_bin / "pkg-config"
    pkg_config.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--cflags\" ]; then printf '%s\\n' '-I/fake/sdk/include'; else printf '%s\\n' '-L/fake/sdk/lib -ldpu'; fi\n",
        encoding="utf-8",
    )
    pkg_config.chmod(0o755)
    plan["commands"]["compatibility"] = [
        "make",
        "-C",
        plan["stage_root"],
        f"STAGE={plan['stage_root']}",
        f"HOST_CC={compiler}",
        "DPU_CC=/fake/dpu-cc",
        f"DPU_PKG_CONFIG={pkg_config}",
        "compatibility",
    ]

    result = runner.compatibility_probe(plan, dict(os.environ), timeout_seconds=10)
    args = args_log.read_text(encoding="utf-8").splitlines()

    assert result["failure_stage"] == "compatibility_preflight"
    assert result["dpu_allocation_attempted"] is False
    assert "-DINT32" in args
    assert "-Werror=implicit-function-declaration" in args
    assert "-Wl,--no-undefined" in args
    assert str(Path(plan["stage_root"]) / "host.c") in args
    assert str(Path(plan["stage_root"]) / "pidcomm_commlib.c") in args
    assert str(Path(plan["stage_root"]) / "include/pidcomm_binary_paths.h") in args
