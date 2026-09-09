"""Controller helper tests: fake admission and harmless owned Linux children."""

import hashlib
import importlib.util
import json
from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from scripts import upmem_cost_guided_execution as execution


SOURCE = "a" * 40


@pytest.fixture
def host(tmp_path, monkeypatch):
    root, evidence = tmp_path / "implementation", tmp_path / "evidence"
    root.mkdir()
    evidence.mkdir()
    proc, ranks = tmp_path / "proc", tmp_path / "sysfs"
    proc.mkdir()
    ranks.mkdir()
    for index in (0, 1):
        rank = ranks / f"dpu_rank{index}"
        rank.mkdir()
        (rank / "is_owned").write_text("0\n")
    (proc / "meminfo").write_text("MemAvailable: 2000000 kB\n")
    device, governor = tmp_path / "rank1", tmp_path / "governor"
    device.write_bytes(b"")
    governor.write_text("performance\n")
    monkeypatch.setattr(execution, "PROC_ROOT", proc)
    monkeypatch.setattr(execution, "RANK_SYSFS", ranks)
    monkeypatch.setattr(execution, "RANK_DEVICE", device)
    monkeypatch.setattr(execution, "GOVERNOR", governor)
    affinity = []
    monkeypatch.setattr(execution.os, "sched_setaffinity", lambda pid, cpus: affinity.append((pid, cpus)))
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(execution.shutil, "disk_usage", lambda path: SimpleNamespace(free=3 * 1024**3))
    commands = {
        ("git", "rev-parse", "HEAD"): SOURCE,
        ("git", "status", "--porcelain"): "",
        ("ps", "-eo", "user,pid,comm,args"): "USER PID COMMAND COMMAND\nuser 100 idle idle\n",
        ("dpu-pkg-config", "--modversion", "dpu"): "2023.1.0",
    }

    def read(args, *, cwd):
        assert cwd == root
        result = commands[tuple(args)]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(execution, "_read", read)
    binaries = {}
    for name in ("host", "dpu", "init"):
        binary = root / name
        binary.write_bytes(name.encode())
        binaries[str(binary)] = hashlib.sha256(name.encode()).hexdigest()
    return SimpleNamespace(root=root, evidence=evidence, proc=proc, ranks=ranks, device=device,
                           governor=governor, binaries=binaries, commands=commands, affinity=affinity)


def test_preflight_records_admission_without_allocation(host):
    result = execution.preflight(host.root, SOURCE, host.binaries, host.evidence)
    assert result["source_sha"] == SOURCE and result["source_clean"] is True
    assert result["rank_ownership"] == {"dpu_rank0": "0", "dpu_rank1": "0"}
    assert result["binary_sha256"] == host.binaries
    assert result["affinity"] == [0] and host.affinity == [(0, {0})]
    assert result["sdk"] == "2023.1.0" and result["cpu0_governor"] == "performance"
    assert result["memory_requirement_bytes"] == 512 * 1024**2 + 1024**3
    assert all((p / "is_owned").read_text() == "0\n" for p in host.ranks.iterdir())


@pytest.mark.parametrize("failure", [
    "source", "dirty", "rank0_owned", "rank1_owned", "rank_unknown", "rank1_missing",
    "missing_rank_state", "device", "cli", "upmem_host", "other_host", "sdk", "governor",
    "memory", "memory_units", "storage", "writable", "binary", "empty_binaries", "affinity",
])
def test_preflight_rejects_admission_failures(host, monkeypatch, failure):
    if failure == "source":
        host.commands[("git", "rev-parse", "HEAD")] = "b" * 40
    elif failure == "dirty":
        host.commands[("git", "status", "--porcelain")] = " M changed.py"
    elif failure in {"rank0_owned", "rank1_owned", "rank_unknown"}:
        name = "dpu_rank0" if failure == "rank0_owned" else "dpu_rank1"
        (host.ranks / name / "is_owned").write_text("unknown" if failure == "rank_unknown" else "1")
    elif failure in {"rank1_missing", "missing_rank_state"}:
        (host.ranks / "dpu_rank1/is_owned").unlink()
        if failure == "rank1_missing":
            (host.ranks / "dpu_rank1").rmdir()
    elif failure == "device":
        host.device.unlink()
    elif failure in {"cli", "upmem_host", "other_host"}:
        token = {"cli": "quantum_bench.cli", "upmem_host": "host_upmem_execution", "other_host": "gwfa_host"}[failure]
        host.commands[("ps", "-eo", "user,pid,comm,args")] += f"user 123 python {token}\n"
    elif failure == "sdk":
        host.commands[("dpu-pkg-config", "--modversion", "dpu")] = "2024.1.0"
    elif failure == "governor":
        host.governor.write_text("")
    elif failure == "memory":
        (host.proc / "meminfo").write_text(f"MemAvailable: {(512 * 1024**2 + 1024**3) // 1024} kB\n")
    elif failure == "memory_units":
        (host.proc / "meminfo").write_text("MemAvailable: 2000000 MB\n")
    elif failure == "storage":
        monkeypatch.setattr(execution.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * 1024**3))
    elif failure == "writable":
        original = os.access
        monkeypatch.setattr(execution.os, "access", lambda path, mode: False if path == host.evidence else original(path, mode))
    elif failure == "binary":
        Path(next(iter(host.binaries))).write_bytes(b"tamper")
    elif failure == "empty_binaries":
        host.binaries.clear()
    elif failure == "affinity":
        monkeypatch.setattr(execution.os, "sched_getaffinity", lambda pid: {1})
    with pytest.raises((RuntimeError, OSError)):
        execution.preflight(host.root, SOURCE, host.binaries, host.evidence)


@pytest.mark.parametrize("failure", [None, "head", "dirty", "owned", "source_error", "rank_error"])
def test_terminal_inspection_preserves_partial_facts_and_fails_closed(host, failure):
    if failure == "head":
        host.commands[("git", "rev-parse", "HEAD")] = "b" * 40
    elif failure == "dirty":
        host.commands[("git", "status", "--porcelain")] = "?? unexpected"
    elif failure == "owned":
        (host.ranks / "dpu_rank0/is_owned").write_text("1")
    elif failure == "source_error":
        host.commands[("git", "rev-parse", "HEAD")] = OSError("unreadable source")
    elif failure == "rank_error":
        (host.ranks / "dpu_rank1/is_owned").unlink()
    result = execution.terminal_inspection(host.root, SOURCE)
    assert result["valid"] is (failure is None)
    assert result["released"] is (failure not in {"owned", "rank_error"})
    assert bool(result["inspection_errors"]) is (failure in {"source_error", "rank_error"})
    assert "worktree_status" in result
    assert not host.affinity


def test_successful_process_logs_and_returns_exit_status(tmp_path):
    with (tmp_path / "process.log").open("wb") as log:
        code, timeout, elapsed = execution.run_bounded(
            [sys.executable, "-c", "print('bounded child'); raise SystemExit(3)"],
            tmp_path, os.environ.copy(), log, 10,
        )
    assert (code, timeout) == (3, False) and elapsed >= 0
    assert b"bounded child" in (tmp_path / "process.log").read_bytes()


@pytest.mark.parametrize("parent_exits", [False, True])
def test_owned_group_cleanup_leaves_unrelated_process_alive(tmp_path, monkeypatch, parent_exits):
    monkeypatch.setattr(execution, "_CLEANUP_GRACES", (
        (signal.SIGINT, 0.1), (signal.SIGTERM, 0.1), (signal.SIGKILL, 1),
    ))
    child_code = "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    parent_code = (
        "import os,pathlib,signal,subprocess,sys,time; "
        "signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        "pathlib.Path('owned.pid').write_text(str(os.getpid())+' '+str(p.pid)); "
        + ("time.sleep(0.1)" if parent_exits else "time.sleep(60)")
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        with (tmp_path / "process.log").open("wb") as log:
            code, timed_out, elapsed = execution.run_bounded(
                [sys.executable, "-c", parent_code], tmp_path, os.environ.copy(), log, 1,
            )
        parent_pid, child_pid = map(int, (tmp_path / "owned.pid").read_text().split())
        assert timed_out is (not parent_exits)
        assert (code == 0) is parent_exits and elapsed >= 0
        assert not execution._live_group(parent_pid)
        child_stat = Path(f"/proc/{child_pid}/stat")
        assert not child_stat.exists() or child_stat.read_text().rsplit(")", 1)[1].split()[0] in {"Z", "X"}
        assert unrelated.poll() is None
    finally:
        os.killpg(unrelated.pid, signal.SIGKILL)
        unrelated.wait(timeout=5)
        if (tmp_path / "owned.pid").exists():
            pgid = int((tmp_path / "owned.pid").read_text().split()[0])
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cleanup_failure_is_bounded_and_explicit(monkeypatch):
    process = Mock(pid=12345)
    monkeypatch.setattr(execution, "_CLEANUP_GRACES", (
        (signal.SIGINT, 0), (signal.SIGTERM, 0), (signal.SIGKILL, 0),
    ))
    monkeypatch.setattr(execution, "_live_group", lambda pid: True)
    kill = Mock()
    monkeypatch.setattr(execution.os, "killpg", kill)
    with pytest.raises(RuntimeError, match="survived SIGKILL"):
        execution._cleanup_owned_group(process)
    assert [call.args[1] for call in kill.call_args_list[:3]] == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert all(call.args[0] == 12345 for call in kill.call_args_list)
    process.wait.assert_called_once_with(timeout=10)


def test_inaccessible_group_inspection_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "PROC_ROOT", tmp_path)
    proc = tmp_path / "123"
    proc.mkdir()
    stat = proc / "stat"
    stat.write_text("123 (comm with spaces) S 1 123 123")
    original = Path.read_text

    def inaccessible(path, *args, **kwargs):
        if path == stat:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", inaccessible)
    with pytest.raises(RuntimeError, match="cannot inspect"):
        execution._live_group(123)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_invalid_timeout_never_starts_process(monkeypatch, tmp_path, timeout):
    spawn = Mock()
    monkeypatch.setattr(execution.subprocess, "Popen", spawn)
    with pytest.raises(ValueError):
        execution.run_bounded(["unused"], tmp_path, {}, None, timeout)
    spawn.assert_not_called()


@pytest.fixture
def stage(tmp_path):
    stage = tmp_path / "stage"
    (stage / "preregistration").mkdir(parents=True)
    (stage / "preregistration/physical.yml").write_text("synthetic packet\n")
    (stage / "preregistration/SHA256SUMS").write_text("nested binding\n")
    (stage / "preregistration.txt").write_text("sorting boundary\n")
    (stage / "partial.log").write_text("preserved partial observations\n")
    return stage


def test_archive_relative_sorted_checksums_include_nested_manifest(stage):
    archive, digest = execution.archive_stage(stage)
    lines = (stage / "SHA256SUMS").read_text().splitlines()
    entries = dict(line.split("  ", 1)[::-1] for line in lines)
    assert list(entries) == sorted(entries)
    assert "preregistration/SHA256SUMS" in entries and "SHA256SUMS" not in entries
    assert all(not Path(name).is_absolute() and execution._digest(stage / name) == sha for name, sha in entries.items())
    assert execution._digest(archive) == digest
    assert Path(str(archive) + ".sha256").read_text() == f"{digest}  {archive.name}\n"
    assert not Path(str(archive) + ".partial").exists()
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        assert f"{stage.name}/preregistration/SHA256SUMS" in names
    before = archive.read_bytes()
    with pytest.raises(RuntimeError, match="never overwrite"):
        execution.archive_stage(stage)
    assert archive.read_bytes() == before


def test_archive_failure_preserves_partial(stage, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("archive interrupted")

    monkeypatch.setattr(tarfile.TarFile, "add", fail)
    with pytest.raises(OSError, match="interrupted"):
        execution.archive_stage(stage)
    assert Path(str(stage) + ".tar.gz.partial").exists()
    assert (stage / "SHA256SUMS").exists()
    assert (stage / "partial.log").read_text() == "preserved partial observations\n"
    assert not Path(str(stage) + ".tar.gz").exists()


def test_archive_publication_race_cannot_overwrite(stage, monkeypatch):
    link = os.link

    def race(source, target):
        Path(target).write_bytes(b"another archive")
        link(source, target)

    monkeypatch.setattr(execution.os, "link", race)
    with pytest.raises(FileExistsError):
        execution.archive_stage(stage)
    assert Path(str(stage) + ".tar.gz").read_bytes() == b"another archive"
    assert Path(str(stage) + ".tar.gz.partial").exists()


def test_archive_rejects_external_symlink(stage, tmp_path):
    external = tmp_path / "external"
    external.write_bytes(b"outside evidence")
    (stage / "linked").symlink_to(external)
    with pytest.raises(RuntimeError, match="symlink"):
        execution.archive_stage(stage)


def test_archive_fsync_order_and_flushed_contents(stage, monkeypatch):
    events = []
    fsync, link, unlink = os.fsync, os.link, Path.unlink

    def sync(descriptor):
        path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        events.append(("fsync", path))
        if path == stage / "SHA256SUMS":
            assert "preregistration/SHA256SUMS" in path.read_text()
        elif path.name.endswith(".partial"):
            with tarfile.open(path) as tar:
                assert f"{stage.name}/SHA256SUMS" in tar.getnames()
        elif path.name.endswith(".sha256"):
            assert path.read_text().endswith(f"  {stage.name}.tar.gz\n")
        fsync(descriptor)

    def publish(source, target):
        events.append(("link", Path(target)))
        link(source, target)

    def remove(path, *args, **kwargs):
        events.append(("unlink", path))
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(execution.os, "fsync", sync)
    monkeypatch.setattr(execution.os, "link", publish)
    monkeypatch.setattr(Path, "unlink", remove)
    archive, _ = execution.archive_stage(stage)
    partial = Path(str(archive) + ".partial")
    assert events == [
        ("fsync", stage / "SHA256SUMS"), ("fsync", stage),
        ("fsync", partial), ("link", archive), ("fsync", stage.parent),
        ("fsync", Path(str(archive) + ".sha256")), ("fsync", stage.parent),
        ("unlink", partial), ("fsync", stage.parent),
    ]


@pytest.mark.parametrize("failure_at", range(1, 8))
def test_archive_fsync_failure_preserves_evidence_and_never_retries(stage, monkeypatch, failure_at):
    fsync = os.fsync
    calls = 0

    def fail_once(descriptor):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected durability failure")
        fsync(descriptor)

    monkeypatch.setattr(execution.os, "fsync", fail_once)
    with pytest.raises(OSError, match="durability failure"):
        execution.archive_stage(stage)
    assert calls == failure_at
    assert (stage / "SHA256SUMS").exists()
    assert (stage / "partial.log").read_text() == "preserved partial observations\n"
    if failure_at >= 3:
        assert Path(str(stage) + ".tar.gz.partial").exists()
    if failure_at >= 4:
        assert Path(str(stage) + ".tar.gz").exists()
    with pytest.raises(RuntimeError, match="never overwrite"):
        execution.archive_stage(stage)
    assert calls == failure_at


@pytest.fixture
def invocation(tmp_path, monkeypatch, request):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from quantum_bench.evidence import canonical_json
    from quantum_bench.experiment import load_experiment_config, default_validation_policy_id
    import upmem_cost_guided_path as coordinator
    from scripts.qualify_quantized_upmem_execution import write_checksums
    from scripts.upmem_path_heuristic import _canonical_bytes, _wave_portable_qasm_configuration

    spec = importlib.util.spec_from_file_location("execution_packet_fixture", root / "tests/test_cost_guided_packet.py")
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    definition = {"kind": "builtin", "name": "quantization_stress",
                  "parameters": {"n_qubits": 6, "repeat_layers": 1}}
    if getattr(request, "param", None) == "qasm":
        circuit = fixture.builtin_circuit(definition["name"], definition["parameters"])
        lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[6];']
        for op in circuit.operations:
            params = "(" + ",".join(str(value) for value in op.params) + ")" if op.params else ""
            lines.append(f"{op.gate}{params} " + ",".join(f"q[{wire}]" for wire in op.wires) + ";")
        source = tmp_path / "fixture.qasm"
        source.write_text("\n".join(lines) + "\n")
        definition = {"kind": "qasm_file", "name": None, "path": str(source),
                      "parameters": {"qasm_sha256": execution._digest(source)}}
    manifest, workload = fixture._packet(definition)
    study = json.loads((root / "configs/upmem_cost_guided_path_study_v1.json").read_text())
    binding = {"source_sha": SOURCE, "executor_source": study["executor"]["source"],
               "study_hash": coordinator.record_hash(study), "workload_hash": study["workload"]["sha256"]}
    monkeypatch.setattr(coordinator, "research_binding", lambda study: binding)
    profile, normalization = {"integer_weights": [2] * 5}, {"scales": [1] * 5}
    for name, record in (("binding", binding), ("profile", profile), ("normalization", normalization)):
        manifest[f"{name}_hash"] = coordinator.record_hash(record)
    config, provenance = fixture.qualifier.prepare_cost_guided_config(
        manifest, workload, execution_root=root, experiment_id="bounded-execution-fixture",
    )
    packet = tmp_path / "packet"
    packet.mkdir()
    for source_binding in provenance["qasm_source_bindings"]:
        staged = packet / source_binding["prepared_path"]
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(Path(definition["path"]).read_bytes())
    (packet / "physical.yml").write_text(yaml.safe_dump(config))
    normalized = json.loads(canonical_json(load_experiment_config(packet / "physical.yml")))
    portable = _wave_portable_qasm_configuration(normalized, packet / "physical.yml", provenance)
    binaries = {str(root / "native/upmem/runtime/bin" / name): digest
                for name, digest in study["executor"]["binaries"].items()}
    budget = {"prior_attempts": 0, "stage_attempts": 8, "cumulative_attempts": 8,
              "prior_physical_stage_elapsed_s": 0.0, "remaining_physical_time_s": 86400.0}
    provenance.update(source_sha=SOURCE, execution_source=study["executor"]["source"],
                      round_manifest_hash=coordinator.record_hash(manifest),
                      configuration_sha256=execution._digest(packet / "physical.yml"),
                      normalized_configuration_sha256=hashlib.sha256(_canonical_bytes(portable)).hexdigest(),
                      execution_budget=budget)
    for name, value in {"round_manifest": manifest, "binding": binding, "profile": profile,
                        "normalization": normalization, "binary_sha256": binaries,
                        "physical.yml.provenance": provenance}.items():
        (packet / f"{name}.json").write_bytes(_canonical_bytes(value))
    write_checksums(packet)
    qualification = {"all_passed": True, "source_sha": SOURCE, "executor_source": study["executor"]["source"],
                     "study_hash": coordinator.record_hash(study), "numeric_policy": study["executor"]["numeric_policy"],
                     "binary_sha256": study["executor"]["binaries"],
                     "validation_policy_id": default_validation_policy_id(),
                     "cpu_raw_sha256": {"manifest.json": "c" * 64}, "sdk_raw_sha256": {"manifest.json": "d" * 64}}
    for name, target in (("cpu", "cpu"), ("sdk", "sdk"), ("candidate_cpu", "cpu")):
        qualification[name] = {"all_passed": True, "purpose": "correctness_only", "target": target,
                               "source_sha": SOURCE, "executor_source": study["executor"]["source"],
                               "study_hash": coordinator.record_hash(study), "validation_policy_id": default_validation_policy_id()}
    qualification["candidate_cpu"]["round_manifest_hash"] = coordinator.record_hash(manifest)
    receipt = {"kind": "upmem_cost_guided_execution_handoff_v1", "source_sha": SOURCE,
               "executor_source": study["executor"]["source"], "study_hash": coordinator.record_hash(study),
               "stage": "initial", "round_manifest_hash": coordinator.record_hash(manifest),
               **{key: manifest[key] for key in ("binding_hash", "profile_hash", "normalization_hash")},
               "packet_checksums_sha256": execution._digest(packet / "SHA256SUMS"), "predecessors": [],
               "qualification_receipt": qualification, "qualification_hash": coordinator.record_hash(qualification),
               "budget": budget}
    handoff = tmp_path / "stage_handoff.json"
    handoff.write_bytes(_canonical_bytes(receipt))
    procedures = []

    def run(args, cwd, env, log, timeout_s):
        procedures.append((args, timeout_s))
        markers = list((tmp_path / "cost-guided-invocations").glob("*.json"))
        assert len(markers) == 1
        marker = json.loads(markers[0].read_text())
        assert marker["preflight_passed"] is True and marker["consumed_before_child_invocation"] is True
        assert cwd == root and env["PYTHONPATH"] == str(root / "src")
        assert "DPU_BACKEND" not in env and "UPMEM_REQUIRE_SDK_SIMULATOR" not in env
        if args[3] == "qualify":
            config_path = Path(args[args.index("--config") + 1])
            assert config_path.parent.name == "preregistration" and config_path.parent != packet
            assert config_path.read_bytes() == (packet / "physical.yml").read_bytes()
            raw = Path(args[args.index("--output") + 1])
            raw.mkdir()
            (raw / "samples.jsonl").write_text('{}\n' * 8)
        log.write(b"fake canonical runner\n")
        return 0, False, 0.25

    original_preflight = execution.preflight
    monkeypatch.setattr(execution, "preflight", lambda *args: {"source_sha": SOURCE})
    monkeypatch.setattr(execution, "run_bounded", run)
    monkeypatch.setattr(execution, "terminal_inspection", lambda *args: {
        "source_sha": SOURCE, "worktree_status": "", "rank_ownership": {"dpu_rank1": "0"},
        "valid": True, "released": True, "inspection_errors": [],
    })
    return SimpleNamespace(packet=packet, output=tmp_path / "stage", study=study, workload=workload,
                           handoff=handoff, handoff_sha256=execution._digest(handoff), lock_path=tmp_path / "lock",
                           procedures=procedures, run=run, receipt=receipt, original_preflight=original_preflight)


def _execute(case, **changes):
    parameters = {key: getattr(case, key) for key in ("handoff", "handoff_sha256", "study", "workload", "lock_path")}
    parameters.update(changes)
    return execution.execute_stage(case.packet, case.output, **parameters)


@pytest.mark.parametrize("invocation", ["qasm"], indirect=True)
def test_execute_validates_relocated_qasm_packet(invocation):
    from quantum_bench.experiment import load_experiment_config

    original = execution._validate_execution_packet(invocation.packet, invocation.study, invocation.workload)
    result = _execute(invocation)
    assert result["success"] is True and result["accepted"] is False
    relocated = invocation.output / "preregistration"
    copied = execution._validate_execution_packet(relocated, invocation.study, invocation.workload)
    original_config = load_experiment_config(invocation.packet / "physical.yml")
    relocated_config = load_experiment_config(relocated / "physical.yml")
    assert original_config["cases"] != relocated_config["cases"]
    assert original_config["experiment_id"] == relocated_config["experiment_id"]
    assert copied["portable_normalized"] == original["portable_normalized"]
    assert copied["packet_checksums_sha256"] == original["packet_checksums_sha256"]
    sources = list((invocation.packet / "qasm").rglob("*.qasm"))
    assert len(sources) == 1
    for source in sources:
        assert (relocated / source.relative_to(invocation.packet)).read_bytes() == source.read_bytes()
    assert [args[3] for args, _ in invocation.procedures] == ["qualify", "verify"]


def test_execute_once_from_archived_packet_and_never_accept(invocation):
    result = _execute(invocation)
    assert result["success"] is True and result["accepted"] is False
    assert [args[3] for args, _ in invocation.procedures] == ["qualify", "verify"]
    assert [timeout for _, timeout in invocation.procedures] == [86400, 300]
    terminal = json.loads((invocation.output / "terminal.json").read_text())
    assert terminal["physical_stage_elapsed_s"] == 0.25
    assert terminal["results"]["physical"]["actual_sample_rows"] == 8
    assert terminal["accepted"] is False and terminal["retries"] == terminal["replacements"] == 0
    assert terminal["source"] == SOURCE and terminal["terminal_inspection_valid"] is True
    assert execution._digest(Path(result["archive"])) == result["archive_sha256"]
    invocation.output = invocation.output.with_name("replacement")
    with pytest.raises(RuntimeError, match="already invoked"):
        _execute(invocation)
    assert len(invocation.procedures) == 2 and not invocation.output.exists()


@pytest.mark.parametrize("outcome", ["failure", "timeout", "exception", "sigterm", "terminal_denied"])
def test_failed_stage_is_retained_and_consumed(invocation, monkeypatch, outcome):
    original_handler = signal.getsignal(signal.SIGTERM)

    def fail(*args):
        raise RuntimeError("injected denial")

    def run(args, cwd, env, log, timeout_s):
        invocation.run(args, cwd, env, log, timeout_s)
        if outcome == "exception":
            raise RuntimeError("injected process exception")
        if outcome == "sigterm":
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return (2, False, 0.25) if outcome == "failure" else (-9, True, 0.25)

    if outcome == "terminal_denied":
        monkeypatch.setattr(execution, "terminal_inspection", fail)
    else:
        monkeypatch.setattr(execution, "run_bounded", run)
    if outcome in {"exception", "sigterm"}:
        with pytest.raises(KeyboardInterrupt if outcome == "sigterm" else RuntimeError):
            _execute(invocation)
    else:
        assert _execute(invocation)["success"] is False
    assert signal.getsignal(signal.SIGTERM) == original_handler
    assert (invocation.output / "terminal.json").is_file()
    assert Path(str(invocation.output) + ".tar.gz").is_file()
    before = len(invocation.procedures)
    invocation.output = invocation.output.with_name("retry-elsewhere")
    with pytest.raises(RuntimeError, match="already invoked"):
        _execute(invocation)
    assert len(invocation.procedures) == before


@pytest.mark.parametrize("denial", ["occupied", "source"])
def test_preflight_denial_under_lock_does_not_consume_stage(invocation, host, monkeypatch, denial):
    import fcntl

    if denial == "occupied":
        (host.ranks / "dpu_rank0/is_owned").write_text("1")
    else:
        host.commands[("git", "rev-parse", "HEAD")] = "b" * 40

    def preflight(*args):
        assert not invocation.output.exists()
        assert not (invocation.lock_path.parent / "cost-guided-invocations").exists()
        with invocation.lock_path.open("a+") as competing_lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return invocation.original_preflight(host.root, SOURCE, host.binaries, host.evidence)

    monkeypatch.setattr(execution, "preflight", preflight)
    with pytest.raises(RuntimeError, match="rank ownership|source HEAD"):
        _execute(invocation)
    assert not invocation.procedures
    assert not invocation.output.exists()
    assert not Path(str(invocation.output) + ".tar.gz").exists()
    assert not (invocation.lock_path.parent / "cost-guided-invocations").exists()
    (host.ranks / "dpu_rank0/is_owned").write_text("0")
    host.commands[("git", "rev-parse", "HEAD")] = SOURCE
    assert _execute(invocation)["success"] is True


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_stop_during_spawn_cleans_captured_child(tmp_path, monkeypatch, stop_signal):
    popen, children = subprocess.Popen, []
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        signal.getsignal(stop_signal)(stop_signal, None)
        return child

    monkeypatch.setattr(execution.subprocess, "Popen", spawn)
    try:
        with (tmp_path / "stopped.log").open("wb") as log:
            with pytest.raises(KeyboardInterrupt, match="during creation"):
                execution.run_bounded([sys.executable, "-c", "import time; time.sleep(60)"],
                                      tmp_path, os.environ.copy(), log, 10)
        assert len(children) == 1 and children[0].poll() is not None
        assert not execution._live_group(children[0].pid)
        assert {sig: signal.getsignal(sig) for sig in before} == before
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def test_second_stop_does_not_interrupt_owned_cleanup(tmp_path, monkeypatch):
    cleanup = execution._cleanup_owned_group
    cleaned = []

    def stop_while_cleaning(process):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        cleanup(process)
        cleaned.append(process)

    monkeypatch.setattr(execution, "_cleanup_owned_group", stop_while_cleaning)
    with (tmp_path / "cleanup.log").open("wb") as log:
        with pytest.raises(KeyboardInterrupt, match="during cleanup"):
            execution.run_bounded([sys.executable, "-c", "import time; time.sleep(60)"],
                                  tmp_path, os.environ.copy(), log, 0.1)
    assert len(cleaned) == 1 and cleaned[0].poll() is not None
    assert not execution._live_group(cleaned[0].pid)


def test_existing_output_is_not_overwritten(invocation):
    invocation.output.mkdir()
    (invocation.output / "preserve").write_text("original")
    with pytest.raises(RuntimeError, match="never overwrite"):
        _execute(invocation)
    assert (invocation.output / "preserve").read_text() == "original"
    assert not invocation.procedures


@pytest.mark.parametrize("mutation", ["external_hash", "qualification", "qualification_hash", "source", "packet_hash", "budget", "prefix", "numeric"])
def test_handoff_tampering_rejects_before_invocation(invocation, mutation):
    from scripts.upmem_cost_guided_path import record_hash

    receipt = deepcopy(invocation.receipt)
    if mutation == "qualification":
        receipt["qualification_receipt"]["all_passed"] = False
        receipt["qualification_hash"] = record_hash(receipt["qualification_receipt"])
    elif mutation == "qualification_hash":
        receipt["qualification_hash"] = "0" * 64
    elif mutation == "source":
        receipt["source_sha"] = "f" * 40
    elif mutation == "packet_hash":
        receipt["packet_checksums_sha256"] = "0" * 64
    elif mutation == "budget":
        receipt["budget"]["remaining_physical_time_s"] = 86401
    elif mutation == "prefix":
        receipt["predecessors"] = [{"stage": "initial"}]
    elif mutation == "numeric":
        receipt["qualification_receipt"]["numeric_policy"] = "complex128"
        receipt["qualification_hash"] = record_hash(receipt["qualification_receipt"])
    invocation.handoff.write_text(json.dumps(receipt))
    if mutation != "external_hash":
        invocation.handoff_sha256 = execution._digest(invocation.handoff)
    with pytest.raises(RuntimeError):
        _execute(invocation)
    assert not invocation.procedures and not invocation.output.exists()


def test_packet_changed_after_export_rejects(invocation):
    (invocation.packet / "profile.json").write_text('{}')
    with pytest.raises(ValueError, match="checksum"):
        _execute(invocation)
    assert not invocation.procedures


@pytest.mark.parametrize("mutation", [None, "attempt_count", "elapsed", "archive_hash", "profile_hash", "budget"])
def test_handoff_budget_uses_unchanged_prefix_records(invocation, monkeypatch, mutation):
    from scripts.upmem_cost_guided_path import record_hash

    frozen = execution._validate_execution_packet(invocation.packet, invocation.study, invocation.workload)
    previous = deepcopy(frozen["round_manifest"])
    accepted = {"stage": "initial", "round_manifest_hash": record_hash(previous),
                "binding_hash": previous["binding_hash"], "rows": [{}] * 8,
                "archives": [{"path": "/local-only/one/archive.tar.gz", "sha256": "a" * 64},
                             {"path": "/local-only/two/archive.tar.gz", "sha256": "a" * 64}],
                "physical_stage_elapsed_s": 17.0}
    frozen["profile"]["accepted_round_hashes"] = [record_hash(accepted)]
    frozen["round_manifest"].update(stage="feedback_1", profile_hash=record_hash(frozen["profile"]))
    budget = {"prior_attempts": 8, "stage_attempts": 8, "cumulative_attempts": 16,
              "prior_physical_stage_elapsed_s": 17.0, "remaining_physical_time_s": 86383.0}
    frozen["provenance"]["execution_budget"] = budget
    receipt = deepcopy(invocation.receipt)
    receipt.update(stage="feedback_1", round_manifest_hash=record_hash(frozen["round_manifest"]),
                   profile_hash=record_hash(frozen["profile"]), budget=budget,
                   predecessors=[{"stage": "initial", "manifest": previous, "accepted": accepted}])
    receipt["qualification_receipt"]["candidate_cpu"]["round_manifest_hash"] = record_hash(frozen["round_manifest"])
    receipt["qualification_hash"] = record_hash(receipt["qualification_receipt"])
    if mutation == "attempt_count":
        previous["expected_attempts"] += 1
        accepted["round_manifest_hash"] = record_hash(previous)
    elif mutation == "elapsed":
        accepted["physical_stage_elapsed_s"] = float("inf")
    elif mutation == "archive_hash":
        accepted["archives"][1]["sha256"] = "b" * 64
    elif mutation == "profile_hash":
        accepted["rows"][0] = {"tampered": True}
    elif mutation == "budget":
        receipt["budget"]["prior_attempts"] = 0
    invocation.handoff.write_text(json.dumps(receipt))
    pin = execution._digest(invocation.handoff)
    monkeypatch.setattr(execution, "_validate_execution_packet", lambda *args: frozen)
    before = deepcopy(receipt)
    if mutation is None:
        checked = execution._validate_handoff(invocation.packet, invocation.handoff, invocation.study, invocation.workload, pin)
        assert checked["provenance"]["execution_budget"]["remaining_physical_time_s"] == 86383
        assert receipt == before
    else:
        with pytest.raises(RuntimeError):
            execution._validate_handoff(invocation.packet, invocation.handoff, invocation.study, invocation.workload, pin)


def test_external_lock_denies_invocation_before_any_work(invocation):
    import fcntl

    with invocation.lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            _execute(invocation)
    assert not invocation.procedures and not invocation.output.exists()


@pytest.mark.parametrize(("report", "field", "value"), [
    ("cpu", "all_passed", False), ("sdk", "all_passed", 1),
    ("sdk", "target", "cpu"), ("cpu", "purpose", "performance"),
    ("candidate_cpu", "round_manifest_hash", "f" * 64),
    ("candidate_cpu", "source_sha", "b" * 40),
    ("sdk", "validation_policy_id", "f" * 64),
])
def test_nested_qualification_reports_are_bound(invocation, report, field, value):
    from upmem_cost_guided_path import record_hash

    receipt = deepcopy(invocation.receipt)
    receipt["qualification_receipt"][report][field] = value
    receipt["qualification_hash"] = record_hash(receipt["qualification_receipt"])
    invocation.handoff.write_text(json.dumps(receipt))
    invocation.handoff_sha256 = execution._digest(invocation.handoff)
    with pytest.raises(RuntimeError, match="qualification"):
        _execute(invocation)
    assert not invocation.procedures


def test_helper_imports_with_actual_script_entry_path(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; from pathlib import Path; "
        f"sys.path[0] = {str(root / 'scripts')!r}; "
        "import upmem_cost_guided_execution as helper; "
        "\ntry: helper._validate_execution_packet(Path('missing-packet'), {}, {})"
        "\nexcept RuntimeError as exc: assert str(exc) == 'packet must be a real directory'"
        "\nelse: raise AssertionError('invalid packet accepted')"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                   env=dict(os.environ, PYTHONPATH=str(root / "src")), check=True, timeout=30)
