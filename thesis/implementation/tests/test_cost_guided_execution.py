"""Controller helper tests: fake admission and harmless owned Linux children."""

import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import upmem_cost_guided_execution as execution


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
