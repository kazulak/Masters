"""Linux controller helpers only: no runner, allocation, lock or acceptance gate."""

import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time


PROC_ROOT = Path("/proc")
RANK_SYSFS = Path("/sys/class/dpu_rank")
RANK_DEVICE = Path("/dev/dpu_rank1")
GOVERNOR = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
DECLARED_MEMORY_BYTES = 512 * 1024**2
EXCLUDED_MEMORY_HEADROOM_BYTES = 1024**3
MINIMUM_STORAGE_BYTES = 2 * 1024**3
_CLEANUP_GRACES = ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 10))


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _read(args, *, cwd):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=30).strip()


def _rank_ownership():
    return {entry.name: (entry / "is_owned").read_text(encoding="ascii").strip()
            for entry in sorted(RANK_SYSFS.glob("dpu_rank*"))}


def _ranks_released(ownership):
    return ownership.get("dpu_rank1") == "0" and set(ownership.values()) == {"0"}


def _mem_available():
    for line in (PROC_ROOT / "meminfo").read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(":")
        if key == "MemAvailable":
            fields = value.split()
            _require(separator and len(fields) == 2 and fields[0].isdigit() and fields[1] == "kB",
                     "MemAvailable is malformed")
            return int(fields[0]) * 1024
    raise RuntimeError("MemAvailable is missing")


def preflight(implementation_root, source_sha, binary_manifest, evidence_directory) -> dict:
    """Observe admission under the caller's lock; pin CPU 0, never allocate ranks."""
    root, evidence = Path(implementation_root), Path(evidence_directory)
    _require(isinstance(source_sha, str) and len(source_sha) == 40
             and all(c in "0123456789abcdef" for c in source_sha), "invalid source SHA")
    head = _read(["git", "rev-parse", "HEAD"], cwd=root)
    status = _read(["git", "status", "--porcelain"], cwd=root)
    _require(head == source_sha and status == "", "source HEAD/worktree is not frozen and clean")
    ownership = _rank_ownership()
    _require(_ranks_released(ownership), "rank ownership is not clear")
    _require(os.access(RANK_DEVICE, os.R_OK | os.W_OK), "selected rank is inaccessible")
    processes = _read(["ps", "-eo", "user,pid,comm,args"], cwd=root)
    _require(not any(token in processes for token in ("quantum_bench.cli", "host_upmem_", "gwfa_host")),
             "competing host process is present")
    sdk = _read(["dpu-pkg-config", "--modversion", "dpu"], cwd=root)
    _require(sdk == "2023.1.0", "SDK is not 2023.1.0")
    governor = GOVERNOR.read_text(encoding="ascii").strip()
    _require(bool(governor), "CPU 0 governor is unavailable")
    _require(evidence.is_dir() and os.access(evidence, os.W_OK | os.X_OK),
             "evidence directory is not writable")
    storage = shutil.disk_usage(evidence).free
    _require(storage > MINIMUM_STORAGE_BYTES, "evidence storage is not above 2 GiB")
    memory = _mem_available()
    required = DECLARED_MEMORY_BYTES + EXCLUDED_MEMORY_HEADROOM_BYTES
    _require(memory > required, "MemAvailable is not above 512 MiB plus 1 GiB excluded overhead")
    _require(bool(binary_manifest), "binary manifest is empty")
    hashes = {}
    for path, expected in sorted(binary_manifest.items()):
        _require(Path(path).is_absolute() and isinstance(expected, str) and len(expected) == 64
                 and all(c in "0123456789abcdef" for c in expected), "invalid binary binding")
        actual = _digest(path)
        _require(actual == expected, f"binary digest mismatch: {path}")
        hashes[str(path)] = actual
    os.sched_setaffinity(0, {0})
    affinity = sorted(os.sched_getaffinity(0))
    _require(affinity == [0], "controller is not restricted to CPU 0")
    return {"source_sha": head, "worktree_status": status, "source_clean": True,
            "rank_ownership": ownership, "competing_process_snapshot": processes,
            "sdk": sdk, "cpu0_governor": governor, "affinity": affinity,
            "storage_free_bytes": storage, "mem_available_bytes": memory,
            "declared_memory_budget_bytes": DECLARED_MEMORY_BYTES,
            "excluded_memory_headroom_bytes": EXCLUDED_MEMORY_HEADROOM_BYTES,
            "memory_requirement_bytes": required, "binary_sha256": hashes}


def terminal_inspection(implementation_root, source_sha) -> dict:
    """Never block retention on inspection failure; released means observed unowned."""
    facts, errors = {}, []
    for name, operation in (
        ("source_sha", lambda: _read(["git", "rev-parse", "HEAD"], cwd=implementation_root)),
        ("worktree_status", lambda: _read(["git", "status", "--porcelain"], cwd=implementation_root)),
        ("rank_ownership", _rank_ownership),
    ):
        try:
            facts[name] = operation()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    facts["source_valid"] = facts.get("source_sha") == source_sha and facts.get("worktree_status") == ""
    facts["released"] = _ranks_released(facts.get("rank_ownership", {}))
    facts["inspection_errors"] = errors
    facts["valid"] = not errors and facts["source_valid"] and facts["released"]
    return facts


def _live_group(pgid):
    """Ignore zombies, but never interpret inaccessible /proc as successful cleanup."""
    _require(PROC_ROOT.is_dir(), "cannot inspect owned process group: /proc unavailable")
    for entry in PROC_ROOT.glob("[0-9]*/stat"):
        try:
            fields = entry.read_text(encoding="ascii").rsplit(")", 1)[1].split()
            if len(fields) < 3:
                raise ValueError("incomplete stat")
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError) as exc:
            raise RuntimeError("cannot inspect owned process group") from exc
    return False


def _cleanup_owned_group(process):
    try:
        for sig, grace in _CLEANUP_GRACES:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + grace
            while _live_group(process.pid) and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.05)
            if not _live_group(process.pid):
                break
        if _live_group(process.pid):
            raise RuntimeError("owned process group survived SIGKILL cleanup grace; partial stage retained")
    except RuntimeError:
        # Even when inspection fails, signal only our own session and do not
        # report verified cleanup. Reaping the direct child remains bounded.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        raise
    process.wait(timeout=10)


def run_bounded(args, cwd, env, log, timeout_s) -> tuple[int, bool, float]:
    """Run a new owned session, returning (returncode, timed_out, elapsed_seconds)."""
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    started = time.monotonic()
    process = subprocess.Popen([str(arg) for arg in args], cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    cleanup_attempted = False
    try:
        result = process.wait(timeout=timeout_s)
        if _live_group(process.pid):
            cleanup_attempted = True
            _cleanup_owned_group(process)
        return result, False, time.monotonic() - started
    except subprocess.TimeoutExpired:
        cleanup_attempted = True
        _cleanup_owned_group(process)
        return process.returncode, True, time.monotonic() - started
    except BaseException:
        if not cleanup_attempted:
            _cleanup_owned_group(process)
        raise


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def archive_stage(stage) -> tuple[Path, str]:
    """Freeze relative checksums and publish a fresh archive; failures retain partials."""
    stage = Path(stage)
    archive = Path(str(stage) + ".tar.gz")
    partial = Path(str(archive) + ".partial")
    checksum = Path(str(archive) + ".sha256")
    root_manifest = stage / "SHA256SUMS"
    _require(stage.is_dir() and not stage.is_symlink(), "stage must be a real directory")
    _require(not any(os.path.lexists(p) for p in (archive, partial, checksum, root_manifest)),
             "stage archive/checksum already exists; never overwrite")
    entries = sorted(stage.rglob("*"))
    _require(all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in entries),
             "stage contains symlink or special file")
    files = sorted((p for p in entries if p.is_file()), key=lambda p: p.relative_to(stage).as_posix())
    names = [p.relative_to(stage).as_posix() for p in files]
    _require(all("\n" not in name and "\r" not in name for name in names), "unsafe checksum filename")
    with root_manifest.open("x", encoding="utf-8") as stream:
        stream.writelines(f"{_digest(path)}  {name}\n" for path, name in zip(files, names, strict=True))
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(stage)
    with tarfile.open(partial, "x:gz") as tar:
        tar.add(stage, arcname=stage.name)
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    # A hard link publishes atomically without replace() overwriting a racer.
    os.link(partial, archive)
    _fsync_directory(stage.parent)
    outer = _digest(archive)
    with checksum.open("x", encoding="ascii") as stream:
        stream.write(f"{outer}  {archive.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(stage.parent)
    partial.unlink()
    try:
        _fsync_directory(stage.parent)
    except OSError:
        # Preserve an explicit partial marker if final publication durability
        # cannot be confirmed. This is retention, not an archive/run retry.
        os.link(archive, partial)
        raise
    return archive, outer
