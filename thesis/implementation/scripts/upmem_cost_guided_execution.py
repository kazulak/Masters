"""Once-only Linux stage invocation and archival; acceptance remains external."""

import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time


PROC_ROOT = Path("/proc")
RANK_SYSFS = Path("/sys/class/dpu_rank")
RANK_DEVICE = Path("/dev/dpu_rank1")
GOVERNOR = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
DECLARED_MEMORY_BYTES = 512 * 1024**2
EXCLUDED_MEMORY_HEADROOM_BYTES = 1024**3
MINIMUM_STORAGE_BYTES = 2 * 1024**3
_CLEANUP_GRACES = ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 10))
PRIVATE_LOCK = Path("/home/tkazulak/evidence/upmem-experiment.lock")
IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]


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
    process = None
    cleanup_attempted = False
    pending_stop = False
    handlers = {}

    def stop(signum, _frame):
        nonlocal pending_stop
        pending_stop = True
        # Capture ownership before raising; a second stop must not interrupt
        # group cleanup. The original handlers are restored on every exit.
        if process is not None and not cleanup_attempted:
            raise KeyboardInterrupt(f"bounded process received signal {signum}")

    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, stop)
        process = subprocess.Popen([str(arg) for arg in args], cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        if pending_stop:
            raise KeyboardInterrupt("bounded process stopped during creation")
        result = process.wait(timeout=timeout_s)
        if _live_group(process.pid):
            cleanup_attempted = True
            _cleanup_owned_group(process)
        if pending_stop:
            raise KeyboardInterrupt("bounded process stopped during cleanup")
        return result, False, time.monotonic() - started
    except subprocess.TimeoutExpired:
        cleanup_attempted = True
        _cleanup_owned_group(process)
        if pending_stop:
            raise KeyboardInterrupt("bounded process stopped during cleanup")
        return process.returncode, True, time.monotonic() - started
    except BaseException:
        if process is not None and not cleanup_attempted:
            cleanup_attempted = True
            _cleanup_owned_group(process)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


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


def _validate_execution_packet(packet, study, workload):
    """Reopen the portable packet without trusting its self-declared hashes alone."""
    import yaml
    from quantum_bench.evidence import canonical_json
    from quantum_bench.experiment import load_experiment_config
    import upmem_cost_guided_path as coordinator
    from qualify_quantized_upmem_execution import verify_checksums
    from qualify_upmem_path_candidates import prepare_cost_guided_config
    from upmem_path_heuristic import _canonical_bytes, _wave_portable_qasm_configuration

    packet = Path(packet)
    _require(packet.is_dir() and not packet.is_symlink(), "packet must be a real directory")
    _require(not any(p.is_symlink() for p in packet.rglob("*")), "packet must not contain symlinks")
    lines = (packet / "SHA256SUMS").read_text().splitlines()
    names = [line.partition("  ")[2] for line in lines]
    _require(len(names) == len(set(names)) and all(
        name and not Path(name).is_absolute() and ".." not in Path(name).parts for name in names
    ), "unsafe or duplicate packet checksum entry")
    verify_checksums(packet)
    records = {name: json.loads((packet / f"{name}.json").read_text())
               for name in ("round_manifest", "binding", "profile", "normalization", "binary_sha256")}
    manifest, binding = records["round_manifest"], records["binding"]
    _require(binding == coordinator.research_binding(study), "packet coordinator source/dependencies changed")
    for name in ("binding", "profile", "normalization"):
        _require(manifest[f"{name}_hash"] == coordinator.record_hash(records[name]), f"packet {name} hash mismatch")
    config_path = packet / "physical.yml"
    raw_config = yaml.safe_load(config_path.read_text())
    expected_config, expected_provenance = prepare_cost_guided_config(
        manifest, workload, execution_root=IMPLEMENTATION_ROOT,
        experiment_id=raw_config["experiment_id"], simulator=False, cpu_reference=False,
    )
    _require(coordinator.record_hash(raw_config) == coordinator.record_hash(expected_config),
             "packet is not the exact selected physical replay configuration")
    normalized = json.loads(canonical_json(load_experiment_config(config_path)))
    provenance = json.loads((packet / "physical.yml.provenance.json").read_text())
    portable = _wave_portable_qasm_configuration(normalized, config_path, provenance)
    for key, value in {
        "round_manifest_hash": coordinator.record_hash(manifest),
        "binding_hash": coordinator.record_hash(binding),
        "profile_hash": manifest["profile_hash"], "normalization_hash": manifest["normalization_hash"],
        "source_sha": binding["source_sha"], "execution_source": study["executor"]["source"],
        "configuration_sha256": _digest(config_path),
        "normalized_configuration_sha256": hashlib.sha256(_canonical_bytes(portable)).hexdigest(),
        "workload_record_sha256": coordinator.record_hash(workload),
        "selected_cells": expected_provenance["selected_cells"],
        "expected_physical_attempts": manifest["expected_attempts"],
        "expected_prepared_attempts": manifest["expected_attempts"],
    }.items():
        _require(provenance.get(key) == value, f"packet provenance {key} mismatch")
    expected_files = {"physical.yml", "physical.yml.provenance.json", "binary_sha256.json",
                      "round_manifest.json", "binding.json", "profile.json", "normalization.json"}
    for source in expected_provenance["qasm_source_bindings"]:
        path = coordinator._contained_path(packet, source["prepared_path"])
        _require(_digest(path) == source["qasm_sha256"], "packet QASM bytes changed")
        expected_files.add(source["prepared_path"])
    _require(set(names) == expected_files, "packet file inventory differs from frozen inputs")
    binaries = {str(IMPLEMENTATION_ROOT / "native/upmem/runtime/bin" / name): digest
                for name, digest in study["executor"]["binaries"].items()}
    _require(records["binary_sha256"] == binaries, "packet frozen binary bindings changed")
    budget = provenance["execution_budget"]
    for key in ("prior_attempts", "stage_attempts", "cumulative_attempts"):
        _require(type(budget.get(key)) is int and budget[key] >= 0, "invalid packet attempt budget")
    _require(budget["stage_attempts"] == manifest["expected_attempts"] > 0
             and budget["prior_attempts"] + budget["stage_attempts"] == budget["cumulative_attempts"]
             and budget["cumulative_attempts"] <= study["campaign"]["effective_attempt_cap"],
             "packet cumulative attempt budget exceeded")
    prior = budget["prior_physical_stage_elapsed_s"]
    remaining = budget["remaining_physical_time_s"]
    _require(type(prior) in (int, float) and math.isfinite(prior) and prior >= 0
             and type(remaining) in (int, float) and math.isfinite(remaining) and remaining > 0
             and prior + remaining == study["campaign"]["physical_stage_timeout_s"] == 86400,
             "packet cumulative time budget exceeded")
    if manifest["stage"] == "initial":
        _require(budget["prior_attempts"] == 0 and prior == 0, "initial stage has prior execution")
    return {**records, "provenance": provenance, "portable_normalized": portable,
            "packet_checksums_sha256": _digest(packet / "SHA256SUMS")}


def _write_record(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(Path(path).parent)


def _validate_handoff(packet, handoff, study, workload, handoff_sha256):
    from quantum_bench.experiment import default_validation_policy_id
    from upmem_cost_guided_path import record_hash, STAGES

    _require(isinstance(handoff_sha256, str) and len(handoff_sha256) == 64
             and all(c in "0123456789abcdef" for c in handoff_sha256), "invalid external handoff SHA")
    data = Path(handoff).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == handoff_sha256, "external handoff digest mismatch")
    receipt = json.loads(data)
    frozen = _validate_execution_packet(packet, study, workload)
    manifest, binding = frozen["round_manifest"], frozen["binding"]
    stage = manifest["stage"]
    for key, value in {
        "kind": "upmem_cost_guided_execution_handoff_v1", "source_sha": binding["source_sha"],
        "executor_source": study["executor"]["source"], "study_hash": record_hash(study), "stage": stage,
        "binding_hash": record_hash(binding), "round_manifest_hash": record_hash(manifest),
        "profile_hash": record_hash(frozen["profile"]), "normalization_hash": record_hash(frozen["normalization"]),
        "packet_checksums_sha256": frozen["packet_checksums_sha256"],
    }.items():
        _require(receipt.get(key) == value, f"handoff {key} mismatch")
    qualification = receipt["qualification_receipt"]
    _require(receipt.get("qualification_hash") == record_hash(qualification), "qualification receipt hash mismatch")
    _require(qualification.get("all_passed") is True, "CPU/SDK qualification did not pass")
    for key, value in {
        "source_sha": binding["source_sha"], "executor_source": study["executor"]["source"],
        "study_hash": record_hash(study), "numeric_policy": "split_complex_float32_v1",
        "binary_sha256": study["executor"]["binaries"],
        "validation_policy_id": default_validation_policy_id(),
    }.items():
        _require(qualification.get(key) == value, f"qualification {key} mismatch")
    _require(study["executor"]["numeric_policy"] == qualification["numeric_policy"], "frozen numerical policy mismatch")
    for name, target in (("cpu", "cpu"), ("sdk", "sdk"), ("candidate_cpu", "cpu")):
        report = qualification.get(name)
        _require(isinstance(report, dict) and report.get("all_passed") is True,
                 f"qualification {name} report did not pass")
        for key, value in {
            "purpose": "correctness_only", "target": target, "source_sha": binding["source_sha"],
            "executor_source": study["executor"]["source"], "study_hash": record_hash(study),
            "validation_policy_id": default_validation_policy_id(),
        }.items():
            _require(report.get(key) == value, f"qualification {name} {key} mismatch")
    _require(qualification["candidate_cpu"].get("round_manifest_hash") == record_hash(manifest),
             "candidate CPU qualification selection mismatch")
    predecessors = receipt["predecessors"]
    _require(isinstance(predecessors, list) and [p["stage"] for p in predecessors] == list(STAGES[:STAGES.index(stage)]),
             "handoff predecessor stage order mismatch")
    prior_attempts, prior_elapsed, accepted_hashes = 0, 0.0, []
    for prior in predecessors:
        name, previous, accepted = prior["stage"], prior["manifest"], prior["accepted"]
        _require(previous["stage"] == accepted["stage"] == name
                 and previous["study_id"] == study["study_id"]
                 and previous["binding_hash"] == accepted["binding_hash"] == record_hash(binding)
                 and previous["normalization_hash"] == record_hash(frozen["normalization"])
                 and accepted["round_manifest_hash"] == record_hash(previous), "predecessor identity mismatch")
        attempts = previous["expected_attempts"]
        derived = 4 * sum(len(c["candidates"]) for c in previous["cells"].values())
        _require(type(attempts) is int and attempts == derived and 0 <= attempts <= (192 if name == "initial" else 144)
                 and len(accepted["rows"]) == attempts, "predecessor attempt count mismatch")
        elapsed = accepted["physical_stage_elapsed_s"]
        _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                 "invalid predecessor physical elapsed")
        archives = accepted["archives"]
        if attempts:
            _require(elapsed > 0 and len(archives) == 2 and archives[0]["path"] != archives[1]["path"]
                     and archives[0]["sha256"] == archives[1]["sha256"]
                     and len(archives[0]["sha256"]) == 64
                     and all(c in "0123456789abcdef" for c in archives[0]["sha256"]),
                     "predecessor requires two verified archive identities")
        else:
            _require(name.startswith("feedback_") and not archives and elapsed == 0,
                     "only empty feedback may omit physical archives")
        prior_attempts += attempts
        prior_elapsed += elapsed
        accepted_hashes.append(record_hash(accepted))
    if predecessors:
        _require(frozen["profile"].get("accepted_round_hashes") == accepted_hashes,
                 "profile does not bind unchanged predecessor acceptances")
    budget = {"prior_attempts": prior_attempts, "stage_attempts": manifest["expected_attempts"],
              "cumulative_attempts": prior_attempts + manifest["expected_attempts"],
              "prior_physical_stage_elapsed_s": prior_elapsed,
              "remaining_physical_time_s": 86400 - prior_elapsed}
    _require(receipt["budget"] == frozen["provenance"]["execution_budget"] == budget,
             "handoff budget differs from verified predecessor records")
    return frozen


def execute_stage(packet, output, *, handoff, handoff_sha256, study, workload, lock_path=PRIVATE_LOCK) -> dict:
    """Consume one frozen stage, invoke qualify once, retain evidence, never accept it."""
    _require(__debug__ and threading.current_thread() is threading.main_thread(),
             "stage execution requires non-optimized Python on the main thread")
    packet, output, lock_path = Path(packet).resolve(), Path(output).absolute(), Path(lock_path)
    _require(not output.resolve().is_relative_to(packet) and not packet.is_relative_to(output.resolve()),
             "output and packet directories must be disjoint")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = _validate_handoff(packet, handoff, study, workload, handoff_sha256)
        manifest, binding = frozen["round_manifest"], frozen["binding"]
        stage, source = manifest["stage"], binding["source_sha"]
        _require(not any(os.path.lexists(p) for p in (
            output, Path(str(output) + ".tar.gz"), Path(str(output) + ".tar.gz.partial"),
            Path(str(output) + ".tar.gz.sha256"),
        )), "output/archive already exists; never overwrite")
        ledger = lock_path.parent / "cost-guided-invocations"
        identity = {"study_id": study["study_id"], "source_sha": source, "stage": stage}
        marker = ledger / f"{study['study_id']}-{source}-{stage}.json"
        _require(not os.path.lexists(marker), "physical stage was already invoked; no retries or replacements")
        admission = preflight(IMPLEMENTATION_ROOT, source, frozen["binary_sha256"], output.parent)
        ledger.mkdir(exist_ok=True)
        _fsync_directory(lock_path.parent)
        _write_record(marker, {**identity, "handoff_sha256": handoff_sha256,
                              "output": str(output), "preflight_passed": True,
                              "consumed_before_child_invocation": True})
        output.mkdir(parents=True, exist_ok=False)
        _fsync_directory(output.parent)
        results, failure = {}, None
        started = time.monotonic()
        physical_started = None
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def terminate(_signum, _frame):
            raise KeyboardInterrupt("stage controller received SIGTERM")

        try:
            signal.signal(signal.SIGTERM, terminate)
            shutil.copytree(packet, output / "preregistration")
            shutil.copyfile(handoff, output / "stage_handoff.json")
            copied = _validate_handoff(output / "preregistration", output / "stage_handoff.json",
                                       study, workload, handoff_sha256)
            _require(copied == frozen, "handoff changed while copying")
            command = [sys.executable, "-m", "quantum_bench.cli", "qualify",
                       "--config", str(output / "preregistration/physical.yml"),
                       "--output", str(output / "raw"), "--allow-physical"]
            env = dict(os.environ, PYTHONPATH=str(IMPLEMENTATION_ROOT / "src"),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0", UPMEM_ALLOW_PHYSICAL_HARDWARE="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
            for name in ("DPU_BACKEND", "UPMEM_REQUIRE_SDK_SIMULATOR"):
                env.pop(name, None)
            budget = frozen["provenance"]["execution_budget"]
            _write_record(output / "preflight.json", {
                "admission": admission, "execution_budget": budget, "handoff_sha256": handoff_sha256,
                "ready_command": command, "invocation_marker": str(marker), "accepted": False,
            })
            with (output / "physical.log").open("xb") as log:
                physical_started = time.monotonic()
                code, timed_out, elapsed = run_bounded(command, IMPLEMENTATION_ROOT, env, log,
                                                     budget["remaining_physical_time_s"])
            results["physical"] = {"returncode": code, "timed_out": timed_out, "elapsed_s": elapsed,
                                   "planned_attempts": manifest["expected_attempts"]}
            physical_started = None
            if code == 0 and not timed_out:
                with (output / "canonical.log").open("xb") as log:
                    code, timed_out, elapsed = run_bounded(
                        [sys.executable, "-m", "quantum_bench.cli", "verify", "--input", str(output / "raw")],
                        IMPLEMENTATION_ROOT, env, log, 300,
                    )
                results["canonical"] = {"returncode": code, "timed_out": timed_out, "elapsed_s": elapsed}
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if physical_started is not None:
                results["physical"] = {"returncode": None, "timed_out": False,
                                       "elapsed_s": time.monotonic() - physical_started,
                                       "planned_attempts": manifest["expected_attempts"]}
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            try:
                inspection = terminal_inspection(IMPLEMENTATION_ROOT, source)
            except Exception as exc:
                inspection = {"valid": False, "released": False,
                              "inspection_errors": [f"terminal inspection: {type(exc).__name__}: {exc}"]}
            samples = output / "raw/samples.jsonl"
            actual_rows = None
            if samples.is_file():
                try:
                    with samples.open("rb") as stream:
                        actual_rows = sum(bool(line.strip()) for line in stream)
                except OSError as exc:
                    inspection["inspection_errors"].append(f"sample count: {exc}")
                    inspection["valid"] = False
            if "physical" in results:
                results["physical"]["actual_sample_rows"] = actual_rows
            terminal = {
                "results": results, "physical_stage_elapsed_s": results.get("physical", {}).get("elapsed_s", 0.0),
                "physical_stage_elapsed_clock": "monotonic_duration", "controller_elapsed_s": time.monotonic() - started,
                "source": inspection.get("source_sha"), "worktree_status": inspection.get("worktree_status"),
                "rank_ownership_at_finally": inspection.get("rank_ownership", {}),
                "rank1_released": inspection["released"], "terminal_inspection_valid": inspection["valid"],
                "terminal_inspection_errors": inspection["inspection_errors"], "failure": failure,
                "private_lock_release": "held_until_controller_context_exit; release not observed in-stage",
                "accepted": False, "retries": 0, "replacements": 0,
            }
            try:
                _write_record(output / "terminal.json", terminal)
            finally:
                archive, archive_sha = archive_stage(output)
        success = inspection["valid"] and actual_rows == manifest["expected_attempts"] and all(
            results.get(name, {}).get("returncode") == 0 and results[name]["timed_out"] is False
            for name in ("physical", "canonical")
        )
        return {"success": success, "accepted": False, "archive": str(archive),
                "archive_sha256": archive_sha, "results": results, "invocation_marker": str(marker)}
