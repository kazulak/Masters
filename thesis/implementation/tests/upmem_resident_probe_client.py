"""Test-only interactive two-launch client; not a runtime backend or admission tool."""

import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import tempfile
import time

import numpy as np

from quantum_bench.numerics import decode_complex_products
from quantum_bench.upmem.wave_protocol import COMPLETION, WaveCompletion
from tests import test_upmem_resident_kernel_simulator as probe

PRIVATE_LOCK = Path("/home/tkazulak/evidence/upmem-experiment.lock")


def _physical_lock(lock_fd):
    if type(lock_fd) is not int or lock_fd < 0:
        raise ValueError("physical execution requires the controller's lock_fd")
    actual = os.fstat(lock_fd)
    expected = PRIVATE_LOCK.stat(follow_symlinks=False)
    if (not stat.S_ISREG(expected.st_mode) or actual.st_nlink != 1 or
            actual.st_uid != os.geteuid() or actual.st_mode & 0o077 or
            (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)):
        raise ValueError("lock_fd is not the existing private experiment lock")
    # Linux fdinfo identifies the flock on THIS open file description. A lock
    # held through another open of the same inode is not sufficient to inherit.
    locks = [line.split()[2:] for line in Path(f"/proc/self/fdinfo/{lock_fd}").read_text().splitlines()
             if line.startswith("lock:")]
    if not any(row[:3] == ["FLOCK", "ADVISORY", "WRITE"] and row[-2:] == ["0", "EOF"]
               for row in locks):
        raise ValueError("lock_fd must already hold an exclusive whole-file flock")
    return (lock_fd,)


def _ready(fd, deadline, write=False):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("resident pair deadline")
    readable, writable, _ = select.select([] if write else [fd], [fd] if write else [], [], remaining)
    if not (readable or writable):
        raise TimeoutError("resident pair pipe deadline")


def _read_exact(fd, size, deadline):
    result = bytearray()
    while len(result) < size:
        _ready(fd, deadline)
        try:
            chunk = os.read(fd, size - len(result))
        except BlockingIOError:
            continue
        if not chunk:
            raise EOFError(f"short resident reply: {len(result)}/{size}")
        result.extend(chunk)
    return bytes(result)


def _write_all(fd, data, deadline):
    view = memoryview(data)
    while view:
        _ready(fd, deadline, write=True)
        try:
            count = os.write(fd, view)
        except BlockingIOError:
            continue
        if count <= 0:
            raise BrokenPipeError("resident request pipe")
        view = view[count:]


def _stop(proc):
    # Closing input unblocks an ordinary host read; TERM also interrupts stdio.
    if not proc.stdin.closed:
        proc.stdin.close()
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def _failure_facts(proc, errors, stage, started):
    errors.seek(0, os.SEEK_END)
    size = errors.tell()
    errors.seek(max(0, size - 65536))
    stderr = errors.read(65536).decode("utf-8", errors="replace")
    metrics = None
    for line in reversed(stderr.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and candidate.get("schema") == "resident_pair_measurement_v1":
            metrics = candidate
            break
    return {"stage": stage, "returncode": None if proc is None else proc.poll(),
            "elapsed_s": time.monotonic() - started, "stderr": stderr,
            "stderr_bytes": size, "stderr_truncated": size > 65536,
            "native_metrics": metrics, "native_metrics_validated": False}


def product_bytes(products, control):
    return b"".join(np.asarray(value, dtype="<f4").tobytes() +
                    probe.GUARD * (span[1] - value.nbytes)
                    for value, span in zip(products, control.planes[4:], strict=True))


def reconstruct(raw, control):
    stride = control.planes[4][1]
    if len(raw) != 4 * stride:
        raise ValueError("product reply size")
    # Match the qualified host decoder and the DPU's positive-zero assembly.
    lanes = tuple(np.add(np.float32(0.0), np.frombuffer(
        raw, dtype="<f4", count=control.m * control.n, offset=i * stride
    ).reshape(control.m, control.n), dtype=np.float32) for i in range(4))
    return decode_complex_products(lanes, 1.0, 1.0, "split_complex_float32_v1")


def _metrics(record, arm, physical, pair_id):
    expected = dict(schema="resident_pair_measurement_v1", status=0, physical=physical,
                    arm=arm, pair_id=pair_id, dpus=1, ranks=1, tasklets=8, allocated=1,
                    release_verified=1, fallback=False, launches=2,
                    h2d_bytes=93832, d2h_bytes=65680, h2d_calls=5, d2h_calls=3)
    if arm == "host_roundtrip":
        expected.update(h2d_bytes=102024, d2h_bytes=82064, h2d_calls=6, d2h_calls=4)
    for key, value in expected.items():
        if type(record.get(key)) is not type(value) or record[key] != value:
            raise ValueError(f"resident metric mismatch: {key}")
    for key in ("session_open_s", "pair_s", "session_close_s", "attempt_wall_s",
                "h2d_s", "d2h_s", "launch_wall_s"):
        value = record.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid duration: {key}")
    return record


def measure_selected(host, binary, arm, *, physical=False, timeout=120, lock_fd=None):
    from tests.test_upmem_resident_corpus import selected_case

    item, _ = selected_case()
    result = _measure(host, binary, item, arm, physical=physical, timeout=timeout, lock_fd=lock_fd)
    result["corpus"] = "Stress16 greedy contract_121->contract_122; fixed retained candidate"
    result["initial_live_sha256"] = hashlib.sha256(item["initial"][:93184]).hexdigest()
    return result


def _measure(host, binary, item, arm, *, physical=False, timeout=120, lock_fd=None):
    """One session only. Caller supplies the fixed CPU-qualified corpus, before timing.

    Simulator output is diagnostic only. Hardware still needs external admission.
    Failure is terminal: never retry allocation or claim verified release on kill.
    """
    if arm not in ("resident", "host_roundtrip") or type(physical) is not bool:
        raise ValueError("explicit arm/backend required")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive finite timeout required")
    env = os.environ.copy()
    if physical and (env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1" or any(
            key in env for key in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE", "UPMEM_REQUIRE_SDK_SIMULATOR"))):
        raise ValueError("physical opt-in or backend selector conflict")
    inherited = _physical_lock(lock_fd) if physical else ()
    info = item["info"]
    pair_id = info["controls"][0].request_sequence
    fixed = probe.make_plan((16, 64, 4), (16, 256, 4), "right", pair_id, 8)
    if info != fixed or len(item["initial"]) < fixed["live"]:
        raise ValueError("not the fixed Stress16 pair layout")
    controls = info["controls"]
    expected = [product_bytes(item[key], control) for key, control in
                zip(("first_products", "second_products"), controls, strict=True)]
    for raw, control in zip(expected, controls, strict=True):
        reconstruct(raw, control)
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for name, path in (("host_sha256", host), ("dpu_sha256", binary))}
    command = [str(host), str(binary), "--measure", arm]
    if physical:
        command += ["--rank-path", "/dev/dpu_rank1"]
    with tempfile.TemporaryFile() as errors:
        started = time.monotonic()
        deadline = started + timeout
        proc, stage = None, "spawn"
        try:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=errors, bufsize=0, start_new_session=True, env=env,
                                    pass_fds=inherited)
            stage = "producer_request"
            os.set_blocking(proc.stdin.fileno(), False)
            os.set_blocking(proc.stdout.fileno(), False)
            _write_all(proc.stdin.fileno(), probe.request(1, item, 0, item["initial"][:93184]), deadline)
            stage = "producer_completion"
            WaveCompletion.from_bytes(_read_exact(proc.stdout.fileno(), COMPLETION.size, deadline),
                                      controls[0], require_success=True)
            patch, offset, second = b"", 0, 3
            producer = None
            if arm == "host_roundtrip":
                stage = "producer_reconstruction"
                producer = _read_exact(proc.stdout.fileno(), len(expected[0]), deadline)
                value = reconstruct(producer, controls[0])
                patch = b"".join(np.asarray(array, dtype="<f4").tobytes() +
                                 probe.GUARD * (span[1] - array.nbytes)
                                 for array, span in zip((value.real, value.imag), info["retained"], strict=True))
                offset, second = info["retained"][0][0], 2
            stage = "consumer_request"
            _write_all(proc.stdin.fileno(), probe.request(second, item, offset, patch), deadline)
            proc.stdin.close()
            stage = "consumer_completion"
            WaveCompletion.from_bytes(_read_exact(proc.stdout.fileno(), COMPLETION.size, deadline),
                                      controls[1], require_success=True)
            stage = "consumer_reconstruction"
            raw = _read_exact(proc.stdout.fileno(), len(expected[1]), deadline)
            value = reconstruct(raw, controls[1])
            stage = "terminal_eof"
            _ready(proc.stdout.fileno(), deadline)
            if os.read(proc.stdout.fileno(), 1):
                raise ValueError("unexpected trailing reply")
            stage = "reap"
            if proc.wait(timeout=max(0, deadline - time.monotonic())) != 0:
                raise ValueError("resident host failed")
            reaped = time.monotonic()
            stage = "producer_oracle"
            if producer is not None and producer != expected[0]:
                raise ValueError("producer product bytes differ from CPU oracle")
            stage = "consumer_oracle"
            if raw != expected[1]:
                raise ValueError("consumer product bytes differ from CPU oracle")
            stage = "terminal_metrics"
            errors.seek(0)
            stderr = errors.read().decode("utf-8")
            record = _metrics(json.loads(stderr.splitlines()[-1]), arm, physical, pair_id)
        except BaseException as error:
            cleanup_error = None
            try:
                if proc is not None:
                    _stop(proc)
            except BaseException as cleanup:
                cleanup_error = f"{type(cleanup).__name__}: {cleanup}"
            try:
                error.resident_probe_failure = _failure_facts(proc, errors, stage, started)
                error.resident_probe_failure["cleanup_error"] = cleanup_error
            except BaseException as capture:
                error.resident_probe_failure = {
                    "stage": stage, "returncode": None if proc is None else proc.returncode,
                    "elapsed_s": time.monotonic() - started, "cleanup_error": cleanup_error,
                    "evidence_error": f"{type(capture).__name__}: {capture}"}
            raise
        finally:
            if proc is not None:
                proc.stdout.close()
        return {**record, **hashes, "subprocess_wall_s": reaped - started,
                "validation_s": time.monotonic() - reaped,
                "client_total_s": time.monotonic() - started,
                "consumer_products_sha256": hashlib.sha256(raw).hexdigest(),
                "consumer_complex64_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                "timing_source": "physical_probe" if physical else "simulator_diagnostic_only",
                "stderr": stderr, "production_residency": False}
