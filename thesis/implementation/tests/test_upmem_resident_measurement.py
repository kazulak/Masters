"""Software-only pair client and C host lifecycle qualification; never uses SDK."""

import json
import ctypes
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from unittest.mock import Mock

import numpy as np
import pytest

from tests import test_upmem_resident_kernel_simulator as probe
from tests import upmem_resident_probe_client as client


@pytest.fixture(scope="module")
def mock_host(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    native = root / "tests/native"
    host = tmp_path_factory.mktemp("resident-host-mock") / "host"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                    "-I", str(native / "resident_mock"), "-I", str(native),
                    "-I", str(root / "native/upmem/runtime"),
                    str(native / "upmem_resident_probe_host.c"), "-o", str(host)],
                   check=True, capture_output=True, timeout=30)
    return host


@pytest.fixture
def item():
    # Deliberately all-zero MOCK oracle, not physical/corpus evidence.
    info = probe.make_plan((16, 64, 4), (16, 256, 4), "right", 1, 8)
    return {"info": info, "initial": bytes(info["live"]),
            "first_products": tuple(np.zeros((16, 64), dtype=np.float32) for _ in range(4)),
            "second_products": tuple(np.zeros((16, 256), dtype=np.float32) for _ in range(4))}


@pytest.mark.parametrize("arm", ["resident", "host_roundtrip"])
def test_interactive_mock_two_launches_and_reap(mock_host, item, arm):
    record = client._measure(mock_host, mock_host, item, arm, timeout=5)
    assert record["launches"] == 2 and record["release_verified"] == 1
    assert record["timing_source"] == "simulator_diagnostic_only"
    assert record["client_total_s"] >= record["subprocess_wall_s"] >= record["attempt_wall_s"]
    assert "kernel_s" not in record
    assert record["stderr"].count("MOCK_FREE") == 1


def _start(host, item):
    proc = subprocess.Popen([str(host), "unused", "--measure", "resident"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=0, start_new_session=True)
    os.set_blocking(proc.stdin.fileno(), False)
    client._write_all(proc.stdin.fileno(), probe.request(1, item, 0, item["initial"]), time.monotonic() + 5)
    return proc


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("phase", ["second_request", "partial_request", "final_eof"])
def test_host_interrupt_blocked_stdio_frees(mock_host, item, sig, phase):
    proc = _start(mock_host, item)
    try:
        deadline = time.monotonic() + 5
        client._read_exact(proc.stdout.fileno(), 72, deadline)
        if phase == "partial_request":
            client._write_all(proc.stdin.fileno(), b"\x03", deadline)
        elif phase == "final_eof":
            client._write_all(proc.stdin.fileno(), probe.request(3, item, 0, b""), deadline)
            client._read_exact(proc.stdout.fileno(), 72 + 65536, deadline)
        time.sleep(0.03)
        proc.send_signal(sig)
        assert proc.wait(timeout=3) == 5
        stderr = proc.stderr.read().decode()
        assert stderr.count("MOCK_ALLOC") == stderr.count("MOCK_FREE") == 1
        assert json.loads(stderr.splitlines()[-1])["release_verified"] == 1
    finally:
        client._stop(proc)
        proc.stdout.close()
        proc.stderr.close()


def test_host_closed_reply_pipe_frees_instead_of_sigpipe(mock_host, item):
    proc = subprocess.Popen([str(mock_host), "unused", "--measure", "resident"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=0, start_new_session=True)
    try:
        proc.stdout.close()
        os.set_blocking(proc.stdin.fileno(), False)
        client._write_all(proc.stdin.fileno(), probe.request(1, item, 0, item["initial"]), time.monotonic() + 5)
        assert proc.wait(timeout=3) == 5
        assert proc.stderr.read().count(b"MOCK_FREE") == 1
    finally:
        client._stop(proc)
        proc.stderr.close()


@pytest.mark.parametrize("bad", ["tasklets", "span", "geometry"])
def test_host_rejects_bad_plan_before_allocation(mock_host, item, bad):
    raw = bytearray(probe.request(1, item, 0, item["initial"]))
    # First control starts at byte20; mutate exact packed fields.
    offset = {"tasklets": 20 + 12, "span": 20 + 80, "geometry": 20 + 60}[bad]
    raw[offset:offset + 4] = (0xffffffff).to_bytes(4, "little")
    result = subprocess.run([str(mock_host), "unused", "--measure", "resident"],
                            input=raw, capture_output=True, timeout=3)
    assert result.returncode == 2 and b"MOCK_ALLOC" not in result.stderr


@pytest.mark.parametrize("failure", [EOFError, TimeoutError, KeyboardInterrupt, BrokenPipeError])
def test_client_failure_closes_and_reaps(mock_host, item, monkeypatch, failure):
    original = client.subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        proc = original(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(client.subprocess, "Popen", launch)
    injected = failure("injected")
    monkeypatch.setattr(client, "_read_exact", Mock(side_effect=injected))
    with pytest.raises(failure) as caught:
        client._measure(mock_host, mock_host, item, "resident", timeout=3)
    assert caught.value is injected
    facts = caught.value.resident_probe_failure
    assert facts["stage"] == "producer_completion"
    assert facts["returncode"] is not None and facts["elapsed_s"] >= 0
    assert not facts["native_metrics_validated"]
    json.dumps(facts)
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_owned_cleanup_escalates_and_reaps(monkeypatch):
    proc = Mock(pid=12345)
    proc.stdin.closed = False
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("mock", 10), -9]
    kill = Mock()
    monkeypatch.setattr(client.os, "killpg", kill)
    client._stop(proc)
    assert [call.args for call in kill.call_args_list] == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert proc.wait.call_count == 2


def test_short_read_and_timeout():
    read, write = os.pipe()
    try:
        os.write(write, b"ab")
        assert client._read_exact(read, 2, time.monotonic() + 1) == b"ab"
        with pytest.raises(TimeoutError):
            client._read_exact(read, 1, time.monotonic() + 0.01)
        os.write(write, b"a")
        os.close(write)
        write = None
        with pytest.raises(EOFError, match="1/2"):
            client._read_exact(read, 2, time.monotonic() + 1)
    finally:
        os.close(read)
        if write is not None:
            os.close(write)


@pytest.mark.parametrize("selector", [None, "DPU_BACKEND", "UPMEM_EXECUTION_MODE", "UPMEM_REQUIRE_SDK_SIMULATOR"])
def test_physical_selectors_reject_before_spawn(item, monkeypatch, selector):
    monkeypatch.delenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", raising=False)
    if selector:
        monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
        monkeypatch.setenv(selector, "")
    launch = Mock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(client.subprocess, "Popen", launch)
    with pytest.raises(ValueError, match="opt-in"):
        client._measure("unused", "unused", item, "resident", physical=True)
    launch.assert_not_called()


@pytest.mark.parametrize("field,value", [("dpus", 2), ("ranks", 2), ("tasklets", 7),
    ("release_verified", 0), ("fallback", True), ("launches", 3), ("h2d_bytes", 93833),
    ("d2h_calls", 4), ("physical", 0), ("pair_s", float("nan"))])
def test_bad_accounting_rejected(mock_host, item, field, value):
    record = client._measure(mock_host, mock_host, item, "resident", timeout=3)
    record[field] = value
    with pytest.raises(ValueError):
        client._metrics(record, "resident", False, 1)


def test_selected_corpus_reconstruction_is_exact():
    from tests.test_upmem_resident_corpus import selected_case

    item, _ = selected_case()
    value = client.reconstruct(client.product_bytes(item["first_products"], item["info"]["controls"][0]),
                               item["info"]["controls"][0])
    assert value.real.astype("<f4").tobytes() + value.imag.astype("<f4").tobytes() == item["host_patch"][2]


def test_decoder_preserves_qualified_signed_zero_policy_and_rejects_nonfinite(item):
    control = item["info"]["controls"][0]
    products = tuple(np.full((16, 64), -0.0, dtype=np.float32) for _ in range(4))
    value = client.reconstruct(client.product_bytes(products, control), control)
    assert value.tobytes() == bytes(value.nbytes)
    products[0][0, 0] = np.inf
    with pytest.raises(ValueError):
        client.reconstruct(client.product_bytes(products, control), control)


@pytest.mark.parametrize("which", ["first_products", "second_products"])
def test_oracle_comparison_only_after_reap(mock_host, item, monkeypatch, which):
    item[which][0][0, 0] = 1.0
    stopped = []
    original = client._stop

    def observe(proc):
        stopped.append(proc.returncode)
        original(proc)

    monkeypatch.setattr(client, "_stop", observe)
    with pytest.raises(ValueError, match="product bytes differ") as caught:
        client._measure(mock_host, mock_host, item, "host_roundtrip", timeout=3)
    assert stopped == [0]  # Both launches, EOF and successful host exit precede comparison.
    facts = caught.value.resident_probe_failure
    assert facts["stage"] == ("producer_oracle" if which == "first_products" else "consumer_oracle")
    assert facts["returncode"] == 0 and facts["native_metrics"]["release_verified"] == 1
    assert facts["native_metrics"]["launches"] == 2 and "MOCK_FREE" in facts["stderr"]
    assert not facts["native_metrics_validated"]  # Native success is not numeric acceptance.


def test_physical_requires_inherited_lock_before_spawn(item, monkeypatch):
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
    for key in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE", "UPMEM_REQUIRE_SDK_SIMULATOR"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="lock_fd"):
        client._measure("unused", "unused", item, "resident", physical=True)


def test_lock_must_match_and_be_exclusively_held_on_passed_description(tmp_path, monkeypatch):
    path = tmp_path / "experiment.lock"
    monkeypatch.setattr(client, "PRIVATE_LOCK", path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with pytest.raises(ValueError, match="exclusive"):
            client._physical_lock(fd)
        fcntl.flock(fd, fcntl.LOCK_SH)
        with pytest.raises(ValueError, match="exclusive"):
            client._physical_lock(fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert client._physical_lock(fd) == (fd,)
        other = os.open(path, os.O_RDWR)
        try:
            with pytest.raises(ValueError, match="exclusive"):
                client._physical_lock(other)
        finally:
            os.close(other)
        replacement = tmp_path / "different.lock"
        replacement.touch(mode=0o600)
        monkeypatch.setattr(client, "PRIVATE_LOCK", replacement)
        with pytest.raises(ValueError, match="existing private"):
            client._physical_lock(fd)
    finally:
        os.close(fd)


def test_inherited_lock_survives_client_sigkill_until_host_cleanup(mock_host, tmp_path):
    # Temporarily adopt the orphaned mock host so this test can reap it itself.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0  # GET_CHILD_SUBREAPER
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    script = r'''
import fcntl, os, signal, sys
from pathlib import Path
import numpy as np
from tests import upmem_resident_probe_client as c
host, directory = sys.argv[1:]
directory = Path(directory)
c.PRIVATE_LOCK = directory / "experiment.lock"
fd = os.open(c.PRIVATE_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
os.environ["UPMEM_ALLOW_PHYSICAL_HARDWARE"] = "1"
for key in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE", "UPMEM_REQUIRE_SDK_SIMULATOR"):
    os.environ.pop(key, None)
original_spawn, original_read = c.subprocess.Popen, c._read_exact
def spawn(command, **kwargs):
    global native
    # Exercise physical-client FD propagation, but ONLY launch the simulator-only
    # local SDK stub. No physical C path, SDK library, or device is touched.
    assert command[-2:] == ["--rank-path", "/dev/dpu_rank1"]
    assert kwargs["pass_fds"] == (fd,)
    with (directory / "host.stderr").open("wb") as errors:
        kwargs["stderr"] = errors
        native = original_spawn(command[:-2], **kwargs)
    return native
def read(*args):
    value = original_read(*args)
    os.kill(native.pid, signal.SIGSTOP)
    (directory / "native.pid").write_text(str(native.pid))
    os.kill(os.getpid(), signal.SIGSTOP)
    return value
c.subprocess.Popen, c._read_exact = spawn, read
info = c.probe.make_plan((16, 64, 4), (16, 256, 4), "right", 1, 8)
item = {"info": info, "initial": bytes(info["live"]),
        "first_products": tuple(np.zeros((16,64), dtype=np.float32) for _ in range(4)),
        "second_products": tuple(np.zeros((16,256), dtype=np.float32) for _ in range(4))}
c._measure(host, host, item, "resident", physical=True, lock_fd=fd, timeout=10)
'''
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{root / 'src'}:{root}"}
    parent, native = None, None
    try:
        parent = subprocess.Popen([sys.executable, "-c", script, str(mock_host), str(tmp_path)], env=env)
        deadline = time.monotonic() + 5
        marker = tmp_path / "native.pid"
        while not marker.exists() and time.monotonic() < deadline and parent.poll() is None:
            time.sleep(0.01)
        assert marker.exists(), "mock client did not reach first reply"
        native = int(marker.read_text())
        parent.kill()
        assert parent.wait(timeout=3) == -signal.SIGKILL
        with (tmp_path / "experiment.lock").open("r+") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.kill(native, signal.SIGCONT)
            while time.monotonic() < deadline:
                pid, status = os.waitpid(native, os.WNOHANG)
                if pid:
                    native = None
                    break
                time.sleep(0.01)
            assert native is None, "mock host did not clean up on parent EOF"
            assert os.waitstatus_to_exitcode(status) == 5
            assert (tmp_path / "host.stderr").read_text().count("MOCK_FREE") == 1
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if parent is not None and parent.poll() is None:
            parent.kill()
            parent.wait()
        if native is not None:
            os.kill(native, signal.SIGKILL)
            os.waitpid(native, 0)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.mark.parametrize("kill", [False, True])
def test_failure_after_first_reply_preserves_native_cleanup_or_kill(mock_host, item, monkeypatch, kill):
    spawn, read = client.subprocess.Popen, client._read_exact
    processes = []
    injected = TimeoutError("after producer reply")

    def launch(*args, **kwargs):
        proc = spawn(*args, **kwargs)
        processes.append(proc)
        return proc

    def fail(*args):
        read(*args)
        if kill:
            os.kill(processes[0].pid, signal.SIGKILL)
            processes[0].wait(timeout=3)
        raise injected

    monkeypatch.setattr(client.subprocess, "Popen", launch)
    monkeypatch.setattr(client, "_read_exact", fail)
    with pytest.raises(TimeoutError) as caught:
        client._measure(mock_host, mock_host, item, "resident", timeout=3)
    assert caught.value is injected
    facts = caught.value.resident_probe_failure
    assert facts["returncode"] == (-signal.SIGKILL if kill else 5)
    assert facts["stage"] == "producer_completion" and "MOCK_ALLOC" in facts["stderr"]
    assert "release_verified" not in facts and not facts["native_metrics_validated"]
    if kill:
        assert facts["native_metrics"] is None and "MOCK_FREE" not in facts["stderr"]
    else:
        assert facts["native_metrics"]["release_verified"] == 1
        assert facts["native_metrics"]["status"] == 5 and "MOCK_FREE" in facts["stderr"]
    assert len(processes) == 1 and processes[0].poll() is not None


def test_failure_stderr_is_bounded_and_metrics_remain_persistable():
    metrics = {"schema": "resident_pair_measurement_v1", "status": 6, "release_verified": 0}
    with tempfile.TemporaryFile() as stream:
        stream.write(b"x" * 100000 + b"\xff\n" + json.dumps(metrics).encode() + b"\n")
        stream.flush()
        facts = client._failure_facts(Mock(poll=lambda: 6), stream, "terminal_metrics", time.monotonic())
    assert facts["stderr_truncated"] and facts["stderr_bytes"] > 65536
    assert len(facts["stderr"]) <= 65536 and facts["native_metrics"] == metrics
    assert json.loads(json.dumps(facts))["returncode"] == 6


def test_spawn_failure_keeps_original_exception_and_stage(mock_host, item, monkeypatch):
    error = OSError("mock exec failure")
    monkeypatch.setattr(client.subprocess, "Popen", Mock(side_effect=error))
    with pytest.raises(OSError) as caught:
        client._measure(mock_host, mock_host, item, "resident", timeout=3)
    assert caught.value is error
    assert error.resident_probe_failure["stage"] == "spawn"
    assert error.resident_probe_failure["returncode"] is None
