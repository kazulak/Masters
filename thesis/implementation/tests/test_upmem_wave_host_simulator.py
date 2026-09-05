"""Exercise prepared cohorts through the existing persistent native host."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest

from quantum_bench.upmem.wave_protocol import (
    COMPLETION, FOUR_PRODUCT_PANEL, IDLE, NO_OPERATION, WaveCompletion,
    WaveControl, product_layout,
)
from quantum_bench.upmem.packed_wave import pack_wave_envelope, unpack_wave_envelope

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/upmem/runtime"
HEADER = struct.Struct("<8s8I4Q32s32s")
OPERATION = struct.Struct("<32s32s4Q2d")


@pytest.fixture(scope="module")
def binaries():
    missing = [name for name in ("make", "dpu-pkg-config", "dpu-upmem-dpurte-clang", "gcc")
               if not shutil.which(name)]
    if missing:
        message = "wave host SDK prerequisites missing: " + ", ".join(missing)
        if os.environ.get("UPMEM_REQUIRE_SDK_SIMULATOR") == "1":
            pytest.fail(message)
        pytest.skip(message)
    paths = {}
    for tasklets in (3, 7, 8, 12, 24):
        subprocess.run(["make", "-C", str(NATIVE), "v4", f"NR_TASKLETS={tasklets}",
                        f"bin/dpu_wave_v5_t{tasklets}"], capture_output=True, check=True)
        paths[tasklets] = tuple(NATIVE / "bin" / name for name in (
            f"host_upmem_execution_plan_v4_t{tasklets}",
            f"dpu_wave_v5_t{tasklets}", f"dpu_simplepim_management_init_t{tasklets}"))
    return paths


def corpus(binary, tasklets=8, mode=0, *, sequence=1, request_start=10):
    """Independent wire fixture: two ready operations and a partially idle subwave."""
    operations = b"".join(OPERATION.pack(
        hashlib.sha256(f"node-{i}".encode()).digest(),
        hashlib.sha256(f"contract-{i}".encode()).digest(), 1, m, n, k,
        0.25 if mode else 1.0, 0.5 if mode else 1.0,
    ) for i, (m, n, k) in enumerate(((2, 3, 130), (1, 1, 3))))
    tiles, inputs, cases = [], [], []
    for wave in range(2):
        for dpu in range(3):
            op = int(dpu == 2)
            m, n, k = (1, 1, 3) if op else (1, 3, 65)
            c = WaveControl(dpu, tasklets, 0, mode, FOUR_PRODUCT_PANEL, op,
                            wave, request_start + wave, 100 + 3 * wave + dpu,
                            0, m, n, k, 0 if op else 65 * wave,
                            product_layout(m, n, k, numeric_mode=mode,
                                           kernel=FOUR_PRODUCT_PANEL))
            if op and wave:
                c = replace(c, flags=IDLE, operation_index=NO_OPERATION, kernel=0,
                            tile_id=0, m=0, n=0, k=0, k_offset=0, planes=((0, 0),) * 8)
                arrays = ()
            else:
                arrays = tuple(((np.arange(rows * cols) + i + dpu + wave) % 11 - 5)
                               .astype(np.int8 if mode else np.float32).reshape(rows, cols)
                               for i, (rows, cols) in enumerate(((m, k), (m, k), (k, n), (k, n))))
                for array, (_, length) in zip(arrays, c.planes[:4]):
                    payload = array.tobytes()
                    inputs.append(payload + b"\0" * (length - len(payload)))
            tiles.append(struct.pack("<2Q", dpu if not op else 0, 0) + c.to_bytes())
            cases.append((c, arrays))
    body = operations + b"".join(tiles) + b"".join(inputs)
    data = HEADER.pack(b"UPWAVE1\0", 1, 136, 3, tasklets, 2, 2, mode, 0,
                       sequence, 6, 136 + len(body), 136 + len(operations) + 6 * 160,
                       hashlib.sha256(b"plan").digest(), hashlib.sha256(binary.read_bytes()).digest()) + body
    return data, cases


def invoke(tmp_path, binaries, data, *, tasklets=8, second=None, dpu_binary=None,
           bad_digest=False, input_kind="regular"):
    host, binary, init = binaries[tasklets]
    commands = []
    for index, payload in enumerate((data,) if second is None else (data, second)):
        name = f"wave-{index}.bin"
        path = tmp_path / name
        if input_kind == "fifo":
            os.mkfifo(path)
        elif input_kind == "symlink":
            target = tmp_path / "target.bin"
            target.write_bytes(payload)
            path.symlink_to(target)
        elif input_kind == "oversized":
            with path.open("wb") as stream:
                stream.truncate(512 * 1024 * 1024 + 1)
        else:
            path.write_bytes(payload)
        digest = "0" * 64 if bad_digest else hashlib.sha256(payload).hexdigest()
        commands.append(f"SUBMIT_PACKED_WAVES {name} {digest}\n")
    result = subprocess.run([str(host), "--wave-v5", "--target", "simulator",
                             "--session-root", str(tmp_path), "--dpus", "3", "--tasklets", str(tasklets),
                             "--initialization-binary", str(init), "--dpu-binary", str(dpu_binary or binary),
                             "--timeout-s", "30"], input="".join(commands) + "CLOSE\n",
                            capture_output=True, text=True, timeout=90,
                            env={**os.environ, "DPU_BACKEND": "simulator"})
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events, result.stderr
    return result, events


def verify_result(tmp_path, response, cases):
    data = (tmp_path / response["response_path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == response["response_sha256"]
    offset = 0
    for control, arrays in cases:
        WaveCompletion.from_bytes(data[offset:offset + COMPLETION.size], control, require_success=True)
        offset += COMPLETION.size
        if not arrays:
            continue
        for a, b in ((0, 2), (1, 3), (0, 3), (1, 2)):
            dtype = np.int32 if control.numeric_mode else np.float32
            expected = arrays[a].astype(dtype) @ arrays[b].astype(dtype)
            raw = expected.tobytes()
            assert data[offset:offset + len(raw)] == raw
            offset += len(raw)
    assert offset == len(data)


@pytest.mark.parametrize("tasklets,mode", [(3, 0), (7, 1), (8, 0), (12, 1), (24, 0)])
def test_python_session_executes_and_reuses_prepared_cohorts(tmp_path, binaries, tasklets, mode):
    from quantum_bench.upmem.native_session import V4Session
    from quantum_bench.upmem.protocol import V4Profile, REQUEST_TRANSPORT_PACKED_WAVE

    host, binary, init = binaries[tasklets]
    profile = V4Profile(dpu_count=3, tasklets_per_dpu=tasklets, numeric_mode=mode,
                        execution_target="sdk_simulator",
                        request_transport=REQUEST_TRANSPORT_PACKED_WAVE, timeout_s=30)
    command = (host, "--target", "simulator", "--session-root", tmp_path,
               "--dpus", "3", "--tasklets", str(tasklets),
               "--initialization-binary", init, "--dpu-binary", binary, "--timeout-s", "30")
    session = V4Session.start(command, session_root=tmp_path, profile=profile)
    try:
        for sequence in (1, 2):
            data, cases = corpus(binary, tasklets, mode, sequence=sequence,
                                 request_start=sequence * 10)
            operations, waves = unpack_wave_envelope(data)
            result = session.submit_waves(
                plan_sha256=hashlib.sha256(b"plan").digest(),
                dpu_binary_sha256=hashlib.sha256(binary.read_bytes()).digest(),
                sequence=sequence, operations=operations, waves=waves,
            )
            verify_result(tmp_path, result, cases)
            assert len(result["results"]) == 2
            for products, (control, arrays) in zip(
                (slot for wave in result["results"] for slot in wave), cases, strict=True
            ):
                if not arrays:
                    assert all(len(p) == 0 for p in products)
                    continue
                dtype = np.int32 if mode else np.float32
                for payload, (a, b) in zip(products, ((0, 2), (1, 3), (0, 3), (1, 2)), strict=True):
                    assert bytes(payload) == (arrays[a].astype(dtype) @ arrays[b].astype(dtype)).tobytes()
            assert result["launch_count"] == 2
    finally:
        release = session.close()
    assert release.release_confirmed


@pytest.mark.parametrize("tasklets", [3, 7, 8, 12, 24])
@pytest.mark.parametrize("mode", [0, 1])
def test_persistent_host_launches_disjoint_operations_and_partial_waves(tmp_path, binaries, tasklets, mode):
    data, cases = corpus(binaries[tasklets][1], tasklets, mode)
    operations, waves = unpack_wave_envelope(data)
    encoded = pack_wave_envelope(
        plan_sha256=hashlib.sha256(b"plan").digest(),
        dpu_binary_sha256=hashlib.sha256(binaries[tasklets][1].read_bytes()).digest(),
        sequence=1, operations=operations, waves=waves, numeric_mode=mode,
    )
    assert encoded == data
    later, later_cases = corpus(binaries[tasklets][1], tasklets, mode, sequence=2, request_start=12)
    result, events = invoke(tmp_path, binaries, encoded, tasklets=tasklets, second=later)
    assert result.returncode == 0, result.stderr
    assert [event["event"] for event in events] == ["READY", "WAVE_RESPONSE", "WAVE_RESPONSE", "RELEASE"]
    assert events[0]["abi"] == "wave_control_v5"
    assert events[0]["request_transport"] == "packed_wave_v1"
    for response, items in zip(events[1:3], (cases, later_cases)):
        assert response["status"] == "completed"
        assert response["launch_count"] == response["completed_wave_count"] == 2
        assert response["completed_result_count"] == 6
        assert response["allocated_dpu_count"] == 3
        assert response["tasklets_per_dpu"] == tasklets
        assert response["target_observed"] == "sdk_simulator"
        assert response["cpu_fallback_used"] is False
        assert response["envelope_bytes"] == response["native_snapshot_bytes"] == len(data)
        assert response["operation_count"] == 2 and response["control_count"] == 6
        assert response["native_output_buffer_bytes"] == 256 * 256 * 4
        assert response["h2d_bytes"] == sum(
            144 + sum(length for _, length in c.planes[:4]) for c, _ in items)
        assert response["d2h_bytes"] == sum(
            72 + sum(length for _, length in c.planes[4:]) for c, _ in items)
        verify_result(tmp_path, response, items)
    assert events[-1]["release_succeeded"] and events[-1]["dpu_free_called_once"]


@pytest.mark.parametrize("corruption", ["digest", "truncated", "binary", "tail", "replay", "group"])
def test_invalid_cohort_fails_closed_without_replacement(tmp_path, binaries, corruption):
    data, _ = corpus(binaries[8][1])
    second = None
    if corruption == "truncated":
        data = data[:-1]
    if corruption == "tail":
        data += b"\0"
    if corruption == "binary":
        data = data[:104] + b"\1" * 32 + data[136:]
    if corruption == "replay":
        second = data
    if corruption == "group":
        # Ownership is checked before interpreting the other operation's geometry.
        offset = 136 + 2 * 112 + 3 * 160 + 16 + 7 * 4
        data = data[:offset] + struct.pack("<I", 1) + data[offset + 4:]
    result, events = invoke(tmp_path, binaries, data, second=second, bad_digest=corruption == "digest")
    assert result.returncode != 0
    responses = [event for event in events if event["event"] == "WAVE_RESPONSE"]
    assert responses[-1]["status"] == "failed"
    assert responses[-1]["launch_count"] == 0
    assert responses[-1]["completed_wave_count"] == 0
    assert events[-1]["event"] == "RELEASE" and events[-1]["release_succeeded"]
    assert len(responses) == (2 if corruption == "replay" else 1)
    if corruption == "group":
        assert "group changes ownership" in responses[-1]["error"]


@pytest.mark.parametrize("wrong", ["legacy", "tasklets"])
def test_wave_host_rejects_wrong_binary_before_ready(tmp_path, binaries, wrong):
    data, _ = corpus(binaries[8][1])
    binary = NATIVE / "bin/dpu_gemm_tile_v4_t8" if wrong == "legacy" else binaries[3][1]
    result, events = invoke(tmp_path, binaries, data, dpu_binary=binary)
    assert result.returncode != 0
    assert not any(event["event"] == "READY" for event in events)
    assert events[0]["failure_stage"] == "tasklet_binary_mismatch"
    assert events[-1]["release_succeeded"]


@pytest.mark.parametrize("client", [False, True])
def test_partial_cohort_failure_preserves_prefix_and_stops_session(tmp_path, binaries, client):
    # Inject a reported second-wave failure in a test-only DPU binary.
    source = (NATIVE / "dpu_wave.c").read_text()
    anchor = "WAVE_COMPLETION.status = UPMEM_WAVE_COMPLETED;"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + """
        if (WAVE_CONTROL.wave_id == 1u && WAVE_CONTROL.dpu_id == 0u) {
            WAVE_COMPLETION.status = UPMEM_WAVE_FAILED;
            WAVE_COMPLETION.failure_stage = UPMEM_WAVE_FAILURE_EXECUTION;
            WAVE_COMPLETION.failing_product = 0u;
            WAVE_COMPLETION.completed_product_mask = 0u;
            WAVE_COMPLETION.processed_elements = 0u;
        }
""")
    fixture = tmp_path / "failure.c"
    fixture.write_text(source)
    binary = tmp_path / "failure-dpu"
    subprocess.run(["dpu-upmem-dpurte-clang", "-O2", "-DNR_TASKLETS=8", "-I", str(NATIVE),
                    "-o", str(binary), str(fixture)], capture_output=True, check=True)
    data, cases = corpus(binary)
    later, _ = corpus(binary, sequence=2, request_start=12)
    if client:
        from quantum_bench.upmem.native_session import V4Session
        from quantum_bench.upmem.protocol import (
            V4Error, V4Profile, REQUEST_TRANSPORT_PACKED_WAVE,
        )
        host, _, init = binaries[8]
        profile = V4Profile(dpu_count=3, tasklets_per_dpu=8,
                            execution_target="sdk_simulator", timeout_s=30,
                            request_transport=REQUEST_TRANSPORT_PACKED_WAVE)
        session = V4Session.start(
            (host, "--target", "simulator", "--session-root", tmp_path,
             "--dpus", "3", "--tasklets", "8", "--initialization-binary", init,
             "--dpu-binary", binary, "--timeout-s", "30"),
            session_root=tmp_path, profile=profile,
        )
        operations, waves = unpack_wave_envelope(data)
        next_event = session._next_event

        def after_native_exit(timeout_s):
            # Exercise RESPONSE, RELEASE and EOF arriving before the caller reads.
            session.process.wait(timeout=30)
            session._stdout_pump.thread.join(timeout=30)
            return next_event(timeout_s)

        session._next_event = after_native_exit
        with pytest.raises(V4Error) as caught:
            session.submit_waves(
                plan_sha256=hashlib.sha256(b"plan").digest(),
                dpu_binary_sha256=hashlib.sha256(binary.read_bytes()).digest(),
                sequence=1, operations=operations, waves=waves,
            )
        response = caught.value.wave_response
        release = session.close()
        assert session.process.poll() is not None
        assert release.event["dpu_free_called_once"] and release.event["release_succeeded"]
    else:
        result, events = invoke(tmp_path, binaries, data, second=later, dpu_binary=binary)
        assert result.returncode != 0
        assert [event["event"] for event in events] == ["READY", "WAVE_RESPONSE", "RELEASE"]
        response = events[1]
        assert events[-1]["release_succeeded"] and events[-1]["dpu_free_called_once"]
    assert response["status"] == "failed"
    assert response["failure_stage"] == "wave_completion_failed"
    assert response["completed_wave_count"] == 1
    assert response["completed_result_count"] == 3
    assert response["launch_count"] == 2
    assert response["failed_wave_index"] == 1
    assert response["failed_dpu_id"] == response["failed_operation_index"] == 0
    assert response["failed_completion_mask"] == response["failed_product"] == 0
    assert response["failed_completion_status"] == response["failed_completion_stage"] == 2
    verify_result(tmp_path, response, cases[:3])


@pytest.mark.parametrize("kind", ["fifo", "symlink", "oversized"])
def test_file_admission_rejects_nonregular_or_oversized_inputs(tmp_path, binaries, kind):
    data, _ = corpus(binaries[8][1])
    result, events = invoke(tmp_path, binaries, data, input_kind=kind)
    assert result.returncode != 0
    assert [event["event"] for event in events] == ["READY", "WAVE_RESPONSE", "RELEASE"]
    assert events[1]["status"] == "failed" and events[1]["launch_count"] == 0
    assert events[1]["response_path"] is None
    assert events[-1]["release_succeeded"]
