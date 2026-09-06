"""Bounded SDK-simulator coverage for the test-only resident pair probe."""
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import numpy as np
import pytest
from quantum_bench.numerics import decode_complex_products
from quantum_bench.upmem.wave_protocol import (
    COMPLETION, FOUR_PRODUCT_PANEL, MRAM_BYTES, WaveCompletion, WaveControl,
    product_layout,
)
ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/upmem/runtime"
TEST_NATIVE = ROOT / "tests/native"
GUARD = b"\xa5"
@pytest.fixture(scope="module")
def sdk(tmp_path_factory):
    missing = [tool for tool in ("cc", "dpu-pkg-config", "dpu-upmem-dpurte-clang")
               if not shutil.which(tool)]
    if missing:
        message = "resident probe SDK prerequisites missing: " + ", ".join(missing)
        if os.environ.get("UPMEM_REQUIRE_SDK_SIMULATOR") == "1":
            pytest.fail(message)
        pytest.skip(message)
    directory = tmp_path_factory.mktemp("resident-sdk")
    flags = subprocess.run(["dpu-pkg-config", "--cflags", "--libs", "dpu"],
                           capture_output=True, text=True, check=True).stdout
    host = directory / "resident-probe"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                    "-I", str(NATIVE), "-I", str(TEST_NATIVE),
                    str(TEST_NATIVE / "upmem_resident_probe_host.c"),
                    "-o", str(host), *shlex.split(flags)], check=True,
                   capture_output=True)
    binaries = {}
    for tasklets in range(1, 25):
        binary = directory / f"resident-t{tasklets}"
        subprocess.run(["dpu-upmem-dpurte-clang", "-O2", f"-DNR_TASKLETS={tasklets}",
                        "-I", str(NATIVE), "-I", str(TEST_NATIVE), "-o", str(binary),
                        str(TEST_NATIVE / "upmem_resident_probe_dpu.c")],
                       check=True, capture_output=True)
        binaries[tasklets] = binary
    return host, binaries


def make_plan(first, second, side, pair_id, tasklets):
    first_planes = product_layout(*first, numeric_mode=0, kernel=FOUR_PRODUCT_PANEL)
    span = (first[0] * first[1] * 4 + 7) // 8 * 8
    cursor = sum(length for _, length in first_planes)
    retained = ((cursor, span), (cursor + span, span))
    cursor += 2 * span
    template = product_layout(*second, numeric_mode=0, kernel=FOUR_PRODUCT_PANEL)
    resident = (0, 1) if side == "left" else (2, 3)
    second_planes = []
    for index, (_, length) in enumerate(template):
        if index in resident:
            second_planes.append(retained[index - resident[0]])
        else:
            second_planes.append((cursor, length))
            cursor += length
    controls = tuple(
        WaveControl(0, tasklets, 0, 0, FOUR_PRODUCT_PANEL, index, index, pair_id,
                    index, 0, *shape, 0, planes=planes)
        for index, (shape, planes) in enumerate(((first, first_planes),
                                                   (second, tuple(second_planes))))
    )
    plan = (struct.pack("<IIQ", 1, side == "right", pair_id) +
            b"".join(control.to_bytes() for control in controls) +
            struct.pack("<4I", *(value for span_item in retained for value in span_item)))
    assert len(plan) == 320
    return {"bytes": plan, "controls": controls, "retained": retained,
            "second_planes": tuple(second_planes), "live": cursor, "side": side}


def expected_product(left, right):
    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.float32)
    for index in range(left.shape[1]):
        result += left[:, index:index + 1] * right[index:index + 1, :]
    return result


def four_products(values):
    return tuple(expected_product(values[left], values[right])
                 for left, right in ((0, 2), (1, 3), (0, 3), (1, 2)))


def case(first, second, side, tasklets, pair_id=1):
    info = make_plan(first, second, side, pair_id, tasklets)
    rng = np.random.default_rng(91 + tasklets)
    first_values = tuple(rng.uniform(-1.5, 1.5, size=shape).astype(np.float32)
                         for shape in ((first[0], first[2]), (first[0], first[2]),
                                       (first[2], first[1]), (first[2], first[1])))
    external_shape = ((second[2], second[1]) if side == "left"
                      else (second[0], second[2]))
    external = tuple(rng.uniform(-1.5, 1.5, size=external_shape).astype(np.float32)
                     for _ in range(2))
    arena = bytearray(GUARD * MRAM_BYTES)
    for span_item, values in zip(info["controls"][0].planes[:4], first_values):
        raw = values.tobytes()
        arena[span_item[0]:span_item[0] + len(raw)] = raw
    indexes = (2, 3) if side == "left" else (0, 1)
    for index, values in zip(indexes, external):
        span_item = info["controls"][1].planes[index]
        raw = values.tobytes()
        arena[span_item[0]:span_item[0] + len(raw)] = raw
    first_products = four_products(first_values)
    lanes = tuple(np.add(np.float32(0.0), value, dtype=np.float32)
                  for value in first_products)
    retained_value = decode_complex_products(
        lanes, 1.0, 1.0, "split_complex_float32_v1")
    retained_values = (np.asarray(retained_value.real, dtype=np.float32),
                       np.asarray(retained_value.imag, dtype=np.float32))
    resident_shape = ((second[0], second[2]) if side == "left"
                      else (second[2], second[1]))
    resident_values = tuple(value.reshape(resident_shape) for value in retained_values)
    second_values = ((resident_values[0], resident_values[1], external[0], external[1])
                     if side == "left" else
                     (external[0], external[1], resident_values[0], resident_values[1]))
    second_products = four_products(second_values)
    decode_complex_products(
        tuple(np.add(np.float32(0.0), value, dtype=np.float32) for value in second_products),
        1.0, 1.0, "split_complex_float32_v1")
    retained_patch = b"".join(
        values.tobytes() + GUARD * (span_item[1] - values.nbytes)
        for span_item, values in zip(info["retained"], retained_values))
    return {"info": info, "initial": bytes(arena), "first_products": first_products,
            "retained_values": retained_values, "second_products": second_products,
            "host_patch": (info["retained"][0][0], len(retained_patch), retained_patch)}


def _request_plan(command, plan_bytes, offset, payload):
    return (struct.pack("<I", command) + plan_bytes +
            struct.pack("<II", offset, len(payload)) + payload)


def request(command, item, offset, payload):
    return _request_plan(command, item["info"]["bytes"], offset, payload)


def run(sdk, tasklets, requests):
    host, binaries = sdk
    result = subprocess.run([str(host), str(binaries[tasklets])],
                            input=b"".join(requests), capture_output=True,
                            timeout=120, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    size = COMPLETION.size + MRAM_BYTES
    assert len(result.stdout) == len(requests) * size
    return [(result.stdout[index * size:index * size + COMPLETION.size],
             result.stdout[index * size + COMPLETION.size:(index + 1) * size])
            for index in range(len(requests))]


def assert_products(arena, spans, products):
    for span_item, values in zip(spans, products):
        raw = values.tobytes()
        assert arena[span_item[0]:span_item[0] + len(raw)] == raw
        assert arena[span_item[0] + len(raw):span_item[0] + span_item[1]] == \
            GUARD * (span_item[1] - len(raw))


def assert_success(record, control):
    WaveCompletion.from_bytes(record[0], control, require_success=True)


def assert_failure(record, stage):
    completion = WaveCompletion.from_bytes(record[0])
    assert completion.status == 2 and completion.completed_product_mask == 0
    assert completion.processed_elements == 0 and completion.failure_stage == stage


@pytest.mark.parametrize("tasklets,side,first,second,live", [
    (1, "left", (2, 4, 3), (2, 5, 4), 0),
    (3, "right", (3, 5, 5), (3, 5, 3), 0),
    (7, "left", (3, 3, 2), (3, 5, 3), 0),
    (8, "right", (2, 3, 4), (2, 3, 2), 0),
    (12, "left", (4, 3, 5), (4, 5, 3), 0),
    (24, "right", (16, 64, 4), (16, 256, 4), 93184),
])
def test_two_launch_resident_arms_match_exactly(sdk, tasklets, side, first, second, live):
    item = case(first, second, side, tasklets)
    info = item["info"]
    assert not live or info["live"] == live
    first_request = request(1, item, 0, item["initial"])
    patch_offset, patch_length, patch = item["host_patch"]
    host_second = request(2, item, patch_offset, patch)
    local_second = request(3, item, 0, b"")
    baseline = run(sdk, tasklets, (first_request, host_second))
    local = run(sdk, tasklets, (first_request, local_second))
    assert_success(baseline[0], info["controls"][0])
    assert_success(baseline[1], info["controls"][1])
    assert_success(local[0], info["controls"][0])
    assert_success(local[1], info["controls"][1])
    assert baseline[0][1] == local[0][1]
    assert baseline[1][1] == local[1][1]
    assert_products(baseline[0][1], info["controls"][0].planes[4:], item["first_products"])
    assert_products(baseline[1][1], info["retained"], item["retained_values"])
    assert_products(baseline[1][1], info["controls"][1].planes[4:], item["second_products"])


def test_empty_eof_is_valid_but_partial_and_bad_patch_frames_are_rejected(sdk):
    host, binaries = sdk
    binary = str(binaries[1])
    empty = subprocess.run([str(host), binary], input=b"", capture_output=True)
    assert empty.returncode == 0 and empty.stdout == b""
    item = case((2, 4, 3), (2, 5, 4), "left", 1)
    partial = subprocess.run([str(host), binary], input=b"\1\0", capture_output=True)
    assert partial.returncode != 0 and partial.stdout == b""
    bad = (struct.pack("<I", 1) + item["info"]["bytes"] +
           struct.pack("<II", MRAM_BYTES, 1) + b"x")
    result = subprocess.run([str(host), binary], input=bad, capture_output=True)
    assert result.returncode != 0 and result.stdout == b""


@pytest.mark.parametrize("kind", ["command", "second", "duplicate", "modified"])
def test_invalid_phase_or_plan_poisons_session(sdk, kind):
    item = case((2, 4, 3), (2, 5, 4), "left", 3)
    first = request(1, item, 0, item["initial"])
    if kind == "command":
        wires = (_request_plan(99, item["info"]["bytes"], 0, item["initial"]),
                 request(1, item, 0, b""))
    elif kind == "second":
        wires = (request(2, item, 0, item["initial"]), first)
    elif kind == "duplicate":
        wires = (first, request(1, item, 0, b""), request(2, item, 0, b""))
    else:
        modified = make_plan((2, 4, 3), (2, 5, 4), "left", 2, 3)["bytes"]
        wires = (first, _request_plan(2, modified, 0, b""), request(2, item, 0, b""))
    records = run(sdk, 3, wires)
    if kind in ("command", "second"):
        for record in records:
            assert_failure(record, 1)
        assert records[0][1] == records[1][1] == item["initial"]
    else:
        assert_success(records[0], item["info"]["controls"][0])
        assert_failure(records[1], 1)
        assert_failure(records[2], 1)
        assert records[1][1] == records[2][1]


@pytest.mark.parametrize("offset,value", [(100, 0), (104, 0), (36, 1), (0, 0), (4, 2), (8, 0), (304, 0xFFFFFFFF), (308, 0xFFFFFFFF)])
def test_corrupt_size_overlap_and_int8_plan_validation_poisons(sdk, offset, value):
    item = case((2, 4, 3), (2, 5, 4), "left", 1)
    bad = bytearray(item["info"]["bytes"])
    struct.pack_into("<I", bad, offset, value)
    records = run(sdk, 1, (_request_plan(1, bytes(bad), 0, item["initial"]),
                           request(1, item, 0, b"")))
    assert_failure(records[0], 1)
    assert_failure(records[1], 1)
    assert records[0][1] == records[1][1] == item["initial"]


@pytest.mark.parametrize("bits", [0x7F800000, 0x7FC00000])
def test_nonfinite_local_reconstruction_preserves_second_outputs_and_poisons(sdk, bits):
    item = case((3, 3, 2), (3, 5, 3), "left", 7)
    first = request(1, item, 0, item["initial"])
    source = item["info"]["controls"][0].planes[4][0]
    corrupt = struct.pack("<I", bits) + b"\0" * 4
    records = run(sdk, 7, (first, request(3, item, source, corrupt),
                           request(3, item, 0, b"")))
    assert_success(records[0], item["info"]["controls"][0])
    assert_failure(records[1], 2)
    for span_item in item["info"]["controls"][1].planes[4:]:
        assert records[1][1][span_item[0]:span_item[0] + span_item[1]] == \
            GUARD * span_item[1]
    assert_failure(records[2], 1)
    assert records[2][1] == records[1][1]


def test_repeated_pairs_accept_strictly_increasing_ids(sdk):
    first = case((2, 4, 3), (2, 5, 4), "left", 8, 1)
    second = case((2, 4, 3), (2, 5, 4), "left", 8, 2)
    patch_offset, _, patch = first["host_patch"]
    records = run(sdk, 8, (
        request(1, first, 0, first["initial"]),
        request(2, first, patch_offset, patch),
        request(1, second, 0, second["initial"]),
        request(3, second, 0, b""),
        request(1, second, 0, b""),
    ))
    assert_success(records[0], first["info"]["controls"][0])
    assert_success(records[1], first["info"]["controls"][1])
    assert_success(records[2], second["info"]["controls"][0])
    assert_success(records[3], second["info"]["controls"][1])
    assert_failure(records[4], 1)
    assert records[4][1] == records[3][1]
