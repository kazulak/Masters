"""One-launch products use the same panel arithmetic as the accepted real kernel."""

from dataclasses import replace
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

import numpy as np
import pytest

from quantum_bench.upmem.wave_protocol import (
    CONTROL, FOUR_PRODUCT_PANEL, IDLE, MRAM_BYTES, NO_OPERATION, REAL_PANEL,
    WaveCompletion, WaveControl, product_layout,
)

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/upmem/runtime"
COMPLETION = struct.Struct("<6I5Q2I")


@pytest.fixture(scope="module")
def sdk(tmp_path_factory):
    missing = [tool for tool in ("cc", "dpu-pkg-config", "dpu-upmem-dpurte-clang")
               if not shutil.which(tool)]
    if missing:
        message = "wave SDK prerequisites missing: " + ", ".join(missing)
        if os.environ.get("UPMEM_REQUIRE_SDK_SIMULATOR") == "1":
            pytest.fail(message)
        pytest.skip(message)
    directory = tmp_path_factory.mktemp("wave-sdk")
    flags = subprocess.run(["dpu-pkg-config", "--cflags", "--libs", "dpu"],
                           capture_output=True, text=True, check=True).stdout
    host = directory / "wave-probe"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
                    "-I", str(NATIVE), str(ROOT / "tests/native/upmem_wave_probe.c"),
                    "-o", str(host), *shlex.split(flags)], check=True, capture_output=True)
    binaries = {}
    for tasklets in range(1, 25):
        binary = directory / f"wave-t{tasklets}"
        subprocess.run(["dpu-upmem-dpurte-clang", "-O2", f"-DNR_TASKLETS={tasklets}",
                        "-o", str(binary), str(NATIVE / "dpu_wave.c")],
                       check=True, capture_output=True)
        binaries[tasklets] = binary
    return host, binaries


def test_all_tasklet_builds(sdk):
    assert set(sdk[1]) == set(range(1, 25))
    assert all(path.is_file() and path.stat().st_size > 0 for path in sdk[1].values())


def control(m, n, k, mode, tasklets, *, kernel=FOUR_PRODUCT_PANEL):
    return WaveControl(0, tasklets, 0, mode, kernel, 2, 11, 13, 17, 0,
                       m, n, k, 0, product_layout(m, n, k, numeric_mode=mode,
                                                 kernel=kernel))


def arena_for(item, arrays):
    arena = bytearray(b"\xa5" * MRAM_BYTES)
    for (offset, length), array in zip(item.planes[:4], arrays):
        if length:
            data = array.tobytes()
            arena[offset:offset + len(data)] = data
    return bytes(arena)


def run(sdk, tasklets, requests):
    host, binaries = sdk
    result = subprocess.run([str(host), str(binaries[tasklets])],
                            input=b"".join(c + a for c, a in requests),
                            capture_output=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    size = COMPLETION.size + MRAM_BYTES
    assert len(result.stdout) == len(requests) * size
    return [(COMPLETION.unpack_from(result.stdout, i * size),
             result.stdout[i * size + COMPLETION.size:(i + 1) * size])
            for i in range(len(requests))]


def values(m, n, k, mode):
    rng = np.random.default_rng(171)
    if mode:
        return tuple(rng.integers(-127, 128, size=shape, dtype=np.int8)
                     for shape in ((m, k), (m, k), (k, n), (k, n)))
    return tuple(rng.uniform(-2, 2, size=shape).astype(np.float32)
                 for shape in ((m, k), (m, k), (k, n), (k, n)))


def expected_product(a, b, mode):
    if mode:
        return (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for k in range(a.shape[1]):
        out += a[:, k:k + 1] * b[k:k + 1, :]
    return out


@pytest.mark.parametrize("tasklets,m,n,k", [
    (1, 1, 1, 1), (3, 2, 3, 65), (7, 8, 35, 67),
    (8, 3, 32, 130), (12, 17, 1, 257), (24, 3, 31, 65),
])
@pytest.mark.parametrize("mode", [0, 1])
def test_fused_products_match_four_launches_and_sequential_replay(sdk, tasklets, m, n, k, mode):
    item = control(m, n, k, mode, tasklets)
    # Noncanonical gaps prove the kernel honors validated explicit offsets.
    item = replace(item, planes=tuple((offset + 8 * (i + 1), length)
                                     for i, (offset, length) in enumerate(item.planes)))
    arrays = values(m, n, k, mode)
    fused_input = arena_for(item, arrays)
    requests = [(item.to_bytes(), fused_input)]
    generic = control(m, n, k, mode, tasklets, kernel=REAL_PANEL)
    products = ((0, 2), (1, 3), (0, 3), (1, 2))
    for i, (a, b) in enumerate(products):
        real = replace(generic, request_sequence=20 + i)
        requests.append((real.to_bytes(), arena_for(real, (arrays[a], arrays[a],
                                                         arrays[b], arrays[b]))))
    results = run(sdk, tasklets, requests)
    completion, fused = results[0]
    WaveCompletion.from_bytes(COMPLETION.pack(*completion), item, require_success=True)
    assert completion[:9] == (0x35574350, 5, 1, 0, 2, 15, 11, 13, 17)
    assert completion[10:] == (4 * m * n, 0, NO_OPERATION)
    touched = bytearray(MRAM_BYTES)
    for i, (a, b) in enumerate(products):
        offset, _ = item.planes[4 + i]
        size = m * n * 4
        output = fused[offset:offset + size]
        generic_offset, _ = generic.planes[4]
        assert output == results[1 + i][1][generic_offset:generic_offset + size]
        assert output == expected_product(arrays[a], arrays[b], mode).tobytes()
        touched[offset:offset + size] = b"\1" * size
    assert all(before == after for i, (before, after) in
               enumerate(zip(fused_input, fused)) if not touched[i])


def test_idle_invalid_and_repeated_launches_do_not_leak_outputs(sdk):
    item = control(2, 3, 65, 0, 8)
    idle = replace(item, flags=IDLE, kernel=0, operation_index=NO_OPERATION,
                   tile_id=0, m=0, n=0, k=0, planes=((0, 0),) * 8)
    bad = list(CONTROL.unpack(item.to_bytes()))
    bad[16] = 1
    arena = arena_for(item, values(2, 3, 65, 0))
    results = run(sdk, 8, [(item.to_bytes(), arena), (idle.to_bytes(), arena),
                           (CONTROL.pack(*bad), arena), (item.to_bytes(), arena)])
    assert results[0][1] == results[3][1]
    assert results[1][1] == results[2][1] == arena
    assert results[1][0][2:6] == (1, 0, NO_OPERATION, 0)
    assert results[1][0][10:] == (0, 0, NO_OPERATION)
    assert results[2][0][2] == 2
    assert results[2][0][5] == 0
    assert results[2][0][10:] == (0, 1, NO_OPERATION)


@pytest.mark.parametrize("index,value", [
    (0, 0), (1, 4), (2, 64), (3, 7), (4, 2), (5, 2), (6, 99),
    (7, 64), (12, 0), (14, 65537), (15, 65536), (16, 1),
    (17, (1 << 32) - 8), (18, (1 << 32) - 8), (19, 0),
])
def test_dpu_rejects_corrupt_control_before_mram_access(sdk, index, value):
    item = control(2, 3, 65, 0, 8)
    fields = list(CONTROL.unpack(item.to_bytes()))
    fields[index] = value
    arena = arena_for(item, values(2, 3, 65, 0))
    [(completion, after)] = run(sdk, 8, [(CONTROL.pack(*fields), arena)])
    assert after == arena
    assert completion[2] == 2
    assert completion[5] == 0
    assert completion[10:] == (0, 1, NO_OPERATION)
