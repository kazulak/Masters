"""Focused SDK-simulator checks for the test-only scalar-MRAM wave binary."""

import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess

import numpy as np
import pytest

from quantum_bench.upmem.wave_protocol import (
    CONTROL,
    COMPLETION_MAGIC,
    FOUR_PRODUCT_PANEL,
    IDLE,
    MRAM_BYTES,
    NO_OPERATION,
    NO_PRODUCT,
    REAL_PANEL,
    WaveCompletion,
    WaveControl,
    product_layout,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/upmem/runtime"
TEST_NATIVE = ROOT / "tests/native"
COMPLETION = struct.Struct("<6I5Q2I")


@pytest.fixture(scope="module")
def sdk(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    missing = [
        name for name in ("cc", "dpu-pkg-config", "dpu-upmem-dpurte-clang")
        if shutil.which(name) is None
    ]
    if missing:
        reason = "scalar reference SDK prerequisites missing: " + ", ".join(missing)
        if os.environ.get("UPMEM_REQUIRE_SDK_SIMULATOR") == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    directory = tmp_path_factory.mktemp("scalar-reference-sdk")
    flags = subprocess.run(
        ["dpu-pkg-config", "--cflags", "--libs", "dpu"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    host = directory / "wave-probe"
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-I",
            str(NATIVE),
            str(TEST_NATIVE / "upmem_wave_probe.c"),
            "-o",
            str(host),
            *shlex.split(flags),
        ],
        check=True,
        capture_output=True,
    )

    scalar = directory / "scalar-reference-t8"
    subprocess.run(
        [
            "dpu-upmem-dpurte-clang",
            "-O2",
            "-DNR_TASKLETS=8",
            "-I",
            str(NATIVE),
            "-o",
            str(scalar),
            str(TEST_NATIVE / "upmem_scalar_reference_dpu.c"),
        ],
        check=True,
        capture_output=True,
    )
    panel = directory / "wram-panel-t8"
    subprocess.run(
        [
            "dpu-upmem-dpurte-clang",
            "-O2",
            "-DNR_TASKLETS=8",
            "-o",
            str(panel),
            str(NATIVE / "dpu_wave.c"),
        ],
        check=True,
        capture_output=True,
    )
    return host, scalar, panel


def _control(
    m: int,
    n: int,
    k: int,
    numeric_mode: int,
    *,
    kernel: int = FOUR_PRODUCT_PANEL,
    request_sequence: int = 13,
    tile_id: int = 17,
) -> WaveControl:
    return WaveControl(
        dpu_id=0,
        tasklets=8,
        flags=0,
        numeric_mode=numeric_mode,
        kernel=kernel,
        operation_index=2,
        wave_id=11,
        request_sequence=request_sequence,
        tile_id=tile_id,
        batch_index=0,
        m=m,
        n=n,
        k=k,
        k_offset=0,
        planes=product_layout(m, n, k, numeric_mode=numeric_mode, kernel=kernel),
    )


def _idle(*, request_sequence: int = 13, wave_id: int = 11) -> WaveControl:
    return WaveControl(
        dpu_id=0,
        tasklets=8,
        flags=IDLE,
        numeric_mode=0,
        kernel=0,
        operation_index=NO_OPERATION,
        wave_id=wave_id,
        request_sequence=request_sequence,
        tile_id=0,
        batch_index=0,
        m=0,
        n=0,
        k=0,
        k_offset=0,
        planes=((0, 0),) * 8,
    )


def _arrays(m: int, n: int, k: int, numeric_mode: int, *, zero: bool = False):
    shapes = ((m, k), (m, k), (k, n), (k, n))
    dtype = np.int8 if numeric_mode else np.float32
    if zero:
        return tuple(np.zeros(shape, dtype=dtype) for shape in shapes)
    if numeric_mode:
        return (
            ((np.arange(m * k, dtype=np.int32) * 3 % 17) - 8)
            .astype(np.int8)
            .reshape(m, k),
            ((np.arange(m * k, dtype=np.int32) * 5 % 19) - 9)
            .astype(np.int8)
            .reshape(m, k),
            ((np.arange(k * n, dtype=np.int32) * 7 % 23) - 11)
            .astype(np.int8)
            .reshape(k, n),
            ((np.arange(k * n, dtype=np.int32) * 11 % 29) - 14)
            .astype(np.int8)
            .reshape(k, n),
        )
    values = (
        ((np.arange(m * k, dtype=np.int32) * 3 % 17) - 8)
        .astype(np.float32)
        .reshape(m, k),
        ((np.arange(m * k, dtype=np.int32) * 5 % 19) - 9)
        .astype(np.float32)
        .reshape(m, k),
        ((np.arange(k * n, dtype=np.int32) * 7 % 23) - 11)
        .astype(np.float32)
        .reshape(k, n),
        ((np.arange(k * n, dtype=np.int32) * 11 % 29) - 14)
        .astype(np.float32)
        .reshape(k, n),
    )
    values[0].flat[0] = np.float32(-0.0)
    values[1].flat[0] = np.float32(-1.25)
    values[2].flat[0] = np.float32(0.0)
    values[3].flat[0] = np.float32(1.5)
    return values


def _arena(control: WaveControl, arrays: tuple[np.ndarray, ...]) -> bytes:
    result = bytearray(b"\xa5" * MRAM_BYTES)
    for (offset, length), array in zip(control.planes[:4], arrays):
        if length == 0:
            continue
        payload = np.ascontiguousarray(array).tobytes()
        assert len(payload) <= length
        result[offset:offset + len(payload)] = payload
    return bytes(result)


def _request(control: WaveControl, arrays: tuple[np.ndarray, ...]) -> tuple[bytes, bytes]:
    return control.to_bytes(), _arena(control, arrays)


def _run(
    sdk: tuple[Path, Path, Path], binary: Path,
    requests: tuple[tuple[bytes, bytes], ...],
) -> list[tuple[tuple[int, ...], bytes]]:
    host, _scalar, _panel = sdk
    result = subprocess.run(
        [str(host), str(binary)],
        input=b"".join(control + arena for control, arena in requests),
        capture_output=True,
        timeout=120,
        check=False,
        env={**os.environ, "DPU_BACKEND": "simulator"},
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    record_bytes = COMPLETION.size + MRAM_BYTES
    assert len(result.stdout) == len(requests) * record_bytes
    return [
        (
            COMPLETION.unpack_from(result.stdout, index * record_bytes),
            result.stdout[
                index * record_bytes + COMPLETION.size:(index + 1) * record_bytes
            ],
        )
        for index in range(len(requests))
    ]


def _expected(a: np.ndarray, b: np.ndarray, numeric_mode: int) -> np.ndarray:
    dtype = np.int32 if numeric_mode else np.float32
    result = np.zeros((a.shape[0], b.shape[1]), dtype=dtype)
    for k in range(a.shape[1]):
        result += a[:, k:k + 1].astype(dtype) * b[k:k + 1, :].astype(dtype)
    return result


def _products(control: WaveControl, arena: bytes, numeric_mode: int):
    dtype = np.dtype("<i4" if numeric_mode else "<f4")
    for index in range(4 if control.kernel != REAL_PANEL else 1):
        offset, length = control.planes[4 + index]
        assert length >= control.m * control.n * 4
        payload = arena[offset:offset + control.m * control.n * 4]
        yield np.frombuffer(payload, dtype=dtype).reshape(control.m, control.n)


def _assert_matches_panel(
    sdk: tuple[Path, Path, Path],
    requests: tuple[tuple[bytes, bytes], ...],
    controls: tuple[WaveControl, ...],
    arrays: tuple[tuple[np.ndarray, ...] | None, ...],
) -> None:
    scalar = _run(sdk, sdk[1], requests)
    panel = _run(sdk, sdk[2], requests)
    assert len(scalar) == len(panel) == len(controls)
    for control, inputs, scalar_item, panel_item in zip(
        controls, arrays, scalar, panel, strict=True
    ):
        scalar_completion, scalar_arena = scalar_item
        panel_completion, panel_arena = panel_item
        assert scalar_completion[:9] == panel_completion[:9]
        assert scalar_completion[10:] == panel_completion[10:]
        assert scalar_arena == panel_arena
        completion = WaveCompletion.from_bytes(
            COMPLETION.pack(*scalar_completion), control, require_success=True
        )
        assert completion.cycles >= 0
        if inputs is None:
            continue
        pairs = ((0, 2),) if control.kernel == REAL_PANEL else (
            (0, 2), (1, 3), (0, 3), (1, 2)
        )
        for actual, (left, right) in zip(
            _products(control, scalar_arena, control.numeric_mode), pairs, strict=True
        ):
            np.testing.assert_array_equal(
                actual, _expected(inputs[left], inputs[right], control.numeric_mode)
            )


def test_scalar_source_is_mram_reference_with_uniform_barriers() -> None:
    source = (TEST_NATIVE / "upmem_scalar_reference_dpu.c").read_text(encoding="ascii")
    assert '#include "panel_compute.h"' not in source
    assert "shared_b_panel" not in source
    assert "mram_read(" in source
    assert "mram_write_unaligned" in source
    assert source.count("barrier_wait(&v4_barrier);") == 4
    assert "WAVE_COMPLETION.failure_stage = UPMEM_WAVE_FAILURE_VALIDATION;" in source
    assert "WAVE_CONTROL.flags != UPMEM_WAVE_IDLE" in source


def test_scalar_reference_matches_wram_panel_at_physical_ablation_shape(sdk) -> None:
    control = _control(32, 32, 32, 0, request_sequence=32, tile_id=32)
    arrays = _arrays(32, 32, 32, 0)
    _assert_matches_panel(
        sdk,
        (_request(control, arrays),),
        (control,),
        (arrays,),
    )


@pytest.mark.parametrize("kernel", [REAL_PANEL, FOUR_PRODUCT_PANEL])
def test_scalar_reference_matches_panel_for_signed_float_tails_and_product_order(
    sdk, kernel: int
) -> None:
    shape = (3, 5, 7) if kernel == REAL_PANEL else (3, 35, 65)
    control = _control(*shape, 0, kernel=kernel)
    arrays = _arrays(*shape, 0)
    _assert_matches_panel(sdk, (_request(control, arrays),), (control,), (arrays,))


def test_scalar_reference_supports_direct_int8_small_reuse(sdk) -> None:
    control = _control(3, 5, 7, 1)
    arrays = _arrays(3, 5, 7, 1)
    _assert_matches_panel(sdk, (_request(control, arrays),), (control,), (arrays,))


def test_scalar_reference_preserves_zero_idle_and_validation_contract(sdk) -> None:
    zero = _control(3, 5, 7, 0, request_sequence=21, tile_id=21)
    idle = _idle(request_sequence=22, wave_id=12)
    zero_arrays = _arrays(3, 5, 7, 0, zero=True)
    idle_arena = bytes(b"\x5a" * MRAM_BYTES)
    idle_request = (idle.to_bytes(), idle_arena)
    requests = (_request(zero, zero_arrays), idle_request)
    _assert_matches_panel(
        sdk, requests, (zero, idle), (zero_arrays, None)
    )

    invalid = list(CONTROL.unpack(zero.to_bytes()))
    invalid[16] = 1
    invalid_request = (CONTROL.pack(*invalid), _arena(zero, zero_arrays))
    scalar = _run(sdk, sdk[1], (invalid_request,))[0]
    panel = _run(sdk, sdk[2], (invalid_request,))[0]
    assert scalar[0][:9] == panel[0][:9]
    assert scalar[0][2] == 2
    assert scalar[0][5] == 0
    assert scalar[0][10:] == (0, 1, NO_PRODUCT)
    assert scalar[1] == panel[1] == invalid_request[1]
    assert scalar[0][0] == COMPLETION_MAGIC
