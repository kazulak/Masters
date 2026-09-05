"""SDK-simulator coverage for the bounded K=1 outer-product sidecar."""

from dataclasses import replace
import struct

import numpy as np
import pytest

from quantum_bench.upmem.wave_protocol import (
    CONTROL,
    FOUR_PRODUCT_OUTER,
    FOUR_PRODUCT_PANEL,
    REAL_OUTER,
    REAL_PANEL,
    WaveCompletion,
)
from tests import test_upmem_wave_kernel_simulator as wave_helpers
from tests.test_upmem_wave_kernel_simulator import (
    COMPLETION,
    arena_for,
    control,
    expected_product,
    run,
    values,
)


sdk_fixture = wave_helpers.sdk
_SENTINEL = b"\xa5"
_TASKLET_CASES = (
    pytest.param(1, 3, 5, 0, "real", "random", id="t1-float-real-odd"),
    pytest.param(3, 7, 33, 1, "fused", "random", id="t3-int8-fused-tail"),
    pytest.param(8, 3, 5, 1, "fused", "extrema", id="t8-int8-fused-extrema"),
    pytest.param(12, 256, 256, 0, "real", "random", id="t12-float-real-256"),
    pytest.param(24, 127, 256, 1, "fused", "extrema", id="t24-int8-fused-edge"),
)


def _extrema_arrays(m: int, n: int) -> tuple[np.ndarray, ...]:
    def tiled(pattern: tuple[int, ...], shape: tuple[int, int]) -> np.ndarray:
        return np.resize(np.asarray(pattern, dtype=np.int8), shape)

    return (
        tiled((-128, 127, -1, 0), (m, 1)),
        tiled((127, -128, 1, 2), (m, 1)),
        tiled((-128, 127, -1, 0, 1), (1, n)),
        tiled((1, -2, 127, -128, 0), (1, n)),
    )


def _signed_zero_arrays() -> tuple[np.ndarray, ...]:
    a_real = np.asarray([[-0.0], [0.0], [1.0]], dtype="<f4")
    a_imag = np.asarray([[1.0], [-1.0], [0.0]], dtype="<f4")
    b_real = np.asarray([[1.0, -1.0, 0.0, -0.0, 2.0]], dtype="<f4")
    b_imag = np.asarray([[2.0, 0.0, -1.0, 1.0, -0.0]], dtype="<f4")
    return a_real, a_imag, b_real, b_imag


def _arrays(m: int, n: int, mode: int, pattern: str) -> tuple[np.ndarray, ...]:
    if pattern == "random":
        return values(m, n, 1, mode)
    if pattern == "extrema":
        return _extrema_arrays(m, n)
    if pattern == "signedzero":
        return _signed_zero_arrays()
    raise AssertionError(f"unknown test pattern: {pattern}")


def _kernel_pair(kind: str) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    if kind == "real":
        return REAL_OUTER, REAL_PANEL, ((0, 2),)
    return FOUR_PRODUCT_OUTER, FOUR_PRODUCT_PANEL, ((0, 2), (1, 3), (0, 3), (1, 2))


def _item(tasklets: int, m: int, n: int, mode: int, kernel: int, sequence: int):
    return replace(
        control(m, n, 1, mode, tasklets, kernel=kernel),
        request_sequence=sequence,
    )


def _assert_success(completion: tuple[int, ...], item) -> None:
    WaveCompletion.from_bytes(
        COMPLETION.pack(*completion), item, require_success=True
    )


def _expected_arena(item, before: bytes, arrays: tuple[np.ndarray, ...]) -> bytes:
    expected = bytearray(before)
    kind = "real" if item.kernel in (REAL_OUTER, REAL_PANEL) else "fused"
    _, _, pair_indexes = _kernel_pair(kind)
    for product, (a_index, b_index) in enumerate(pair_indexes):
        offset, length = item.planes[4 + product]
        payload = expected_product(
            arrays[a_index], arrays[b_index], item.numeric_mode
        ).tobytes()
        assert len(payload) == item.m * item.n * 4
        assert len(payload) <= length
        expected[offset : offset + len(payload)] = payload
        assert before[offset + len(payload) : offset + length] == _SENTINEL * (
            length - len(payload)
        )
    return bytes(expected)


def _assert_output_bytes(
    item,
    before: bytes,
    after: bytes,
    arrays: tuple[np.ndarray, ...],
    completion: tuple[int, ...],
) -> None:
    _assert_success(completion, item)
    assert after == _expected_arena(item, before, arrays)


@pytest.mark.parametrize(
    "tasklets,m,n,mode,kind,pattern",
    _TASKLET_CASES,
)
def test_outer_matches_panel_exact_bytes_and_preserves_padding(
    sdk_fixture, tasklets, m, n, mode, kind, pattern
):
    outer_kernel, panel_kernel, _ = _kernel_pair(kind)
    arrays = _arrays(m, n, mode, pattern)
    outer = _item(tasklets, m, n, mode, outer_kernel, 101)
    panel = _item(tasklets, m, n, mode, panel_kernel, 102)
    outer_before = arena_for(outer, arrays)
    panel_before = arena_for(panel, arrays)
    assert outer_before == panel_before

    results = run(
        sdk_fixture,
        tasklets,
        [
            (outer.to_bytes(), outer_before),
            (panel.to_bytes(), panel_before),
        ],
    )
    (outer_completion, outer_after), (panel_completion, panel_after) = results
    _assert_output_bytes(outer, outer_before, outer_after, arrays, outer_completion)
    _assert_output_bytes(panel, panel_before, panel_after, arrays, panel_completion)
    assert outer_after == panel_after


def test_t7_float32_signed_zero_bytes_match_panel_and_zero_then_add_semantics(
    sdk_fixture,
):
    arrays = _signed_zero_arrays()
    outer = _item(7, 3, 5, 0, REAL_OUTER, 201)
    panel = _item(7, 3, 5, 0, REAL_PANEL, 202)
    before = arena_for(outer, arrays)
    results = run(
        sdk_fixture,
        7,
        [(outer.to_bytes(), before), (panel.to_bytes(), before)],
    )
    (outer_completion, outer_after), (panel_completion, panel_after) = results
    _assert_output_bytes(outer, before, outer_after, arrays, outer_completion)
    _assert_output_bytes(panel, before, panel_after, arrays, panel_completion)
    assert outer_after == panel_after

    offset, _ = outer.planes[4]
    bits = np.frombuffer(outer_after[offset : offset + 3 * 5 * 4], dtype="<u4")
    assert np.all(bits[:10] == 0)
    assert struct.unpack_from("<f", outer_after, offset + 2 * 5 * 4)[0] == 1.0


def test_repeated_mixed_outer_and_panel_launches_reuse_staging_safely(sdk_fixture):
    tasklets = 8
    cases = (
        ("real", 5, 7, 0, "random"),
        ("fused", 7, 9, 1, "extrema"),
        ("real", 1, 33, 1, "extrema"),
        ("fused", 3, 5, 0, "random"),
    )
    requests = []
    expected = []
    for index, (kind, m, n, mode, pattern) in enumerate(cases):
        outer_kernel, panel_kernel, _ = _kernel_pair(kind)
        arrays = _arrays(m, n, mode, pattern)
        outer = _item(tasklets, m, n, mode, outer_kernel, 300 + 2 * index)
        panel = _item(tasklets, m, n, mode, panel_kernel, 301 + 2 * index)
        outer_before = arena_for(outer, arrays)
        panel_before = arena_for(panel, arrays)
        requests.extend(
            [
                (outer.to_bytes(), outer_before),
                (panel.to_bytes(), panel_before),
            ]
        )
        expected.extend(
            [
                (outer, outer_before, arrays),
                (panel, panel_before, arrays),
            ]
        )

    results = run(sdk_fixture, tasklets, requests)
    assert len(results) == len(expected)
    for (completion, after), (item, before, arrays) in zip(results, expected):
        _assert_output_bytes(item, before, after, arrays, completion)
    for index in range(len(cases)):
        assert results[2 * index][1] == results[2 * index + 1][1]


@pytest.mark.parametrize("kernel", [REAL_OUTER, FOUR_PRODUCT_OUTER])
def test_outer_rejects_k_other_than_one_before_mram_access(sdk_fixture, kernel):
    tasklets, m, n, mode = 8, 3, 5, 1
    arrays = _extrema_arrays(m, n)
    item = _item(tasklets, m, n, mode, kernel, 401)
    before = arena_for(item, arrays)
    fields = list(CONTROL.unpack(item.to_bytes()))
    fields[14] = 2

    [(completion, after)] = run(
        sdk_fixture,
        tasklets,
        [(CONTROL.pack(*fields), before)],
    )
    assert after == before
    assert completion[2] == 2
    assert completion[5] == 0
    assert completion[10:] == (0, 1, (1 << 32) - 1)
