from dataclasses import replace
import struct

import pytest

from quantum_bench.upmem.wave_protocol import (
    FAILURE_EXECUTION,
    FAILURE_NONE,
    FOUR_PRODUCT_PANEL,
    IDLE,
    NO_OPERATION,
    NO_PRODUCT,
    REAL_PANEL,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    WaveCompletion,
    WaveControl,
    product_layout,
)
from quantum_bench.upmem.packed_wave import WaveTile
from quantum_bench.upmem.wave_result import decode_wave_results


def _active(
    *,
    dpu_id: int,
    wave_id: int,
    request_sequence: int,
    tile_id: int,
    kernel: int = FOUR_PRODUCT_PANEL,
    numeric_mode: int = 0,
    m: int = 1,
    n: int = 2,
    k: int = 1,
) -> WaveControl:
    return WaveControl(
        dpu_id=dpu_id,
        tasklets=4,
        flags=0,
        numeric_mode=numeric_mode,
        kernel=kernel,
        operation_index=0,
        wave_id=wave_id,
        request_sequence=request_sequence,
        tile_id=tile_id,
        batch_index=0,
        m=m,
        n=n,
        k=k,
        k_offset=0,
        planes=product_layout(m, n, k, numeric_mode=numeric_mode, kernel=kernel),
    )


def _idle(*, dpu_id: int, wave_id: int, request_sequence: int) -> WaveControl:
    return WaveControl(
        dpu_id=dpu_id,
        tasklets=4,
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


def _tile(control: WaveControl) -> WaveTile:
    return WaveTile(control, 0, 0, (b"", b"", b"", b""))


def _completion(control: WaveControl) -> bytes:
    count = (
        0
        if control.flags == IDLE
        else 1
        if control.kernel == REAL_PANEL
        else 4
    )
    return WaveCompletion(
        status=STATUS_COMPLETED,
        dpu_id=control.dpu_id,
        operation_index=control.operation_index,
        completed_product_mask=(1 << count) - 1 if count else 0,
        wave_id=control.wave_id,
        request_sequence=control.request_sequence,
        tile_id=control.tile_id,
        cycles=17,
        processed_elements=control.m * control.n * count,
        failure_stage=FAILURE_NONE,
        failing_product=NO_PRODUCT,
    ).to_bytes(control, require_success=True)


def _stream(
    entries: tuple[tuple[WaveControl, tuple[bytes, ...]], ...],
) -> bytes:
    return b"".join(_completion(control) + b"".join(products)
                     for control, products in entries)


def test_fused_real_and_idle_slots_decode_in_dense_order() -> None:
    fused = _active(dpu_id=0, wave_id=10, request_sequence=20, tile_id=30)
    idle = _idle(dpu_id=1, wave_id=10, request_sequence=20)
    real = _active(
        dpu_id=0,
        wave_id=11,
        request_sequence=21,
        tile_id=31,
        kernel=REAL_PANEL,
    )
    later_idle = _idle(dpu_id=1, wave_id=11, request_sequence=21)
    fused_products = tuple(
        struct.pack("<2f", float(index + 1), float(-(index + 1)))
        for index in range(4)
    )
    real_product = struct.pack("<2f", 9.0, -9.0)
    waves = ((_tile(fused), _tile(idle)), (_tile(real), _tile(later_idle)))
    data = _stream(
        (
            (fused, fused_products),
            (idle, ()),
            (real, (real_product,)),
            (later_idle, ()),
        )
    )

    decoded = decode_wave_results(data, waves)

    assert len(decoded) == 2
    assert len(decoded[0]) == len(decoded[1]) == 2
    assert tuple(item.tobytes() for item in decoded[0][0]) == fused_products
    assert tuple(item.tobytes() for item in decoded[0][1]) == (b"",) * 4
    assert decoded[1][0][0].tobytes() == real_product
    assert tuple(item.tobytes() for item in decoded[1][0][1:]) == (b"",) * 3
    assert tuple(item.tobytes() for item in decoded[1][1]) == (b"",) * 4
    for wave in decoded:
        for slot in wave:
            for product in slot:
                assert isinstance(product, memoryview)
                assert product.readonly
                assert isinstance(product.obj, bytes)


def test_int32_payload_accepts_the_full_signed_domain_without_extra_policy() -> None:
    control = _active(
        dpu_id=0,
        wave_id=1,
        request_sequence=2,
        tile_id=3,
        numeric_mode=1,
        m=1,
        n=2,
    )
    product = struct.pack("<ii", -(1 << 31), (1 << 31) - 1)
    data = _stream(((control, (product, product, product, product)),))

    decoded = decode_wave_results(data, ((_tile(control),),))

    assert tuple(item.tobytes() for item in decoded[0][0]) == (product,) * 4


@pytest.mark.parametrize(
    "mutator",
    [
        lambda data: data[:-1],
        lambda data: data + b"tail",
    ],
    ids=["truncated", "trailing"],
)
def test_stream_length_must_be_exact(mutator) -> None:
    control = _active(dpu_id=0, wave_id=1, request_sequence=2, tile_id=3)
    product = struct.pack("<2f", 1.0, 2.0)
    data = mutator(_stream(((control, (product,) * 4),)))

    with pytest.raises(ValueError, match="truncated|trailing"):
        decode_wave_results(data, ((_tile(control),),))


def test_idle_slot_completion_is_required() -> None:
    active = _active(dpu_id=0, wave_id=1, request_sequence=2, tile_id=3)
    idle = _idle(dpu_id=1, wave_id=1, request_sequence=2)
    product = struct.pack("<2f", 1.0, 2.0)
    data = _completion(active) + product * 4

    with pytest.raises(ValueError, match="truncated completion"):
        decode_wave_results(data, ((_tile(active), _tile(idle)),))


def test_reordered_completion_is_rejected_by_control_correlation() -> None:
    first = _active(dpu_id=0, wave_id=1, request_sequence=2, tile_id=3,
                    kernel=REAL_PANEL)
    second = _active(dpu_id=1, wave_id=1, request_sequence=2, tile_id=4,
                     kernel=REAL_PANEL)
    product = struct.pack("<2f", 1.0, 2.0)
    data = _completion(second) + product + _completion(first) + product

    with pytest.raises(ValueError, match="completion dpu_id"):
        decode_wave_results(data, ((_tile(first), _tile(second)),))


@pytest.mark.parametrize("field", ["wave_id", "request_sequence", "tile_id"])
def test_stale_completion_identity_is_rejected(field: str) -> None:
    control = _active(dpu_id=0, wave_id=1, request_sequence=2, tile_id=3,
                      kernel=REAL_PANEL)
    stale = replace(
        WaveCompletion(
            status=STATUS_COMPLETED,
            dpu_id=control.dpu_id,
            operation_index=control.operation_index,
            completed_product_mask=1,
            wave_id=control.wave_id,
            request_sequence=control.request_sequence,
            tile_id=control.tile_id,
            cycles=0,
            processed_elements=control.m * control.n,
            failure_stage=FAILURE_NONE,
            failing_product=NO_PRODUCT,
        ),
        **{field: getattr(control, field) + 1},
    ).to_bytes()
    product = struct.pack("<2f", 1.0, 2.0)

    with pytest.raises(ValueError, match=f"completion {field}"):
        decode_wave_results(stale + product, ((_tile(control),),))


@pytest.mark.parametrize("status", [STATUS_PENDING, STATUS_FAILED])
def test_pending_and_failed_completions_are_not_results(status: int) -> None:
    control = _active(dpu_id=0, wave_id=1, request_sequence=2, tile_id=3,
                      kernel=REAL_PANEL)
    if status == STATUS_PENDING:
        completion = WaveCompletion(
            status=STATUS_PENDING,
            dpu_id=control.dpu_id,
            operation_index=control.operation_index,
            completed_product_mask=0,
            wave_id=control.wave_id,
            request_sequence=control.request_sequence,
            tile_id=control.tile_id,
            cycles=0,
            processed_elements=0,
            failure_stage=FAILURE_NONE,
            failing_product=NO_PRODUCT,
        )
    else:
        completion = WaveCompletion(
            status=STATUS_FAILED,
            dpu_id=control.dpu_id,
            operation_index=control.operation_index,
            completed_product_mask=0,
            wave_id=control.wave_id,
            request_sequence=control.request_sequence,
            tile_id=control.tile_id,
            cycles=0,
            processed_elements=0,
            failure_stage=FAILURE_EXECUTION,
            failing_product=0,
        )

    with pytest.raises(ValueError, match="completed status"):
        decode_wave_results(completion.to_bytes() + b"x" * 8, ((_tile(control),),))


def test_nonfinite_float_output_is_corruption() -> None:
    control = _active(dpu_id=0, wave_id=1, request_sequence=2, tile_id=3,
                      kernel=REAL_PANEL)
    data = _completion(control) + struct.pack("<2f", float("nan"), 1.0)

    with pytest.raises(ValueError, match="nonfinite"):
        decode_wave_results(data, ((_tile(control),),))


def test_corrupted_completion_identity_is_rejected() -> None:
    control = _active(dpu_id=0, wave_id=1, request_sequence=2, tile_id=3,
                      kernel=REAL_PANEL)
    completion = bytearray(_completion(control))
    completion[0] ^= 1
    product = struct.pack("<2f", 1.0, 2.0)

    with pytest.raises(ValueError, match="identity"):
        decode_wave_results(bytes(completion) + product, ((_tile(control),),))


def test_wave_shape_is_dense_and_nested() -> None:
    control = _active(dpu_id=0, wave_id=1, request_sequence=2, tile_id=3,
                      kernel=REAL_PANEL)
    product = struct.pack("<2f", 1.0, 2.0)
    data = _completion(control) + product

    with pytest.raises(TypeError, match="nested tuple"):
        decode_wave_results(data, [_tile(control)])
    with pytest.raises(ValueError, match="same dense slot count"):
        decode_wave_results(data, ((_tile(control),), (_tile(control), _tile(control))))
