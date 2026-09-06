"""Bit-level native reconstruction corners, independently of panel arithmetic."""

import numpy as np
import pytest

from quantum_bench.numerics import decode_complex_products
from tests import test_upmem_resident_kernel_simulator as probe


resident_sdk = probe.sdk


@pytest.mark.parametrize("overflow", [False, True])
def test_local_reconstruction_preserves_ieee_corners_or_rejects_overflow(resident_sdk, overflow):
    item = probe.case((3, 3, 2), (3, 5, 3), "left", 7)
    info = item["info"]
    tiny = np.nextafter(np.float32(0), np.float32(1))
    rr = np.array([-0., 0., tiny, -tiny, 1., -1., 1., tiny, -0.], dtype=np.float32).reshape(3, 3)
    ii = np.array([-0., -0., 0., 0., 1., -1., 0., -tiny, 0.], dtype=np.float32).reshape(3, 3)
    products = (rr, ii, rr.copy(), -ii)
    if overflow:
        products[0][0, 0] = np.finfo(np.float32).max
        products[1][0, 0] = -np.finfo(np.float32).max
    assembled = tuple(np.add(np.float32(0), p, dtype=np.float32) for p in products)
    raw_products = b"".join(p.tobytes() + probe.GUARD * (span[1] - p.nbytes)
                            for p, span in zip(products, info["controls"][0].planes[4:]))
    start = info["controls"][0].planes[4][0]
    first = probe.request(1, item, 0, item["initial"])
    local = probe.run(resident_sdk, 7, (first, probe.request(3, item, start, raw_products)))
    probe.assert_success(local[0], info["controls"][0])
    if overflow:
        with pytest.raises(ValueError, match="nonfinite"):
            decode_complex_products(assembled, 1., 1., "split_complex_float32_v1")
        probe.assert_failure(local[1], 2)
        for offset, length in info["controls"][1].planes[4:]:
            assert local[1][1][offset:offset + length] == probe.GUARD * length
        return
    value = decode_complex_products(assembled, 1., 1., "split_complex_float32_v1")
    components = (np.asarray(value.real, dtype=np.float32), np.asarray(value.imag, dtype=np.float32))
    raw_components = b"".join(p.tobytes() + probe.GUARD * (span[1] - p.nbytes)
                              for p, span in zip(components, info["retained"]))
    baseline = probe.run(resident_sdk, 7, (first, probe.request(2, item, start, raw_products + raw_components)))
    probe.assert_success(baseline[1], info["controls"][1])
    probe.assert_success(local[1], info["controls"][1])
    probe.assert_products(local[1][1], info["retained"], components)
    assert local[1][1] == baseline[1][1]


def test_kernel_completion_does_not_bypass_final_host_numerical_gate(resident_sdk):
    item = probe.case((2, 4, 3), (2, 5, 4), "left", 8)
    info = item["info"]
    arena = bytearray(item["initial"])
    # The producer and reconstructed intermediate are finite, but the consumer
    # overflows. Execution completion is not numerical qualification.
    for offset, length in info["controls"][0].planes[:4]:
        arena[offset:offset + length] = np.ones(length // 4, dtype="<f4").tobytes()
    for offset, length in info["controls"][1].planes[2:4]:
        arena[offset:offset + length] = np.full(length // 4, np.finfo(np.float32).max, dtype="<f4").tobytes()
    first, record = probe.run(resident_sdk, 8, (probe.request(1, item, 0, bytes(arena)),
                                               probe.request(3, item, 0, b"")))
    probe.assert_success(first, info["controls"][0])
    probe.assert_success(record, info["controls"][1])
    control = info["controls"][1]
    products = tuple(np.frombuffer(record[1], dtype="<f4", count=control.m * control.n, offset=span[0])
                     for span in control.planes[4:])
    assert any(not np.all(np.isfinite(product)) for product in products)
    with pytest.raises(ValueError, match="finite"):
        decode_complex_products(products, 1., 1., "split_complex_float32_v1")
