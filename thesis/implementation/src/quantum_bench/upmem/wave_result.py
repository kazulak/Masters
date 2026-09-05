"""Pure decoder for the prepared-wave result stream.

The native wave host writes one :class:`WaveCompletion` for every dense
wave/DPU slot, followed by the live output bytes for that slot in product
order.  The returned shape is ``waves -> slots -> (rr, ii, ri, ir)``.

Every product entry is a read-only ``memoryview`` over the immutable input
``bytes`` object.  Empty product entries are zero-length views.  The views
retain the input bytes as their backing storage, so callers own the returned
tuple and can safely keep the views without copying each product.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from quantum_bench.upmem.packed_wave import WaveTile
from quantum_bench.upmem.wave_protocol import (
    COMPLETION_BYTES,
    FOUR_PRODUCT_KERNELS,
    IDLE,
    REAL_KERNELS,
    WaveCompletion,
)


ProductResults: TypeAlias = tuple[memoryview, memoryview, memoryview, memoryview]
WaveResults: TypeAlias = tuple[tuple[ProductResults, ...], ...]


def _product_count(tile: WaveTile) -> int:
    control = tile.control
    if control.flags == IDLE:
        return 0
    if control.kernel in REAL_KERNELS:
        return 1
    if control.kernel in FOUR_PRODUCT_KERNELS:
        return 4
    # WaveCompletion.from_bytes() validates the control before this point.
    raise ValueError("wave control has an unsupported product kernel")


def _validate_product_bytes(payload: memoryview, numeric_mode: int, *, slot: str) -> None:
    if numeric_mode == 0:
        values = np.frombuffer(payload, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError(f"{slot} contains a nonfinite float32 result")
        return

    # Every four-byte pattern is a valid signed int32.  The exact output
    # length is already enforced by the control geometry, so no narrower
    # application range is valid to impose here.
    if numeric_mode == 1:
        return

    raise ValueError(f"{slot} has an unsupported numeric mode")


def decode_wave_results(
    data: bytes,
    waves: tuple[tuple[WaveTile, ...], ...],
) -> WaveResults:
    """Decode and validate one complete prepared-cohort result stream.

    The stream is consumed in the exact order represented by ``waves``.  A
    successful terminal completion is required for every slot, including idle
    slots.  Product views are read-only slices backed by ``data``; no product
    payload is copied or converted, and the caller should treat the returned
    nested tuple as the owned decoded result.
    """

    if not isinstance(data, bytes):
        raise TypeError("result data must be bytes")
    if type(waves) is not tuple or not waves:
        raise TypeError("waves must be a nonempty nested tuple")

    source = memoryview(data)
    empty = source[:0]
    cursor = 0
    slot_count: int | None = None
    decoded_waves: list[tuple[ProductResults, ...]] = []

    for wave_index, wave in enumerate(waves):
        if type(wave) is not tuple or not wave:
            raise TypeError(f"wave {wave_index} must be a nonempty tuple")
        if slot_count is None:
            slot_count = len(wave)
        elif len(wave) != slot_count:
            raise ValueError("all waves must have the same dense slot count")

        decoded_slots: list[ProductResults] = []
        for slot_index, tile in enumerate(wave):
            if not isinstance(tile, WaveTile):
                raise TypeError(f"wave {wave_index} slot {slot_index} must be a WaveTile")
            control = tile.control
            if cursor > len(source) - COMPLETION_BYTES:
                raise ValueError(
                    f"truncated completion at wave {wave_index} slot {slot_index}"
                )

            completion_end = cursor + COMPLETION_BYTES
            # The existing completion API performs both structural validation
            # and exact identity/progress correlation with this control.
            WaveCompletion.from_bytes(
                source[cursor:completion_end], control, require_success=True
            )
            cursor = completion_end

            products = [empty, empty, empty, empty]
            count = _product_count(tile)
            product_bytes = control.m * control.n * 4
            for product in range(count):
                if cursor > len(source) - product_bytes:
                    raise ValueError(
                        f"truncated product {product} at wave {wave_index} "
                        f"slot {slot_index}"
                    )
                product_end = cursor + product_bytes
                payload = source[cursor:product_end]
                _validate_product_bytes(
                    payload,
                    control.numeric_mode,
                    slot=f"wave {wave_index} slot {slot_index} product {product}",
                )
                products[product] = payload
                cursor = product_end
            decoded_slots.append((products[0], products[1], products[2], products[3]))
        decoded_waves.append(tuple(decoded_slots))

    if cursor != len(source):
        raise ValueError("trailing wave result bytes")
    return tuple(decoded_waves)
