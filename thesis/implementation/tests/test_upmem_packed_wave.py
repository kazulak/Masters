from __future__ import annotations

from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from quantum_bench.upmem.packed_wave import (
    HEADER,
    HEADER_BYTES,
    OPERATION_BYTES,
    TILE_BYTES,
    WaveOperation,
    WaveTile,
    pack_wave_envelope,
    unpack_wave_envelope,
    validate_wave_envelope,
)
from quantum_bench.upmem.wave_protocol import (
    FOUR_PRODUCT_PANEL,
    IDLE,
    NO_OPERATION,
    WaveControl,
    product_layout,
)


PLAN = bytes.fromhex("01" * 32)
BINARY = bytes.fromhex("02" * 32)
NODE_A = bytes.fromhex("11" * 32)
NODE_B = bytes.fromhex("12" * 32)
CONTRACT_A = bytes.fromhex("21" * 32)
CONTRACT_B = bytes.fromhex("22" * 32)


def operation(node: bytes = NODE_A, contract: bytes = CONTRACT_A, *, m: int = 4,
              n: int = 4, k: int = 2, batch_count: int = 1,
              left_scale: float = 1.0, right_scale: float = 1.0) -> WaveOperation:
    return WaveOperation(node, contract, batch_count, m, n, k, left_scale, right_scale)


def control(*, dpu: int, wave: int = 1, request: int = 1, tile: int = 1,
            operation_index: int = 0, flags: int = 0, batch: int = 0,
            m: int = 2, n: int = 2, k: int = 2, k_offset: int = 0,
            numeric_mode: int = 0, tasklets: int = 4,
            mram_shift: int = 0) -> WaveControl:
    planes = product_layout(m, n, k, numeric_mode=numeric_mode,
                            kernel=FOUR_PRODUCT_PANEL)
    if mram_shift:
        planes = tuple((offset + mram_shift, length) if length else (0, 0)
                       for offset, length in planes)
    return WaveControl(
        dpu, tasklets, flags, numeric_mode, FOUR_PRODUCT_PANEL,
        operation_index, wave, request, tile, batch, m, n, k, k_offset, planes,
    )


def idle(*, dpu: int, wave: int = 1, request: int = 1, tasklets: int = 4) -> WaveControl:
    return WaveControl(
        dpu, tasklets, IDLE, 0, 0, NO_OPERATION, wave, request, 0, 0,
        0, 0, 0, 0, ((0, 0),) * 8,
    )


def payloads(item: WaveControl, *, value: float = 1.0) -> tuple[bytes, bytes, bytes, bytes]:
    logical = (item.m * item.k * 4, item.m * item.k * 4,
               item.k * item.n * 4, item.k * item.n * 4)
    return tuple(
        struct.pack(f"<{size // 4}f", *([value] * (size // 4)))
        + b"\0" * (item.planes[index][1] - size)
        if item.planes[index][1]
        else b""
        for index, size in enumerate(logical)
    )


def tile(item: WaveControl, *, m_offset: int = 0, n_offset: int = 0,
         value: float = 1.0) -> WaveTile:
    return WaveTile(item, m_offset, n_offset, payloads(item, value=value))


def envelope(*, operations=(operation(),), waves=None, numeric_mode=0) -> bytes:
    if waves is None:
        waves = ((tile(control(dpu=0)),),)
    return pack_wave_envelope(
        plan_sha256=PLAN,
        dpu_binary_sha256=BINARY,
        sequence=7,
        operations=operations,
        waves=waves,
        numeric_mode=numeric_mode,
    )


def test_deterministic_layout_counts_and_roundtrip():
    first_control = control(dpu=0, wave=3, request=5)
    first = envelope(waves=((tile(first_control),),))
    second = envelope(waves=((tile(first_control),),))
    assert first == second
    assert len(first) == HEADER_BYTES + OPERATION_BYTES + TILE_BYTES + sum(
        len(plane) for plane in payloads(first_control)
    )
    header = HEADER.unpack(first[:HEADER_BYTES])
    assert header[:9] == (b"UPWAVE1\0", 1, 136, 1, 4, 1, 1, 0, 0)
    assert header[9:13] == (7, 1, len(first), HEADER_BYTES + OPERATION_BYTES + TILE_BYTES)
    operations, waves = unpack_wave_envelope(first)
    assert operations == (operation(),)
    assert waves[0][0].control == first_control
    assert waves[0][0].inputs == payloads(first_control)
    validate_wave_envelope(first)


def test_counts_and_overflow_are_rejected_before_sparse_parse():
    data = bytearray(envelope())
    fields = list(HEADER.unpack(data[:HEADER_BYTES]))
    fields[6] = 0xFFFFFFFF
    fields[10] = fields[3] * fields[6]
    data[:HEADER_BYTES] = HEADER.pack(*fields)
    with pytest.raises(ValueError, match="truncated|count"):
        validate_wave_envelope(data)
    with pytest.raises(ValueError, match="operation_count"):
        pack_wave_envelope(
            plan_sha256=PLAN, dpu_binary_sha256=BINARY, sequence=1,
            operations=(operation(), operation(NODE_B, CONTRACT_B)),
            waves=((tile(control(dpu=0)),),), numeric_mode=0,
        )


@pytest.mark.parametrize("mutator", [
    lambda data: data[:-1],
    lambda data: data + b"\0",
    lambda data: data[:HEADER_BYTES - 1],
])
def test_truncation_and_trailing_bytes_are_rejected(mutator):
    with pytest.raises(ValueError, match="truncated|total_bytes|trailing"):
        validate_wave_envelope(mutator(envelope()))


@pytest.mark.parametrize("field", ["plan", "binary", "node", "contract"])
def test_digest_identity_is_nonzero(field):
    kwargs = dict(plan_sha256=PLAN, dpu_binary_sha256=BINARY, sequence=1,
                  operations=(operation(),), waves=((tile(control(dpu=0)),),),
                  numeric_mode=0)
    if field == "plan":
        kwargs["plan_sha256"] = b"\0" * 32
    elif field == "binary":
        kwargs["dpu_binary_sha256"] = b"\0" * 32
    else:
        kwargs["operations"] = (operation(**{
            "node" if field == "node" else "contract": b"\0" * 32,
        }),)
    with pytest.raises(ValueError, match="digest"):
        pack_wave_envelope(**kwargs)


def test_duplicate_operations_and_duplicate_tile_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate wave operation"):
        envelope(
            operations=(operation(), operation(NODE_A, CONTRACT_B)),
            waves=(
                (
                    tile(control(dpu=0)),
                    tile(control(dpu=1), m_offset=2),
                ),
            ),
        )
    duplicate_id = (
        tile(control(dpu=0, tile=9)),
        tile(control(dpu=1, tile=9), m_offset=2),
    )
    with pytest.raises(ValueError, match="duplicate wave tile"):
        envelope(
            operations=(operation(),),
            waves=(duplicate_id,),
        )


def test_overlap_is_rejected_regardless_of_k_offset_but_disjoint_nodes_are_allowed():
    overlapping = (
        tile(control(dpu=0, tile=1, k_offset=0)),
        tile(control(dpu=1, tile=2, k_offset=0), value=2.0),
    )
    with pytest.raises(ValueError, match="overlapping"):
        envelope(waves=(overlapping,))
    disjoint_nodes = (
        tile(control(dpu=0, operation_index=0, tile=1)),
        tile(control(dpu=1, operation_index=1, tile=1), m_offset=2),
    )
    assert envelope(
        operations=(operation(), operation(NODE_B, CONTRACT_B)),
        waves=(disjoint_nodes,),
    )


def test_idle_offsets_and_empty_payload_are_checked():
    bad_idle = WaveTile(idle(dpu=0), 1, 0, (b"", b"", b"", b""))
    with pytest.raises(ValueError, match="idle"):
        envelope(waves=((bad_idle,),))
    bad_payload = WaveTile(control(dpu=0), 0, 0, (b"x", b"", b"", b""))
    with pytest.raises(ValueError, match="length"):
        envelope(waves=((bad_payload,),))


def test_padding_and_numeric_payload_validation():
    item = control(dpu=0, m=1, n=1, k=1)
    inputs = list(payloads(item))
    inputs[0] = inputs[0][:-1] + b"\1"
    with pytest.raises(ValueError, match="padding"):
        envelope(waves=((WaveTile(item, 0, 0, tuple(inputs)),),))
    nonfinite = list(payloads(item))
    nonfinite[0] = struct.pack("<f", float("inf")) + nonfinite[0][4:]
    with pytest.raises(ValueError, match="nonfinite"):
        envelope(waves=((WaveTile(item, 0, 0, tuple(nonfinite)),),))
    int8_control = control(dpu=0, numeric_mode=1)
    int8_inputs = tuple(
        bytes([128]) + b"\0" * (length - 1) if length else b""
        for _, length in int8_control.planes[:4]
    )
    with pytest.raises(ValueError, match="int8"):
        envelope(waves=((WaveTile(int8_control, 0, 0, int8_inputs),),), numeric_mode=1)


def test_multiple_waves_require_monotonic_identity_and_use_all_operations():
    waves = (
        (tile(control(dpu=0, wave=1, request=4, tile=1)),),
        (tile(control(dpu=0, wave=2, request=5, tile=2)),),
    )
    packed = envelope(waves=waves)
    assert unpack_wave_envelope(packed)[1][1][0].control.wave_id == 2
    bad = (
        (tile(control(dpu=0, wave=2, request=4, tile=1)),),
        (tile(control(dpu=0, wave=1, request=5, tile=2)),),
    )
    with pytest.raises(ValueError, match="increase"):
        envelope(waves=bad)
    idle_wave = ((tile(control(dpu=0, wave=1)),), (WaveTile(idle(dpu=0, wave=2), 0, 0,
                                                              (b"", b"", b"", b"")),))
    with pytest.raises(ValueError, match="no active"):
        envelope(waves=idle_wave)


def test_dpu_ownership_survives_idle_subwaves_and_rejects_switches():
    allowed = (
        (
            tile(control(dpu=0, wave=1, request=1, tile=1)),
            tile(control(dpu=1, wave=1, request=1, tile=2), m_offset=2),
        ),
        (
            WaveTile(idle(dpu=0, wave=2, request=2), 0, 0,
                     (b"", b"", b"", b"")),
            tile(control(dpu=1, wave=2, request=2, tile=3), m_offset=2),
        ),
        (
            tile(control(dpu=0, wave=3, request=3, tile=4)),
            tile(control(dpu=1, wave=3, request=3, tile=5), m_offset=2),
        ),
    )
    envelope(waves=allowed)

    illegal = (
        (
            tile(control(dpu=0, wave=1, request=1, tile=1)),
            tile(control(dpu=1, wave=1, request=1, tile=2), m_offset=2),
        ),
        (
            tile(control(dpu=0, wave=2, request=2, tile=3,
                        operation_index=1, m=2, n=2)),
            tile(control(dpu=1, wave=2, request=2, tile=4,
                        operation_index=1, m=2, n=2), m_offset=2),
        ),
    )
    with pytest.raises(ValueError, match="ownership"):
        envelope(
            operations=(operation(), operation(NODE_B, CONTRACT_B)),
            waves=illegal,
        )


def test_float_scales_and_geometry_limits():
    with pytest.raises(ValueError, match="unit scales"):
        envelope(operations=(operation(left_scale=2.0),))
    with pytest.raises(ValueError, match="canonical"):
        envelope(waves=((tile(control(dpu=0), m_offset=3),),))
    with pytest.raises(ValueError, match="scale"):
        envelope(operations=(operation(left_scale=float("nan")),))


@pytest.fixture(scope="module")
def c_wave_inspector(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None:
        pytest.fail("C compiler is required for the packed wave wire contract")
    root = Path(__file__).resolve().parents[1] / "native/upmem/runtime"
    directory = tmp_path_factory.mktemp("packed-wave-c")
    source = directory / "inspect.c"
    binary = directory / "inspect"
    source.write_text(
        r'''
#include <stdio.h>
#include <stdlib.h>
#include "wave_envelope.h"
int main(int argc, char **argv) {
    unsigned char *data = NULL, block[4096];
    size_t size = 0, capacity = 0, count;
    char *message = NULL;
    upmem_wave_envelope_t envelope = {0};
    if (argc != 4) return 2;
    while ((count = fread(block, 1, sizeof(block), stdin)) != 0) {
        if (size + count > capacity) {
            size_t next = capacity ? capacity * 2 : 4096;
            while (next < size + count) next *= 2;
            data = realloc(data, next);
            if (!data) return 3;
            capacity = next;
        }
        for (size_t i = 0; i < count; ++i) data[size + i] = block[i];
        size += count;
    }
    envelope.data = data;
    envelope.size = size;
    int result = upmem_wave_envelope_validate(
        &envelope, argv[1], (uint32_t)strtoul(argv[2], NULL, 10),
        (uint32_t)strtoul(argv[3], NULL, 10), &message);
    if (result != 0) fprintf(stderr, "%s\n", message ? message : "invalid");
    free(message);
    free(data);
    return result;
}
''',
        encoding="ascii",
    )
    result = subprocess.run(
        [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(root),
         str(source), str(root / "wave_envelope.c"), str(root / "plan.c"),
         "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


def test_native_codec_accepts_python_bytes(c_wave_inspector):
    data = envelope()
    result = subprocess.run(
        [str(c_wave_inspector), BINARY.hex(), "1", "4"],
        input=data,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("offset,format_,value", [
    (0, "8s", b"wrong\0\0\0"), (8, "I", 2), (16, "I", 65), (20, "I", 0),
    (24, "I", 0), (28, "I", 0), (32, "I", 2), (36, "I", 1),
    (48, "Q", 1 << 63), (56, "Q", (1 << 64) - 1), (64, "Q", 0),
    (72, "32s", bytes(32)), (104, "32s", bytes(32)),
    (136 + 64, "Q", 0), (136 + 88, "Q", 65537), (136 + 96, "d", float("inf")),
    (248, "Q", (1 << 64) - 1), (248 + 16 + 8, "I", 1),
    (248 + 16 + 80, "I", (1 << 32) - 8),
    (248 + 16 + 84, "I", (1 << 32) - 8), (408, "f", float("nan")),
])
def test_python_and_native_reject_same_corrupt_wire(c_wave_inspector, offset, format_, value):
    data = bytearray(envelope())
    struct.pack_into("<" + format_, data, offset, value)
    with pytest.raises((ValueError, TypeError)):
        validate_wave_envelope(data)
    result = subprocess.run([str(c_wave_inspector), BINARY.hex(), "1", "4"],
                            input=bytes(data), capture_output=True)
    assert result.returncode == 1, result.stderr


def test_native_count_extent_padding_and_truncation_checks(c_wave_inspector):
    data = envelope(waves=((tile(control(dpu=0, m=1, n=1, k=1)),),))
    fields = list(HEADER.unpack(data[:HEADER_BYTES]))
    fields[6] = fields[10] = (1 << 32) - 1
    huge_count = HEADER.pack(*fields) + data[HEADER_BYTES:]
    padding = bytearray(data)
    padding[408 + 4] = 1
    for bad in (data[:135], data[:-1], data + b"\0", huge_count, bytes(padding)):
        with pytest.raises((ValueError, TypeError)):
            validate_wave_envelope(bad)
        result = subprocess.run([str(c_wave_inspector), BINARY.hex(), "1", "4"],
                                input=bad, capture_output=True)
        assert result.returncode == 1, result.stderr
