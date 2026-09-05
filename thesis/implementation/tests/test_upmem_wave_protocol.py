"""Private launch controls have explicit byte order, identities and live planes."""

from dataclasses import replace
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from quantum_bench.upmem.wave_protocol import (
    CONTROL, FOUR_PRODUCT_PANEL, IDLE, INT8_COMPONENT_PRODUCT, MAX_K,
    MRAM_BYTES, NO_OPERATION, REAL_PANEL, REAL_OUTER, FOUR_PRODUCT_OUTER,
    WaveControl, aligned_bytes, product_layout,
)


def control(*, kernel=FOUR_PRODUCT_PANEL, numeric_mode=0, m=3, n=5, k=7):
    return WaveControl(
        dpu_id=3, tasklets=8, flags=0, numeric_mode=numeric_mode, kernel=kernel,
        operation_index=2, wave_id=11, request_sequence=13, tile_id=17,
        batch_index=0, m=m, n=n, k=k, k_offset=0,
        planes=product_layout(m, n, k, numeric_mode=numeric_mode, kernel=kernel),
    )


@pytest.mark.parametrize("kernel", [REAL_PANEL, FOUR_PRODUCT_PANEL])
@pytest.mark.parametrize("numeric_mode", [0, 1])
def test_exact_plane_counts_and_roundtrip(kernel, numeric_mode):
    item = control(kernel=kernel, numeric_mode=numeric_mode)
    data = item.to_bytes()
    assert len(data) == 144
    assert data[:8] == struct.pack("<II", 0x35574354, 5)
    assert WaveControl.from_bytes(data) == item
    assert sum(length > 0 for _, length in item.planes) == (3 if kernel == REAL_PANEL else 8)
    for offset, length in item.planes:
        assert offset % 8 == length % 8 == 0
        assert length <= MRAM_BYTES - offset


def test_fusion_uses_four_distinct_product_regions():
    item = control()
    assert len({offset for offset, _ in item.planes[4:]}) == 4
    a = aligned_bytes(3 * 7 * 4)
    b = aligned_bytes(7 * 5 * 4)
    c = aligned_bytes(3 * 5 * 4)
    assert sum(length for _, length in item.planes) == 2 * a + 2 * b + 4 * c


def test_old_admitted_tile_does_not_imply_fused_admission():
    assert sum(length for _, length in product_layout(
        128, 256, 256, numeric_mode=0, kernel=REAL_PANEL)) == MRAM_BYTES
    with pytest.raises(ValueError, match="working set"):
        product_layout(128, 256, 256, numeric_mode=0, kernel=FOUR_PRODUCT_PANEL)


def test_exact_arena_boundary_and_shared_scale_operand_width():
    fused = control(m=64, n=128, k=256)
    assert sum(length for _, length in fused.planes) == MRAM_BYTES
    assert WaveControl.from_bytes(fused.to_bytes()) == fused
    quantized = control(m=64, n=256, k=256, numeric_mode=1)
    assert sum(length for _, length in quantized.planes) == 425984


@pytest.mark.parametrize("field,value", [
    ("dpu_id", 64), ("tasklets", 0), ("tasklets", 25), ("tasklets", True),
    ("numeric_mode", 2), ("kernel", 3), ("flags", 2), ("operation_index", 64),
    ("m", 0), ("n", 257), ("k", 65537), ("k_offset", 65536),
    ("wave_id", 1 << 64), ("request_sequence", -1), ("tile_id", True),
])
def test_reject_invalid_fields(field, value):
    with pytest.raises(ValueError):
        replace(control(), **{field: value}).to_bytes()


@pytest.mark.parametrize("span", [(MRAM_BYTES, 88), (1, 88), (0, 80), (1 << 32, 88)])
def test_reject_corrupt_plane(span):
    item = control()
    with pytest.raises(ValueError):
        replace(item, planes=(span,) + item.planes[1:]).to_bytes()


def test_reject_overlap_and_mutable_descriptors():
    item = control()
    with pytest.raises(ValueError, match="overlapping"):
        replace(item, planes=(item.planes[1],) + item.planes[1:]).to_bytes()
    with pytest.raises(ValueError, match="immutable"):
        replace(item, planes=list(item.planes)).to_bytes()
    with pytest.raises(ValueError, match="uint32"):
        aligned_bytes((1 << 32) - 1)


def test_idle_descriptor_has_no_operation_or_buffers():
    item = replace(control(), flags=IDLE, kernel=0, operation_index=NO_OPERATION,
                   tile_id=0, m=0, n=0, k=0, planes=((0, 0),) * 8)
    assert WaveControl.from_bytes(item.to_bytes()) == item
    with pytest.raises(ValueError, match="idle"):
        replace(item, operation_index=0).to_bytes()
    with pytest.raises(ValueError, match="idle"):
        replace(item, tile_id=5).to_bytes()


@pytest.mark.parametrize("index,value", [(0, 0), (1, 4), (16, 1)])
def test_reject_wrong_identity_or_reserved_fields(index, value):
    fields = list(CONTROL.unpack(control().to_bytes()))
    fields[index] = value
    with pytest.raises(ValueError, match="identity"):
        WaveControl.from_bytes(CONTROL.pack(*fields))


def test_reject_truncated_and_trailing_data():
    data = control().to_bytes()
    for value in (data[:-1], data + b"\0"):
        with pytest.raises(ValueError, match="length"):
            WaveControl.from_bytes(value)


@pytest.fixture(scope="module")
def c_inspector(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("wave-control")
    cc = shutil.which("cc")
    if cc is None:
        pytest.fail("C compiler is required for the private native contract")
    header = Path(__file__).resolve().parents[1] / "native/upmem/runtime"
    binary = tmp_path / "inspect-control"
    source = r'''
#include <stddef.h>
#include <stdio.h>
#include "wave_protocol.h"
int main(void) {
    upmem_wave_control_t c;
    if (fread(&c, 1, sizeof(c), stdin) != sizeof(c)) return 1;
    printf("%zu %zu %zu %u %u %u %llu %llu %llu %u %u %u %u %d\n",
        sizeof(c), sizeof(upmem_wave_completion_t), offsetof(upmem_wave_control_t, planes),
        c.dpu_id, c.kernel, c.operation_index,
        (unsigned long long)c.wave_id, (unsigned long long)c.request_sequence,
        (unsigned long long)c.tile_id, c.m, c.n, c.k, c.planes[UPMEM_WAVE_IR].length,
        upmem_wave_control_valid(&c, 3, 8));
    return 0;
}
'''
    result = subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(header),
                             "-x", "c", "-", "-o", str(binary)], input=source,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return binary


def test_c_layout_matches_python_control(c_inspector):
    result = subprocess.run([str(c_inspector)], input=control().to_bytes(), capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().strip() == "144 72 80 3 2 2 11 13 17 3 5 7 64 1"


@pytest.mark.parametrize("index,value", [
    (0, 0), (1, 4), (2, 64), (3, 7), (4, 2), (5, 2), (6, 3), (7, 64),
    (12, 257), (13, 0), (14, 65537), (15, 65536), (16, 1),
    (17, 1), (17, MRAM_BYTES), (17, (1 << 32) - 1),
    (18, (1 << 32) - 1), (19, 0), (32, 0),
])
def test_native_rejects_corrupt_controls(c_inspector, index, value):
    fields = list(CONTROL.unpack(control().to_bytes()))
    fields[index] = value
    data = CONTROL.pack(*fields)
    result = subprocess.run([str(c_inspector)], input=data, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().split()[-1] == "0"


@pytest.mark.parametrize("kernel", [REAL_PANEL, FOUR_PRODUCT_PANEL])
@pytest.mark.parametrize("mode", [0, 1])
def test_native_accepts_each_numeric_and_product_mode(c_inspector, kernel, mode):
    result = subprocess.run([str(c_inspector)],
                            input=control(kernel=kernel, numeric_mode=mode).to_bytes(),
                            capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().split()[-1] == "1"


def test_explicit_int8_component_bound():
    assert MAX_K * INT8_COMPONENT_PRODUCT <= (1 << 31) - 1


@pytest.mark.parametrize("kernel,panel", [(REAL_OUTER, REAL_PANEL),
                                         (FOUR_PRODUCT_OUTER, FOUR_PRODUCT_PANEL)])
@pytest.mark.parametrize("mode", [0, 1])
def test_outer_selectors_preserve_layout_and_require_k_one(c_inspector, kernel, panel, mode):
    item = control(kernel=kernel, numeric_mode=mode, m=13, n=35, k=1)
    assert item.planes == control(kernel=panel, numeric_mode=mode, m=13, n=35, k=1).planes
    assert WaveControl.from_bytes(item.to_bytes()) == item
    result = subprocess.run([str(c_inspector)], input=item.to_bytes(), capture_output=True)
    assert result.returncode == 0 and result.stdout.decode().split()[-1] == "1"
    with pytest.raises(ValueError, match="requires K=1"):
        product_layout(13, 35, 2, numeric_mode=mode, kernel=kernel)
    fields = list(CONTROL.unpack(item.to_bytes()))
    fields[14] = 2
    result = subprocess.run([str(c_inspector)], input=CONTROL.pack(*fields), capture_output=True)
    assert result.returncode == 0 and result.stdout.decode().split()[-1] == "0"


def test_offsets_are_explicit_not_implicitly_contiguous(c_inspector):
    item = control()
    item = replace(item, planes=tuple((offset + 8, size) for offset, size in item.planes))
    assert WaveControl.from_bytes(item.to_bytes()) == item
    result = subprocess.run([str(c_inspector)], input=item.to_bytes(), capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().split()[-1] == "1"


def test_native_idle_descriptor(c_inspector):
    item = replace(control(), flags=IDLE, kernel=0, operation_index=NO_OPERATION,
                   tile_id=0, m=0, n=0, k=0, planes=((0, 0),) * 8)
    result = subprocess.run([str(c_inspector)], input=item.to_bytes(), capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().split()[-1] == "1"
