from dataclasses import replace
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from quantum_bench.upmem.wave_protocol import (
    COMPLETION,
    COMPLETION_MAGIC,
    COMPLETED_MASK_FUSED,
    COMPLETED_MASK_IDLE,
    COMPLETED_MASK_REAL,
    FAILURE_EXECUTION,
    FAILURE_NONE,
    FAILURE_VALIDATION,
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


def control(*, kernel=FOUR_PRODUCT_PANEL, m=3, n=5, k=7) -> WaveControl:
    return WaveControl(
        dpu_id=3,
        tasklets=8,
        flags=0,
        numeric_mode=0,
        kernel=kernel,
        operation_index=2,
        wave_id=11,
        request_sequence=13,
        tile_id=17,
        batch_index=0,
        m=m,
        n=n,
        k=k,
        k_offset=0,
        planes=product_layout(m, n, k, numeric_mode=0, kernel=kernel),
    )


def idle_control() -> WaveControl:
    return WaveControl(
        dpu_id=3,
        tasklets=8,
        flags=IDLE,
        numeric_mode=0,
        kernel=0,
        operation_index=NO_OPERATION,
        wave_id=11,
        request_sequence=13,
        tile_id=0,
        batch_index=0,
        m=0,
        n=0,
        k=0,
        k_offset=0,
        planes=((0, 0),) * 8,
    )


def completed(control_item: WaveControl, *, cycles: int = 123) -> WaveCompletion:
    product_count = (
        0
        if control_item.flags == IDLE
        else 1
        if control_item.kernel == REAL_PANEL
        else 4
    )
    return WaveCompletion(
        status=STATUS_COMPLETED,
        dpu_id=control_item.dpu_id,
        operation_index=control_item.operation_index,
        completed_product_mask=(1 << product_count) - 1 if product_count else 0,
        wave_id=control_item.wave_id,
        request_sequence=control_item.request_sequence,
        tile_id=control_item.tile_id,
        cycles=cycles,
        processed_elements=control_item.m * control_item.n * product_count,
        failure_stage=FAILURE_NONE,
        failing_product=NO_PRODUCT,
    )


def failed(
    control_item: WaveControl,
    *,
    failure_stage: int,
    failing_product: int = NO_PRODUCT,
    completed_product_mask: int = 0,
    processed_elements: int = 0,
) -> WaveCompletion:
    return WaveCompletion(
        status=STATUS_FAILED,
        dpu_id=control_item.dpu_id,
        operation_index=control_item.operation_index,
        completed_product_mask=completed_product_mask,
        wave_id=control_item.wave_id,
        request_sequence=control_item.request_sequence,
        tile_id=control_item.tile_id,
        cycles=0,
        processed_elements=processed_elements,
        failure_stage=failure_stage,
        failing_product=failing_product,
    )


@pytest.fixture(scope="module")
def c_inspector(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler is required for the native completion ABI check")
    tmp_path = tmp_path_factory.mktemp("wave-completion")
    header = Path(__file__).resolve().parents[1] / "native/upmem/runtime"
    binary = tmp_path / "inspect-completion"
    source = r'''
#include <stddef.h>
#include <stdio.h>
#include "wave_protocol.h"
int main(void) {
    upmem_wave_completion_t completion;
    if (fread(&completion, 1, sizeof(completion), stdin) != sizeof(completion)) return 1;
    printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n",
        sizeof(completion), offsetof(upmem_wave_completion_t, status),
        offsetof(upmem_wave_completion_t, dpu_id),
        offsetof(upmem_wave_completion_t, completed_product_mask),
        offsetof(upmem_wave_completion_t, wave_id),
        offsetof(upmem_wave_completion_t, tile_id),
        offsetof(upmem_wave_completion_t, cycles),
        offsetof(upmem_wave_completion_t, processed_elements),
        offsetof(upmem_wave_completion_t, failure_stage),
        offsetof(upmem_wave_completion_t, failing_product));
    return 0;
}
'''
    result = subprocess.run(
        [
            cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(header),
            "-x",
            "c",
            "-",
            "-o",
            str(binary),
        ],
        input=source,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


def test_completion_has_current_native_layout_and_roundtrips(c_inspector: Path) -> None:
    item = completed(control())
    data = item.to_bytes()

    assert COMPLETION.size == 72
    assert len(data) == 72
    assert data[:8] == struct.pack("<II", COMPLETION_MAGIC, 5)
    assert WaveCompletion.from_bytes(data) == item

    result = subprocess.run([str(c_inspector)], input=data, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().strip() == "72 8 12 20 24 40 48 56 64 68"


@pytest.mark.parametrize(
    ("item", "expected_mask", "expected_elements"),
    [
        (completed(control(kernel=REAL_PANEL)), COMPLETED_MASK_REAL, 15),
        (completed(control()), COMPLETED_MASK_FUSED, 60),
        (completed(idle_control()), COMPLETED_MASK_IDLE, 0),
    ],
)
def test_successful_completion_correlates_product_mask_and_count(
    item: WaveCompletion, expected_mask: int, expected_elements: int
) -> None:
    assert item.completed_product_mask == expected_mask
    assert item.processed_elements == expected_elements
    item.validate(
        control(kernel=REAL_PANEL)
        if expected_mask == COMPLETED_MASK_REAL
        else idle_control()
        if expected_mask == COMPLETED_MASK_IDLE
        else control(),
        require_success=True,
    )


def test_validation_failure_is_parseable_diagnostic_but_not_success() -> None:
    item = failed(control(), failure_stage=FAILURE_VALIDATION)
    parsed = WaveCompletion.from_bytes(item.to_bytes())

    parsed.validate(control())
    with pytest.raises(ValueError, match="completed status"):
        parsed.validate(control(), require_success=True)


@pytest.mark.parametrize(
    ("failing_product", "completed_product_mask"),
    [(0, 0), (1, 1), (2, 3), (3, 7)],
)
def test_execution_failure_preserves_prefix_and_processed_count(
    failing_product: int, completed_product_mask: int
) -> None:
    item = failed(
        control(m=4, n=6),
        failure_stage=FAILURE_EXECUTION,
        failing_product=failing_product,
        completed_product_mask=completed_product_mask,
        processed_elements=4 * 6 * failing_product,
    )

    parsed = WaveCompletion.from_bytes(item.to_bytes(), control(m=4, n=6))
    parsed.validate(control(m=4, n=6))
    with pytest.raises(ValueError, match="completed status"):
        parsed.validate(control(m=4, n=6), require_success=True)


def test_pending_is_never_accepted_as_terminal() -> None:
    item = WaveCompletion(
        status=STATUS_PENDING,
        dpu_id=3,
        operation_index=2,
        completed_product_mask=0,
        wave_id=11,
        request_sequence=13,
        tile_id=17,
        cycles=0,
        processed_elements=0,
        failure_stage=FAILURE_NONE,
        failing_product=NO_PRODUCT,
    )
    item.validate(control())
    with pytest.raises(ValueError, match="completed status"):
        item.validate(control(), require_success=True)
    with pytest.raises(ValueError, match="WaveControl"):
        item.validate(require_success=True)


@pytest.mark.parametrize(
    ("completed_product_mask", "processed_elements"), [(3, 3 * 5 * 2), (15, 3 * 5 * 4)]
)
def test_pending_prefix_progress_remains_diagnostic(
    completed_product_mask: int, processed_elements: int
) -> None:
    item = WaveCompletion(
        status=STATUS_PENDING,
        dpu_id=3,
        operation_index=2,
        completed_product_mask=completed_product_mask,
        wave_id=11,
        request_sequence=13,
        tile_id=17,
        cycles=0,
        processed_elements=processed_elements,
        failure_stage=FAILURE_NONE,
        failing_product=NO_PRODUCT,
    )

    item.validate(control())
    with pytest.raises(ValueError, match="completed status"):
        item.validate(control(), require_success=True)


@pytest.mark.parametrize(
    "field",
    ["dpu_id", "operation_index", "wave_id", "request_sequence", "tile_id"],
)
def test_stale_completion_identity_is_rejected(field: str) -> None:
    item = completed(control())
    value = getattr(item, field) + 1
    stale = replace(item, **{field: value})
    with pytest.raises(ValueError, match="match control"):
        stale.validate(control(), require_success=True)


@pytest.mark.parametrize("field", ["dpu_id", "operation_index"])
def test_diagnostic_bytes_preserve_uint32_identity_until_correlation(field: str) -> None:
    item = replace(completed(control()), **{field: 64})
    parsed = WaveCompletion.from_bytes(item.to_bytes())

    parsed.validate()
    with pytest.raises(ValueError, match="match control"):
        parsed.validate(control())


def test_control_specific_count_mismatch_is_rejected() -> None:
    item = replace(completed(control()), processed_elements=59)
    with pytest.raises(ValueError, match="progress"):
        item.to_bytes(control(), require_success=True)

    wrong_kernel = replace(completed(control(kernel=REAL_PANEL)), completed_product_mask=15)
    with pytest.raises(ValueError, match="progress"):
        wrong_kernel.validate(control(kernel=REAL_PANEL), require_success=True)


@pytest.mark.parametrize(
    "item",
    [
        replace(completed(control()), status=3),
        replace(completed(control()), failure_stage=3),
        replace(completed(control()), completed_product_mask=2),
        replace(
            failed(control(), failure_stage=FAILURE_EXECUTION, failing_product=0),
            completed_product_mask=1,
        ),
        replace(
            failed(control(), failure_stage=FAILURE_EXECUTION, failing_product=0),
            failing_product=4,
        ),
    ],
)
def test_invalid_status_masks_and_failure_counts_are_rejected(item: WaveCompletion) -> None:
    with pytest.raises(ValueError):
        item.to_bytes()


def test_malformed_completion_bytes_and_identity_are_rejected() -> None:
    data = completed(control()).to_bytes()
    for value in (data[:-1], data + b"\0"):
        with pytest.raises(ValueError, match="length"):
            WaveCompletion.from_bytes(value)

    for index, value in ((0, 0), (1, 4)):
        fields = list(COMPLETION.unpack(data))
        fields[index] = value
        with pytest.raises(ValueError, match="identity"):
            WaveCompletion.from_bytes(COMPLETION.pack(*fields))


def test_diagnostic_option_is_explicitly_available_for_failed_bytes() -> None:
    item = failed(control(), failure_stage=FAILURE_VALIDATION)
    data = item.to_bytes()

    WaveCompletion.from_bytes(data, control())
    with pytest.raises(ValueError, match="completed status"):
        WaveCompletion.from_bytes(data, control(), require_success=True)


def test_require_success_is_an_exact_boolean_and_requires_correlation() -> None:
    item = completed(control())
    with pytest.raises(ValueError, match="bool"):
        item.validate(control(), require_success=1)
    with pytest.raises(ValueError, match="WaveControl"):
        item.validate(require_success=True)
