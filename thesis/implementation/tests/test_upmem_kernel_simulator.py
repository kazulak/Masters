from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "upmem" / "runtime"
SOURCE = NATIVE / "dpu.c"
MAKEFILE = NATIVE / "Makefile"


def test_active_wram_panel_target_uses_the_v4_binary() -> None:
    makefile = MAKEFILE.read_text(encoding="ascii")

    active_v4_rule = makefile.split("v4:", 1)[1].split("bin:", 1)[0]
    assert "bin/dpu_gemm_tile_v4_t$(NR_TASKLETS)" in active_v4_rule
    assert "dpu_gemm_tile_v4_wram_panel_internal" not in makefile
    assert not (NATIVE / "dpu_wram_panel_internal.c").exists()


def test_active_wram_panel_source_uses_global_staging_and_bounded_dma() -> None:
    source = SOURCE.read_text(encoding="ascii")

    assert "#define KC 64u" in source
    assert "#define NC 32u" in source
    assert "#define B_PANEL_DATA_BYTES (KC * NC * sizeof(float))" in source
    assert "#define B_PANEL_ROW_STRIDE_BYTES (NC * sizeof(float))" in source
    assert "#define A_BUFFER_DATA_BYTES (KC * sizeof(float))" in source
    assert "#define OUTPUT_BUFFER_DATA_BYTES (NC * sizeof(uint32_t))" in source
    assert "#define UNALIGNED_SCRATCH_BYTES 288u" in source
    assert "__dma_aligned uint8_t shared_b_panel" in source
    assert "__dma_aligned uint8_t tasklet_a_buffer" in source
    assert "__dma_aligned v4_output_slot_t tasklet_output_buffer" in source
    assert "__dma_aligned uint8_t tasklet_unaligned_scratch" in source
    assert "mram_read_unaligned" in source
    assert "mram_write_unaligned" in source
    assert "const uint32_t dst_align = mram_c_offset & 7u;" in source
    assert "tasklet_unaligned_scratch[tid] + dst_align" in source
    assert "B_CONTIGUOUS_CHUNK_BYTES <= 2048u" in source
    assert "barrier_wait(&v4_barrier);" in source

    b_stage = source.split("if (actual_kc == KC", 1)[1].split(
        "barrier_wait(&v4_barrier);", 1
    )[0]
    arithmetic = source.split("if (is_int8)", 1)[1].split(
        "barrier_wait(&v4_barrier);", 1
    )[0]
    assert "mram_read(" in b_stage
    assert "mram_read(" not in arithmetic
    assert "mram_read_unaligned" not in arithmetic


@pytest.mark.parametrize("tasklets", [1, 8, 24])
def test_active_wram_panel_binary_builds_when_sdk_compiler_is_available(tasklets: int) -> None:
    if shutil.which("dpu-upmem-dpurte-clang") is None:
        pytest.skip("UPMEM DPU compiler is unavailable")

    result = subprocess.run(
        [
            "make",
            "-C",
            str(NATIVE),
            f"bin/dpu_gemm_tile_v4_t{tasklets}",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (NATIVE / "bin" / f"dpu_gemm_tile_v4_t{tasklets}").is_file()
