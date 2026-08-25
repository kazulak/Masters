from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from quantum_bench.upmem.plan import UpmemWorkUnit
from quantum_bench.upmem.runtime import _wram_panel_operation_facts


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "upmem" / "runtime"
SOURCE = NATIVE / "dpu.c"
MAKEFILE = NATIVE / "Makefile"


def _work_unit(*, m_size: int, n_size: int, k_size: int) -> UpmemWorkUnit:
    return UpmemWorkUnit(
        node_id="contract",
        stable_tile_id="contract:tile",
        wave=0,
        logical_rank=0,
        logical_dpu=0,
        batch_start=0,
        batch_size=1,
        m_start=0,
        m_size=m_size,
        n_start=0,
        n_size=n_size,
        k_start=0,
        k_size=k_size,
        estimated_input_bytes=0,
        estimated_output_bytes=0,
        aligned_mram_bytes=0,
        estimated_arithmetic_work=m_size * n_size * k_size,
    )


def test_active_wram_panel_target_uses_the_v4_binary() -> None:
    makefile = MAKEFILE.read_text(encoding="ascii")

    active_v4_rule = makefile.split("v4:", 1)[1].split("bin:", 1)[0]
    assert "bin/dpu_gemm_tile_v4_t$(NR_TASKLETS)" in active_v4_rule
    assert "dpu_gemm_tile_v4_wram_panel_internal" not in makefile
    assert not (NATIVE / "dpu_wram_panel_internal.c").exists()


def test_active_wram_panel_source_uses_global_staging_and_bounded_dma() -> None:
    source = SOURCE.read_text(encoding="ascii")

    assert "#define KC EXECUTION_PLAN_V4_WRAM_PANEL_KC" in source
    assert "#define NC EXECUTION_PLAN_V4_WRAM_PANEL_NC" in source
    assert "#define B_PANEL_DATA_BYTES (KC * NC * sizeof(float))" in source
    assert "#define B_PANEL_ROW_STRIDE_BYTES (NC * sizeof(float))" in source
    assert "#define A_BUFFER_DATA_BYTES (KC * sizeof(float))" in source
    assert "#define OUTPUT_BUFFER_DATA_BYTES (NC * sizeof(uint32_t))" in source
    assert "#define UNALIGNED_SCRATCH_BYTES EXECUTION_PLAN_V4_WRAM_PANEL_UNALIGNED_SCRATCH_BYTES" in source
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


def test_wram_panel_facts_count_four_real_products_for_a_full_panel() -> None:
    facts = _wram_panel_operation_facts(
        (_work_unit(m_size=2, n_size=32, k_size=64),),
        numeric_policy="split_complex_float32_v1",
        tasklets_per_dpu=1,
    )

    assert facts == {
        "origin": "wram_panel_algorithm_v1",
        "lane_count": 4,
        "operand_read_helper_calls_exact": 24,
        "output_partial_read_helper_calls_exact": 0,
        "output_write_helper_calls_exact": 8,
        "mram_requested_payload_bytes_exact": 35_840,
        "mram_aligned_transfer_bytes_estimate": 35_840,
        "barrier_events_exact": 24,
        "barrier_tasklet_calls_exact": 24,
        "real_mac_count_exact": 16_384,
        "wram_shared_bytes_exact": 8_192,
        "wram_private_bytes_per_tasklet_exact": 672,
        "wram_active_bytes_exact": 8_864,
        "mram_helper_count_scope": "source_level_helper_calls",
        "mram_aligned_bytes_scope": "geometric_aligned_span_estimate",
    }


def test_wram_panel_facts_account_for_tail_helpers_and_tasklets() -> None:
    facts = _wram_panel_operation_facts(
        (_work_unit(m_size=3, n_size=35, k_size=65),),
        numeric_policy="split_complex_int8_shared_scale_v1",
        tasklets_per_dpu=8,
    )

    assert facts["operand_read_helper_calls_exact"] > 0
    assert facts["output_partial_read_helper_calls_exact"] > 0
    assert facts["mram_aligned_transfer_bytes_estimate"] >= facts[
        "mram_requested_payload_bytes_exact"
    ]
    assert facts["barrier_tasklet_calls_exact"] == 8 * facts["barrier_events_exact"]
    assert facts["real_mac_count_exact"] == 4 * 3 * 35 * 65
    assert facts["wram_active_bytes_exact"] == 8_192 + 8 * 672


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
