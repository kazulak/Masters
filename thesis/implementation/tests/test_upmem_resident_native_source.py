from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/upmem/simplepim/upmem_sdk_generic_loop_resident/dpu.c"


def _dpu_compiler() -> str:
    compiler = shutil.which("dpu-upmem-dpurte-clang")
    if compiler is None:
        pytest.skip("UPMEM SDK compiler is unavailable")
    return compiler


def _compile_variant(tmp_path: Path, abi_version: int) -> str:
    compiler = _dpu_compiler()
    output = tmp_path / f"resident-v{abi_version}.o"
    defines = [
        f"-DRESIDENT_OPERATION_ABI_VERSION={abi_version}",
        "-DNR_TASKLETS=1",
        "-DRESIDENT_COMPLETION_VERSION=1",
        "-DUPMEM_GENERIC_HARDWARE_MVP=1",
    ]
    subprocess.run(
        [compiler, "-O2", *defines, "-E", "-o", str(output.with_suffix(".i")), str(SOURCE)],
        check=True,
    )
    subprocess.run(
        [compiler, "-O2", *defines, "-c", "-o", str(output), str(SOURCE)],
        check=True,
    )
    return output.with_suffix(".i").read_text(encoding="ascii")


def test_resident_output_write_implementation_is_abi_selected(tmp_path: Path) -> None:
    source = SOURCE.read_text(encoding="ascii")
    v1 = _compile_variant(tmp_path, 1)
    v2 = _compile_variant(tmp_path, 2)

    assert "resident_output_tile[NR_TASKLETS][RESIDENT_OUTPUT_TILE_ELEMS + 2u]" in source
    assert "resident_write_output_range(" not in v1
    assert re.search(r"mram_write\(\s*resident_output_tile\[tid\]", v1)

    assert "resident_write_output_range(" in v2
    assert v2.count("resident_write_output_range(") == 4
    assert "resident_output_window[NR_TASKLETS][2]" in source


def test_resident_completion_cycles_are_populated_for_both_abis(tmp_path: Path) -> None:
    v1 = _compile_variant(tmp_path, 1)
    v2 = _compile_variant(tmp_path, 2)

    assert v1.count("RESIDENT_COMPLETION.dpu_run_time_cycles = 0ULL") == 1
    assert v2.count("RESIDENT_COMPLETION.dpu_run_time_cycles = 0ULL") == 1
    assert v1.count("RESIDENT_COMPLETION.dpu_run_time_cycles = (uint64_t)(end_cycles - start_cycles_shared)") == 1
    assert v2.count("RESIDENT_COMPLETION.dpu_run_time_cycles = (uint64_t)(end_cycles - start_cycles_shared)") == 1


def test_resident_makefile_defaults_to_explicit_v1_build() -> None:
    makefile = (SOURCE.parent / "Makefile").read_text(encoding="ascii")

    assert ".PHONY: all v1 v2 clean" in makefile
    assert "all: v1" in makefile
    assert "v1: bin/host bin/dpu_resident" in makefile
    assert "v2: bin/host_v2 bin/dpu_resident_v2" in makefile


def test_resident_output_tile_rejects_odd_size(tmp_path: Path) -> None:
    compiler = _dpu_compiler()
    completed = subprocess.run(
        [
            compiler,
            "-O2",
            "-DRESIDENT_OPERATION_ABI_VERSION=1",
            "-DNR_TASKLETS=1",
            "-DRESIDENT_COMPLETION_VERSION=1",
            "-DUPMEM_GENERIC_HARDWARE_MVP=1",
            "-DRESIDENT_OUTPUT_TILE_ELEMS=3",
            "-c",
            "-o",
            str(tmp_path / "resident-odd.o"),
            str(SOURCE),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "positive and even" in completed.stderr
