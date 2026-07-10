from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

from quantum_bench.routing import GenericTaskPreparationCaps


ROOT = Path(__file__).resolve().parents[1]


def _load_runner_module():
    runner_path = ROOT / "native" / "upmem" / "simplepim" / "upmem_sdk_generic_loop_runner.py"
    spec = importlib.util.spec_from_file_location("upmem_sdk_generic_loop_runner_for_tests", runner_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generic_loop_runner_packs_all_default_rank_seven_entries() -> None:
    runner = _load_runner_module()
    max_rank = GenericTaskPreparationCaps().max_rank
    unsigned_keys = (
        "left_shape",
        "right_shape",
        "output_shape",
        "contracted_dims",
        "left_strides",
        "right_strides",
        "output_strides",
    )
    signed_keys = (
        "output_to_left_axes",
        "output_to_right_axes",
        "contracted_to_left_axes",
        "contracted_to_right_axes",
    )
    native = {
        "left_rank": max_rank,
        "right_rank": max_rank,
        "output_rank": max_rank,
        "contracted_rank": max_rank,
        "left_shape": [1] * max_rank,
        "right_shape": [1] * max_rank,
        "output_shape": [1] * max_rank,
        "contracted_dims": [1] * max_rank,
        "left_strides": list(range(max_rank)),
        "right_strides": list(range(10, 10 + max_rank)),
        "output_strides": list(range(20, 20 + max_rank)),
        "output_to_left_axes": list(range(30, 30 + max_rank)),
        "output_to_right_axes": list(range(40, 40 + max_rank)),
        "contracted_to_left_axes": list(range(50, 50 + max_rank)),
        "contracted_to_right_axes": list(range(60, 60 + max_rank)),
        "output_element_count": 1,
        "contracted_combination_count": 1,
    }

    packed = runner._pack_args(native)
    unsigned_count = 9 + len(unsigned_keys) * max_rank
    words = struct.unpack("<" + "I" * unsigned_count + "i" * (len(signed_keys) * max_rank), packed)

    assert runner.DEFAULT_MAX_RANK == max_rank == 7
    assert len(packed) == struct.calcsize("<" + "I" * unsigned_count + "i" * (len(signed_keys) * max_rank))
    for index, key in enumerate(unsigned_keys):
        start = 9 + index * max_rank
        assert words[start : start + max_rank] == tuple(native[key])
    for index, key in enumerate(signed_keys):
        start = unsigned_count + index * max_rank
        assert words[start : start + max_rank] == tuple(native[key])
