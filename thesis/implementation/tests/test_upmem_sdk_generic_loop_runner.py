from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_generic_loop_runner_packs_all_rank_sixteen_entries() -> None:
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

    assert runner.DEFAULT_MAX_RANK == max_rank == 16
    assert len(packed) == struct.calcsize("<" + "I" * unsigned_count + "i" * (len(signed_keys) * max_rank))
    for index, key in enumerate(unsigned_keys):
        start = 9 + index * max_rank
        assert words[start : start + max_rank] == tuple(native[key])
    for index, key in enumerate(signed_keys):
        start = unsigned_count + index * max_rank
        assert words[start : start + max_rank] == tuple(native[key])


def test_tiled_metadata_counts_aligned_mram_traffic() -> None:
    runner = _load_runner_module()

    metadata = runner._tile_metadata({"output_element_count": 3, "contracted_combination_count": 2})

    assert metadata["generic_output_tile_count"] == 1
    assert metadata["mram_tiled_task_count"] == 0
    assert metadata["mram_read_bytes_model"] == 3 * 2 * 2 * 8
    assert metadata["mram_write_bytes_model"] == 16


def _write_non_gemm_manifest(root: Path, *, mode: str) -> Path:
    left_shape = (9, 8, 8)
    right_shape = (8, 8, 8)
    output_shape = (9, 8, 8, 8)
    left = np.arange(np.prod(left_shape), dtype=np.float32).reshape(left_shape) / 97.0
    right = np.arange(np.prod(right_shape), dtype=np.float32).reshape(right_shape) / 89.0
    if mode == "int8_scaled":
        left_blob = np.rint(left * 10).astype(np.int8)
        right_blob = np.rint(right * 10).astype(np.int8)
        expected = np.einsum("abc,cde->abde", left_blob.astype(np.int32), right_blob.astype(np.int32), optimize=False)
        operand_mode = mode
        dequantization = {"output_scale": 1.0, "left": {"scale": 1.0, "zero_point": 0}, "right": {"scale": 1.0, "zero_point": 0}}
        quantization_mode = "per_task_input_quantize"
    else:
        left_blob = left.astype(np.float32)
        right_blob = right.astype(np.float32)
        expected = np.einsum("abc,cde->abde", left_blob, right_blob, optimize=False).astype(np.float32)
        operand_mode = "float32_no_quant"
        dequantization = {"output_scale": 1.0, "left": {"scale": 1.0, "zero_point": 0}, "right": {"scale": 1.0, "zero_point": 0}}
        quantization_mode = "none"

    operands = root / "operands"
    references = root / "references"
    operands.mkdir(parents=True)
    references.mkdir(parents=True)
    np.save(operands / "left.npy", left_blob, allow_pickle=False)
    np.save(operands / "right.npy", right_blob, allow_pickle=False)
    np.save(references / "expected.npy", expected, allow_pickle=False)
    def blob(path, array, role):
        return {
            "relative_path": path,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "role": role,
        }
    metadata = {
        "operand_mode": operand_mode,
        "quantization_mode": quantization_mode,
        "native_index_metadata": True,
        "simplepim_api_used": False,
    }
    native = {
        "operand_mode": operand_mode,
        "left_rank": 3,
        "right_rank": 3,
        "output_rank": 4,
        "contracted_rank": 1,
        "left_shape": list(left_shape),
        "right_shape": list(right_shape),
        "output_shape": list(output_shape),
        "contracted_dims": [8],
        "left_strides": [64, 8, 1],
        "right_strides": [64, 8, 1],
        "output_strides": [512, 64, 8, 1],
        "output_to_left_axes": [0, 1, -1, -1],
        "output_to_right_axes": [-1, -1, 1, 2],
        "contracted_to_left_axes": [2],
        "contracted_to_right_axes": [0],
        "output_element_count": 4608,
        "contracted_combination_count": 8,
    }
    manifest = {
        "schema_version": "generic_contraction_bridge_v1",
        "bridge_id": "upmem_generic_contraction_bridge_v1",
        "manifest_kind": "generic_contraction_bridge_input",
        "route_id": "generic_loop_fallback",
        "backend_id": "upmem_sdk_simulator_generic_loop",
        "kernel_family": "generic_loop_fallback",
        "task_id": "non_gemm_9x8x8",
        "operands": {"left": blob("operands/left.npy", left_blob, "left"), "right": blob("operands/right.npy", right_blob, "right")},
        "expected_quantized_reference_output": blob("references/expected.npy", expected, "expected"),
        "full_precision_reference_output": blob("references/expected.npy", expected, "full_precision"),
        "input_shapes": [list(left_shape), list(right_shape)],
        "output_shape": list(output_shape),
        "native_index_metadata": native,
        "metadata": metadata,
        "dequantization": dequantization,
    }
    path = root / "input_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ("float32_no_quant", "int8_scaled"))
def test_sdk_runner_real_non_gemm_shape_when_sdk_tools_are_present(tmp_path: Path, monkeypatch, mode: str) -> None:
    runner = _load_runner_module()
    missing = tuple(name for name, path in runner._required_tools(os.environ).items() if path is None)
    if missing:
        pytest.skip(f"UPMEM SDK simulator unavailable: missing {', '.join(missing)}")

    manifest = _write_non_gemm_manifest(tmp_path, mode=mode)
    output = tmp_path / "output_manifest.json"
    monkeypatch.setattr(sys, "argv", ["runner", "--input-manifest", str(manifest), "--output-manifest", str(output)])
    monkeypatch.setenv("UPMEM_GENERIC_MAX_ELEMS", "65536")
    assert runner.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "upmem_sdk_simulator_generic_loop_executed"
    assert result["validation_metrics"]["passed"] is True
    metadata = result["metadata"]
    assert metadata["generic_kernel_strategy"] == "mram_resident_output_tiled_v1"
    assert metadata["generic_output_tile_elements"] == 256
    assert metadata["generic_output_tile_count"] == 18
    assert metadata["mram_tiled_task_count"] == 1
    assert metadata["mram_read_bytes_model"] == 4608 * 8 * 2 * 8
    assert metadata["mram_write_bytes_model"] == 4608 * 4
