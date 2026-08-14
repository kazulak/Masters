from __future__ import annotations

from typing import Any

import numpy as np

from quantum_bench.core.records import ContractionTask
from quantum_bench.formats.fixed_point import FixedPointSpec, quantize_fixed_point
from quantum_bench.tn.contract import contract_binary_task


class Float32RealPolicy:
    name = "float32_real"

    def contract(
        self,
        task: ContractionTask,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        left_array = _real_float32(left)
        right_array = _real_float32(right)
        output = contract_binary_task(task, left_array, right_array, dtype=np.float32)
        return np.asarray(output, dtype=np.float32), {
            "input_dtype": "float32",
            "accumulator_dtype": "float32",
            "quantization": False,
        }


class HostPackedInt8Policy:
    """Reference for host packing plus int8 x int8 -> int32 accumulation.

    The result is dequantized before entering the store so the next task can
    be packed independently. This deliberately models the simple CPU policy;
    a resident packed store can replace it without changing the executor.
    """

    name = "host_packed_int8_per_task_v1"

    def contract(
        self,
        task: ContractionTask,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        left_q = quantize_fixed_point(
            _real_float32(left), FixedPointSpec(route_dtype="int8")
        )
        right_q = quantize_fixed_point(
            _real_float32(right), FixedPointSpec(route_dtype="int8")
        )
        integer_output = contract_binary_task(
            task, left_q.array, right_q.array, dtype=np.int32
        )
        scale = left_q.record.scale * right_q.record.scale
        output = np.asarray(integer_output, dtype=np.float32) * np.float32(scale)
        return output, {
            "input_dtype": "int8",
            "accumulator_dtype": "int32",
            "quantization": True,
            "left_scale": float(left_q.record.scale),
            "right_scale": float(right_q.record.scale),
            "packed_input_bytes": int(
                left_q.record.converted_bytes + right_q.record.converted_bytes
            ),
            "zero_scale_fallback_used": bool(
                left_q.record.scale == 1.0 and not np.any(left_q.array)
            )
            or bool(right_q.record.scale == 1.0 and not np.any(right_q.array)),
            "saturation_count": int(
                left_q.record.saturation_count + right_q.record.saturation_count
            ),
        }


def _real_float32(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if np.iscomplexobj(value) and np.any(np.imag(value) != 0):
        raise ValueError("Numeric policy requires real-valued tensors")
    return np.asarray(np.real(value), dtype=np.float32)
