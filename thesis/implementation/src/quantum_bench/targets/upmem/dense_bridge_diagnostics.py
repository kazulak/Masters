from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict
from quantum_bench.targets.upmem.dense_bridge import run_mock_dense_bridge


DENSE_BRIDGE_DIAGNOSTICS_SCHEMA_VERSION = "dense_bridge_validation_diagnostics_v1"


def write_dense_bridge_validation_diagnostics(
    bridge_dir: Path,
    *,
    case_id: str | None = None,
    task_index: int | None = None,
    task_id: str | None = None,
    compare_mock: bool = False,
) -> JsonDict:
    diagnostics = dense_bridge_validation_diagnostics(
        bridge_dir,
        case_id=case_id,
        task_index=task_index,
        task_id=task_id,
        compare_mock=compare_mock,
    )
    write_json(bridge_dir / "validation_diagnostics.json", diagnostics)
    return diagnostics


def dense_bridge_validation_diagnostics(
    bridge_dir: Path,
    *,
    case_id: str | None = None,
    task_index: int | None = None,
    task_id: str | None = None,
    compare_mock: bool = False,
) -> JsonDict:
    bridge_dir = bridge_dir.resolve()
    try:
        manifest = _read_json(bridge_dir / "input_manifest.json")
        output_manifest = _read_optional_json(bridge_dir / "output_manifest.json")
        left = _load_manifest_array(bridge_dir, manifest["operands"]["left"]["relative_path"])
        right = _load_manifest_array(bridge_dir, manifest["operands"]["right"]["relative_path"])
        expected = _load_manifest_array(bridge_dir, manifest["expected_output"]["relative_path"])
        direct = reconstruct_dense_bridge_output(manifest, left, right)
        simulator_output = _load_output_array(bridge_dir, output_manifest)

        comparisons: JsonDict = {
            "direct_python_reconstruction_vs_expected": compare_arrays(direct, expected),
        }
        conclusion = "direct_reconstruction_matches_expected"
        if simulator_output is not None:
            comparisons["simulator_output_vs_expected"] = compare_arrays(simulator_output, expected)
            comparisons["simulator_output_vs_direct_python_reconstruction"] = compare_arrays(simulator_output, direct)
            if not comparisons["simulator_output_vs_expected"]["passed"]:
                conclusion = "simulator_output_mismatch"
        else:
            comparisons["simulator_output_vs_expected"] = {"status": "not_available", "reason": "simulator_output_blob_missing"}

        mock_payload = None
        if compare_mock:
            mock_payload = _run_mock_comparison(bridge_dir, expected, direct)
            comparisons["mock_output_vs_expected"] = mock_payload["mock_output_vs_expected"]
            if mock_payload["mock_status"] != "mock_executed":
                conclusion = "mock_output_mismatch"

        return {
            "schema_version": DENSE_BRIDGE_DIAGNOSTICS_SCHEMA_VERSION,
            "case_id": case_id,
            "task_index": task_index,
            "task_id": task_id or manifest.get("task_id"),
            "input_tensor_ids": manifest.get("input_tensor_ids"),
            "output_tensor_id": manifest.get("output_tensor_id"),
            "output_labels": manifest.get("output_labels"),
            "gemm": {
                "m": manifest.get("gemm_m"),
                "k": manifest.get("gemm_k"),
                "n": manifest.get("gemm_n"),
            },
            "labels": {
                "left": manifest.get("left_labels"),
                "right": manifest.get("right_labels"),
                "contracted": manifest.get("contracted_labels"),
                "left_free": manifest.get("left_free_labels"),
                "right_free": manifest.get("right_free_labels"),
                "gemm_output": manifest.get("gemm_output_labels"),
                "output": manifest.get("output_labels"),
            },
            "operands": {
                "left": _array_payload(left, manifest["operands"]["left"]),
                "right": _array_payload(right, manifest["operands"]["right"]),
            },
            "expected_output": _array_payload(expected, manifest["expected_output"]),
            "simulator_output": _array_payload(simulator_output, (output_manifest or {}).get("output_blob")) if simulator_output is not None else None,
            "conversion": {
                "left": _conversion_payload(manifest, "left"),
                "right": _conversion_payload(manifest, "right"),
            },
            "complex_execution_mode": _complex_mode(manifest),
            "comparisons": comparisons,
            "mock_comparison": mock_payload,
            "conclusion": conclusion,
            "artifact_paths": {
                "input_manifest": "input_manifest.json",
                "output_manifest": "output_manifest.json" if (bridge_dir / "output_manifest.json").exists() else None,
                "diagnostics": "validation_diagnostics.json",
            },
        }
    except Exception as exc:
        return {
            "schema_version": DENSE_BRIDGE_DIAGNOSTICS_SCHEMA_VERSION,
            "case_id": case_id,
            "task_index": task_index,
            "task_id": task_id,
            "conclusion": "diagnostic_input_invalid",
            "error": str(exc),
        }


def reconstruct_dense_bridge_output(manifest: JsonDict, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_deq = dict(manifest["dequantization"]["left"])
    right_deq = dict(manifest["dequantization"]["right"])
    left_scale = float(left_deq["scale"])
    right_scale = float(right_deq["scale"])
    left_rep = str(left_deq["representation"])
    right_rep = str(right_deq["representation"])
    if left_rep == "real" and right_rep == "real":
        matrix = left.astype(np.int32) @ right.astype(np.int32)
        return _restore_output_order(matrix.astype(np.float64) * left_scale * right_scale, manifest)
    if left_rep == "split_complex_real_imag" and right_rep == "split_complex_real_imag":
        ar_br = left[..., 0].astype(np.int32) @ right[..., 0].astype(np.int32)
        ai_bi = left[..., 1].astype(np.int32) @ right[..., 1].astype(np.int32)
        ar_bi = left[..., 0].astype(np.int32) @ right[..., 1].astype(np.int32)
        ai_br = left[..., 1].astype(np.int32) @ right[..., 0].astype(np.int32)
        matrix = ((ar_br * left_scale * right_scale) - (ai_bi * left_scale * right_scale)) + 1j * (
            (ar_bi * left_scale * right_scale) + (ai_br * left_scale * right_scale)
        )
        return _restore_output_order(matrix, manifest)
    raise ValueError(f"Unsupported dense bridge representations: {left_rep}, {right_rep}")


def compare_arrays(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1.0e-9, rtol: float = 1.0e-9) -> JsonDict:
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "shape_mismatch": True,
            "actual_shape": tuple(int(dim) for dim in actual.shape),
            "expected_shape": tuple(int(dim) for dim in expected.shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
        }
    diff = actual - expected
    abs_diff = np.abs(diff)
    max_index = tuple(int(item) for item in np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)) if abs_diff.size else ()
    l2_error = float(np.linalg.norm(diff.ravel())) if diff.size else 0.0
    expected_norm = float(np.linalg.norm(expected.ravel()))
    relative_l2_error = 0.0 if expected_norm == 0.0 and l2_error == 0.0 else (None if expected_norm == 0.0 else l2_error / expected_norm)
    return {
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "shape_mismatch": False,
        "actual_shape": tuple(int(dim) for dim in actual.shape),
        "expected_shape": tuple(int(dim) for dim in expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "max_abs_error": float(abs_diff[max_index]) if max_index else 0.0,
        "l2_error": l2_error,
        "relative_l2_error": relative_l2_error,
        "max_abs_error_index": max_index,
        "expected_value_at_max_error": _scalar_payload(expected[max_index]) if max_index else None,
        "actual_value_at_max_error": _scalar_payload(actual[max_index]) if max_index else None,
        "policy": {"comparison": "np.allclose", "atol": atol, "rtol": rtol},
    }


def _run_mock_comparison(bridge_dir: Path, expected: np.ndarray, direct: np.ndarray) -> JsonDict:
    mock_dir = bridge_dir / "diagnostics" / "mock_bridge"
    if mock_dir.exists():
        shutil.rmtree(mock_dir)
    mock_dir.mkdir(parents=True)
    shutil.copy2(bridge_dir / "input_manifest.json", mock_dir / "input_manifest.json")
    shutil.copytree(bridge_dir / "operands", mock_dir / "operands")
    shutil.copytree(bridge_dir / "references", mock_dir / "references")
    result = run_mock_dense_bridge(mock_dir / "input_manifest.json")
    mock_output = None
    if result.output_blob_path is not None:
        mock_output = np.load(result.output_blob_path, allow_pickle=False)
    comparison = compare_arrays(mock_output, expected) if mock_output is not None else {"status": "not_available", "reason": result.error}
    direct_comparison = compare_arrays(mock_output, direct) if mock_output is not None else {"status": "not_available", "reason": result.error}
    return {
        "mock_status": result.status,
        "mock_error": result.error,
        "mock_output_vs_expected": comparison,
        "mock_output_vs_direct_python_reconstruction": direct_comparison,
        "artifact_paths": {
            "mock_input_manifest": "diagnostics/mock_bridge/input_manifest.json",
            "mock_output_manifest": "diagnostics/mock_bridge/output_manifest.json",
            "mock_output_blob": "diagnostics/mock_bridge/outputs/mock_dequantized_output.npy" if mock_output is not None else None,
        },
    }


def _restore_output_order(matrix_output: np.ndarray, manifest: JsonDict) -> np.ndarray:
    left_free = tuple(int(label) for label in manifest["left_free_labels"])
    right_free = tuple(int(label) for label in manifest["right_free_labels"])
    output_labels = tuple(int(label) for label in manifest["output_labels"])
    output_shape = tuple(int(dim) for dim in manifest["output_shape"])
    gemm_labels = left_free + right_free
    gemm_shape = tuple(output_shape[output_labels.index(label)] for label in gemm_labels)
    tensor_output = np.asarray(matrix_output).reshape(gemm_shape)
    if gemm_labels == output_labels:
        return tensor_output
    axes = tuple(gemm_labels.index(label) for label in output_labels)
    return np.transpose(tensor_output, axes)


def _load_output_array(bridge_dir: Path, output_manifest: JsonDict | None) -> np.ndarray | None:
    if not output_manifest or not output_manifest.get("output_blob"):
        return None
    return _load_manifest_array(bridge_dir, output_manifest["output_blob"]["relative_path"])


def _load_manifest_array(bridge_dir: Path, relative_path: str) -> np.ndarray:
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Bridge diagnostic path escapes bridge directory: {relative_path}")
    return np.load(bridge_dir / rel, allow_pickle=False)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return _read_json(path)


def _array_payload(array: np.ndarray | None, metadata: Any) -> JsonDict | None:
    if array is None:
        return None
    payload = {
        "shape": tuple(int(dim) for dim in array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }
    if isinstance(metadata, dict):
        payload["manifest_shape"] = metadata.get("shape")
        payload["manifest_dtype"] = metadata.get("dtype")
        payload["relative_path"] = metadata.get("relative_path")
        payload["representation"] = metadata.get("representation")
    return payload


def _conversion_payload(manifest: JsonDict, side: str) -> JsonDict:
    deq = dict(manifest["dequantization"][side])
    record = dict((manifest.get("conversion_records") or {}).get(side) or {})
    return {
        "representation": deq.get("representation"),
        "route_dtype": deq.get("route_dtype"),
        "scale": deq.get("scale"),
        "zero_point": deq.get("zero_point"),
        "source_dtype": record.get("source_dtype"),
        "source_shape": record.get("shape"),
        "quantization_error": record.get("quantization_error"),
        "dequantization_error": record.get("dequantization_error"),
    }


def _complex_mode(manifest: JsonDict) -> str:
    left = str(manifest["dequantization"]["left"]["representation"])
    right = str(manifest["dequantization"]["right"]["representation"])
    if left == "real" and right == "real":
        return "real_single_gemm"
    if left == "split_complex_real_imag" and right == "split_complex_real_imag":
        return "split_complex_four_gemm"
    return f"unsupported:{left},{right}"


def _scalar_payload(value: Any) -> Any:
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, complex):
        return {"real": float(scalar.real), "imag": float(scalar.imag)}
    if isinstance(scalar, float):
        return float(scalar)
    if isinstance(scalar, int):
        return int(scalar)
    return scalar
