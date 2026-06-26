from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


DENSE_BRIDGE_SCHEMA_VERSION = "dense_bridge_v1"
DENSE_BRIDGE_ID = "upmem_dense_bridge_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-executing SimplePIM dense bridge stub")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--backend-id", default="simplepim_external_stub")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    started = time.perf_counter()
    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] | None = None
    try:
        manifest = json.loads(input_path.read_text(encoding="utf-8"))
        _validate_input_manifest(manifest, input_path.parent)
        payload = _output_manifest(
            backend=args.backend_id,
            status="stub_executed",
            input_manifest_path=input_path,
            manifest=manifest,
            total_time_s=float(time.perf_counter() - started),
            error=None,
            reason="external_stub_contract_executed",
            error_type=None,
        )
        _write_json(output_path, payload)
        return 0
    except Exception as exc:
        payload = _output_manifest(
            backend=args.backend_id,
            status="failed",
            input_manifest_path=input_path,
            manifest=manifest,
            total_time_s=float(time.perf_counter() - started),
            error=str(exc),
            reason="simplepim_external_stub_failed",
            error_type="simplepim_external_stub_failed",
        )
        _write_json(output_path, payload)
        return 1


def _validate_input_manifest(manifest: dict[str, Any], bridge_dir: Path) -> None:
    if manifest.get("schema_version") != DENSE_BRIDGE_SCHEMA_VERSION:
        raise ValueError("unsupported dense bridge schema_version")
    if manifest.get("bridge_id") != DENSE_BRIDGE_ID:
        raise ValueError("unsupported dense bridge_id")
    if manifest.get("manifest_kind") != "dense_bridge_input":
        raise ValueError("input manifest_kind must be dense_bridge_input")
    for role in ("left", "right"):
        blob = manifest["operands"][role]
        array = np.load(_resolve_manifest_path(bridge_dir, str(blob["relative_path"])), allow_pickle=False)
        _validate_blob(array, blob, role)
    expected = manifest["expected_output"]
    expected_array = np.load(_resolve_manifest_path(bridge_dir, str(expected["relative_path"])), allow_pickle=False)
    _validate_blob(expected_array, expected, "expected_output")


def _validate_blob(array: np.ndarray, metadata: dict[str, Any], role: str) -> None:
    expected_shape = tuple(int(dim) for dim in metadata["shape"])
    expected_dtype = np.dtype(str(metadata["dtype"]))
    if tuple(array.shape) != expected_shape:
        raise ValueError(f"{role} blob shape {array.shape} does not match manifest shape {expected_shape}")
    if array.dtype != expected_dtype:
        raise ValueError(f"{role} blob dtype {array.dtype} does not match manifest dtype {expected_dtype}")


def _resolve_manifest_path(base_dir: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Bridge manifest path must be relative and stay inside the bridge directory: {relative_path}")
    return base_dir / rel


def _output_manifest(
    *,
    backend: str,
    status: str,
    input_manifest_path: Path,
    manifest: dict[str, Any] | None,
    total_time_s: float,
    error: str | None,
    reason: str,
    error_type: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": DENSE_BRIDGE_SCHEMA_VERSION,
        "bridge_id": DENSE_BRIDGE_ID,
        "manifest_kind": "dense_bridge_output",
        "backend": backend,
        "status": status,
        "input_manifest": input_manifest_path.name,
        "route_id": str(manifest.get("route_id", "")) if manifest is not None else "",
        "task_id": str(manifest.get("task_id", "")) if manifest is not None else "",
        "output_blob": None,
        "accumulator_blob": None,
        "validation_metrics": {
            "status": "not_applicable",
            "reason": "stub_writes_no_output_blob",
        },
        "compute_time_s": 0.0,
        "write_time_s": 0.0,
        "total_time_s": total_time_s,
        "external_command_executed": True,
        "execution_implemented": False,
        "error": error,
        "metadata": {
            "reason": reason,
            "error_type": error_type,
            "native_kernel_executed": False,
            "simplepim_or_native_execution_implemented": False,
            "stub_note": "External bridge contract validated; no SimplePIM or native UPMEM kernel executed.",
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
