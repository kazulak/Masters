from __future__ import annotations

# Makefile glue for thesis evidence shortcuts, not a public benchmark CLI.

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Small helpers for Makefile evidence shortcuts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    suite_parser = sub.add_parser("suite-id")
    suite_parser.add_argument("run_dir")

    gpu_parser = sub.add_parser("check-gpu")
    gpu_parser.add_argument("run_dir")

    upmem_parser = sub.add_parser("check-upmem")
    upmem_parser.add_argument("run_dir")

    upmem_hardware_parser = sub.add_parser("check-upmem-hardware")
    upmem_hardware_parser.add_argument("run_dir")

    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if args.command == "suite-id":
        print(_suite_id(run_dir))
        return 0
    if args.command == "check-gpu":
        return _check_gpu(run_dir)
    if args.command == "check-upmem":
        return _check_upmem(run_dir)
    if args.command == "check-upmem-hardware":
        return _check_upmem_hardware(run_dir)
    raise AssertionError(args.command)


def _suite_id(run_dir: Path) -> str:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing run manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite_id = payload.get("suite_id")
    if not suite_id:
        raise SystemExit(
            f"run_manifest.json does not contain suite_id: {manifest_path}"
        )
    return str(suite_id)


def _check_gpu(run_dir: Path) -> int:
    records = _load_records(run_dir)
    matches = [
        record
        for record in records
        if record.get("contraction_execution_target") == "gpu"
        and _truthy(record.get("gpu_backend_verified"))
        and _truthy(record.get("gpu_program_executed"))
        and record.get("gpu_device_name")
        and record.get("validation_status") == "passed"
    ]
    if not matches:
        print(
            "GPU blocker: no verified GPU benchmark row found in runs/latest/normalized_records.jsonl. "
            "Inspect build/gpu_verification/ and the latest run summary.",
            file=sys.stderr,
        )
        return 2
    devices = sorted(
        {str(record.get("gpu_device_name") or "unknown") for record in matches}
    )
    print(f"Verified GPU benchmark rows: {len(matches)}; devices: {', '.join(devices)}")
    return 0


def _check_upmem(run_dir: Path) -> int:
    records = _load_records(run_dir)
    matches = [
        record
        for record in records
        if record.get("contraction_execution_target") == "upmem"
        and record.get("upmem_execution_mode") == "sdk_simulator"
        and _truthy(record.get("upmem_program_executed"))
        and not _truthy(record.get("cpu_fallback_used"))
    ]
    if not matches:
        print(
            "UPMEM SDK simulator blocker: no strict UPMEM SDK simulator benchmark row found in "
            "runs/latest/normalized_records.jsonl. Inspect the latest run summary and UPMEM environment check.",
            file=sys.stderr,
        )
        return 2
    print(f"Verified UPMEM SDK simulator benchmark rows: {len(matches)}")
    return 0


def _check_upmem_hardware(run_dir: Path) -> int:
    records = _load_records(run_dir)
    matches = [
        record
        for record in records
        if record.get("contraction_execution_target") == "upmem"
        and record.get("target_requested") == "hardware"
        and record.get("target_observed") == "hardware"
        and record.get("backend_id") == "upmem_sdk_hardware_dense"
        and record.get("hardware_profile_version") == "hardware_mvp_l1_v1"
        and record.get("requested_dpu_count") == 1
        and record.get("allocated_dpu_count") == 1
        and record.get("tasklets_per_dpu") == 1
        and _truthy(record.get("hardware_allocation_verified"))
        and _truthy(record.get("hardware_kernel_executed"))
        and not _truthy(record.get("simulator_kernel_executed"))
        and not _truthy(record.get("cpu_fallback_used"))
        and _truthy(record.get("exact_integer_match"))
        and record.get("validation_status") == "passed"
        and not _truthy(record.get("hardware_speedup_applicable"))
    ]
    if not matches:
        print(
            "UPMEM hardware MVP blocker: no verified single-DPU exact hardware row found. "
            "Inspect normalized_records.jsonl, output manifests, and retained native logs.",
            file=sys.stderr,
        )
        return 2
    print(
        f"Verified UPMEM hardware MVP rows: {len(matches)}; single-DPU exact functionality evidence only"
    )
    return 0


def _load_records(run_dir: Path) -> list[dict[str, Any]]:
    records_path = run_dir / "normalized_records.jsonl"
    if not records_path.exists():
        raise SystemExit(f"Missing normalized records: {records_path}")
    records: list[dict[str, Any]] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"No normalized records found in: {records_path}")
    return records


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
