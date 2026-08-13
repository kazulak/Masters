"""Small, hardware-only launcher for the additive execution-plan v3 ABI.

This module is intentionally independent of the Block 1/2 Python adapters.
It validates the fixed binary sidecar, builds an artifact directory keyed by
tasklet count, and invokes the v3 host.  It has no simulator or CPU fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence


V3_HEADER_FORMAT = "<8s16I32s32s"
V3_RECORD_FORMAT = "<8I"
V3_MAGIC = b"UPXDPV3\0"
V3_VERSION = 3
V3_HEADER_BYTES = struct.calcsize(V3_HEADER_FORMAT)
V3_RECORD_BYTES = struct.calcsize(V3_RECORD_FORMAT)
V3_MAX_DPUS = 64
V3_MAX_TASKLETS = 24
V3_MAX_ELEMS = 65536
V3_MRAM_POOL_BYTES = 512 * 1024
V3_OUTPUT_TILE_ELEMS = 2
V3_MAX_WARMUPS = 4
V3_MAX_REPETITIONS = 16
V3_PARTITION_OUTPUT = 1
V3_PARTITION_CONTRACTED = 2
V3_NUMERIC_FLOAT32 = 0
V3_NUMERIC_INT8_REQUANTIZE = 1
V3_NUMERIC_NAMES = {
    V3_NUMERIC_FLOAT32: "float32",
    V3_NUMERIC_INT8_REQUANTIZE: "per_task_resident_requantize",
}


class V3RunnerError(ValueError):
    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(message)


@dataclass(frozen=True)
class V3WorkUnit:
    package_operation_index: int
    operation_id: int
    partition_mode: int
    dpu_id: int
    output_offset: int
    output_elements: int
    contracted_offset: int
    contracted_elements: int


@dataclass(frozen=True)
class V3Plan:
    dpu_count: int
    tasklets_per_dpu: int
    partition_mode: int
    numeric_mode: int
    package_operation_index: int
    operation_id: int
    output_elements: int
    contracted_elements: int
    output_slot: int
    package_sha256: bytes
    operation_sha256: bytes
    work_units: tuple[V3WorkUnit, ...]


def _sha256_bytes(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def _validate_count(name: str, value: int, lower: int, upper: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise V3RunnerError("hardware_profile_violation", f"{name} must be in [{lower},{upper}]")


def _as_work_unit(value: V3WorkUnit | Sequence[int]) -> V3WorkUnit:
    if isinstance(value, V3WorkUnit):
        return value
    if len(value) != 8:
        raise V3RunnerError("execution_plan_compile_failed", "v3 work records contain eight uint32 fields")
    return V3WorkUnit(*(int(item) for item in value))


def pack_v3_sidecar(
    *,
    package_bytes: bytes,
    operation_bytes: bytes,
    dpu_count: int,
    tasklets_per_dpu: int,
    partition_mode: int,
    numeric_mode: int,
    output_elements: int,
    contracted_elements: int,
    output_slot: int,
    work_units: Iterable[V3WorkUnit | Sequence[int]],
    package_operation_index: int = 0,
    operation_id: int = 0,
) -> bytes:
    """Serialize and validate one v3 sidecar."""

    units = tuple(_as_work_unit(item) for item in work_units)
    _validate_count("dpu_count", dpu_count, 1, V3_MAX_DPUS)
    _validate_count("tasklets_per_dpu", tasklets_per_dpu, 1, V3_MAX_TASKLETS)
    _validate_count("work_unit_count", len(units), 1, V3_MAX_DPUS)
    if len(units) != dpu_count:
        raise V3RunnerError("hardware_profile_violation", "v3 has exactly one work record per selected DPU")
    if partition_mode not in (V3_PARTITION_OUTPUT, V3_PARTITION_CONTRACTED):
        raise V3RunnerError("hardware_profile_violation", "unsupported v3 partition mode")
    if numeric_mode not in V3_NUMERIC_NAMES:
        raise V3RunnerError("hardware_profile_violation", "unsupported v3 numeric mode")
    _validate_count("output_elements", output_elements, 1, 0xFFFFFFFF)
    _validate_count("contracted_elements", contracted_elements, 1, 0xFFFFFFFF)
    seen: set[int] = set()
    cursor = 0
    for index, unit in enumerate(units):
        if (
            unit.package_operation_index != package_operation_index
            or unit.operation_id != operation_id
            or unit.partition_mode != partition_mode
            or unit.dpu_id in seen
            or not 0 <= unit.dpu_id < dpu_count
            or unit.output_elements <= 0
            or unit.contracted_elements <= 0
        ):
            raise V3RunnerError("hardware_profile_violation", "invalid or duplicate v3 work-unit identity")
        if partition_mode == V3_PARTITION_OUTPUT:
            if (
                unit.output_offset != cursor
                or unit.output_offset % 2
                or unit.output_offset + unit.output_elements > output_elements
                or unit.contracted_offset != 0
                or unit.contracted_elements != contracted_elements
                or (index + 1 < len(units) and unit.output_elements % 2)
            ):
                raise V3RunnerError("hardware_profile_violation", "v3 output partition is not covered on aligned boundaries")
            cursor += unit.output_elements
        else:
            if (
                unit.output_offset != 0
                or unit.output_elements != output_elements
                or unit.contracted_offset != cursor
                or unit.contracted_offset + unit.contracted_elements > contracted_elements
            ):
                raise V3RunnerError("hardware_profile_violation", "v3 contracted partition is not covered")
            cursor += unit.contracted_elements
        seen.add(unit.dpu_id)
    expected_cursor = output_elements if partition_mode == V3_PARTITION_OUTPUT else contracted_elements
    if cursor != expected_cursor or seen != set(range(dpu_count)):
        raise V3RunnerError("hardware_profile_violation", "v3 partition coverage or DPU density is invalid")
    header = struct.pack(
        V3_HEADER_FORMAT,
        V3_MAGIC,
        V3_VERSION,
        V3_HEADER_BYTES,
        len(units),
        dpu_count,
        tasklets_per_dpu,
        1,
        partition_mode,
        numeric_mode,
        package_operation_index,
        operation_id,
        output_elements,
        contracted_elements,
        output_slot,
        V3_RECORD_BYTES,
        0,
        0,
        _sha256_bytes(package_bytes),
        _sha256_bytes(operation_bytes),
    )
    records = b"".join(struct.pack(V3_RECORD_FORMAT, *unit.__dict__.values()) for unit in units)
    return header + records


def parse_v3_sidecar(payload: bytes, *, expected_tasklets: int | None = None) -> V3Plan:
    if len(payload) < V3_HEADER_BYTES:
        raise V3RunnerError("distributed_plan_v3_parse_failed", "v3 sidecar is truncated")
    fields = struct.unpack_from(V3_HEADER_FORMAT, payload, 0)
    (
        magic, version, header_bytes, work_count, dpu_count, tasklets, provider_count,
        partition_mode, numeric_mode, package_index, operation_id, output_elements,
        contracted_elements, output_slot, record_bytes, reserved0, reserved1,
        package_sha256, operation_sha256,
    ) = fields
    if magic != V3_MAGIC or version != V3_VERSION or header_bytes != V3_HEADER_BYTES or record_bytes != V3_RECORD_BYTES:
        raise V3RunnerError("hardware_profile_violation", "v3 magic, version, header, or record size is invalid")
    if provider_count != 1 or reserved0 != 0 or reserved1 != 0:
        raise V3RunnerError("hardware_profile_violation", "v3 provider or reserved header fields are invalid")
    _validate_count("dpu_count", dpu_count, 1, V3_MAX_DPUS)
    _validate_count("tasklets_per_dpu", tasklets, 1, V3_MAX_TASKLETS)
    if expected_tasklets is not None and tasklets != expected_tasklets:
        raise V3RunnerError("hardware_profile_violation", "sidecar tasklet count differs from the selected build")
    if work_count != dpu_count or len(payload) != V3_HEADER_BYTES + work_count * V3_RECORD_BYTES:
        raise V3RunnerError("hardware_profile_violation", "v3 sidecar length or work-unit count is invalid")
    units = tuple(V3WorkUnit(*struct.unpack_from(V3_RECORD_FORMAT, payload, V3_HEADER_BYTES + i * V3_RECORD_BYTES)) for i in range(work_count))
    # Reuse the serializer's coverage checks without changing the bytes.
    pack_v3_sidecar(
        package_bytes=b"", operation_bytes=b"", dpu_count=dpu_count, tasklets_per_dpu=tasklets,
        partition_mode=partition_mode, numeric_mode=numeric_mode, output_elements=output_elements,
        contracted_elements=contracted_elements, output_slot=output_slot, work_units=units,
        package_operation_index=package_index, operation_id=operation_id,
    )
    return V3Plan(dpu_count, tasklets, partition_mode, numeric_mode, package_index, operation_id,
                  output_elements, contracted_elements, output_slot, package_sha256, operation_sha256, units)


def validate_v3_sidecar(path: Path, *, package_bytes: bytes, operation_bytes: bytes, expected_tasklets: int | None = None) -> V3Plan:
    plan = parse_v3_sidecar(Path(path).read_bytes(), expected_tasklets=expected_tasklets)
    if plan.package_sha256 != _sha256_bytes(package_bytes) or plan.operation_sha256 != _sha256_bytes(operation_bytes):
        raise V3RunnerError("hardware_profile_violation", "v3 sidecar package or operation binding is stale")
    return plan


def _hardware_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if environment is None else environment)
    env.pop("DPU_BACKEND", None)
    return env


def build(build_dir: Path, *, tasklets_per_dpu: int, source_dir: Path | None = None, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    _validate_count("tasklets_per_dpu", tasklets_per_dpu, 1, V3_MAX_TASKLETS)
    source = Path(source_dir or Path(__file__).resolve().parent / "upmem_sdk_execution_plan").resolve()
    if not source.is_dir():
        raise V3RunnerError("native_build_failed", "v3 source directory is missing")
    keyed = Path(build_dir).resolve() / f"tasklets-{tasklets_per_dpu}"
    staged_root = keyed / "simplepim"
    staged = staged_root / "upmem_sdk_execution_plan"
    # The Makefile references the resident kernel and the staged frontier
    # sibling, so preserve the local SimplePIM source-tree shape.
    shutil.copytree(source.parent, staged_root, dirs_exist_ok=True)
    simplepim_root = Path(__file__).resolve().parents[3] / "external" / "SimplePIM"
    command = (
        "make", "clean", "v3", f"NR_TASKLETS={tasklets_per_dpu}",
        f"MAX_ELEMS={V3_MAX_ELEMS}", f"RESIDENT_MRAM_POOL_BYTES={V3_MRAM_POOL_BYTES}",
        f"RESIDENT_OUTPUT_TILE_ELEMS={V3_OUTPUT_TILE_ELEMS}", f"SIMPLEPIM_ROOT={simplepim_root}",
    )
    completed = subprocess.run(command, cwd=staged, env=_hardware_environment(environment), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise V3RunnerError("native_build_failed", (completed.stderr or completed.stdout or "v3 build failed")[-4000:])
    host = staged / "bin" / f"host_upmem_execution_plan_v3_t{tasklets_per_dpu}"
    dpu = staged / "bin" / f"dpu_resident_v3_t{tasklets_per_dpu}"
    initialization = staged / "bin" / "dpu_simplepim_management_init"
    if not host.is_file() or not dpu.is_file() or not initialization.is_file():
        raise V3RunnerError("native_build_failed", "v3 build did not produce tasklet-keyed artifacts")
    if host.parent != initialization.parent:
        raise V3RunnerError("native_build_failed", "v3 initialization binary is not beside the host")
    return {
        "status": "built", "tasklets_per_dpu": tasklets_per_dpu, "build_dir": str(keyed),
        "host_binary": str(host), "dpu_binary": str(dpu),
        "initialization_binary": str(initialization),
        "host_binary_identity": host.name, "dpu_binary_identity": dpu.name,
        "host_binary_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
        "staged_dpu_binary_sha256": hashlib.sha256(dpu.read_bytes()).hexdigest(),
        "initialization_binary_sha256": hashlib.sha256(initialization.read_bytes()).hexdigest(),
        "max_elements": V3_MAX_ELEMS, "mram_pool_bytes": V3_MRAM_POOL_BYTES,
        "output_tile_elements": V3_OUTPUT_TILE_ELEMS, "build_command": list(command),
    }


def execute(
    host_binary: Path,
    *,
    resident_manifest: Path,
    sidecar: Path,
    response: Path,
    tasklets_per_dpu: int,
    warmups: int = 1,
    repetitions: int = 1,
    timeout_s: float = 60.0,
    environment: Mapping[str, str] | None = None,
    policy_reference: Path | None = None,
    policy_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    requested_environment = dict(os.environ if environment is None else environment)
    if requested_environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise V3RunnerError("hardware_opt_in_missing", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    if requested_environment.get("DPU_BACKEND"):
        raise V3RunnerError("hardware_profile_violation", "DPU_BACKEND must be unset for the physical v3 route")
    if not requested_environment.get("UPMEM_HW_RANK_PATH", "").strip():
        raise V3RunnerError("hardware_rank_path_missing", "UPMEM_HW_RANK_PATH is required")
    _validate_count("tasklets_per_dpu", tasklets_per_dpu, 1, V3_MAX_TASKLETS)
    if (
        isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0 or warmups > V3_MAX_WARMUPS
        or isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1
        or repetitions > V3_MAX_REPETITIONS
    ):
        raise V3RunnerError("execution_plan_compile_failed", "invalid v3 repetition bounds")
    if isinstance(policy_tolerance, bool) or not isinstance(policy_tolerance, (float, int)) or not math.isfinite(policy_tolerance) or policy_tolerance < 0:
        raise V3RunnerError("policy_reference_validation_failed", "policy tolerance must be finite and non-negative")
    reference_path = Path(policy_reference or Path(sidecar).resolve().parent / "reference_f32.bin").resolve()
    if not reference_path.is_file():
        raise V3RunnerError("policy_reference_validation_failed", "supplied CPU policy reference is missing")
    reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    command = (
        str(Path(host_binary).resolve()), "--execute-plan", "--resident-package", str(Path(resident_manifest).resolve()),
        "--distributed-plan-v3", str(Path(sidecar).resolve()), "--policy-reference", str(reference_path),
        "--policy-reference-sha256", reference_sha256, "--policy-tolerance", str(float(policy_tolerance)),
        "--response", str(Path(response).resolve()), "--warmups", str(warmups), "--repetitions", str(repetitions),
        "--timeout-s", str(max(1, int(timeout_s))),
    )
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=Path(host_binary).resolve().parent, env=_hardware_environment(requested_environment), capture_output=True, text=True, timeout=timeout_s, check=False)
    if not Path(response).is_file():
        raise V3RunnerError("output_manifest_failed", "v3 native response is missing")
    payload = json.loads(Path(response).read_text(encoding="utf-8"))
    payload["runner_wall_time_s"] = time.perf_counter() - started
    if completed.returncode != 0 or payload.get("status") != "completed":
        raise V3RunnerError(str(payload.get("failure_stage") or "kernel_launch_failed"), str(payload.get("error") or "v3 native execution failed"))
    if payload.get("simulator_kernel_executed") is not False or payload.get("cpu_fallback_used") is not False:
        raise V3RunnerError("output_manifest_failed", "v3 response exposed a simulator or CPU fallback")
    for field in ("native_kernel_executed", "hardware_kernel_executed", "hardware_release_verified", "hardware_allocation_verified"):
        if payload.get(field) is not True:
            raise V3RunnerError("output_manifest_failed", f"v3 response did not verify {field}")
    if payload.get("requested_rank_path") != requested_environment["UPMEM_HW_RANK_PATH"] or payload.get("observed_rank_count") != 1:
        raise V3RunnerError("output_manifest_failed", "v3 response did not verify the requested physical rank")
    partition_strategy = payload.get("partition_strategy")
    if partition_strategy == "output":
        expected_collective = "none"
        expected_reconstruction = "host_owned_range_assembly_v1"
    elif partition_strategy == "contracted":
        expected_collective = "host_mediated_sum_v1"
        expected_reconstruction = "host_float64_reduction_v1"
    else:
        raise V3RunnerError("output_manifest_failed", "v3 response lacks a valid partition strategy")
    expected_providers = {
        "allocation_provider": "upmem_sdk_rank_profile_v1",
        "simplepim_role": "initialization_binary_and_management_state_only",
        "kernel_provider": "thesis_resident_generic_c_v3",
        "transfer_provider": "upmem_sdk_synchronous_v1",
        "collective_provider": expected_collective,
        "reconstruction_provider": expected_reconstruction,
    }
    for field, expected_value in expected_providers.items():
        if payload.get(field) != expected_value:
            raise V3RunnerError(
                "output_manifest_failed",
                f"v3 response did not verify {field}={expected_value}",
            )
    policy = payload.get("policy_reference_validation")
    if not isinstance(policy, Mapping) or policy.get("status") != "passed" or policy.get("passed") is not True or policy.get("reference_sha256") != reference_sha256:
        raise V3RunnerError("policy_reference_validation_failed", "v3 response did not pass the supplied CPU policy reference")
    max_abs_error = policy.get("max_abs_error")
    response_tolerance = policy.get("tolerance")
    if (
        policy.get("finite") is not True
        or not isinstance(max_abs_error, (float, int))
        or not isinstance(response_tolerance, (float, int))
        or not math.isfinite(max_abs_error)
        or not math.isfinite(response_tolerance)
        or max_abs_error < 0
        or max_abs_error > float(policy_tolerance)
        or max_abs_error > response_tolerance
    ):
        raise V3RunnerError("policy_reference_validation_failed", "v3 policy-reference error exceeds its tolerance")
    if (
        not isinstance(payload.get("host_binary_sha256"), str)
        or not isinstance(payload.get("staged_dpu_binary_sha256"), str)
        or not isinstance(payload.get("initialization_binary_sha256"), str)
    ):
        raise V3RunnerError("output_manifest_failed", "v3 response lacks binary hashes")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hardware-only UPMEM execution-plan v3 runner")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--tasklets", type=int, required=True)
    parser.add_argument("--resident-manifest", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--host-binary", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.build_dir is not None and not args.execute:
            print(json.dumps(build(args.build_dir, tasklets_per_dpu=args.tasklets), sort_keys=True))
            return 0
        if not args.execute or args.host_binary is None or args.resident_manifest is None or args.sidecar is None or args.response is None:
            parser.error("--execute requires --host-binary, --resident-manifest, --sidecar, and --response")
        print(json.dumps(execute(args.host_binary, resident_manifest=args.resident_manifest, sidecar=args.sidecar, response=args.response, tasklets_per_dpu=args.tasklets), sort_keys=True))
        return 0
    except (OSError, subprocess.TimeoutExpired, V3RunnerError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "failure_stage": getattr(exc, "stage", "runner_failed"), "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
