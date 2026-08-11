"""KISS M5.1 output-partition workflow.

The route owns one real float32 contraction and three output-partition plans.
Native execution remains behind the existing execution-plan host boundary; this
module only prepares the package/sidecars, invokes that boundary, and validates
the returned final buffer against the same CPU contraction.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Protocol

import numpy as np

from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import create_run_dir, EVIDENCE_ARTIFACT_KIND
from quantum_bench.core.jsonio import write_json
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.distributed_plan_v2 import (
    CONTRACTED_PARTIAL_SUM,
    build_output_tile_plan_v2,
    build_contracted_partial_sum_plan_v2,
    serialize_upxdpv2_contracted_partition,
    serialize_upxdpv2_output_partition,
)
from quantum_bench.targets.upmem.generic_boundary import build_generic_boundary_workload
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    build_resident_graph_package,
)
from quantum_bench.targets.upmem.simplepim_taskgraph_executor import build as build_native
from quantum_bench.bench.upmem_simplepim_taskgraph import _lower_real_float32
from quantum_bench.tn.contract import contract_binary_task


SUITE_ID = "upmem_hardware_distributed_m5_1"
ROUTE_LABEL = "upmem_hw_m5_1"
ROUTE_ID = "upmem_tn_hardware_distributed_output_partition_m5_1"
BACKEND_ID = "upmem_sdk_hardware_distributed_m5_1"
SCHEMA_VERSION = "upmem_hardware_distributed_m5_1_v1"
NATIVE_RESPONSE_SCHEMA = "upmem_execution_plan_native_v2"
DPUS = (1, 2, 4)
WARMUPS = 0
REPETITIONS = 1
TOLERANCE = 1.0e-6


class M51NativeTarget(Protocol):
    """Small seam used by hardware-independent tests."""

    def set_environment(self, environment: Mapping[str, str]) -> None: ...

    def build(self, build_dir: Path) -> Mapping[str, Any]: ...

    def validate(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]: ...

    def execute(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]: ...


class _DefaultNativeTarget:
    def __init__(self) -> None:
        self._environment = dict(os.environ)

    def set_environment(self, environment: Mapping[str, str]) -> None:
        self._environment = dict(environment)

    def build(self, build_dir: Path) -> Mapping[str, Any]:
        built = build_native(build_dir, prepare_only=True)
        source = Path(str(built["native_source_dir"]))
        host = source / "bin" / "host_upmem_execution_plan_v2"
        dpu = source / "bin" / "dpu_resident_v2"
        if not host.is_file() or not dpu.is_file():
            raise RuntimeError("native_build_failed: M5.1 v2 binaries were not produced")
        return {**built, "host_binary_v2": str(host), "dpu_binary_v2": str(dpu)}

    def validate(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]:
        return self._invoke(request, "--validate-plan", timeout_s=timeout_s, expected="validated")

    def execute(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]:
        return self._invoke(request, "--execute-plan", timeout_s=timeout_s, expected="completed")

    def _invoke(
        self,
        request: Mapping[str, Any],
        mode: str,
        *,
        timeout_s: float,
        expected: str,
    ) -> Mapping[str, Any]:
        response_path = Path(str(request["response_path"]))
        host = Path(str(request["host_binary"]))
        response_path.unlink(missing_ok=True)
        command = [
            str(host),
            mode,
            "--resident-package",
            str(request["resident_manifest"]),
            "--distributed-plan-v2",
            str(request["distributed_plan"]),
            "--response",
            str(response_path),
            "--warmups",
            str(WARMUPS),
            "--repetitions",
            str(REPETITIONS),
            "--timeout-s",
            str(max(1, math.ceil(timeout_s))),
        ]
        completed = subprocess.run(
            command,
            cwd=host.parent.parent,
            env=self._environment,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if not response_path.is_file():
            raise RuntimeError("output_manifest_failed: native M5.1 response is missing")
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("output_manifest_failed: native M5.1 response is not an object")
        if completed.returncode != 0 or payload.get("status") != expected:
            raise RuntimeError(str(payload.get("error") or "native M5.1 request failed"))
        return payload


def prepare(
    root_dir: Path,
    *,
    build: bool = False,
    native_target: M51NativeTarget | None = None,
) -> dict[str, Any]:
    """Prepare and parser-validate the three plans without allocating a DPU."""

    if not build:
        raise ValueError("prepare_requires_build: --prepare-only requires --build")
    plan_dir = _unique_dir(root_dir / "build" / f"{SUITE_ID}_plan")
    target = native_target or _DefaultNativeTarget()
    result = _prepare_bundle(plan_dir, target=target, build=build)
    artifact = plan_dir / f"{SUITE_ID}_plan.json"
    prepared = {"status": "prepared", **result}
    write_json(artifact, prepared)
    return {"plan_dir": str(plan_dir), "artifact": str(artifact), **prepared}


def execute(
    root_dir: Path,
    *,
    environment: Mapping[str, str] | None = None,
    native_target: M51NativeTarget | None = None,
) -> dict[str, Any]:
    """Prepare, validate, execute, and record all 1/2/4-DPU partitions."""

    env = dict(os.environ if environment is None else environment)
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    if env.get("DPU_BACKEND"):
        raise ValueError("hardware_profile_violation: DPU_BACKEND must be unset for physical M5.1")

    target = native_target or _DefaultNativeTarget()
    target.set_environment(env)

    run_dir = create_run_dir(
        root_dir,
        SUITE_ID,
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
    )
    captured_environment = capture_environment(root_dir)
    captured_environment["m5_1_native_environment"] = {
        "UPMEM_ALLOW_PHYSICAL_HARDWARE": env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE"),
        "DPU_BACKEND": env.get("DPU_BACKEND"),
        "UPMEM_HOME": env.get("UPMEM_HOME"),
    }
    write_json(run_dir / "environment.json", captured_environment)
    summary_name = f"{SUITE_ID}_summary.json"
    manifest = write_run_manifest(
        run_dir,
        run_kind=SCHEMA_VERSION,
        suite_id=SUITE_ID,
        suite_path=None,
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
        route_id=ROUTE_ID,
        backend_id=BACKEND_ID,
        execution_scope="one_operation_real_float32_output_partition_1_2_4_dpus",
        evidence_type="physical_hardware_functionality_only",
        normalized_records="normalized_records.jsonl",
        summary=summary_name,
        upmem_execution_mode="execution_plan_v2_output_partition",
        command="UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5-1",
        root_dir=root_dir,
    )
    bundle = _prepare_bundle(run_dir, target=target, build=True)
    records: list[dict[str, Any]] = []
    for dpu_count in DPUS:
        item = bundle["plans"][str(dpu_count)]
        try:
            output_path = Path(item["output_path"])
            output_path.unlink(missing_ok=True)
            response = target.execute(item["request"], timeout_s=60.0)
            _validate_execute_response(response, item)
            if not output_path.is_file():
                raise ValueError("result_transfer_failed: native final output is missing")
            output = np.fromfile(output_path, dtype="<f4")
            reference = np.fromfile(item["reference_path"], dtype="<f4")
            if output.shape != reference.shape or not np.all(np.isfinite(output)):
                raise ValueError("output_validation_failed: native output shape or finiteness mismatch")
            max_abs_error = float(np.max(np.abs(output - reference), initial=0.0))
            l2_error = float(np.linalg.norm(output - reference))
            if max_abs_error > TOLERANCE:
                raise ValueError(f"output_validation_failed: max_abs_error={max_abs_error}")
            records.append(_record(item, response, output, reference, max_abs_error, l2_error))
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            records.append(
                {
                    "case_id": "m5_1_one_operation",
                    "workload_id": "generic_rank3_boundary",
                    "route_id": ROUTE_ID,
                    "backend_id": BACKEND_ID,
                    "requested_dpu_count": dpu_count,
                    "status": "failed",
                    "validation_status": "failed",
                    "scientific_validation_status": "failed",
                    "failure_stage": str(exc).split(":", 1)[0],
                    "reason": str(exc),
                    "execution_plan_kind": "distributed_plan_v2_output_partition",
                    "execution_plan_executed": False,
                }
            )
    write_normalized_records(run_dir, records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "status": "completed" if records and all(item["status"] == "completed" for item in records) else "failed",
        "row_count": len(records),
        "dpu_counts": list(DPUS),
        "claim_boundary": "functionality_only_no_speedup_claim",
        "native_protocol": "--validate-plan/--execute-plan --distributed-plan-v2",
        "normalized_records": "normalized_records.jsonl",
        "plans": bundle["plans"],
    }
    summary_path = run_dir / summary_name
    write_json(summary_path, summary)
    manifest["hardware_available"] = "verified_by_execution" if summary["status"] == "completed" else "not_verified"
    write_json(run_dir / "run_manifest.json", manifest)
    return {"run_dir": str(run_dir), "artifact": str(summary_path), "status": summary["status"], "row_count": len(records)}


def _prepare_bundle(
    root: Path,
    *,
    target: M51NativeTarget,
    build: bool,
    partition_kind: str = "output_tile",
    case_id: str = "m5_1_one_operation",
    suite_id: str = SUITE_ID,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    workload = build_generic_boundary_workload()
    package_graph, package_network = _lower_real_float32(workload.graph, workload.network)
    task = package_graph.tasks[0]
    reference = np.asarray(
        contract_binary_task(
            task,
            package_network.tensors[0].array,
            package_network.tensors[1].array,
            dtype=np.float32,
        ),
        dtype="<f4",
    ).ravel()
    native = target.build(root / "native_build") if build else {}
    dpu_binary = Path(str(native.get("dpu_binary_v2", root / "native_build" / "dpu_resident_v2")))
    host_binary = Path(str(native.get("host_binary_v2", root / "native_build" / "host_upmem_execution_plan_v2")))
    package = build_resident_graph_package(
        package_graph,
        package_network,
        case_id=case_id,
        suite_id=suite_id,
        quantization_mode="none",
        full_precision_output=reference,
        allow_slot_reuse=False,
        operation_abi_version=2,
    ).write(root, dpu_binary=dpu_binary, request_id=case_id)
    if package.package_path is None or package.manifest_path is None:
        raise RuntimeError("package_preparation_failed: resident package paths are missing")
    package_bytes = package.package_path.read_bytes()
    resident_manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(resident_manifest, dict):
        raise RuntimeError("package_preparation_failed: resident manifest is not an object")
    operation_bytes = package.operations[0].to_bytes(
        operation_abi_version=package.operation_abi_version
    )
    reference_path = root / "cpu_reference_float32.bin"
    reference.tofile(reference_path)
    plans: dict[str, Any] = {}
    for dpu_count in DPUS:
        plan_dir = root / f"dpu_{dpu_count}"
        plan_dir.mkdir(parents=True, exist_ok=True)
        if partition_kind == CONTRACTED_PARTIAL_SUM:
            plan = build_contracted_partial_sum_plan_v2(
                logical_operation_id=task.id,
                logical_task_id=task.id,
                total_output_elements=int(reference.size),
                total_contracted_elements=int(task.gemm_k),
                contraction_plan_hash=package_graph.contraction_plan_hash,
                dpu_count=dpu_count,
            )
        else:
            plan = build_output_tile_plan_v2(
                logical_operation_id=task.id,
                logical_task_id=task.id,
                total_output_elements=int(reference.size),
                total_contracted_elements=int(task.gemm_k),
                contraction_plan_hash=package_graph.contraction_plan_hash,
                dpu_count=dpu_count,
            )
        plan_path = plan_dir / "distributed_plan_v2.json"
        sidecar_path = plan_dir / "distributed_plan_v2.bin"
        write_json(plan_path, plan.to_json_dict())
        serializer = (
            serialize_upxdpv2_contracted_partition
            if partition_kind == CONTRACTED_PARTIAL_SUM
            else serialize_upxdpv2_output_partition
        )
        sidecar_path.write_bytes(serializer(
            plan,
            package_bytes=package_bytes,
            operation_bytes=operation_bytes,
            output_slot=package.operations[0].slot_out_real,
        ))
        output_path = plan_dir / "native_final_output.bin"
        manifest_payload = json.loads(json.dumps(resident_manifest))
        final_outputs = manifest_payload.get("final_outputs")
        if not isinstance(final_outputs, list) or len(final_outputs) != 1:
            raise RuntimeError("package_preparation_failed: expected one resident final output")
        final_outputs[0]["output_path"] = str(output_path.relative_to(root))
        manifest_payload["session_id"] = f"{case_id}_dpu_{dpu_count}"
        manifest_payload["requested_dpus"] = dpu_count
        resident_manifest_path = root / f"m5_1_dpu_{dpu_count}_resident_request.json"
        write_json(resident_manifest_path, manifest_payload)
        request = {
            "dpu_count": dpu_count,
            "host_binary": str(host_binary),
            "resident_manifest": str(resident_manifest_path),
            "distributed_plan": str(sidecar_path),
            "response_path": str(plan_dir / "native_response.json"),
            "output_path": str(output_path),
            "partition_kind": partition_kind,
            "work_units": [unit.__dict__ for unit in plan.work_units],
        }
        validation = target.validate(request, timeout_s=60.0)
        _validate_plan_response(validation, dpu_count)
        plans[str(dpu_count)] = {
            "dpu_count": dpu_count,
            "execution_plan_hash": plan.execution_plan_hash,
            "contraction_plan_hash": plan.contraction_plan_hash,
            "plan_path": str(plan_path),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": _sha256_file(sidecar_path),
            "package_path": str(package.package_path),
            "package_sha256": _sha256_file(package.package_path),
            "output_path": str(output_path),
            "reference_path": str(reference_path),
            "request": request,
            "native_validation": dict(validation),
            "work_units": [unit.__dict__ for unit in plan.work_units],
        }
    return {"schema_version": SCHEMA_VERSION, "package_manifest": str(package.manifest_path), "plans": plans}


def _validate_plan_response(response: Mapping[str, Any], dpu_count: int) -> None:
    required = {
        "schema_version": NATIVE_RESPONSE_SCHEMA,
        "status": "validated",
        "target_requested": "hardware",
        "target_observed": "not_allocated",
        "requested_dpu_count": dpu_count,
        "allocated_dpu_count": 0,
    }
    for key, expected in required.items():
        if key not in response or response[key] != expected:
            raise ValueError(f"execution_plan_compile_failed: invalid native validation field {key}")
        if type(expected) is int and type(response[key]) is not int:
            raise ValueError(f"execution_plan_compile_failed: invalid native validation field {key}")


def _validate_execute_response(
    response: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    partition_kind: str = "output_tile",
) -> None:
    dpu_count = int(item["dpu_count"])
    contracted_reduction = partition_kind == CONTRACTED_PARTIAL_SUM
    required = {
        "schema_version": NATIVE_RESPONSE_SCHEMA,
        "status": "completed",
        "failure_stage": None,
        "error": None,
        "target_requested": "hardware",
        "target_observed": "physical_hardware",
        "requested_dpu_count": dpu_count,
        "allocated_dpu_count": dpu_count,
        "tasklets_per_dpu": 1,
        "requested_warmups": WARMUPS,
        "requested_repetitions": REPETITIONS,
        "validation_status": "native_completion_verified",
    }
    for key, expected in required.items():
        if key not in response or response[key] != expected:
            raise ValueError(f"output_manifest_failed: invalid native response field {key}")
        if type(expected) is int and type(response[key]) is not int:
            raise ValueError(f"output_manifest_failed: invalid native response field {key}")

    required_flags = {
        "hardware_allocation_verified": True,
        "allocation_succeeded": True,
        "allocation_was_confirmed": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "hardware_functionality_evidence": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": True,
    }
    for key, expected in required_flags.items():
        if response.get(key) is not expected:
            raise ValueError(f"output_manifest_failed: invalid native response flag {key}")

    allocation = _required_mapping(response, "allocation")
    for key, expected in {"attempted": True, "release_confirmed": True}.items():
        if allocation.get(key) is not expected:
            raise ValueError(f"output_manifest_failed: invalid allocation flag {key}")
    if allocation.get("confirmed") is not True:
        raise ValueError("output_manifest_failed: allocation confirmation history is missing")

    metrics = _required_mapping(response, "metrics")
    metric_names = (
        "descriptor_h2d_bytes",
        "operand_h2d_bytes",
        "reset_h2d_bytes",
        "active_operation_h2d_bytes",
        "completion_d2h_bytes",
        "cross_d2h_bytes",
        "cross_h2d_bytes",
        "final_d2h_bytes",
        "actual_h2d_bytes",
        "actual_d2h_bytes",
        "actual_transfer_bytes",
        "launch_count",
        "synchronize_count",
        "completion_reads",
        "cross_dpu_edge_count",
    )
    metric_values = {name: _required_non_negative_int(metrics, name) for name in metric_names}
    reduction_metric_names = (
        "reduction_d2h_bytes",
        "reduction_partial_reads",
        "reduction_element_additions",
        "reduction_accumulator_resets",
        "reduction_participant_count",
    )
    if contracted_reduction:
        metric_values.update(
            {name: _required_non_negative_int(metrics, name) for name in reduction_metric_names}
        )
    else:
        metric_values.update({name: 0 for name in reduction_metric_names})
    expected_launches = dpu_count * (WARMUPS + REPETITIONS)
    for name in ("launch_count", "synchronize_count", "completion_reads"):
        if metric_values[name] != expected_launches or metric_values[name] <= 0:
            raise ValueError(f"output_manifest_failed: inconsistent {name}")
    expected_h2d = sum(
        metric_values[name]
        for name in (
            "descriptor_h2d_bytes",
            "operand_h2d_bytes",
            "reset_h2d_bytes",
            "active_operation_h2d_bytes",
            "cross_h2d_bytes",
        )
    )
    expected_d2h = sum(
        metric_values[name]
        for name in (
            "completion_d2h_bytes",
            "cross_d2h_bytes",
            "final_d2h_bytes",
            "reduction_d2h_bytes",
        )
    )
    if metric_values["actual_h2d_bytes"] != expected_h2d or expected_h2d <= 0:
        raise ValueError("output_manifest_failed: inconsistent actual_h2d_bytes")
    if metric_values["actual_d2h_bytes"] != expected_d2h or expected_d2h <= 0:
        raise ValueError("output_manifest_failed: inconsistent actual_d2h_bytes")
    if metric_values["actual_transfer_bytes"] != expected_h2d + expected_d2h:
        raise ValueError("output_manifest_failed: inconsistent actual_transfer_bytes")
    if metric_values["cross_dpu_edge_count"] != 0:
        raise ValueError("output_manifest_failed: output partition unexpectedly transferred between DPUs")
    if contracted_reduction:
        if response.get("provider_identities", {}).get("communication") != "host_mediated_sum_v1":
            raise ValueError("output_manifest_failed: host-mediated reduction provider is missing")
        if metric_values["reduction_partial_reads"] != dpu_count:
            raise ValueError("output_manifest_failed: every contracted partial must be read")
        if metric_values["reduction_participant_count"] != dpu_count:
            raise ValueError("output_manifest_failed: reduction participant evidence is incomplete")
        if metric_values["reduction_accumulator_resets"] != WARMUPS + REPETITIONS:
            raise ValueError("output_manifest_failed: reduction accumulator reset evidence is inconsistent")
        expected_additions = int(item["work_units"][0]["output_elements"]) * (dpu_count - 1) * (WARMUPS + REPETITIONS)
        if metric_values["reduction_element_additions"] != expected_additions:
            raise ValueError("output_manifest_failed: one-participant reduction must perform zero additions")
        reduction = _required_mapping(response, "reduction")
        if reduction.get("provider") != "host_mediated_sum_v1" or reduction.get("order") != "ascending_dpu_id":
            raise ValueError("output_manifest_failed: deterministic host reduction contract is missing")
        if reduction.get("element_additions") != expected_additions:
            raise ValueError("output_manifest_failed: reduction addition evidence is inconsistent")
    elif any(metric_values[name] != 0 for name in (
        "reduction_d2h_bytes",
        "reduction_partial_reads",
        "reduction_element_additions",
        "reduction_accumulator_resets",
        "reduction_participant_count",
    )):
        raise ValueError("output_manifest_failed: output partition reported reduction metrics")

    timing = _required_mapping(response, "timing")
    timing_names = (
        "allocation_time_s",
        "binary_load_time_s",
        "descriptor_h2d_time_s",
        "operand_h2d_time_s",
        "cross_dpu_transfer_time_s",
        "launch_sync_time_s",
        "final_d2h_time_s",
        "output_write_time_s",
        "release_time_s",
    )
    timing_values = [_required_non_negative_number(timing, name) for name in timing_names]
    if sum(timing_values) <= 0.0:
        raise ValueError("output_manifest_failed: native timing evidence is empty")
    if contracted_reduction and _required_non_negative_number(timing, "reduction_time_s") <= 0.0:
        raise ValueError("output_manifest_failed: host reduction timing evidence is empty")

    expected_units = item.get("work_units")
    if not isinstance(expected_units, list) or len(expected_units) != dpu_count:
        raise ValueError("output_manifest_failed: expected work-unit evidence is invalid")
    assignments = _required_list(response, "operation_assignments", dpu_count)
    completed = _required_list(response, "completed_per_dpu", dpu_count)
    completed_expected = WARMUPS + REPETITIONS
    if any(type(value) is not int or value != completed_expected for value in completed):
        raise ValueError("output_manifest_failed: per-DPU completion evidence is inconsistent")
    completed_units = _required_list(response, "distributed_work_units", dpu_count)
    for expected, assignment, completed_unit in zip(
        expected_units, assignments, completed_units, strict=True
    ):
        _validate_work_unit_evidence(
            expected,
            assignment,
            completed_unit,
            partition_kind=partition_kind,
        )
    if response.get("cross_dpu_transfers") != []:
        raise ValueError("output_manifest_failed: output partition reported cross-DPU transfers")


def _record(
    item: Mapping[str, Any],
    response: Mapping[str, Any],
    output: np.ndarray,
    reference: np.ndarray,
    max_abs_error: float,
    l2_error: float,
) -> dict[str, Any]:
    metrics = _required_mapping(response, "metrics")
    timing = _required_mapping(response, "timing")
    return {
        "case_id": "m5_1_one_operation",
        "workload_id": "generic_rank3_boundary",
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "benchmark_role": "physical_m5_1_output_partition_functionality",
        "execution_model": "distributed_single_contraction_output_partition",
        "execution_plan_kind": "distributed_plan_v2_output_partition",
        "execution_plan_executed": True,
        "requested_dpu_count": item["dpu_count"],
        "allocated_dpu_count": response["allocated_dpu_count"],
        "partition_kind": "output_tile",
        "communication_provider": "none",
        "dtype": "float32",
        "execution_plan_hash": item["execution_plan_hash"],
        "contraction_plan_hash": item["contraction_plan_hash"],
        "distributed_plan_v2_sha256": item["sidecar_sha256"],
        "package_file_sha256": item["package_sha256"],
        "work_units": item["work_units"],
        "native_response": dict(response),
        "hardware_allocation_verified": response["hardware_allocation_verified"],
        "native_kernel_executed": response["native_kernel_executed"],
        "hardware_kernel_executed": response["hardware_kernel_executed"],
        "simulator_kernel_executed": response["simulator_kernel_executed"],
        "cpu_fallback_used": response["cpu_fallback_used"],
        "release_confirmed": response["allocation"]["release_confirmed"],
        "descriptor_h2d_bytes": metrics["descriptor_h2d_bytes"],
        "operand_h2d_bytes": metrics["operand_h2d_bytes"],
        "reset_h2d_bytes": metrics["reset_h2d_bytes"],
        "active_operation_h2d_bytes": metrics["active_operation_h2d_bytes"],
        "completion_d2h_bytes": metrics["completion_d2h_bytes"],
        "cross_d2h_bytes": metrics["cross_d2h_bytes"],
        "cross_h2d_bytes": metrics["cross_h2d_bytes"],
        "final_d2h_bytes": metrics["final_d2h_bytes"],
        "actual_h2d_bytes": metrics["actual_h2d_bytes"],
        "actual_d2h_bytes": metrics["actual_d2h_bytes"],
        "actual_transfer_bytes": metrics["actual_transfer_bytes"],
        "launch_count": metrics["launch_count"],
        "synchronize_count": metrics["synchronize_count"],
        "completion_reads": metrics["completion_reads"],
        "allocation_time_s": timing["allocation_time_s"],
        "binary_load_time_s": timing["binary_load_time_s"],
        "descriptor_h2d_time_s": timing["descriptor_h2d_time_s"],
        "operand_h2d_time_s": timing["operand_h2d_time_s"],
        "cross_dpu_transfer_time_s": timing["cross_dpu_transfer_time_s"],
        "launch_sync_time_s": timing["launch_sync_time_s"],
        "final_d2h_time_s": timing["final_d2h_time_s"],
        "output_write_time_s": timing["output_write_time_s"],
        "release_time_s": timing["release_time_s"],
        "timing_is_bringup_only": response["timing_is_bringup_only"],
        "output_elements": int(reference.size),
        "output_dtype": str(output.dtype),
        "reference_output_sha256": _sha256_array(reference),
        "native_output_sha256": _sha256_array(output),
        "max_abs_error": max_abs_error,
        "l2_error": l2_error,
        "validation_status": "passed",
        "scientific_validation_status": "passed",
        "status": "completed",
        "hardware_speedup_applicable": False,
        "claim_boundary": "functionality_only_no_speedup_claim",
    }


def _validate_work_unit_evidence(
    expected: Mapping[str, Any],
    assignment: Any,
    completed: Any,
    *,
    partition_kind: str = "output_tile",
) -> None:
    if not isinstance(assignment, Mapping) or not isinstance(completed, Mapping):
        raise ValueError("output_manifest_failed: work-unit evidence is not an object")
    expected_fields = {
        "dpu_id": expected["dpu_id"],
        "output_offset": expected["output_offset"],
        "output_elements": expected["output_elements"],
        "contracted_offset": expected["contracted_offset"],
        "contracted_elements": expected["contracted_elements"],
    }
    if expected_fields["output_elements"] <= 0 or expected_fields["contracted_elements"] <= 0:
        raise ValueError("output_manifest_failed: work-unit assignment is empty")
    for name, value in expected_fields.items():
        if (
            type(assignment.get(name)) is not int
            or type(completed.get(name)) is not int
            or assignment[name] != value
            or completed[name] != value
        ):
            raise ValueError(f"output_manifest_failed: inconsistent work-unit field {name}")
    expected_partition_mode = "contracted" if partition_kind == CONTRACTED_PARTIAL_SUM else "output"
    if assignment.get("partition_mode") != expected_partition_mode:
        raise ValueError("output_manifest_failed: native partition mode is not output")
    if assignment.get("package_operation_index") != 0 or assignment.get("operation_id") != 0:
        raise ValueError("output_manifest_failed: native operation assignment is inconsistent")
    if _required_positive_int(completed, "runtime_cycles") <= 0:
        raise ValueError("output_manifest_failed: DPU runtime cycle evidence is empty")
    if _required_positive_int(completed, "processed_elements") != expected["output_elements"]:
        raise ValueError("output_manifest_failed: processed element evidence is inconsistent")
    if _required_positive_int(completed, "completion_count") != WARMUPS + REPETITIONS:
        raise ValueError("output_manifest_failed: work-unit completion count is inconsistent")
    checksum = completed.get("output_checksum_fnv1a64")
    if not isinstance(checksum, str) or len(checksum) != 16:
        raise ValueError("output_manifest_failed: work-unit checksum evidence is invalid")
    try:
        int(checksum, 16)
    except ValueError as exc:
        raise ValueError(
            "output_manifest_failed: work-unit checksum evidence is invalid"
        ) from exc


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"output_manifest_failed: native response field {key} is missing")
    return nested


def _required_list(value: Mapping[str, Any], key: str, length: int) -> list[Any]:
    nested = value.get(key)
    if not isinstance(nested, list) or len(nested) != length:
        raise ValueError(f"output_manifest_failed: native response field {key} is invalid")
    return nested


def _required_non_negative_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int or item < 0:
        raise ValueError(f"output_manifest_failed: native response field {key} is invalid")
    return item


def _required_positive_int(value: Mapping[str, Any], key: str) -> int:
    item = _required_non_negative_int(value, key)
    if item <= 0:
        raise ValueError(f"output_manifest_failed: native response field {key} is not positive")
    return item


def _required_non_negative_number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if type(item) not in (int, float) or not math.isfinite(item) or item < 0.0:
        raise ValueError(f"output_manifest_failed: native response field {key} is invalid")
    return float(item)


def _unique_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / "latest"
    index = 1
    while candidate.exists():
        candidate = parent / f"run_{index:02d}"
        index += 1
    candidate.mkdir(parents=True)
    return candidate


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


__all__ = ["BACKEND_ID", "DPUS", "ROUTE_ID", "SCHEMA_VERSION", "execute", "prepare"]
