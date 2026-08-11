"""KISS M5.2 host-mediated contracted-axis reduction workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir
from quantum_bench.core.jsonio import write_json
from quantum_bench.environment import capture_environment
from quantum_bench.bench import upmem_hardware_distributed_m5_1 as m51
from quantum_bench.targets.upmem.distributed_plan_v2 import CONTRACTED_PARTIAL_SUM


SUITE_ID = "upmem_hardware_distributed_m5_2"
ROUTE_LABEL = "upmem_hw_m5_2"
ROUTE_ID = "upmem_tn_hardware_distributed_host_reduction_m5_2"
BACKEND_ID = "upmem_sdk_hardware_distributed_m5_2"
SCHEMA_VERSION = "upmem_hardware_distributed_m5_2_v1"
DPUS = (1, 2, 4)
WARMUPS = 0
REPETITIONS = 1
TOLERANCE = 1.0e-6


def reduce_float32_partials(partials: Mapping[int, np.ndarray]) -> np.ndarray:
    """Sum complete float32 partials in ascending DPU-ID order."""

    if not partials:
        raise ValueError("host_reduction_requires_partials")
    ids = sorted(partials)
    first = np.asarray(partials[ids[0]], dtype="<f4")
    if first.ndim != 1 or not np.all(np.isfinite(first)):
        raise ValueError("host_reduction_partial_is_not_finite_float32")
    result = np.zeros(first.shape, dtype="<f4")
    for dpu_id in ids:
        partial = np.asarray(partials[dpu_id], dtype="<f4")
        if partial.shape != first.shape or not np.all(np.isfinite(partial)):
            raise ValueError("host_reduction_partials_have_inconsistent_shape_or_finiteness")
        np.add(result, partial, out=result, casting="unsafe")
        if not np.all(np.isfinite(result)):
            raise ValueError("host_reduction_produced_non_finite_float32")
    return result


def prepare(
    root_dir: Path,
    *,
    build: bool = False,
    native_target: m51.M51NativeTarget | None = None,
) -> dict[str, Any]:
    """Prepare and native-parser-validate M5.2 plans without allocating a DPU."""

    if not build:
        raise ValueError("prepare_requires_build: --prepare-only requires --build")
    plan_dir = m51._unique_dir(root_dir / "build" / f"{SUITE_ID}_plan")
    target = native_target or m51._DefaultNativeTarget()
    bundle = m51._prepare_bundle(
        plan_dir,
        target=target,
        build=True,
        partition_kind=CONTRACTED_PARTIAL_SUM,
        case_id="m5_2_one_operation",
        suite_id=SUITE_ID,
    )
    result = {**bundle, "status": "prepared", "schema_version": SCHEMA_VERSION}
    artifact = plan_dir / f"{SUITE_ID}_plan.json"
    write_json(artifact, result)
    return {"plan_dir": str(plan_dir), "artifact": str(artifact), **result}


def execute(
    root_dir: Path,
    *,
    environment: Mapping[str, str] | None = None,
    native_target: m51.M51NativeTarget | None = None,
) -> dict[str, Any]:
    """Execute one real float32 contraction with host-mediated partial reduction."""

    env = dict(os.environ if environment is None else environment)
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    if env.get("DPU_BACKEND"):
        raise ValueError("hardware_profile_violation: DPU_BACKEND must be unset for physical M5.2")

    target = native_target or m51._DefaultNativeTarget()
    target.set_environment(env)
    run_dir = create_run_dir(
        root_dir,
        SUITE_ID,
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
    )
    captured_environment = capture_environment(root_dir)
    captured_environment["m5_2_native_environment"] = {
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
        execution_scope="one_operation_real_float32_contracted_partition_host_reduction_1_2_4_dpus",
        evidence_type="physical_hardware_functionality_only",
        normalized_records="normalized_records.jsonl",
        summary=summary_name,
        upmem_execution_mode="execution_plan_v2_contracted_partition_host_mediated_sum_v1",
        command="UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5-2",
        root_dir=root_dir,
    )
    bundle = m51._prepare_bundle(
        run_dir,
        target=target,
        build=True,
        partition_kind=CONTRACTED_PARTIAL_SUM,
        case_id="m5_2_one_operation",
        suite_id=SUITE_ID,
    )
    records: list[dict[str, Any]] = []
    for dpu_count in DPUS:
        item = bundle["plans"][str(dpu_count)]
        try:
            output_path = Path(item["output_path"])
            output_path.unlink(missing_ok=True)
            response = target.execute(item["request"], timeout_s=60.0)
            m51._validate_execute_response(
                response, item, partition_kind=CONTRACTED_PARTIAL_SUM
            )
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
            row = m51._record(item, response, output, reference, max_abs_error, l2_error)
            row.update(
                {
                    "case_id": "m5_2_one_operation",
                    "route_id": ROUTE_ID,
                    "backend_id": BACKEND_ID,
                    "benchmark_role": "physical_m5_2_host_mediated_reduction_functionality",
                    "execution_model": "distributed_single_contraction_contracted_partition_host_reduction",
                    "execution_plan_kind": "distributed_plan_v2_contracted_partition",
                    "partition_kind": CONTRACTED_PARTIAL_SUM,
                    "communication_provider": "host_mediated_sum_v1",
                    "reduction_order": "ascending_dpu_id",
                    "reduction_d2h_bytes": response["metrics"]["reduction_d2h_bytes"],
                    "reduction_partial_reads": response["metrics"]["reduction_partial_reads"],
                    "reduction_element_additions": response["metrics"]["reduction_element_additions"],
                    "reduction_accumulator_resets": response["metrics"]["reduction_accumulator_resets"],
                    "reduction_participant_count": response["metrics"]["reduction_participant_count"],
                    "status": "completed",
                }
            )
            records.append(row)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            records.append(
                {
                    "case_id": "m5_2_one_operation",
                    "workload_id": "generic_rank3_boundary",
                    "route_id": ROUTE_ID,
                    "backend_id": BACKEND_ID,
                    "requested_dpu_count": dpu_count,
                    "status": "failed",
                    "validation_status": "failed",
                    "scientific_validation_status": "failed",
                    "failure_stage": str(exc).split(":", 1)[0],
                    "reason": str(exc),
                    "execution_plan_kind": "distributed_plan_v2_contracted_partition",
                    "communication_provider": "host_mediated_sum_v1",
                    "execution_plan_executed": False,
                }
            )
    write_normalized_records(run_dir, records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "status": "completed" if records and all(row["status"] == "completed" for row in records) else "failed",
        "row_count": len(records),
        "dpu_counts": list(DPUS),
        "communication_provider": "host_mediated_sum_v1",
        "reduction_order": "ascending_dpu_id",
        "claim_boundary": "functionality_only_no_speedup_claim",
        "native_protocol": "--validate-plan/--execute-plan --distributed-plan-v2 UPXDPV2_v2",
        "normalized_records": "normalized_records.jsonl",
        "plans": bundle["plans"],
    }
    summary_path = run_dir / summary_name
    write_json(summary_path, summary)
    manifest["hardware_available"] = "verified_by_execution" if summary["status"] == "completed" else "not_verified"
    write_json(run_dir / "run_manifest.json", manifest)
    return {"run_dir": str(run_dir), "artifact": str(summary_path), "status": summary["status"], "row_count": len(records)}


__all__ = ["DPUS", "execute", "prepare", "reduce_float32_partials"]
