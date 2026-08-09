"""M4.2 SimplePIM rank-1 operator qualification.

This is deliberately a small adapter around the thesis-owned native route. It
admits one rank-1 contraction task only and does not claim generic TaskGraph
execution or end-to-end tensor operand transport.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Any, Mapping

import yaml

from quantum_bench.bench.provider_qualification import SIMULATOR_ENV_KEYS
from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict
from quantum_bench.environment import capture_environment


SCHEMA_VERSION = "upmem_hardware_simplepim_rank1_m4_2_v1"
NATIVE_SCHEMA_VERSION = "simplepim_rank1_dot_m4_2_v1"
PROFILE_ID = "hardware_simplepim_rank1_dot_m4_2_v1"
BACKEND_ID = "upmem_sdk_hardware_simplepim_rank1_dot_m4_2"
ROUTE_ID = "upmem_tn_hardware_simplepim_rank1_dot_m4_2"
PROVIDER_ID = "simplepim"
SUITE_ID = "upmem_hardware_simplepim_rank1_m4_2"
NATIVE_REL = Path("native/upmem/simplepim/upmem_sdk_rank1_dot_m4_2")
WARMUPS = 1
REPEATS = 5
DPUS = 2
TASKLETS = 12
VECTOR_LENGTH = 256
TIMEOUT_S = 120.0
SIMPLEPIM_SOURCE_COMMIT = "1d639c53532555f01e9f71d872e7712b166d6cba"
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
CLAIM_BOUNDARY = (
    "SimplePIM operator qualification for one deterministic rank-1 task; "
    "not general TaskGraph execution, operand transport, persistence, scaling, "
    "speedup, or energy evidence"
)
OPERATOR_SEQUENCE = [
    "simplepim_scatter",
    "simplepim_scatter",
    "table_zip_virtual",
    "table_map_pair_product",
    "table_gen_red_host_reduce",
]


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_suite() -> Path:
    return _root() / "configs" / "suites" / "upmem_hardware_simplepim_rank1_m4_2.yml"


def _profile(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "hardware_profile_version": PROFILE_ID,
        "target": "hardware",
        "backend_id": BACKEND_ID,
        "route_id": ROUTE_ID,
        "requested_dpu_count": DPUS,
        "initialization_tasklets_per_dpu": 1,
        "operator_tasklets_per_dpu": TASKLETS,
        "numeric_mode": "int32_inputs_int64_accumulator",
        "vector_length": VECTOR_LENGTH,
        "synchronous_execution": True,
        "performance_claim_applicable": False,
    }
    if dict(raw) != expected:
        raise ValueError("hardware_profile_violation: M4.2 profile is not the fixed profile")
    return expected


def load_suite(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved != _canonical_suite().resolve():
        raise ValueError("hardware_profile_violation: M4.2 requires the committed suite")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("suite_id") != SUITE_ID:
        raise ValueError("hardware_profile_violation: M4.2 suite identity mismatch")
    if raw.get("schema_version") != 1 or raw.get("fail_fast") is not True:
        raise ValueError("hardware_profile_violation: M4.2 suite schema mismatch")
    defaults = raw.get("defaults")
    workloads = raw.get("workloads")
    if defaults != {"warmups": WARMUPS, "repeats": REPEATS, "timeout_s": TIMEOUT_S}:
        raise ValueError("hardware_profile_violation: M4.2 repeat profile mismatch")
    if not isinstance(workloads, list) or len(workloads) != 1 or not isinstance(workloads[0], dict):
        raise ValueError("hardware_profile_violation: M4.2 requires one workload")
    workload = workloads[0]
    fixture = workload.get("qualification_fixture")
    if not isinstance(fixture, dict) or fixture.get("task_count") != 1 or fixture.get("task_kind") != "rank1_contraction":
        raise ValueError("hardware_profile_violation: one fixed rank1 qualification task is required")
    if fixture.get("rank") != 1 or fixture.get("vector_length") != VECTOR_LENGTH:
        raise ValueError("hardware_profile_violation: unsupported qualification fixture")
    profile = _profile(raw.get("metadata", {}).get("hardware_profile", {}))
    if workload.get("numeric_mode") != profile["numeric_mode"]:
        raise ValueError("hardware_profile_violation: workload numeric mode mismatch")
    return {
        "raw": raw,
        "path": resolved,
        "profile": profile,
        "workload": workload,
    }


def _run(command: list[str], *, cwd: Path, env: Mapping[str, str], timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "timed_out": False,
            "elapsed_s": time.perf_counter() - started,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "timed_out": True,
            "elapsed_s": time.perf_counter() - started,
            "stdout": str(exc.stdout or "")[-4000:],
            "stderr": str(exc.stderr or "")[-4000:],
        }
    except OSError as exc:
        return {
            "command": command,
            "returncode": None,
            "timed_out": False,
            "elapsed_s": time.perf_counter() - started,
            "stdout": "",
            "stderr": str(exc),
        }


def _native_dir(root: Path) -> Path:
    return root / NATIVE_REL


def _response(native_dir: Path, name: str) -> dict[str, Any]:
    path = native_dir / "build" / "simplepim_rank1_dot_m4_2" / name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"response_transport_failed: invalid native response {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("response_transport_failed: native response is not an object")
    return payload


def _require_response(payload: Mapping[str, Any], *, parser: bool) -> None:
    checks: dict[str, Any] = {
        "schema_version": NATIVE_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "backend_id": BACKEND_ID,
        "route_id": ROUTE_ID,
        "provider_id": PROVIDER_ID,
        "target_requested": "physical_hardware",
        "requested_dpu_count": DPUS,
        "initialization_tasklets_per_dpu": 1,
        "operator_tasklets_per_dpu": TASKLETS,
        "warmup_count": WARMUPS,
        "repeat_count": REPEATS,
        "operator_sequence": OPERATOR_SEQUENCE,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": True,
        "hardware_target_observation_method": "default_sdk_allocation_and_observed_dpu_count_after_simulator_selector_rejection",
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(f"response_evidence_invalid: {key} mismatch")
    if parser:
        for key in (
            "allocation_attempted",
            "provider_initialized",
            "simplepim_operator_api_used",
            "map_kernel_executed",
            "genred_kernel_executed",
            "release_attempted",
            "release_confirmed",
            "hardware_kernel_executed",
        ):
            if payload.get(key) is not False:
                raise ValueError(f"parser_allocation_check_failed: {key} must be false")
        if payload.get("parser_mode") is not True or payload.get("status") != "prepared":
            raise ValueError("parser_allocation_check_failed: parser response is not prepared")
        return
    required_true = (
        "provider_initialized",
        "allocation_attempted",
        "simplepim_operator_api_used",
        "virtual_zip",
        "map_kernel_executed",
        "genred_kernel_executed",
        "host_mediated_reduction",
        "all_tasks_completed",
        "exact_integer_match",
        "hardware_kernel_executed",
        "release_attempted",
        "release_confirmed",
        "hardware_functionality_evidence",
        "operator_validations_passed",
        "operator_metadata_checks_passed",
        "persistent_allocation_requested",
        "persistent_allocation_observed",
        "simplepim_managed_allocation",
    )
    for key in required_true:
        if payload.get(key) is not True:
            raise ValueError(f"response_evidence_invalid: {key} must be true")
    if payload.get("target_observed") != "physical_hardware" or payload.get("allocated_dpu_count") != DPUS:
        raise ValueError("response_evidence_invalid: physical allocation identity mismatch")
    if payload.get("status") != "completed" or payload.get("validation_status") != "passed":
        raise ValueError("response_evidence_invalid: native run is not completed and validated")
    if payload.get("parser_mode") is not False:
        raise ValueError("response_evidence_invalid: execute response has parser mode")
    if payload.get("thesis_direct_raw_sdk_allocation_used") is not False:
        raise ValueError("native_schema_correction_required: thesis_direct_raw_sdk_allocation_used=false is missing")
    if payload.get("simplepim_managed_allocation") is not True:
        raise ValueError("native_schema_correction_required: simplepim_managed_allocation=true is missing")
    repetitions = payload.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != WARMUPS + REPEATS:
        raise ValueError("response_evidence_invalid: repetition count mismatch")
    if not all(isinstance(row, dict) for row in repetitions):
        raise ValueError("response_evidence_invalid: every repetition must be an object")
    warmups = [row for row in repetitions if row.get("warmup") is True]
    measured = [row for row in repetitions if row.get("warmup") is False]
    if len(warmups) != WARMUPS or len(measured) != REPEATS or {row.get("repeat_id") for row in measured} != set(range(REPEATS)):
        raise ValueError("response_evidence_invalid: warmup/measured repetition structure mismatch")
    for row in repetitions:
        if not _HEX16.fullmatch(str(row.get("input_hash", ""))) or not _HEX16.fullmatch(str(row.get("output_hash", ""))) or row.get("exact_integer_match") is not True:
            raise ValueError("response_evidence_invalid: repetition validation/hash fields are incomplete")
        for field in ("scatter_time_s", "virtual_zip_time_s", "map_time_s", "reduction_time_s", "total_time_s"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"response_evidence_invalid: repetition timing {field} is invalid")
        component_sum = sum(float(row[field]) for field in ("scatter_time_s", "virtual_zip_time_s", "map_time_s", "reduction_time_s"))
        if not math.isclose(float(row["total_time_s"]), component_sum, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("response_evidence_invalid: repetition timing total is inconsistent")
    per_iteration = (
        "logical_payload_h2d_bytes_per_iteration",
        "logical_payload_d2h_bytes_per_iteration",
        "logical_payload_transfer_bytes_per_iteration",
    )
    totals = (
        "logical_payload_h2d_bytes_total_session",
        "logical_payload_d2h_bytes_total_session",
        "logical_payload_transfer_bytes_total_session",
    )
    if any(not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool) or payload.get(key) < 0 for key in (*per_iteration, *totals)):
        raise ValueError("native_schema_correction_required: corrected logical payload fields are missing")
    h2d, d2h, transfer = (payload[key] for key in per_iteration)
    total_h2d, total_d2h, total_transfer = (payload[key] for key in totals)
    if transfer != h2d + d2h or total_transfer != total_h2d + total_d2h:
        raise ValueError("response_evidence_invalid: native payload transfer invariant failed")
    if (total_h2d, total_d2h, total_transfer) != ((WARMUPS + REPEATS) * h2d, (WARMUPS + REPEATS) * d2h, (WARMUPS + REPEATS) * transfer):
        raise ValueError("response_evidence_invalid: native session payload totals failed")
    if payload.get("source_commit") != SIMPLEPIM_SOURCE_COMMIT:
        raise ValueError("response_evidence_invalid: pinned SimplePIM source commit mismatch")
    for key in ("staged_source_tree_sha256", "staged_overlay_tree_sha256", "patch_sha256"):
        if not _HEX64.fullmatch(str(payload.get(key, ""))):
            raise ValueError(f"response_evidence_invalid: provenance field {key} is not a sha256")
    if not _HEX16.fullmatch(str(payload.get("stage_manifest_hash", ""))):
        raise ValueError("response_evidence_invalid: stage_manifest_hash is invalid")
    for key in ("host_binary_hash", "initialization_binary_hash", "map_binary_hash", "genred_binary_hash", "genred_reduce_shared_object_hash"):
        if not _HEX16.fullmatch(str(payload.get(key, ""))):
            raise ValueError(f"response_evidence_invalid: runtime artifact hash {key} is invalid")
    for key in ("allocation_time_s", "handle_compile_time_s", "release_time_s", "total_route_time_s"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"response_evidence_invalid: route timing {key} is invalid")
    component_total = sum(float(row["total_time_s"]) for row in repetitions)
    if not math.isclose(float(payload["total_route_time_s"]), component_total + float(payload["allocation_time_s"]) + float(payload["handle_compile_time_s"]) + float(payload["release_time_s"]), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("response_evidence_invalid: route timing total is inconsistent")
    for key in ("expected_table_count_session", "observed_table_count", "mram_layout_bound_bytes_per_dpu", "mram_high_water_bytes_per_dpu"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"response_evidence_invalid: layout field {key} is invalid")
    if payload["expected_table_count_session"] != 30 or payload["observed_table_count"] != 30:
        raise ValueError("response_evidence_invalid: table high-water count mismatch")
    if payload["mram_high_water_bytes_per_dpu"] > payload["mram_layout_bound_bytes_per_dpu"]:
        raise ValueError("response_evidence_invalid: MRAM layout high-water exceeds bound")
    if payload.get("mram_capacity_verified") is not False:
        raise ValueError("response_evidence_invalid: MRAM capacity must remain unverified")


def _identity(suite: Mapping[str, Any]) -> str:
    payload = {
        "case_id": suite["workload"]["id"],
        "qualification_fixture": suite["workload"]["qualification_fixture"],
        "fixture_version": suite["workload"]["fixture_version"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _records(payload: Mapping[str, Any], suite: Mapping[str, Any], source: str) -> list[JsonDict]:
    workload = suite["workload"]
    identity = _identity(suite)
    measured = [row for row in payload["repetitions"] if row.get("warmup") is False]
    h2d = int(payload["logical_payload_h2d_bytes_per_iteration"])
    d2h = int(payload["logical_payload_d2h_bytes_per_iteration"])
    records: list[JsonDict] = []
    for row in measured:
        records.append({
            "case_id": workload["id"],
            "workload_id": workload["id"],
            "route_id": ROUTE_ID,
            "backend_id": BACKEND_ID,
            "provider_id": PROVIDER_ID,
            "benchmark_role": "operator_qualification",
            "execution_model": "fixed_rank1_kernel_qualification_fixture",
            "qualification_task_count": 1,
            "task_graph_integrated": False,
            "qualification_fixture_identity": identity,
            "contraction_plan_identity": None,
            "qualification_fixture_status": "fixed_rank1_supported",
            "operand_transport_integrated": False,
            "operand_transport_status": "native_fixed_deterministic_operands",
            "thesis_direct_raw_sdk_allocation_used": False,
            "simplepim_managed_allocation": True,
            "simplepim_allocation_used": True,
            "simplepim_operator_api_used": True,
            "operator_metadata_checks_passed": True,
            "operator_validations_passed": True,
            "simplepim_kernel_executed": True,
            "pid_comm_invoked": False,
            "communication_provider": "host_mediated",
            "atim_integrated": False,
            "hardware_execution": True,
            "hardware_functionality_evidence": True,
            "hardware_speedup_applicable": False,
            "cpu_fallback_used": False,
            "simulator_kernel_executed": False,
            "target_requested": "physical_hardware",
            "target_observed": "physical_hardware",
            "execution_class": "rank1_contraction",
            "kernel_strategy": "virtual_zip_map_int64_genred",
            "hardware_profile_version": PROFILE_ID,
            "requested_dpu_count": DPUS,
            "allocated_dpu_count": DPUS,
            "initialization_tasklets_per_dpu": 1,
            "operator_tasklets_per_dpu": TASKLETS,
            "hardware_allocation_verified": True,
            "hardware_kernel_executed": True,
            "exact_integer_match": True,
            "reference_int64": row["reference_int64"],
            "result_int64": row["result_int64"],
            "scatter_time_s": row["scatter_time_s"],
            "virtual_zip_time_s": row["virtual_zip_time_s"],
            "map_time_s": row["map_time_s"],
            "reduction_time_s": row["reduction_time_s"],
            "validation_status": "passed",
            "validation_max_abs_error": 0.0,
            "application_visible_h2d_bytes": h2d,
            "application_visible_d2h_bytes": d2h,
            "application_visible_transfer_bytes": h2d + d2h,
            "timing_scope": payload["timing_scope"],
            "timing_is_bringup_only": True,
            "total_route_time_s": row["total_time_s"],
            "warmup": False,
            "repeat_id": row["repeat_id"],
            "input_hash": row["input_hash"],
            "output_hash": row["output_hash"],
            "source_commit": payload["source_commit"],
            "hostname": payload.get("hostname"),
            "host_binary_hash": payload.get("host_binary_hash"),
            "dpu_binary_hash": payload.get("map_binary_hash"),
            "native_response_artifact": source,
            "expected_table_count_session": payload["expected_table_count_session"],
            "observed_table_count": payload["observed_table_count"],
            "mram_layout_bound_bytes_per_dpu": payload["mram_layout_bound_bytes_per_dpu"],
            "mram_high_water_bytes_per_dpu": payload["mram_high_water_bytes_per_dpu"],
            "mram_capacity_verified": False,
        })
    return records


def prepare(root_dir: Path, *, suite_path: Path, build: bool = False, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    suite = load_suite(suite_path)
    plan_dir = root_dir / "build" / "upmem_hardware_simplepim_rank1_m4_2_plan" / time.strftime("%Y-%m-%d_%H-%M-%S")
    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "config").mkdir()
    shutil.copy2(suite["path"], plan_dir / "config" / "resolved_suite.yml")
    write_json(plan_dir / "config" / "hardware_profile.json", suite["profile"])
    write_json(plan_dir / "environment.json", capture_environment(root_dir))
    result: dict[str, Any] = {"status": "prepared", "plan_dir": str(plan_dir), "dpu_allocation_attempted": False, "dpu_launch_attempted": False}
    if build:
        native = _native_dir(root_dir)
        env = dict(os.environ if environment is None else environment)
        parser_response = native / "build" / "simplepim_rank1_dot_m4_2" / "parser_response.json"
        parser_response.unlink(missing_ok=True)
        completed = _run(["make", "clean", "parser"], cwd=native, env=env, timeout_s=TIMEOUT_S)
        result["native_build"] = completed
        if completed["returncode"] != 0 or completed["timed_out"]:
            result["status"] = "failed"
        else:
            try:
                parser = _response(native, "parser_response.json")
                _require_response(parser, parser=True)
            except ValueError as exc:
                result.update({"status": "failed", "failure_stage": str(exc).split(":", 1)[0], "failure_reason": str(exc)})
            else:
                shutil.copy2(parser_response, plan_dir / "parser_response.json")
    write_json(plan_dir / "upmem_hardware_simplepim_rank1_m4_2_plan.json", {**result, "schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "route_id": ROUTE_ID, "backend_id": BACKEND_ID, "profile": suite["profile"], "claim_boundary": CLAIM_BOUNDARY})
    result["summary_path"] = str(plan_dir / "upmem_hardware_simplepim_rank1_m4_2_plan.json")
    return result


def execute(root_dir: Path, *, suite_path: Path, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if environment is None else environment)
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    selected = [key for key in SIMULATOR_ENV_KEYS if key in env]
    if selected:
        raise ValueError("hardware_profile_violation: simulator selector keys are forbidden: " + ", ".join(selected))
    suite = load_suite(suite_path)
    run_dir = create_run_dir(root_dir, SUITE_ID, artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_hw_m4_2")
    shutil.copy2(suite["path"], run_dir / "config" / "resolved_suite.yml")
    write_json(run_dir / "config" / "hardware_profile.json", suite["profile"])
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    write_run_manifest(run_dir, run_kind=SCHEMA_VERSION, suite_id=SUITE_ID, suite_path=str(suite["path"]), artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_hw_m4_2", route_id=ROUTE_ID, backend_id=BACKEND_ID, execution_scope="physical_simplepim_rank1_operator_qualification", evidence_type="functionality_only", summary="upmem_hardware_simplepim_rank1_m4_2_summary.json", command=shlex.join(["UPMEM_ALLOW_PHYSICAL_HARDWARE=1", "make", "upmem-hw-m4-2"]), root_dir=root_dir)
    native = _native_dir(root_dir)
    response_file = native / "build" / "simplepim_rank1_dot_m4_2" / "execute_response.json"
    response_file.unlink(missing_ok=True)
    completed = _run(["make", "clean", "execute"], cwd=native, env=env, timeout_s=TIMEOUT_S)
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "route_id": ROUTE_ID, "backend_id": BACKEND_ID, "status": "failed", "command": completed, "claim_boundary": CLAIM_BOUNDARY, "communication_provider": "host_mediated", "pid_comm_invoked": False, "atim_integrated": False, "qualification_fixture": "one fixed rank1 kernel task", "task_graph_integrated": False, "operand_transport_integrated": False}
    if response_file.exists():
        shutil.copy2(response_file, run_dir / "native_execute_response.json")
    try:
        if completed["returncode"] != 0 or completed["timed_out"]:
            raise ValueError("native_host_failed: response cannot be admitted after nonzero or timed-out subprocess")
        payload = _response(native, "execute_response.json")
        _require_response(payload, parser=False)
        records = _records(payload, suite, "native_execute_response.json")
        write_json(run_dir / "native_response_summary.json", payload)
        write_normalized_records(run_dir, records)
        summary.update({"status": "completed", "validation_status": "passed", "row_count": len(records), "warmup_count": WARMUPS, "repeat_count": REPEATS, "native_response": "native_execute_response.json", "session_logical_h2d_bytes": payload["logical_payload_h2d_bytes_total_session"], "session_logical_d2h_bytes": payload["logical_payload_d2h_bytes_total_session"], "session_logical_transfer_bytes": payload["logical_payload_transfer_bytes_total_session"]})
    except ValueError as exc:
        summary.update({"failure_stage": str(exc).split(":", 1)[0], "failure_reason": str(exc)})
    summary_path = run_dir / "upmem_hardware_simplepim_rank1_m4_2_summary.json"
    write_json(summary_path, summary)
    return {"run_dir": str(run_dir), "artifact": str(summary_path), "status": summary["status"], "row_count": summary.get("row_count", 0)}


__all__ = ["load_suite", "prepare", "execute", "_require_response", "_records", "PROFILE_ID", "BACKEND_ID", "ROUTE_ID"]
