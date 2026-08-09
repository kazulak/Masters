"""M4.4 Python orchestration for the physical SimplePIM chain fixture.

The native worker is intentionally kept outside this module.  This wrapper
prepares the deterministic fixture, invokes that worker, and admits only a
physical, native TaskGraph response.  It never selects a simulator or CPU
fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
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
from quantum_bench.core.records import to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.simplepim_chain_task import (
    CHAIN_CASE_ID,
    CHAIN_FIXTURE_VERSION,
    build_simplepim_chain_graph_binding,
    build_simplepim_chain_workload,
    validate_simplepim_chain_workload,
)


SCHEMA_VERSION = "upmem_hardware_simplepim_chain_m4_4_v1"
NATIVE_SCHEMA_VERSION = "simplepim_chain_m4_4_v1"
PROFILE_ID = "hardware_simplepim_chain_m4_4_v1"
BACKEND_ID = "upmem_sdk_hardware_simplepim_chain_m4_4"
ROUTE_ID = "upmem_tn_hardware_simplepim_chain_m4_4"
SUITE_ID = "upmem_hardware_simplepim_chain_m4_4"
NATIVE_REL = Path("native/upmem/simplepim/upmem_sdk_chain_m4_4")
NATIVE_BUILD_REL = Path("build/simplepim_chain_m4_4")
WARMUPS = 1
REPEATS = 5
DPUS = 1
TASKLETS = 1
TIMEOUT_S = 120.0
CLAIM_BOUNDARY = (
    "one-DPU native SimplePIM two-task real int8 chain with device-resident "
    "intermediate and host final reduction; "
    "no speedup, energy, multi-DPU, PID-Comm, ATiM, or general TN claim"
)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_suite() -> Path:
    return _root() / "configs" / "suites" / f"{SUITE_ID}.yml"


def load_suite(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved != _canonical_suite().resolve():
        raise ValueError("hardware_profile_violation: M4.4 requires the committed suite")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("suite_id") != SUITE_ID:
        raise ValueError("hardware_profile_violation: M4.4 suite identity mismatch")
    if raw.get("fail_fast") is not True:
        raise ValueError("hardware_profile_violation: M4.4 fail_fast is required")
    if raw.get("defaults") != {"warmups": WARMUPS, "repeats": REPEATS, "timeout_s": TIMEOUT_S}:
        raise ValueError("hardware_profile_violation: M4.4 repeat profile mismatch")
    expected = {
        "hardware_profile_version": PROFILE_ID,
        "target": "hardware",
        "backend_id": BACKEND_ID,
        "route_id": ROUTE_ID,
        "requested_dpu_count": DPUS,
        "tasklets_per_dpu": TASKLETS,
        "effective_operator_tasklets": TASKLETS,
        "final_reduction_location": "host",
        "intermediate_residency": "device_mram",
        "numeric_mode": "real_int8_int32_accumulator",
        "chain_length": 256,
        "task_count": 2,
        "synchronous_execution": True,
        "performance_claim_applicable": False,
    }
    if raw.get("metadata", {}).get("hardware_profile") != expected:
        raise ValueError("hardware_profile_violation: M4.4 profile is not fixed")
    workloads = raw.get("workloads")
    if (
        not isinstance(workloads, list)
        or len(workloads) != 1
        or workloads[0].get("id") != CHAIN_CASE_ID
        or workloads[0].get("fixture_version") != CHAIN_FIXTURE_VERSION
    ):
        raise ValueError("hardware_profile_violation: M4.4 requires the chain fixture")
    task_graph = workloads[0].get("task_graph")
    expected_task_graph = {
        "binding_protocol": "M44_GRAPH_BINDING_V1",
        "task_count": 2,
        "dependencies": [["task_1", "task_0"]],
        "input_shapes": [[256], [256], [256]],
        "output_shape": [],
        "operation_kinds": [
            "elementwise_product_i8_i8",
            "scalar_product_i32_i8_reduce_i64",
        ],
    }
    if task_graph != expected_task_graph:
        raise ValueError("hardware_profile_violation: M4.4 task graph contract mismatch")
    return {"raw": raw, "path": resolved, "profile": expected, "workload": workloads[0]}


def _run(command: list[str], *, cwd: Path, env: Mapping[str, str], timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=cwd, env=dict(env), capture_output=True, text=True, check=False, timeout=timeout_s)
        return {"command": command, "returncode": result.returncode, "timed_out": False, "elapsed_s": time.perf_counter() - started, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}
    except subprocess.TimeoutExpired as exc:
        return {"command": command, "returncode": None, "timed_out": True, "elapsed_s": time.perf_counter() - started, "stdout": str(exc.stdout or "")[-4000:], "stderr": str(exc.stderr or "")[-4000:]}
    except OSError as exc:
        return {"command": command, "returncode": None, "timed_out": False, "elapsed_s": time.perf_counter() - started, "stdout": "", "stderr": str(exc)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_artifacts(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    workload = build_simplepim_chain_workload()
    validate_simplepim_chain_workload(workload)
    operands = run_dir / "operands.bin"
    operands.write_bytes(b"".join(array.tobytes() for array in workload.operands))
    input_sha256 = _sha256(operands)
    binding = build_simplepim_chain_graph_binding(workload, input_sha256=input_sha256)
    binding_path = run_dir / "graph_binding.txt"
    binding_path.write_bytes(binding.text.encode("ascii"))
    manifest = {
        "schema_version": "m4_4_chain_input_v1",
        "case_id": CHAIN_CASE_ID,
        "operands_file": operands.name,
        "input_sha256": input_sha256,
        "graph_binding_file": binding_path.name,
        "graph_binding_sha256": binding.sha256,
        "fixture_version": CHAIN_FIXTURE_VERSION,
        "input_dtype": "int8",
        "byte_layout": "chain_a_int8[256],chain_b_int8[256],chain_c_int8[256]",
        "task_count": 2,
        "tile_length": 64,
        "operand_sha256": workload.operand_sha256,
        "expected_path": [list(step) for step in workload.graph.path],
        **binding.fields,
        "reference_int64": workload.reference_int64,
    }
    write_json(run_dir / "input_manifest.json", manifest)
    write_json(run_dir / "taskgraph_manifest.json", {"graph": to_jsonable(workload.graph), "case_id": CHAIN_CASE_ID})
    return operands, manifest


def _environment(root_dir: Path) -> dict[str, Any]:
    result = capture_environment(root_dir)
    result["upmem_m4_4"] = {"native_route": str(NATIVE_REL), "profile_id": PROFILE_ID, "backend_id": BACKEND_ID}
    return result


def _response_path(native: Path) -> Path:
    candidates = (
        native / NATIVE_BUILD_REL / "execute_response.json",
        native / NATIVE_BUILD_REL / "response.json",
        native / "build" / "execute_response.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _native_command(
    native: Path,
    operands: Path,
    binding: Path,
    response: Path,
    *,
    input_sha256: str,
    graph_binding_sha256: str,
) -> tuple[list[str], Path]:
    binary_candidates = (
        native / NATIVE_BUILD_REL / "staged" / "benchmarks" / "chain_m4_4" / "bin" / "chain_host",
        native / NATIVE_BUILD_REL / "chain_host",
    )
    binary = next((path for path in binary_candidates if path.is_file()), binary_candidates[0])
    command = [
        str(binary),
        "--mode",
        "execute",
        "--response",
        str(response),
        "--operands-file",
        str(operands),
        "--input-sha256",
        input_sha256,
        "--graph-binding",
        str(binding),
        "--graph-binding-sha256",
        graph_binding_sha256,
    ]
    return command, binary.parent.parent


def _validate_response(payload: Mapping[str, Any], *, manifest: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("response_evidence_invalid: native response must be a mapping")
    expected = {
        "schema_version": NATIVE_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "backend_id": BACKEND_ID,
        "route_id": ROUTE_ID,
        "target_requested": "physical_hardware",
        "target_observed": "physical_hardware",
        "requested_dpu_count": DPUS,
        "allocated_dpu_count": DPUS,
        "tasklets_per_dpu": TASKLETS,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "hardware_speedup_applicable": False,
        "status": "completed",
        "validation_status": "passed",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"response_evidence_invalid: {key} mismatch")
    binding_identity = {
        "case_id": manifest["case_id"],
        "fixture_version": manifest["fixture_version"],
        "circuit_semantics_hash": manifest["circuit_semantics_hash"],
        "tensor_network_hash": manifest["tensor_network_hash"],
        "contraction_plan_hash": manifest["contraction_plan_hash"],
        "contraction_path_structure_hash": manifest["contraction_path_structure_hash"],
        "input_sha256": manifest["input_sha256"],
        "reference_int64": manifest["reference_int64"],
        "graph_binding_sha256": manifest["graph_binding_sha256"],
        "input_dtype": "int8",
        "accumulator_dtype": "int32",
        "length": 256,
        "task_count": 2,
        "path": manifest["expected_path"],
        "task_order": ["task_0", "task_1"],
        "task_dependencies": [[], ["task_0"]],
        "operation_kinds": [
            "elementwise_product_i8_i8",
            "scalar_product_i32_i8_reduce_i64",
        ],
    }
    for key, value in binding_identity.items():
        if payload.get(key) != value:
            raise ValueError(f"response_evidence_invalid: graph binding {key} mismatch")
    if payload.get("graph_binding_validated") is not True:
        raise ValueError("response_evidence_invalid: graph binding was not validated")
    if payload.get("native_taskgraph_protocol") is not True or payload.get("task_graph_integrated") is not True:
        raise ValueError("response_evidence_invalid: native TaskGraph integration was not admitted")
    for key in ("provider_initialized", "simplepim_operator_api_used", "native_kernel_executed", "hardware_kernel_executed", "all_tasks_completed", "exact_integer_match", "release_confirmed", "hardware_functionality_evidence"):
        if payload.get(key) is not True:
            raise ValueError(f"response_evidence_invalid: {key} must be true")
    if payload.get("effective_operator_tasklets") != TASKLETS:
        raise ValueError("response_evidence_invalid: effective operator tasklet count mismatch")
    if payload.get("final_reduction_location") != "host":
        raise ValueError("response_evidence_invalid: final host reduction must be explicit")
    if payload.get("intermediate_residency") != "device_mram":
        raise ValueError("response_evidence_invalid: intermediate device residency must be explicit")
    if payload.get("input_sha256") != manifest["input_sha256"] or payload.get("reference_int64") != manifest["reference_int64"]:
        raise ValueError("response_evidence_invalid: input identity mismatch")
    repetitions = payload.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != WARMUPS + REPEATS:
        raise ValueError("response_evidence_invalid: repetition count mismatch")
    if any(not isinstance(row, Mapping) for row in repetitions):
        raise ValueError("response_evidence_invalid: repetition rows must be mappings")
    warmups = [row for row in repetitions if row.get("warmup") is True]
    measured = [row for row in repetitions if row.get("warmup") is False]
    if len(warmups) != WARMUPS:
        raise ValueError("response_evidence_invalid: warmup marker structure mismatch")
    if len(measured) != REPEATS or {row.get("repeat_id") for row in measured} != set(range(REPEATS)):
        raise ValueError("response_evidence_invalid: measured repeat structure mismatch")
    for row in repetitions:
        if not isinstance(row.get("warmup"), bool) or not isinstance(row.get("repeat_id"), int):
            raise ValueError("response_evidence_invalid: repetition markers are invalid")
        if row.get("result_int64") != manifest["reference_int64"] or row.get("exact_integer_match") is not True:
            raise ValueError("response_evidence_invalid: result validation mismatch")
        timing_fields = ("scatter_time_s", "virtual_zip_time_s", "map_time_s", "reduction_time_s", "total_time_s")
        timings = []
        for field in timing_fields:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"response_evidence_invalid: invalid repetition timing {field}")
            timings.append(float(value))
        if not math.isclose(timings[-1], sum(timings[:-1]), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("response_evidence_invalid: repetition timing components do not sum")
    h2d = payload.get("application_visible_h2d_bytes")
    d2h = payload.get("application_visible_d2h_bytes")
    total = payload.get("application_visible_transfer_bytes")
    if (h2d, d2h, total) != (768, 8, 776):
        raise ValueError("response_evidence_invalid: fixed chain transfer contract mismatch")
    if not all(isinstance(value, int) and value >= 0 for value in (h2d, d2h, total)) or total != h2d + d2h:
        raise ValueError("response_evidence_invalid: transfer invariant failed")
    intermediate_fields = ("intermediate_h2d_bytes", "intermediate_d2h_bytes", "intermediate_transfer_bytes")
    present = [field for field in intermediate_fields if field in payload]
    if present:
        if len(present) != len(intermediate_fields) or any(not isinstance(payload[field], int) or payload[field] < 0 for field in intermediate_fields):
            raise ValueError("response_evidence_invalid: intermediate transfer fields are incomplete")
        if tuple(payload[field] for field in intermediate_fields) != (0, 0, 0):
            raise ValueError("response_evidence_invalid: intermediate transfer must be zero")


def _records(payload: Mapping[str, Any], manifest: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    return [
        {
            "case_id": CHAIN_CASE_ID,
            "fixture_version": manifest["fixture_version"],
            "circuit_semantics_hash": manifest["circuit_semantics_hash"],
            "tensor_network_hash": manifest["tensor_network_hash"],
            "contraction_plan_hash": manifest["contraction_plan_hash"],
            "contraction_path_structure_hash": manifest["contraction_path_structure_hash"],
            "graph_binding_sha256": manifest["graph_binding_sha256"],
            "input_sha256": manifest["input_sha256"],
            "path": manifest["expected_path"],
            "route_id": ROUTE_ID,
            "backend_id": BACKEND_ID,
            "benchmark_role": "physical_simplepim_native_taskgraph_functionality",
            "execution_model": "one_dpu_one_tasklet_resident_chain",
            "intermediate_residency": "device_mram",
            "final_reduction_location": "host",
            "effective_operator_tasklets": TASKLETS,
            "task_graph_integrated": True,
            "native_taskgraph_protocol": True,
            "graph_binding_validated": True,
            "source_task_count": 2,
            "source_task_completion_count": 2,
            "input_dtype": "int8",
            "accumulator_dtype": "int32",
            "reference_int64": row["reference_int64"],
            "result_int64": row["result_int64"],
            "exact_integer_match": True,
            "validation_status": "passed",
            "hardware_execution": True,
            "hardware_functionality_evidence": True,
            "target_requested": "physical_hardware",
            "target_observed": "physical_hardware",
            "requested_dpu_count": DPUS,
            "allocated_dpu_count": DPUS,
            "tasklets_per_dpu": TASKLETS,
            "native_kernel_executed": True,
            "cpu_fallback_used": False,
            "simulator_kernel_executed": False,
            "hardware_speedup_applicable": False,
            "application_visible_h2d_bytes": payload["application_visible_h2d_bytes"],
            "application_visible_d2h_bytes": payload["application_visible_d2h_bytes"],
            "application_visible_transfer_bytes": payload["application_visible_transfer_bytes"],
            "total_route_time_s": row.get("total_route_time_s", row.get("total_time_s")),
            "timing_scope": "physical SimplePIM native chain bring-up",
            "timing_is_bringup_only": True,
            "repeat_id": row["repeat_id"],
            "warmup": row["warmup"],
            "native_response_artifact": source,
            "input_file_sha256": manifest["input_sha256"],
            "native_binary_hash": payload.get("native_binary_hash"),
            "hostname": payload.get("hostname"),
        }
        for row in payload["repetitions"]
    ]


def prepare(root_dir: Path, *, suite_path: Path, build: bool = False, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    suite = load_suite(suite_path)
    plan_dir = root_dir / "build" / f"{SUITE_ID}_plan" / time.strftime("%Y-%m-%d_%H-%M-%S")
    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "config").mkdir()
    shutil.copy2(suite["path"], plan_dir / "config" / "resolved_suite.yml")
    write_json(plan_dir / "config" / "hardware_profile.json", suite["profile"])
    write_json(plan_dir / "environment.json", _environment(root_dir))
    operands, manifest = _input_artifacts(plan_dir)
    result: dict[str, Any] = {"status": "prepared", "plan_dir": str(plan_dir), "dpu_allocation_attempted": False, "dpu_launch_attempted": False, "input_sha256": manifest["input_sha256"], "reference_int64": manifest["reference_int64"]}
    if build:
        native = root_dir / NATIVE_REL
        build_env = dict(os.environ if environment is None else environment)
        # The native Makefile may parse a physical-build guard.  Preparation
        # only compiles; it never invokes the host or allocates a DPU.
        build_env["UPMEM_ALLOW_PHYSICAL_HARDWARE"] = "1"
        result["native_build"] = _run(["make", "clean", "build"], cwd=native, env=build_env, timeout_s=TIMEOUT_S)
        if result["native_build"]["returncode"] != 0 or result["native_build"]["timed_out"]:
            result["status"] = "failed"
    artifact = plan_dir / f"{SUITE_ID}_plan.json"
    write_json(artifact, {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "route_id": ROUTE_ID, "backend_id": BACKEND_ID, "profile": suite["profile"], "claim_boundary": CLAIM_BOUNDARY, **result})
    result["artifact"] = str(artifact)
    return result


def execute(root_dir: Path, *, suite_path: Path, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if environment is None else environment)
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    selected = [key for key in SIMULATOR_ENV_KEYS if key in env]
    if selected:
        raise ValueError("hardware_profile_violation: simulator selector keys are forbidden: " + ", ".join(selected))
    suite = load_suite(suite_path)
    run_dir = create_run_dir(root_dir, SUITE_ID, artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_hw_m4_4")
    shutil.copy2(suite["path"], run_dir / "config" / "resolved_suite.yml")
    write_json(run_dir / "config" / "hardware_profile.json", suite["profile"])
    write_json(run_dir / "environment.json", _environment(root_dir))
    operands, manifest = _input_artifacts(run_dir)
    binding = run_dir / manifest["graph_binding_file"]
    write_run_manifest(run_dir, run_kind=SCHEMA_VERSION, suite_id=SUITE_ID, suite_path=str(suite["path"]), artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_hw_m4_4", route_id=ROUTE_ID, backend_id=BACKEND_ID, execution_scope="physical_one_dpu_native_simplepim_chain", evidence_type="functionality_only", summary=f"{SUITE_ID}_summary.json", command=shlex.join(["UPMEM_ALLOW_PHYSICAL_HARDWARE=1", "make", "upmem-hw-m4-4"]), root_dir=root_dir)
    native = root_dir / NATIVE_REL
    response = _response_path(native)
    response.unlink(missing_ok=True)
    build_result = _run(["make", "clean", "build"], cwd=native, env=env, timeout_s=TIMEOUT_S)
    command_result = build_result
    if build_result["returncode"] == 0 and not build_result["timed_out"]:
        command, cwd = _native_command(
            native,
            operands,
            binding,
            response,
            input_sha256=manifest["input_sha256"],
            graph_binding_sha256=manifest["graph_binding_sha256"],
        )
        command_result = _run(["timeout", "--signal=TERM", "--kill-after=5s", f"{int(TIMEOUT_S)}s", *command], cwd=cwd, env=env, timeout_s=TIMEOUT_S + 10)
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "route_id": ROUTE_ID, "backend_id": BACKEND_ID, "status": "failed", "claim_boundary": CLAIM_BOUNDARY, "command": command_result, "dpu_allocation_attempted": command_result is not build_result, "dpu_launch_attempted": command_result is not build_result, "input_sha256": manifest["input_sha256"], "reference_int64": manifest["reference_int64"]}
    if response.exists():
        shutil.copy2(response, run_dir / "native_execute_response.json")
    try:
        payload: dict[str, Any] | None = None
        if response.exists():
            loaded = json.loads(response.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
                write_json(run_dir / "native_response_summary.json", payload)
        if command_result["returncode"] != 0 or command_result["timed_out"]:
            if payload is None:
                raise ValueError("native_host_failed: build or physical native command failed")
            summary.update(
                {
                    key: payload[key]
                    for key in (
                        "failure_stage",
                        "reason",
                        "allocation_attempted",
                        "launch_attempted",
                        "release_confirmed",
                    )
                    if key in payload
                }
            )
            # A failed or timed-out native process cannot report a completed run,
            # even if its response payload is malformed or stale.
            summary["status"] = "failed"
            summary["failure_reason"] = payload.get("reason", "native command failed")
            raise ValueError(
                f"{summary.get('failure_stage', 'native_host_failed')}: "
                f"{summary['failure_reason']}"
            )
        if payload is None:
            raise ValueError("native_response_missing: native worker did not produce a response")
        _validate_response(payload, manifest=manifest)
        records = _records(payload, manifest, "native_execute_response.json")
        write_normalized_records(run_dir, records)
        summary.update({"status": "completed", "validation_status": "passed", "row_count": len(records), "hardware_functionality_evidence": True, "native_taskgraph_protocol": True, "cpu_fallback_used": False})
    except (OSError, TypeError, ValueError, KeyError, AttributeError, json.JSONDecodeError) as exc:
        summary.setdefault("failure_stage", str(exc).split(":", 1)[0])
        summary.setdefault("failure_reason", str(exc))
    artifact = run_dir / f"{SUITE_ID}_summary.json"
    write_json(artifact, summary)
    return {"run_dir": str(run_dir), "artifact": str(artifact), "status": summary["status"], "row_count": summary.get("row_count", 0)}


__all__ = ["load_suite", "prepare", "execute", "_validate_response", "PROFILE_ID", "BACKEND_ID", "ROUTE_ID"]
