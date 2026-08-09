"""M4.3: transport one real TaskGraph rank-1 task to the M4.2 primitive.

This wrapper is intentionally narrow.  It requires the native M4.2 host to
support ``--operands-file``; an older host is rejected rather than silently
executing its fixed qualification vectors.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import yaml

from quantum_bench.bench.provider_qualification import SIMULATOR_ENV_KEYS
from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import ContractionTask, JsonDict, TaskGraph
from quantum_bench.environment import capture_environment
from quantum_bench.tn import build_execution_bundle, execution_identity_metadata, with_execution_identity
from quantum_bench.tn.execution_bundle import canonical_hash


SCHEMA_VERSION = "upmem_hardware_simplepim_taskgraph_m4_3_v1"
OPERAND_BINDING_SCHEMA = "m4_3_taskgraph_operand_binding_v1"
NATIVE_SCHEMA_VERSION = "simplepim_rank1_dot_m4_2_v1"
NATIVE_PROFILE_ID = "hardware_simplepim_rank1_dot_m4_2_v1"
NATIVE_BACKEND_ID = "upmem_sdk_hardware_simplepim_rank1_dot_m4_2"
NATIVE_ROUTE_ID = "upmem_tn_hardware_simplepim_rank1_dot_m4_2"
PROFILE_ID = "hardware_simplepim_rank1_taskgraph_m4_3_v1"
BACKEND_ID = "upmem_sdk_hardware_simplepim_taskgraph_m4_3"
ROUTE_ID = "upmem_tn_hardware_simplepim_taskgraph_m4_3"
SUITE_ID = "upmem_hardware_simplepim_taskgraph_m4_3"
NATIVE_REL = Path("native/upmem/simplepim/upmem_sdk_rank1_dot_m4_2")
WARMUPS = 1
REPEATS = 5
DPUS = 2
TASKLETS = 12
VECTOR_LENGTH = 256
TIMEOUT_S = 120.0
CLAIM_BOUNDARY = (
    "one real bounded rank-1 ContractionTask-derived operand adapter transported to the SimplePIM operator; "
    "no general TN, speedup, energy, PID-Comm, ATiM, or scaling claim"
)
_HEX64 = set("0123456789abcdef")


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_suite() -> Path:
    return _root() / "configs" / "suites" / f"{SUITE_ID}.yml"


def _load_workload() -> Any:
    """Load the thesis-owned fixture lazily so plan validation stays explicit."""

    try:
        from quantum_bench.targets.upmem.simplepim_rank1_task import (
            build_rank1_taskgraph_workload,
            validate_rank1_task,
        )
    except ImportError as exc:
        raise ValueError(
            "hardware_profile_violation: simplepim_rank1_task target adapter is required"
        ) from exc
    workload = build_rank1_taskgraph_workload()
    result = validate_rank1_task(workload)
    if result is False:
        raise ValueError("hardware_profile_violation: rank1 TaskGraph fixture validation failed")
    return workload


def load_suite(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved != _canonical_suite().resolve():
        raise ValueError("hardware_profile_violation: M4.3 requires the committed suite")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("suite_id") != SUITE_ID or raw.get("schema_version") != 1:
        raise ValueError("hardware_profile_violation: M4.3 suite identity mismatch")
    if raw.get("fail_fast") is not True:
        raise ValueError("hardware_profile_violation: M4.3 fail_fast is required")
    if raw.get("defaults") != {"warmups": WARMUPS, "repeats": REPEATS, "timeout_s": TIMEOUT_S}:
        raise ValueError("hardware_profile_violation: M4.3 repeat profile mismatch")
    profile = raw.get("metadata", {}).get("hardware_profile")
    expected = {
        "hardware_profile_version": PROFILE_ID,
        "target": "hardware",
        "backend_id": BACKEND_ID,
        "route_id": ROUTE_ID,
        "requested_dpu_count": DPUS,
        "tasklets_per_dpu": TASKLETS,
        "numeric_mode": "int8_inputs_int32_accumulator",
        "vector_length": VECTOR_LENGTH,
        "synchronous_execution": True,
        "performance_claim_applicable": False,
    }
    if profile != expected:
        raise ValueError("hardware_profile_violation: M4.3 profile is not fixed")
    workloads = raw.get("workloads")
    if not isinstance(workloads, list) or len(workloads) != 1 or not isinstance(workloads[0], dict):
        raise ValueError("hardware_profile_violation: M4.3 requires one workload")
    return {"raw": raw, "path": resolved, "profile": expected, "workload": workloads[0]}


def _run(command: list[str], *, cwd: Path, env: Mapping[str, str], timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=cwd, env=dict(env), capture_output=True, text=True, check=False, timeout=timeout_s)
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
        return {"command": command, "returncode": None, "timed_out": False, "elapsed_s": time.perf_counter() - started, "stdout": "", "stderr": str(exc)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _int64_output_hash(value: object) -> str:
    if not _is_integer(value):
        raise ValueError("response_evidence_invalid: result_int64 is not an integer")
    return _fnv1a64(np.asarray([int(value)], dtype="<i8").tobytes())


def _m4_3_environment(root_dir: Path) -> dict[str, Any]:
    """Capture M4.3-specific tool identity without changing global probes."""

    environment = capture_environment(root_dir)
    patch = root_dir / NATIVE_REL / "simplepim_rank1_hardening.patch"
    dpu_compiler = shutil.which("dpu-upmem-dpurte-clang")
    sdk_config = shutil.which("dpu-pkg-config")
    sdk_version = None
    if sdk_config:
        result = subprocess.run([sdk_config, "--modversion", "dpu"], capture_output=True, text=True, check=False, timeout=5)
        sdk_version = result.stdout.strip() if result.returncode == 0 else None
    environment["simplepim"] = {
        "integration": True,
        "available": True,
        "probe_status": "m4_3_native_adapter",
        "pinned_source_commit": "1d639c53532555f01e9f71d872e7712b166d6cba",
        "adapter_path": str(NATIVE_REL),
        "adapter_patch_sha256": _sha256(patch) if patch.is_file() else None,
        "dpu_compiler": dpu_compiler,
        "sdk_config": sdk_config,
        "sdk_version": sdk_version,
    }
    environment["upmem_sdk"] = {"compiler_path": dpu_compiler, "pkg_config_path": sdk_config, "version": sdk_version, "available": bool(dpu_compiler and sdk_config)}
    environment["thesis_source_commit"] = environment.get("git_commit")
    environment["simplepim_source_commit"] = "1d639c53532555f01e9f71d872e7712b166d6cba"
    return environment


def _hex64(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and set(text) <= _HEX64


def _graph(workload: Any) -> TaskGraph:
    graph = with_execution_identity(workload.graph)
    if not isinstance(graph, TaskGraph) or len(graph.tasks) != 1:
        raise ValueError("hardware_profile_violation: M4.3 requires exactly one TaskGraph task")
    task = graph.tasks[0]
    if task.dependencies or task.input_shapes != ((VECTOR_LENGTH,), (VECTOR_LENGTH,)) or task.output_shape != ():
        raise ValueError("hardware_profile_violation: unsupported rank1 task shape or dependency")
    if (task.gemm_m, task.gemm_k, task.gemm_n) != (1, VECTOR_LENGTH, 1):
        raise ValueError("hardware_profile_violation: M4.3 GEMM shape must be 1x256x1")
    if len(task.contracted_labels) != 1:
        raise ValueError("hardware_profile_violation: M4.3 requires one contracted label")
    return graph


def _write_operands(run_dir: Path, workload: Any) -> tuple[Path, str, str, int]:
    left = np.asarray(workload.left)
    right = np.asarray(workload.right)
    if left.shape != (VECTOR_LENGTH,) or right.shape != (VECTOR_LENGTH,) or left.dtype.kind not in "iu" or right.dtype.kind not in "iu":
        raise ValueError("hardware_profile_violation: operands must be two real vectors of 256 elements")
    left_i8 = left.astype(np.int8, copy=False)
    right_i8 = right.astype(np.int8, copy=False)
    path = run_dir / "operands.bin"
    path.write_bytes(left_i8.tobytes(order="C") + right_i8.tobytes(order="C"))
    reference = int(np.dot(left_i8.astype(np.int64), right_i8.astype(np.int64)))
    raw = path.read_bytes()
    return path, _sha256(path), _fnv1a64(raw), reference


def _identity_payload(graph: TaskGraph, *, case_id: str) -> JsonDict:
    return {
        **execution_identity_metadata(graph, plan_reused=False),
        "execution_bundle": build_execution_bundle(graph, case_id=case_id, suite_id=SUITE_ID),
    }


def _operand_binding_payload(graph: TaskGraph, task: ContractionTask, *, input_sha256: str) -> JsonDict:
    """Canonical host binding for the bytes passed to the fixed native operator."""

    return {
        "schema_version": OPERAND_BINDING_SCHEMA,
        "contraction_plan_hash": graph.contraction_plan_hash,
        "circuit_semantics_hash": graph.circuit_semantics_hash,
        "tensor_network_hash": graph.tensor_network_hash,
        "task": {
            "id": task.id,
            "input_tensor_ids": task.input_tensor_ids,
            "output_tensor_id": task.output_tensor_id,
            "dependencies": task.dependencies,
            "index_expression": task.index_expression,
            "input_shapes": task.input_shapes,
            "output_shape": task.output_shape,
            "left_labels": task.left_labels,
            "right_labels": task.right_labels,
            "contracted_labels": task.contracted_labels,
            "output_labels": task.output_labels,
            "gemm_m": task.gemm_m,
            "gemm_k": task.gemm_k,
            "gemm_n": task.gemm_n,
            "structure": task.structure,
        },
        "input_file_sha256": input_sha256,
    }


def _operand_binding(graph: TaskGraph, task: ContractionTask, *, input_sha256: str) -> JsonDict:
    payload = _operand_binding_payload(graph, task, input_sha256=input_sha256)
    return {**payload, "binding_hash": canonical_hash(payload)}


def _require_operand_binding(identities: Mapping[str, Any], graph: TaskGraph, *, input_sha256: str) -> JsonDict:
    expected = _operand_binding(graph, graph.tasks[0], input_sha256=input_sha256)
    actual = identities.get("operand_binding")
    if actual != expected:
        raise ValueError("response_evidence_invalid: host operand binding mismatch")
    return expected


def _native_response(native_dir: Path, name: str = "execute_response.json") -> dict[str, Any]:
    path = native_dir / "build" / "simplepim_rank1_dot_m4_2" / name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"response_transport_failed: invalid native response {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("response_transport_failed: native response is not an object")
    return payload


def _require_response(
    payload: Mapping[str, Any],
    *,
    identities: Mapping[str, Any],
    input_sha256: str,
    input_hash: str,
    reference: int,
    expected_output_hash: str | None = None,
) -> None:
    if not _hex64(input_sha256) or len(input_hash) != 16:
        raise ValueError("response_evidence_invalid: wrapper operand hashes are invalid")
    if not _is_integer(reference):
        raise ValueError("response_evidence_invalid: reference_int64 is not an integer")
    expected = {
        "schema_version": NATIVE_SCHEMA_VERSION,
        "profile_id": NATIVE_PROFILE_ID,
        "backend_id": NATIVE_BACKEND_ID,
        "route_id": NATIVE_ROUTE_ID,
        "target_requested": "physical_hardware",
        "target_observed": "physical_hardware",
        "allocation_profile": "backend=hw",
        "requested_dpu_count": DPUS,
        "allocated_dpu_count": DPUS,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "hardware_speedup_applicable": False,
        "warmup_count": WARMUPS,
        "repeat_count": REPEATS,
        "external_operand_transport": True,
        "operand_input_length_bytes": 512,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"response_evidence_invalid: {key} mismatch")
    if payload.get("operand_input_hash") != input_hash:
        raise ValueError("response_evidence_invalid: native operand hash mismatch")
    required_true = ("provider_initialized", "simplepim_operator_api_used", "operator_validations_passed", "operator_metadata_checks_passed", "all_tasks_completed", "exact_integer_match", "hardware_kernel_executed", "release_attempted", "release_confirmed", "hardware_functionality_evidence", "persistent_allocation_observed", "simplepim_managed_allocation")
    if any(payload.get(key) is not True for key in required_true):
        raise ValueError("response_evidence_invalid: hardware execution/release flags incomplete")
    if payload.get("status") != "completed" or payload.get("validation_status") != "passed":
        raise ValueError("response_evidence_invalid: native run is not completed and validated")
    repetitions = payload.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != WARMUPS + REPEATS:
        raise ValueError("response_evidence_invalid: repetition count mismatch")
    if len([row for row in repetitions if row.get("warmup") is True]) != WARMUPS or len([row for row in repetitions if row.get("warmup") is False]) != REPEATS:
        raise ValueError("response_evidence_invalid: warmup/measured repetition structure mismatch")
    measured = [row for row in repetitions if row.get("warmup") is False]
    if {row.get("repeat_id") for row in measured} != set(range(REPEATS)):
        raise ValueError("response_evidence_invalid: measured repeat ids mismatch")
    for row in repetitions:
        if row.get("input_hash") != input_hash or row.get("reference_int64") != reference or row.get("result_int64") != reference or row.get("exact_integer_match") is not True:
            raise ValueError("response_evidence_invalid: repetition validation mismatch")
        result_hash = _int64_output_hash(row.get("result_int64"))
        if row.get("output_hash") != result_hash:
            raise ValueError("response_evidence_invalid: native output hash mismatch")
        if expected_output_hash is not None and row.get("output_hash") != expected_output_hash:
            raise ValueError("response_evidence_invalid: wrapper output hash mismatch")
        for field in ("scatter_time_s", "virtual_zip_time_s", "map_time_s", "reduction_time_s", "total_time_s"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"response_evidence_invalid: invalid timing {field}")
        if not math.isclose(float(row["total_time_s"]), sum(float(row[field]) for field in ("scatter_time_s", "virtual_zip_time_s", "map_time_s", "reduction_time_s")), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("response_evidence_invalid: timing components do not sum")
    for key in ("logical_payload_h2d_bytes_per_iteration", "logical_payload_d2h_bytes_per_iteration", "logical_payload_transfer_bytes_per_iteration"):
        if not isinstance(payload.get(key), int) or payload[key] < 0:
            raise ValueError("response_evidence_invalid: transfer fields are incomplete")
    if payload["logical_payload_transfer_bytes_per_iteration"] != payload["logical_payload_h2d_bytes_per_iteration"] + payload["logical_payload_d2h_bytes_per_iteration"]:
        raise ValueError("response_evidence_invalid: transfer invariant failed")
    if (payload["logical_payload_h2d_bytes_per_iteration"], payload["logical_payload_d2h_bytes_per_iteration"], payload["logical_payload_transfer_bytes_per_iteration"]) != (2048, 16, 2064):
        raise ValueError("response_evidence_invalid: M4.2 expanded int8-to-int32 transfer contract mismatch")
    session_fields = ("logical_payload_h2d_bytes_total_session", "logical_payload_d2h_bytes_total_session", "logical_payload_transfer_bytes_total_session")
    if any(not isinstance(payload.get(key), int) or payload[key] < 0 for key in session_fields):
        raise ValueError("response_evidence_invalid: session transfer fields are incomplete")
    if tuple(payload[key] for key in session_fields) != (12288, 96, 12384):
        raise ValueError("response_evidence_invalid: session transfer totals mismatch")
    for key in ("host_binary_hash", "initialization_binary_hash", "map_binary_hash", "genred_binary_hash", "genred_reduce_shared_object_hash"):
        if not payload.get(key):
            raise ValueError(f"response_evidence_invalid: missing provenance field {key}")
    simplepim_commit = payload.get("simplepim_source_commit") or payload.get("source_commit")
    if simplepim_commit != "1d639c53532555f01e9f71d872e7712b166d6cba":
        raise ValueError("response_evidence_invalid: pinned SimplePIM source commit mismatch")


def _records(payload: Mapping[str, Any], identities: Mapping[str, Any], task: ContractionTask, source: str, *, input_sha256: str | None = None, operand_binding_hash: str | None = None, thesis_source_commit: str | None = None) -> list[JsonDict]:
    rows = []
    for row in payload["repetitions"]:
        if row.get("warmup"):
            continue
        rows.append({
            "case_id": "m4_3_rank1_taskgraph_256",
            "workload_id": "m4_3_rank1_taskgraph_256",
            "route_id": ROUTE_ID,
            "backend_id": BACKEND_ID,
            "benchmark_role": "physical_taskgraph_derived_operand_adapter_functionality",
            "execution_model": "one_task_rank1_taskgraph_derived_operand_adapter",
            "task_graph_integrated": False,
            "taskgraph_derived_operand_adapter": True,
            "host_taskgraph_operand_binding": True,
            "native_taskgraph_protocol": False,
            "native_plan_identity_binding": False,
            "qualification_task_count": 1,
            "contraction_plan_identity": identities["contraction_plan_hash"],
            "circuit_semantics_hash": identities["circuit_semantics_hash"],
            "tensor_network_hash": identities["tensor_network_hash"],
            "contraction_plan_hash": identities["contraction_plan_hash"],
            "contraction_path_structure_hash": identities["contraction_path_structure_hash"],
            "execution_bundle_artifact": "execution_bundle.json",
            "operand_binding_hash": operand_binding_hash,
            "task_id": task.id,
            "source_task_count": 1,
            "source_task_completion_count": 1,
            "input_tensor_ids": list(task.input_tensor_ids),
            "output_tensor_id": task.output_tensor_id,
            "input_shapes": [list(shape) for shape in task.input_shapes],
            "output_shape": list(task.output_shape),
            "input_file_sha256": input_sha256,
            "scientific_input_file_bytes": 512,
            "input_dtype": "int8",
            "native_table_dtype": "int32",
            "input_to_table_expansion_factor": 4,
            "input_transport_dtype": "int32",
            "input_hash": row["input_hash"],
            "output_hash": row.get("output_hash"),
            "reference_int64": row["reference_int64"],
            "result_int64": row["result_int64"],
            "exact_integer_match": True,
            "validation_status": "passed",
            "hardware_execution": True,
            "hardware_functionality_evidence": True,
            "target_requested": "physical_hardware",
            "target_observed": "physical_hardware",
            "allocation_profile": payload["allocation_profile"],
            "requested_dpu_count": DPUS,
            "allocated_dpu_count": DPUS,
            "hardware_allocation_verified": True,
            "native_kernel_executed": True,
            "release_confirmed": True,
            "tasklets_per_dpu": TASKLETS,
            "cpu_fallback_used": False,
            "simulator_kernel_executed": False,
            "pid_comm_invoked": False,
            "atim_integrated": False,
            "hardware_speedup_applicable": False,
            "application_visible_h2d_bytes": payload["logical_payload_h2d_bytes_per_iteration"],
            "application_visible_d2h_bytes": payload["logical_payload_d2h_bytes_per_iteration"],
            "application_visible_transfer_bytes": payload["logical_payload_transfer_bytes_per_iteration"],
            "per_iteration_operator_time_s": row["total_time_s"],
            "native_session_setup_time_s": payload.get("session_setup_time_s"),
            "native_session_total_time_s": payload.get("total_route_time_s"),
            "host_binary_hash": payload.get("host_binary_hash"),
            "initialization_binary_hash": payload.get("initialization_binary_hash"),
            "map_binary_hash": payload.get("map_binary_hash"),
            "genred_binary_hash": payload.get("genred_binary_hash"),
            "genred_reduce_shared_object_hash": payload.get("genred_reduce_shared_object_hash"),
            "timing_scope": "physical SimplePIM one-task bring-up; API orchestration included",
            "timing_is_bringup_only": True,
            "repeat_id": row["repeat_id"],
            "warmup": False,
            "native_response_artifact": source,
            "thesis_source_commit": thesis_source_commit or payload.get("thesis_source_commit"),
            "simplepim_source_commit": payload.get("simplepim_source_commit") or payload.get("source_commit"),
            "hostname": payload.get("hostname"),
        })
    return rows


def _artifact_inputs(run_dir: Path, workload: Any, graph: TaskGraph) -> tuple[Path, str, str, int, JsonDict]:
    operands, input_sha256, input_hash, reference = _write_operands(run_dir, workload)
    identities = _identity_payload(graph, case_id="m4_3_rank1_taskgraph_256")
    binding = _operand_binding(graph, graph.tasks[0], input_sha256=input_sha256)
    execution_bundle = {**identities["execution_bundle"], "operand_binding": binding}
    write_json(run_dir / "execution_bundle.json", execution_bundle)
    write_json(run_dir / "input_manifest.json", {"schema_version": "m4_3_input_v1", "operands_file": operands.name, "input_file_sha256": input_sha256, "input_hash": input_hash, "reference_int64": reference, "byte_layout": "left int8[256] followed by right int8[256]", "input_dtype": "int8", "native_table_dtype": "int32", "input_transport_dtype": "int32", "input_to_table_expansion_factor": 4, "taskgraph_derived_operand_adapter": True, "host_taskgraph_operand_binding": True, "native_taskgraph_protocol": False, "native_plan_identity_binding": False, "operand_binding": binding, **{key: identities[key] for key in ("circuit_semantics_hash", "tensor_network_hash", "contraction_plan_hash")}})
    identities = {**identities, "execution_bundle": execution_bundle, "operand_binding": binding}
    return operands, input_sha256, input_hash, reference, identities


def prepare(root_dir: Path, *, suite_path: Path, build: bool = False, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    suite = load_suite(suite_path)
    workload = _load_workload()
    graph = _graph(workload)
    plan_dir = root_dir / "build" / f"{SUITE_ID}_plan" / time.strftime("%Y-%m-%d_%H-%M-%S")
    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "config").mkdir()
    shutil.copy2(suite["path"], plan_dir / "config" / "resolved_suite.yml")
    write_json(plan_dir / "config" / "hardware_profile.json", suite["profile"])
    write_json(plan_dir / "environment.json", _m4_3_environment(root_dir))
    _, input_sha256, _, reference, identities = _artifact_inputs(plan_dir, workload, graph)
    result: dict[str, Any] = {"status": "prepared", "plan_dir": str(plan_dir), "dpu_allocation_attempted": False, "dpu_launch_attempted": False, "input_file_sha256": input_sha256, "reference_int64": reference, **{key: identities[key] for key in ("circuit_semantics_hash", "tensor_network_hash", "contraction_plan_hash")}}
    if build:
        env = dict(os.environ if environment is None else environment)
        native = root_dir / NATIVE_REL
        result["native_build"] = _run(["make", "clean", "parser"], cwd=native, env=env, timeout_s=TIMEOUT_S)
        if result["native_build"]["returncode"] != 0 or result["native_build"]["timed_out"]:
            result["status"] = "failed"
    summary_path = plan_dir / f"{SUITE_ID}_plan.json"
    write_json(summary_path, {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "route_id": ROUTE_ID, "backend_id": BACKEND_ID, "profile": suite["profile"], "claim_boundary": CLAIM_BOUNDARY, **result})
    result["artifact"] = str(summary_path)
    return result


def execute(root_dir: Path, *, suite_path: Path, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if environment is None else environment)
    if env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    selected = [key for key in SIMULATOR_ENV_KEYS if key in env]
    if selected:
        raise ValueError("hardware_profile_violation: simulator selector keys are forbidden: " + ", ".join(selected))
    suite = load_suite(suite_path)
    workload = _load_workload()
    graph = _graph(workload)
    run_dir = create_run_dir(root_dir, SUITE_ID, artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_hw_m4_3")
    shutil.copy2(suite["path"], run_dir / "config" / "resolved_suite.yml")
    write_json(run_dir / "config" / "hardware_profile.json", suite["profile"])
    environment_metadata = _m4_3_environment(root_dir)
    write_json(run_dir / "environment.json", environment_metadata)
    operands, input_sha256, input_hash, reference, identities = _artifact_inputs(run_dir, workload, graph)
    _require_operand_binding(identities, graph, input_sha256=input_sha256)
    manifest = write_run_manifest(run_dir, run_kind=SCHEMA_VERSION, suite_id=SUITE_ID, suite_path=str(suite["path"]), artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label="upmem_hw_m4_3", route_id=ROUTE_ID, backend_id=BACKEND_ID, execution_scope="physical_two_dpu_one_task_taskgraph_fixture", evidence_type="functionality_only", summary=f"{SUITE_ID}_summary.json", command=shlex.join(["UPMEM_ALLOW_PHYSICAL_HARDWARE=1", "make", "upmem-hw-m4-3"]), root_dir=root_dir)
    sdk_available = bool(environment_metadata["upmem_sdk"].get("available"))
    manifest.update({"hardware_available": False, "upmem_sdk_available": sdk_available, "simplepim_source_commit": environment_metadata["simplepim_source_commit"], "thesis_source_commit": environment_metadata.get("thesis_source_commit"), "taskgraph_derived_operand_adapter": True, "host_taskgraph_operand_binding": True, "native_taskgraph_protocol": False, "native_plan_identity_binding": False})
    write_json(run_dir / "run_manifest.json", manifest)
    native = root_dir / NATIVE_REL
    response = native / "build" / "simplepim_rank1_dot_m4_2" / "execute_response.json"
    response.unlink(missing_ok=True)
    binary = native / "build" / "simplepim_rank1_dot_m4_2" / "staged" / "benchmarks" / "rank1_dot" / "bin" / "rank1_dot_host"
    stage = native / "build" / "simplepim_rank1_dot_m4_2" / "staged" / "simplepim_stage_manifest.json"
    build_result = _run(["make", "clean", "build"], cwd=native, env=env, timeout_s=TIMEOUT_S)
    command_result = build_result
    if build_result["returncode"] == 0 and not build_result["timed_out"]:
        stage_bench = native / "build" / "simplepim_rank1_dot_m4_2" / "staged" / "benchmarks" / "rank1_dot"
        command_result = _run(["timeout", "--signal=TERM", "--kill-after=5s", f"{int(TIMEOUT_S)}s", str(binary), "--mode", "execute", "--response", str(response), "--stage-manifest", str(stage), "--operands-file", str(operands)], cwd=stage_bench, env=env, timeout_s=TIMEOUT_S + 10)
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "suite_id": SUITE_ID, "route_id": ROUTE_ID, "backend_id": BACKEND_ID, "status": "failed", "claim_boundary": CLAIM_BOUNDARY, "command": command_result, "native_interface": "M4.2 host must accept --operands-file and report operand_input_hash", "task_graph_integrated": False, "taskgraph_derived_operand_adapter": True, "host_taskgraph_operand_binding": True, "native_taskgraph_protocol": False, "native_plan_identity_binding": False, "operand_binding_hash": identities["operand_binding"]["binding_hash"], "execution_bundle_artifact": "execution_bundle.json", "source_task_count": 1, "source_task_completion_count": 0, "input_dtype": "int8", "native_table_dtype": "int32", "input_transport_dtype": "int32", "input_to_table_expansion_factor": 4, "cpu_fallback_used": False, "simulator_kernel_executed": False, "hardware_available": False}
    if response.exists():
        shutil.copy2(response, run_dir / "native_execute_response.json")
    try:
        if command_result["returncode"] != 0 or command_result["timed_out"]:
            raise ValueError("native_host_failed: no admission after nonzero or timed-out process")
        payload = _native_response(native)
        _require_response(payload, identities=identities, input_sha256=input_sha256, input_hash=input_hash, reference=reference)
        records = _records(payload, identities, graph.tasks[0], "native_execute_response.json", input_sha256=input_sha256, operand_binding_hash=identities["operand_binding"]["binding_hash"], thesis_source_commit=environment_metadata.get("thesis_source_commit"))
        write_json(run_dir / "native_response_summary.json", payload)
        write_normalized_records(run_dir, records)
        manifest["hardware_available"] = True
        write_json(run_dir / "run_manifest.json", manifest)
        summary.update({"status": "completed", "validation_status": "passed", "claim_verdict": "taskgraph_derived_operand_adapter_functionality_evidence", "row_count": len(records), "warmup_count": WARMUPS, "repeat_count": REPEATS, "source_task_count": 1, "source_task_completion_count": 1, "input_file_sha256": input_sha256, "operand_binding_hash": identities["operand_binding"]["binding_hash"], "execution_bundle_artifact": "execution_bundle.json", "reference_int64": reference, "taskgraph_derived_operand_adapter": True, "host_taskgraph_operand_binding": True, "native_taskgraph_protocol": False, "native_plan_identity_binding": False, "hardware_available": True, "thesis_source_commit": environment_metadata.get("thesis_source_commit"), "simplepim_source_commit": environment_metadata.get("simplepim_source_commit"), **{key: identities[key] for key in ("circuit_semantics_hash", "tensor_network_hash", "contraction_plan_hash")}})
    except ValueError as exc:
        summary.update({"failure_stage": str(exc).split(":", 1)[0], "failure_reason": str(exc)})
    summary_path = run_dir / f"{SUITE_ID}_summary.json"
    write_json(summary_path, summary)
    return {"run_dir": str(run_dir), "artifact": str(summary_path), "status": summary["status"], "row_count": summary.get("row_count", 0)}


__all__ = ["load_suite", "prepare", "execute", "_require_response", "_records", "PROFILE_ID", "BACKEND_ID", "ROUTE_ID"]
