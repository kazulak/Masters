"""Public M5 distributed-plan-v3 hardware study.

This module deliberately owns orchestration only.  Circuit lowering and task
input replay belong to ``m5_task_selection``; native plan construction and
execution belong to the v3 target.  Keeping those boundaries here makes the
study useful with a fake target in CI and prevents accidental CPU/simulator
substitution on a physical run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
from pathlib import Path
import shutil
import socket
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import yaml

from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, create_run_dir
from quantum_bench.circuits import load_circuit
from quantum_bench.core.jsonio import write_json
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.execution_plan_v3 import DEFAULT_TIMEOUT_S
from quantum_bench.targets.upmem.hardware_session import hardware_environment_metadata
from quantum_bench.tn import build_tensor_network, plan_task_graph_with_config
from quantum_bench.tn.execution_bundle import canonical_hash
from quantum_bench.core.records import to_jsonable
from quantum_bench.formats import conversion_error_metrics


SUITE_ID = "upmem_hardware_distributed_m5"
ROUTE_LABEL = "upmem_hw_m5"
ROUTE_ID = "upmem_tn_hardware_distributed_m5"
BACKEND_ID = "upmem_sdk_hardware_distributed_m5"
# These values mirror V3_ROUTE_ID/V3_BACKEND_ID/V3_PROFILE_VERSION in the
# native Makefile, which are passed to host_v3.c as RESIDENT_* macros.
NATIVE_ROUTE_ID = "upmem_tn_hardware_taskgraph_resident"
NATIVE_BACKEND_ID = "upmem_sdk_hardware_taskgraph_resident"
NATIVE_HARDWARE_PROFILE_VERSION = "hardware_taskgraph_distributed_single_contraction_m5_v3"
SCHEMA_VERSION = "upmem_hardware_distributed_m5_v1"
NATIVE_PLAN_KIND = "distributed_plan_v3"
NATIVE_RESPONSE_SCHEMA = "upmem_execution_plan_native_v3"
DEFAULT_DPU_COUNTS = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_TASKLETS = 8
MIN_TASKLETS = 1
MAX_TASKLETS = 24
WARMUPS = 2
REPEATS = 7
TOTAL_REPETITIONS = WARMUPS + REPEATS
DEFAULT_CAPACITY = 64
MAX_TIMEOUT_S = 1800.0
SUBPROCESS_TIMEOUT_GRACE_S = 30.0
QUANTIZATION_MODES = ("none", "per_task_resident_requantize")
SUPPORTED_QUANTIZATION_MODES = (*QUANTIZATION_MODES, "host_packed_int8")
PARTITION_STRATEGIES = ("output", "contracted")
NATIVE_OUTPUT_LIMIT_BYTES = 64 * 1024
ERROR_TEXT_LIMIT_BYTES = 4 * 1024
SDK_PROBE_TIMEOUT_S = 2.0
SDK_VERSION_OUTPUT_LIMIT_BYTES = 512
CORE_UPMEM_SDK_TOOLS = ("dpu-upmem-dpurte-clang", "dpu-pkg-config")


class NativeExecutionError(RuntimeError):
    """A native invocation failed but still produced a structured response."""

    def __init__(
        self,
        message: str,
        *,
        response: Mapping[str, Any] | None = None,
        returncode: int | None = None,
    ) -> None:
        self.response = dict(response) if isinstance(response, Mapping) else None
        self.returncode = returncode
        failure_stage = self.response.get("failure_stage") if self.response else None
        self.failure_stage = failure_stage if isinstance(failure_stage, str) and failure_stage else None
        super().__init__(message)


class M5NativeTarget(Protocol):
    """Native seam used by the physical runner and hardware-free tests."""

    def set_environment(self, environment: Mapping[str, str]) -> None: ...

    def build(self, build_dir: Path, *, tasklets: int) -> Mapping[str, Any]: ...

    def prepare_request(
        self,
        *,
        case: Mapping[str, Any],
        materialized: Mapping[str, Any],
        dpu_count: int,
        tasklets: int,
        quantization_mode: str,
        partition_strategy: str,
        build: Mapping[str, Any],
        root: Path,
    ) -> Mapping[str, Any]: ...

    def validate(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]: ...

    def execute(self, request: Mapping[str, Any], *, timeout_s: float) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class M5StudyConfig:
    suite_id: str
    dpu_counts: tuple[int, ...]
    tasklets: int
    warmups: int
    repeats: int
    timeout_s: float
    capacity: int
    quantization_modes: tuple[str, ...]
    partition_strategies: tuple[str, ...]
    cases: tuple[Mapping[str, Any], ...]
    suite_path: Path


class _DefaultNativeTarget:
    """Exact adapter for the additive execution-plan v3 target."""

    def __init__(self) -> None:
        self._environment = dict(os.environ)
        self._module: Any | None = None

    def set_environment(self, environment: Mapping[str, str]) -> None:
        self._environment = dict(environment)

    def _load(self) -> Any:
        if self._module is None:
            self._module = importlib.import_module(
                "quantum_bench.targets.upmem.execution_plan_v3"
            )
        return self._module

    def build(self, build_dir: Path, *, tasklets: int) -> Mapping[str, Any]:
        result = self._load().build(
            build_dir, tasklets=tasklets, environment=self._environment
        )
        return {
            **dict(result),
            "selected_rank_path": self._environment.get("UPMEM_HW_RANK_PATH"),
        }

    def prepare_request(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._load().prepare_request(**kwargs)

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
        host = Path(str(request.get("host_binary") or request.get("runner")))
        response_path = Path(str(request["response_path"]))
        policy_reference = _reference_binding(request, "policy_reference")
        integer_reference = _integer_reference_binding(request)
        response_path.unlink(missing_ok=True)
        command = [
            str(host),
            mode,
            "--resident-package",
            str(request["resident_manifest"]),
            "--distributed-plan-v3",
            str(request["distributed_plan"]),
            "--policy-reference",
            policy_reference["path"],
            "--policy-reference-sha256",
            policy_reference["sha256"],
            "--policy-tolerance",
            str(policy_reference["max_abs_tolerance"]),
            "--response",
            str(response_path),
            "--warmups",
            str(WARMUPS),
            "--repetitions",
            str(REPEATS),
            "--timeout-s",
            str(max(1, math.ceil(timeout_s))),
        ]
        if integer_reference is not None:
            command.extend(
                [
                    "--integer-reference",
                    integer_reference["path"],
                    "--integer-reference-sha256",
                    integer_reference["sha256"],
                ]
            )
        try:
            completed = subprocess.run(
                command,
                cwd=host.parent,
                env=dict(self._environment),
                capture_output=True,
                text=True,
                timeout=timeout_s + SUBPROCESS_TIMEOUT_GRACE_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output_artifacts = _write_native_output_artifacts(
                response_path, getattr(exc, "stdout", None), getattr(exc, "stderr", None)
            )
            payload = _read_native_response(response_path) or {
                "schema_version": NATIVE_RESPONSE_SCHEMA,
                "status": "failed",
            }
            payload["native_output_artifacts"] = output_artifacts
            failure_stage = payload.get("failure_stage")
            if not isinstance(failure_stage, str) or not failure_stage:
                failure_stage = "outer_process_timeout"
            payload["failure_stage"] = failure_stage
            payload.setdefault("error", "native subprocess exceeded the outer timeout")
            payload["native_timeout_s"] = timeout_s
            payload["outer_timeout_s"] = timeout_s + SUBPROCESS_TIMEOUT_GRACE_S
            _write_native_response(response_path, payload)
            raise NativeExecutionError(
                f"{failure_stage}: native subprocess exceeded {timeout_s + SUBPROCESS_TIMEOUT_GRACE_S}s",
                response=payload,
            ) from exc
        output_artifacts = _write_native_output_artifacts(
            response_path, getattr(completed, "stdout", None), getattr(completed, "stderr", None)
        )
        if not response_path.is_file():
            payload = {
                "schema_version": NATIVE_RESPONSE_SCHEMA,
                "status": "failed",
                "failure_stage": "output_manifest_failed",
                "error": "native_response_missing: v3 response was not written",
                "native_output_artifacts": output_artifacts,
            }
            _write_native_response(response_path, payload)
            raise NativeExecutionError(payload["error"], response=payload, returncode=completed.returncode)
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload = {
                "schema_version": NATIVE_RESPONSE_SCHEMA,
                "status": "failed",
                "failure_stage": "output_manifest_failed",
                "error": f"native_response_invalid: {exc}",
                "native_output_artifacts": output_artifacts,
            }
            _write_native_response(response_path, payload)
            raise NativeExecutionError(payload["error"], response=payload, returncode=completed.returncode) from exc
        if not isinstance(payload, dict):
            payload = {
                "schema_version": NATIVE_RESPONSE_SCHEMA,
                "status": "failed",
                "failure_stage": "output_manifest_failed",
                "error": "native_response_invalid: v3 response is not an object",
                "native_output_artifacts": output_artifacts,
            }
            _write_native_response(response_path, payload)
            raise NativeExecutionError(payload["error"], response=payload, returncode=completed.returncode)
        payload["native_output_artifacts"] = output_artifacts
        _write_native_response(response_path, payload)
        if completed.returncode != 0 or payload.get("status") != expected:
            raise NativeExecutionError(
                str(payload.get("error") or "native v3 request failed"),
                response=payload,
                returncode=completed.returncode,
            )
        policy_validation = payload.get("policy_reference_validation")
        if not isinstance(policy_validation, Mapping):
            raise NativeExecutionError(
                "policy_reference_validation_failed: native response lacks policy-reference evidence",
                response=payload,
                returncode=completed.returncode,
            )
        if expected == "validated":
            if (
                policy_validation.get("status") != "not_run"
                or policy_validation.get("passed") is not False
            ):
                raise NativeExecutionError(
                    "policy_reference_validation_failed: validate-only response must not claim reference execution",
                    response=payload,
                    returncode=completed.returncode,
                )
        else:
            if (
                policy_validation.get("status") != "passed"
                or policy_validation.get("passed") is not True
                or policy_validation.get("reference_sha256")
                != policy_reference["sha256"]
            ):
                raise NativeExecutionError(
                    "policy_reference_validation_failed: native response did not pass the prepared policy reference",
                    response=payload,
                    returncode=completed.returncode,
                )
            payload["full_precision_accuracy"] = _full_precision_accuracy(request)
            if integer_reference is not None:
                integer_validation = payload.get("exact_integer_validation")
                if (
                    not isinstance(integer_validation, Mapping)
                    or integer_validation.get("required") is not True
                    or integer_validation.get("passed") is not True
                    or integer_validation.get("exact_match") is not True
                    or integer_validation.get("mismatch_count") != 0
                    or integer_validation.get("reference_sha256")
                    != integer_reference["sha256"]
                ):
                    raise NativeExecutionError(
                        "integer_reference_validation_failed: native response did not prove exact int32 output",
                        response=payload,
                        returncode=completed.returncode,
                    )
        payload.update(hardware_environment_metadata(self._environment))
        return payload


def load_m5_suite(path: Path, *, dpu_counts: Sequence[int] | None = None, tasklets: int | None = None) -> M5StudyConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("suite_id") != SUITE_ID:
        raise ValueError("suite_invalid: expected the committed upmem_hardware_distributed_m5 suite")
    defaults = raw.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("suite_invalid: defaults must be a mapping")
    warmups = int(defaults.get("warmups", WARMUPS))
    repeats = int(defaults.get("repeats", REPEATS))
    if (warmups, repeats) != (WARMUPS, REPEATS):
        raise ValueError("suite_invalid: M5 requires warmups=2 and repeats=7")
    timeout_s = _validate_timeout(defaults.get("timeout_s", DEFAULT_TIMEOUT_S))
    quantization_modes = _parse_suite_options(
        defaults.get("quantization_modes", QUANTIZATION_MODES),
        "quantization_modes",
        SUPPORTED_QUANTIZATION_MODES,
    )
    partition_strategies = _parse_suite_options(
        defaults.get("partition_strategies", PARTITION_STRATEGIES),
        "partition_strategies",
        PARTITION_STRATEGIES,
    )
    capacity = int((raw.get("metadata") or {}).get("hardware_profile", {}).get("max_dpu_count", DEFAULT_CAPACITY))
    if capacity < 1:
        raise ValueError("suite_invalid: max_dpu_count must be positive")
    selected_counts = _parse_positive_ints(dpu_counts) if dpu_counts is not None else tuple(
        int(value) for value in defaults.get("dpu_counts", DEFAULT_DPU_COUNTS)
    )
    if not selected_counts:
        raise ValueError("dpu_counts_invalid: at least one DPU count is required")
    configured_tasklets = defaults.get("tasklets", DEFAULT_TASKLETS)
    selected_tasklets = configured_tasklets if tasklets is None else tasklets
    _validate_tasklets(selected_tasklets)
    cases = raw.get("workloads")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite_invalid: workloads must be non-empty")
    normalized_cases: list[Mapping[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or not case.get("id"):
            raise ValueError("suite_invalid: every workload needs an id")
        normalized_case = {**case, "case_id": str(case["id"]), "workload_id": str(case["id"])}
        _strategies_for(normalized_case, partition_strategies)
        normalized_cases.append(normalized_case)
    return M5StudyConfig(
        suite_id=str(raw["suite_id"]),
        dpu_counts=tuple(selected_counts),
        tasklets=selected_tasklets,
        warmups=warmups,
        repeats=repeats,
        timeout_s=timeout_s,
        capacity=capacity,
        quantization_modes=quantization_modes,
        partition_strategies=partition_strategies,
        cases=tuple(normalized_cases),
        suite_path=path,
    )


def prepare(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    dpu_counts: Sequence[int] | None = None,
    tasklets: int | None = None,
    native_target: M5NativeTarget | None = None,
    task_selector: Any | None = None,
) -> dict[str, Any]:
    if not build:
        raise ValueError("prepare_requires_build: --prepare-only requires --build")
    config = load_m5_suite(suite_path, dpu_counts=dpu_counts, tasklets=tasklets)
    target = native_target or _DefaultNativeTarget()
    plan_dir = _unique_dir(root_dir / "build" / f"{SUITE_ID}_plan")
    target.set_environment(dict(os.environ))
    native_build = target.build(plan_dir / "native_build", tasklets=config.tasklets)
    plans, preparation_rows = _prepare_plans(
        root_dir, plan_dir, config, target, native_build, task_selector=task_selector
    )
    counts = _status_counts(preparation_rows)
    status = (
        "failed"
        if counts["failed_count"]
        else "prepared"
        if counts["prepared_count"]
        else "failed"
    )
    artifact = plan_dir / f"{SUITE_ID}_plan.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "suite_id": config.suite_id,
        "suite_path": str(suite_path),
        "dpu_counts": list(config.dpu_counts),
        "tasklets": config.tasklets,
        "warmups": config.warmups,
        "repeats": config.repeats,
        "timeout_s": config.timeout_s,
        "capacity": config.capacity,
        "quantization_modes": list(config.quantization_modes),
        "partition_strategies": list(config.partition_strategies),
        "native_plan_kind": NATIVE_PLAN_KIND,
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
        "plans": plans,
        "preparation_rows": preparation_rows,
        **counts,
    }
    write_json(artifact, payload)
    return {
        "plan_dir": str(plan_dir),
        "artifact": str(artifact),
        "status": status,
        "prepared_count": counts["prepared_count"],
        "unsupported_count": counts["unsupported_count"],
        "failed_count": counts["failed_count"],
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
    }


def execute(
    root_dir: Path,
    *,
    suite_path: Path,
    dpu_counts: Sequence[int] | None = None,
    tasklets: int | None = None,
    environment: Mapping[str, str] | None = None,
    native_target: M5NativeTarget | None = None,
    task_selector: Any | None = None,
) -> dict[str, Any]:
    env = dict(os.environ if environment is None else environment)
    _require_physical_environment(env)
    config = load_m5_suite(suite_path, dpu_counts=dpu_counts, tasklets=tasklets)
    using_default_native_target = native_target is None
    provider_kind = "default_native" if using_default_native_target else "injected_test_only"
    target = _DefaultNativeTarget() if using_default_native_target else native_target
    assert target is not None
    target.set_environment(env)
    run_dir = create_run_dir(root_dir, SUITE_ID, artifact_kind=EVIDENCE_ARTIFACT_KIND, route_label=ROUTE_LABEL)
    sdk_provenance = _upmem_sdk_provenance(env)
    environment_payload = capture_environment(root_dir)
    environment_payload.update({
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "m5_native_provider_kind": provider_kind,
        "upmem_sdk_provenance": sdk_provenance,
    })
    environment_payload["m5_native_environment"] = {
        "UPMEM_ALLOW_PHYSICAL_HARDWARE": env.get("UPMEM_ALLOW_PHYSICAL_HARDWARE"),
        "DPU_BACKEND": env.get("DPU_BACKEND"),
        "UPMEM_EXECUTION_MODE": env.get("UPMEM_EXECUTION_MODE"),
        "provider_kind": provider_kind,
        "requested_rank_path": sdk_provenance["requested_rank_path"],
        "effective_profile": sdk_provenance["effective_profile"],
        "upmem_sdk_tools": sdk_provenance["tools"],
        **hardware_environment_metadata(env),
    }
    write_json(run_dir / "environment.json", environment_payload)
    summary_name = f"{SUITE_ID}_summary.json"
    manifest = write_run_manifest(
        run_dir,
        run_kind=SCHEMA_VERSION,
        suite_id=config.suite_id,
        suite_path=str(suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label=ROUTE_LABEL,
        route_id=ROUTE_ID,
        backend_id=BACKEND_ID,
        execution_scope="execution_plan_v3_distributed_study",
        evidence_type="physical_hardware_functionality_and_repeat_timing",
        normalized_records="normalized_records.jsonl",
        summary=summary_name,
        upmem_execution_mode=NATIVE_PLAN_KIND,
        policies=config.partition_strategies,
        quantization_modes=config.quantization_modes,
        command=_public_execution_command(config, sdk_provenance["requested_rank_path"]),
        root_dir=root_dir,
    )
    manifest.update(hardware_environment_metadata(env))
    manifest.update({
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "requested_rank_path": sdk_provenance["requested_rank_path"],
        "effective_profile": sdk_provenance["effective_profile"],
        "upmem_sdk_tools": sdk_provenance["tools"],
    })
    manifest["timeout_s"] = config.timeout_s
    manifest["native_provider_kind"] = provider_kind
    manifest["claims"] = _false_claims()
    manifest["status"] = "running"
    write_json(run_dir / "run_manifest.json", manifest)

    native_build_dir = run_dir / "native_build"
    try:
        native_build = target.build(native_build_dir, tasklets=config.tasklets)
    except Exception as exc:
        failure_artifacts = _write_build_failure_artifact(native_build_dir, exc)
        failure_stage = _failure_stage(exc)
        failure_row = _failure_record(
            {
                "status": "failed",
                "case_id": "__native_build__",
                "workload_id": "__native_build__",
                "benchmark_role": "m5_native_build",
                "quantum_case": "not_applicable",
                "failure_stage": failure_stage,
                "reason": str(exc),
                "native_output_artifacts": failure_artifacts,
                "native_response": getattr(exc, "response", None),
                "native_provider_kind": provider_kind,
            },
            env,
            provider_kind=provider_kind,
        )
        write_normalized_records(run_dir, [failure_row])
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "suite_id": config.suite_id,
            "suite_path": str(suite_path),
            "route_id": ROUTE_ID,
            "backend_id": BACKEND_ID,
            "native_plan_kind": NATIVE_PLAN_KIND,
            "dpu_counts": list(config.dpu_counts),
            "tasklets": config.tasklets,
            "warmups": config.warmups,
            "repeats": config.repeats,
            "timeout_s": config.timeout_s,
            "quantization_modes": list(config.quantization_modes),
            "partition_strategies": list(config.partition_strategies),
            "row_count": 1,
            "preparation_row_count": 0,
            "unsupported_count": 0,
            "failed_count": 1,
            "completed_count": 0,
            "prepared_count": 0,
            "preparation_unsupported_count": 0,
            "preparation_failed_count": 1,
            "failure_stage": failure_stage,
            "failure_artifacts": failure_artifacts,
            "claims": _false_claims(),
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
            "normalized_records": "normalized_records.jsonl",
        }
        summary_path = run_dir / summary_name
        write_json(summary_path, summary)
        manifest["status"] = "failed"
        manifest["failure_stage"] = failure_stage
        manifest["failure_artifacts"] = failure_artifacts
        manifest["hardware_available"] = "not_verified"
        manifest["upmem_sdk_available"] = "not_verified_by_execution"
        write_json(run_dir / "run_manifest.json", manifest)
        return {
            "run_dir": str(run_dir),
            "artifact": str(summary_path),
            "status": "failed",
            "row_count": 1,
            "dpu_allocation_attempted": False,
            "dpu_launch_attempted": False,
        }
    plans, preparation_rows = _prepare_plans(
        root_dir, run_dir, config, target, native_build, task_selector=task_selector
    )
    records: list[dict[str, Any]] = []
    for row in preparation_rows:
        if row["status"] != "prepared":
            records.append(_failure_record(row, env, provider_kind=provider_kind))
            continue
        plan = plans[row["plan_key"]]
        request = plan["request"]
        response: Mapping[str, Any] | None = None
        try:
            response = target.execute(request, timeout_s=float(request["timeout_s"]))
            _validate_execute_response(response, request, config)
            repetitions = _measured_repetitions(response, config)
            for repeat_id, timing in repetitions:
                records.append(
                    _normalized_record(
                        plan, response, timing, repeat_id, env, provider_kind=provider_kind
                    )
                )
        except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            native_response = (
                dict(response) if isinstance(response, Mapping)
                else getattr(exc, "response", None)
            )
            records.append(_failure_record({
                **row,
                "status": "failed",
                "reason": str(exc),
                "failure_stage": _failure_stage(exc),
                "native_response": native_response,
                "native_provider_kind": provider_kind,
            }, env, provider_kind=provider_kind))
    write_normalized_records(run_dir, records)
    preparation_counts = _status_counts(preparation_rows)
    failed_count = sum(row["status"] == "failed" for row in records)
    completed_count = sum(row["status"] == "completed" for row in records)
    if failed_count:
        status = "failed"
    elif completed_count:
        status = "completed"
    else:
        status = "unsupported"
    allocation_attempted, launch_attempted = _native_attempt_flags(records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "suite_id": config.suite_id,
        "suite_path": str(suite_path),
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "native_plan_kind": NATIVE_PLAN_KIND,
        "dpu_counts": list(config.dpu_counts),
        "tasklets": config.tasklets,
        "warmups": config.warmups,
        "repeats": config.repeats,
        "timeout_s": config.timeout_s,
        "quantization_modes": list(config.quantization_modes),
        "partition_strategies": list(config.partition_strategies),
        "row_count": len(records),
        "preparation_row_count": len(preparation_rows),
        "unsupported_count": sum(row["status"] == "unsupported" for row in records),
        "failed_count": failed_count,
        "completed_count": completed_count,
        "prepared_count": preparation_counts["prepared_count"],
        "preparation_unsupported_count": preparation_counts["unsupported_count"],
        "preparation_failed_count": preparation_counts["failed_count"],
        "claims": _false_claims(),
        "dpu_allocation_attempted": allocation_attempted,
        "dpu_launch_attempted": launch_attempted,
        "normalized_records": "normalized_records.jsonl",
    }
    summary_path = run_dir / summary_name
    write_json(summary_path, summary)
    physical_completed = _has_completed_physical_rows(records)
    manifest["status"] = status
    manifest["hardware_available"] = (
        "verified_by_execution"
        if physical_completed and using_default_native_target
        else "not_verified"
    )
    manifest["upmem_sdk_available"] = (
        "verified_by_execution"
        if physical_completed and using_default_native_target
        else "not_verified_by_execution"
    )
    write_json(run_dir / "run_manifest.json", manifest)
    return {
        "run_dir": str(run_dir),
        "artifact": str(summary_path),
        "status": status,
        "row_count": len(records),
        "dpu_allocation_attempted": allocation_attempted,
        "dpu_launch_attempted": launch_attempted,
    }


def _public_execution_command(config: M5StudyConfig, rank_path: str) -> str:
    if "host_packed_int8" in config.quantization_modes:
        target = (
            "upmem-hw-m5-4-smoke"
            if config.dpu_counts == (1, 2, 4, 8)
            else "upmem-hw-m5-4"
        )
    else:
        target = "upmem-hw-m5"
    return (
        f"UPMEM_HW_RANK_PATH={rank_path} "
        f"UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make {target}"
    )


def _prepare_plans(
    root_dir: Path,
    output_root: Path,
    config: M5StudyConfig,
    target: M5NativeTarget,
    native_build: Mapping[str, Any],
    *,
    task_selector: Any | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    plans: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for case in config.cases:
        selection: Mapping[str, Any] | None = None
        try:
            selection = _select_and_materialize(case, root_dir, task_selector)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            for dpu_count in config.dpu_counts:
                for mode in config.quantization_modes:
                    for strategy in _strategies_for(case, config.partition_strategies):
                        rows.append(_base_row(case, dpu_count, config.tasklets, mode, strategy) | {
                            "status": "failed", "failure_stage": "task_selection", "reason": str(exc)
                        })
            continue
        for dpu_count in config.dpu_counts:
            for mode in config.quantization_modes:
                for strategy in _strategies_for(case, config.partition_strategies):
                    row = _base_row(case, dpu_count, config.tasklets, mode, strategy) | {
                        "selection": _selection_evidence(selection),
                        **_identity_hashes(selection),
                    }
                    key = _plan_key(row)
                    if dpu_count > config.capacity:
                        rows.append(row | {
                            "status": "unsupported", "failure_stage": "capacity",
                            "reason": f"requested_dpu_count={dpu_count} exceeds capacity={config.capacity}",
                            "plan_key": key,
                        })
                        continue
                    try:
                        case_root = output_root / "plans" / _safe_name(key)
                        request = target.prepare_request(
                            case=case,
                            materialized=selection,
                            dpu_count=dpu_count,
                            tasklets=config.tasklets,
                            quantization_mode=mode,
                            partition_strategy=strategy,
                            build=native_build,
                            root=case_root,
                        )
                        if not isinstance(request, Mapping):
                            raise ValueError("native_v3_plan_invalid: request builder returned a non-mapping")
                        request = {**request, "timeout_s": config.timeout_s}
                        validation = target.validate(request, timeout_s=float(request["timeout_s"]))
                        _validate_plan_response(validation, dpu_count, config.tasklets)
                        plan = {
                            **row,
                            "status": "prepared",
                            "plan_key": key,
                            "request": dict(request),
                            "native_validation": dict(validation),
                            "selection": _selection_evidence(selection),
                            "hashes": _identity_hashes(selection, request),
                        }
                        plans[key] = plan
                        rows.append(plan)
                    except (OSError, RuntimeError, TimeoutError, ValueError, TypeError) as exc:
                        rows.append(_preparation_failure_row(row, key, exc))
    return plans, rows


def _preparation_failure_row(
    row: Mapping[str, Any], plan_key: str, exc: BaseException
) -> dict[str, Any]:
    reason = str(exc)
    stage = _failure_stage(exc)
    classification = _unsupported_preparation_stage(stage, reason)
    return {
        **row,
        "status": "unsupported" if classification is not None else "failed",
        "failure_stage": classification or stage,
        "reason": reason,
        "plan_key": plan_key,
    }


def _unsupported_preparation_stage(stage: str, reason: str) -> str | None:
    """Classify non-runnable resource and partition cases without masking defects."""

    detail = f"{stage}: {reason}".lower()
    if stage == "partition_unsupported" or "partition" in detail:
        return "partition_incompatible"
    if any(token in detail for token in ("capacity", "resource", "mram", "dpu count", "tasklet")):
        return "resource_limited"
    if stage in {"hardware_profile_violation", "unsupported", "not_supported"}:
        return "hardware_profile_violation"
    return None


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "prepared_count": sum(row.get("status") == "prepared" for row in rows),
        "unsupported_count": sum(row.get("status") == "unsupported" for row in rows),
        "failed_count": sum(row.get("status") == "failed" for row in rows),
    }


def _select_and_materialize(case: Mapping[str, Any], root_dir: Path, helper: Any | None) -> Mapping[str, Any]:
    if helper is None:
        try:
            helper = importlib.import_module("quantum_bench.targets.upmem.m5_task_selection")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "task_selection_unavailable: quantum_bench.targets.upmem.m5_task_selection is required"
            ) from exc
    if case.get("non_quantum") is True or case.get("quantum_case") == "non_quantum":
        # Diagnostics intentionally have no circuit identity, but retain the
        # same materialized-input boundary in their request metadata.
        return {
            "status": "selected",
            "case_id": case["case_id"],
            "non_quantum": True,
            "matrix_shapes": case.get("matrix_shapes"),
            **_diagnostic_hashes(case),
        }
    function = helper if callable(helper) else getattr(helper, "select_and_materialize_task", None)
    if function is None:
        function = getattr(helper, "select_and_materialize", None)
    if function is None:
        function = getattr(helper, "select_highest_work_supported_task", None)
    if function is None:
        function = getattr(helper, "select_and_materialize_highest_work_task", None)
    if function is None:
        raise RuntimeError("task_selection_unavailable: shared helper has no selection entry point")
    try:
        result = _invoke_flexible(function, case=case, root_dir=root_dir, root=root_dir)
    except TypeError:
        circuit = load_circuit(dict(case), root_dir)
        network = build_tensor_network(circuit)
        graph = plan_task_graph_with_config(network, dict(case.get("planner") or {}))
        result = _invoke_flexible(function, graph, network)
    if isinstance(result, Mapping):
        status = str(result.get("status", "materialized"))
        if status not in {"materialized", "initial_inputs_available", "selected", "prepared"}:
            raise ValueError(f"task_selection_{status}: {result.get('reason') or result.get('error') or status}")
        return _retain_real_selection_context(case, dict(result), root_dir)
    if hasattr(result, "to_json_dict"):
        return _retain_real_selection_context(
            case, {**result.to_json_dict(), "selection_object": result}, root_dir
        )
    raise TypeError("task_selection_invalid: shared helper returned an unsupported result")


def _retain_real_selection_context(
    case: Mapping[str, Any], selection: dict[str, Any], root_dir: Path
) -> Mapping[str, Any]:
    """Keep the selected graph/task and arrays live through native preparation."""

    if case.get("non_quantum") is True or case.get("quantum_case") == "non_quantum":
        return selection
    try:
        circuit = load_circuit(dict(case), root_dir)
        network = build_tensor_network(circuit)
        graph = plan_task_graph_with_config(network, dict(case.get("planner") or {}))
        task_id = selection.get("task_id") or getattr(selection.get("selection_object"), "task_id", None)
        task = next((item for item in graph.tasks if item.id == task_id), None)
        selected_object = selection.get("selection_object")
        left = selection.get("left_operand")
        if left is None:
            left = getattr(selected_object, "left_operand", None)
        right = selection.get("right_operand")
        if right is None:
            right = getattr(selected_object, "right_operand", None)
        if task is not None and left is not None and right is not None:
            selection.update({
                "_source_graph": graph,
                "_source_network": network,
                "_selected_task": task,
                "_left_operand": left,
                "_right_operand": right,
                "task_hash": selection.get("task_hash") or canonical_hash(to_jsonable(task)),
            })
    except (OSError, RuntimeError, TypeError, ValueError):
        # Custom test selectors may intentionally provide only identity fields;
        # their fake target does not need the real lowering context.
        pass
    return selection


def _selection_evidence(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Drop live graph/array objects before writing the plan artifact."""

    excluded = {
        "selection_object", "_source_graph", "_source_network", "_selected_task",
        "_left_operand", "_right_operand", "graph", "network", "task",
        "left_operand", "right_operand",
    }
    return {key: value for key, value in selection.items() if key not in excluded}


def _validate_plan_response(response: Mapping[str, Any], dpu_count: int, tasklets: int) -> None:
    if response.get("status") != "validated":
        raise ValueError("native_v3_validation_failed: response is not validated")
    if response.get("schema_version") not in {NATIVE_RESPONSE_SCHEMA, "upmem_execution_plan_native_v3"}:
        raise ValueError("native_v3_validation_failed: response schema is not v3")
    if response.get("target_observed") not in {"not_allocated", "hardware_unallocated"}:
        raise ValueError("native_v3_validation_failed: validation allocated or selected a target")
    if response.get("requested_dpu_count") != dpu_count or response.get("allocated_dpu_count") != 0:
        raise ValueError("native_v3_validation_failed: allocation evidence is not zero")
    if response.get("tasklets_per_dpu", tasklets) != tasklets:
        raise ValueError("native_v3_validation_failed: tasklet count mismatch")


def _validate_execute_response(response: Mapping[str, Any], request: Mapping[str, Any], config: M5StudyConfig) -> None:
    partition_strategy = str(request["partition_strategy"])
    if partition_strategy == "output":
        expected_collective = "none"
        expected_reconstruction = "host_owned_range_assembly_v1"
        expected_checksum_policy = "output_slice_per_dpu"
    else:
        expected_collective = "host_mediated_sum_v1"
        expected_reconstruction = "host_float64_reduction_v1"
        expected_checksum_policy = "final_reference_validation_only"
    quantization_mode = request.get("quantization_mode")
    if quantization_mode == "none":
        expected_numeric_mode = "float32"
        expected_numeric_arithmetic = "float32"
        expected_requantization_scope = "none"
        expected_transport = "float32_mram"
        expected_packed_transfer = False
    elif quantization_mode == "per_task_resident_requantize":
        expected_numeric_mode = "per_task_resident_requantize"
        expected_numeric_arithmetic = "int8_requantized"
        expected_requantization_scope = "per_task_on_dpu"
        expected_transport = "float32_mram"
        expected_packed_transfer = False
    elif quantization_mode == "host_packed_int8":
        expected_numeric_mode = "host_packed_int8"
        expected_numeric_arithmetic = "int8_multiply_int32_accumulate"
        expected_requantization_scope = "host_initial_once_final_host_dequantize"
        expected_transport = "host_packed_int8_mram"
        expected_packed_transfer = True
        if partition_strategy == "contracted":
            expected_reconstruction = "host_int64_reduction_v1"
    else:
        raise ValueError(
            f"native_v3_request_invalid: unsupported quantization_mode={quantization_mode!r}"
        )
    expected = {
        "status": "completed",
        "target_requested": "hardware",
        "target_observed": "physical_hardware",
        "route_id": NATIVE_ROUTE_ID,
        "backend_id": NATIVE_BACKEND_ID,
        "hardware_profile_version": NATIVE_HARDWARE_PROFILE_VERSION,
        "requested_dpu_count": int(request["dpu_count"]),
        "allocated_dpu_count": int(request["dpu_count"]),
        "tasklets_per_dpu": config.tasklets,
        "observed_rank_count": 1,
        "requested_warmups": WARMUPS,
        "requested_repetitions": REPEATS,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "partition_strategy": partition_strategy,
        "numeric_mode": expected_numeric_mode,
        "numeric_arithmetic": expected_numeric_arithmetic,
        "numeric_transport": expected_transport,
        "requantization_scope": expected_requantization_scope,
        "packed_int8_transfer": expected_packed_transfer,
        "allocation_provider": "upmem_sdk_rank_profile_v1",
        "simplepim_role": "initialization_binary_and_management_state_only",
        "kernel_provider": "thesis_resident_generic_c_v3",
        "transfer_provider": "upmem_sdk_synchronous_v1",
        "collective_provider": expected_collective,
        "reconstruction_provider": expected_reconstruction,
        "output_checksum_policy": expected_checksum_policy,
        "dispatch_mode": "bulk_set_synchronous_v1",
        "kernel_launch_api_calls": TOTAL_REPETITIONS,
        "explicit_sync_api_calls": 0,
        "host_quantization": expected_packed_transfer,
        "dpu_intermediate_requantization": False,
    }
    for key, value in expected.items():
        if response.get(key) != value:
            raise ValueError(f"native_v3_response_invalid: {key}={response.get(key)!r}, expected {value!r}")
    if response.get("schema_version") != NATIVE_RESPONSE_SCHEMA:
        raise ValueError("native_v3_response_invalid: native response is not execution-plan-v3")
    if response.get("failure_stage") not in {None, ""} or response.get("error") not in {None, ""}:
        raise ValueError("native_v3_response_invalid: native response reports a failure")
    if response.get("fallback_used", False) is not False:
        raise ValueError("native_v3_response_invalid: fallback is forbidden")
    for key in (
        "hardware_allocation_verified",
        "native_kernel_executed",
        "hardware_kernel_executed",
        "hardware_release_verified",
    ):
        if response.get(key) is not True:
            raise ValueError(f"native_v3_response_invalid: {key} must be true")
    expected_hashes = {
        "package_file_sha256": "package_sha256",
        "distributed_plan_v3_sha256": "sidecar_sha256",
        "host_binary_sha256": "host_binary_sha256",
        "staged_dpu_binary_sha256": "dpu_binary_sha256",
        "initialization_binary_sha256": "initialization_binary_sha256",
    }
    for response_key, request_key in expected_hashes.items():
        expected_hash = request.get(request_key)
        if not isinstance(expected_hash, str) or response.get(response_key) != expected_hash:
            raise ValueError(
                "native_v3_response_invalid: "
                f"{response_key} does not match the prepared request"
            )
    allocation = response.get("allocation")
    if not isinstance(allocation, Mapping) or allocation.get("confirmed") is not True or allocation.get("release_confirmed") is not True:
        raise ValueError("native_v3_response_invalid: allocation/release confirmation is missing")
    policy_validation = response.get("policy_reference_validation")
    if not isinstance(policy_validation, Mapping) or policy_validation.get("passed") is not True:
        raise ValueError("native_v3_response_invalid: policy reference validation did not pass")
    policy_reference = _reference_binding(request, "policy_reference")
    if policy_validation.get("reference_sha256") != policy_reference["sha256"]:
        raise ValueError("native_v3_response_invalid: policy reference hash does not match the prepared request")
    if not _is_finite_nonnegative(policy_validation.get("max_abs_error")):
        raise ValueError("native_v3_response_invalid: policy reference error is not numeric")
    integer_reference = _integer_reference_binding(request)
    integer_validation = response.get("exact_integer_validation")
    if integer_reference is not None:
        if (
            not isinstance(integer_validation, Mapping)
            or integer_validation.get("required") is not True
            or integer_validation.get("passed") is not True
            or integer_validation.get("exact_match") is not True
            or integer_validation.get("mismatch_count") != 0
            or integer_validation.get("reference_sha256") != integer_reference["sha256"]
        ):
            raise ValueError(
                "integer_reference_validation_failed: exact int32 evidence did not pass"
            )
    elif isinstance(integer_validation, Mapping) and integer_validation.get("required") is True:
        raise ValueError(
            "integer_reference_validation_failed: float32 execution cannot require int32 evidence"
        )
    _validate_repetition_transfer_invariants(response)
    if _full_precision_required(request, response):
        full_precision = response.get("full_precision_accuracy")
        if (
            not isinstance(full_precision, Mapping)
            or full_precision.get("passed") is not True
            or not _is_finite_nonnegative(full_precision.get("max_abs_error"))
        ):
            raise ValueError("full_precision_accuracy_failed: mandatory full-precision accuracy did not pass")


def _measured_repetitions(response: Mapping[str, Any], config: M5StudyConfig) -> list[tuple[int, Mapping[str, Any]]]:
    repetitions = response.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != config.warmups + config.repeats:
        raise ValueError("repeat_evidence_invalid: native response must contain 2 warmups and 7 repeats")
    if any(not isinstance(item, Mapping) for item in repetitions):
        raise ValueError("repeat_evidence_invalid: repetitions must be mappings")
    if any(item.get("warmup") is not True for item in repetitions[:config.warmups]):
        raise ValueError("repeat_evidence_invalid: warmups must precede measured repetitions")
    if any(item.get("warmup") is not False for item in repetitions[config.warmups:]):
        raise ValueError("repeat_evidence_invalid: measured repetitions are not canonical")
    # Native repeat IDs are evidence, but normalized rows use the canonical
    # measured sequence 0..repeats-1 across all providers.
    measured = [(index, item) for index, item in enumerate(repetitions[config.warmups:])]
    if len(measured) != config.repeats:
        raise ValueError("repeat_evidence_invalid: measured repeat count is not seven")
    return measured


def _validate_repetition_transfer_invariants(response: Mapping[str, Any]) -> None:
    repetitions = response.get("repetitions")
    if not isinstance(repetitions, list):
        raise ValueError("transfer_evidence_invalid: native response has no repetitions")
    for index, repetition in enumerate(repetitions):
        if not isinstance(repetition, Mapping):
            raise ValueError(f"transfer_evidence_invalid: repetition {index} is not a mapping")
        transfers = repetition.get("transfers", repetition.get("transfer", {}))
        if not isinstance(transfers, Mapping):
            raise ValueError(f"transfer_evidence_invalid: repetition {index} has no transfers")
        fields = _application_visible_transfer_fields(transfers)
        if any(fields[name] is None for name in fields):
            raise ValueError(
                f"transfer_evidence_invalid: repetition {index} lacks finite nonnegative transfer bytes"
            )
        if fields["application_visible_transfer_bytes"] != (
            fields["application_visible_h2d_bytes"]
            + fields["application_visible_d2h_bytes"]
        ):
            raise ValueError(
                f"transfer_evidence_invalid: repetition {index} total does not equal h2d+d2h"
            )
        if _first_transfer_number(transfers, "reset_h2d_bytes") != 0:
            raise ValueError(
                f"transfer_evidence_invalid: repetition {index} re-uploaded resident state"
            )


def _normalized_record(
    plan: Mapping[str, Any],
    response: Mapping[str, Any],
    timing: Mapping[str, Any],
    repeat_id: int,
    environment: Mapping[str, str],
    *,
    provider_kind: str = "default_native",
) -> dict[str, Any]:
    native_evidence = _native_evidence(response)
    policy_validation = _validation_payload(response.get("policy_reference_validation"))
    full_precision_accuracy, quantization_error_vs_float32, scientific_status = _scientific_validation_fields(
        plan.get("request", {}), response
    )
    exact_integer_validation = _validation_payload(
        response.get("exact_integer_validation")
    )
    per_repeat_transfers = timing.get("transfers", timing.get("transfer", {}))
    if not isinstance(per_repeat_transfers, Mapping):
        per_repeat_transfers = {}
    run_global_transfers = response.get(
        "run_total_transfers", response.get("transfers", response.get("transfer", {}))
    )
    if not isinstance(run_global_transfers, Mapping):
        run_global_transfers = {}
    row = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "suite_id": SUITE_ID,
        "case_id": plan["case_id"],
        "workload_id": plan["workload_id"],
        "benchmark_role": plan["benchmark_role"],
        "quantum_case": plan["quantum_case"],
        "workload_kind": (
            "synthetic"
            if plan["quantum_case"] == "non_quantum"
            else "quantum_circuit"
        ),
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "native_route_id": response.get("route_id"),
        "native_backend_id": response.get("backend_id"),
        "native_hardware_profile_version": response.get("hardware_profile_version"),
        "output_checksum_policy": response.get("output_checksum_policy"),
        "backend_family": "upmem_sdk",
        "target_requested": "hardware",
        "target_observed": "physical_hardware",
        "hardware_functionality_evidence": _physical_functionality_evidence(
            response, provider_kind=provider_kind
        ),
        "native_provider_kind": provider_kind,
        "execution_class": response.get("execution_class"),
        "kernel_strategy": response.get("kernel_strategy"),
        "requested_rank_path": response.get("requested_rank_path"),
        "rank_count": response.get("rank_count"),
        "one_rank": response.get("one_rank"),
        "single_rank": response.get("single_rank"),
        "upmem_execution_mode": NATIVE_PLAN_KIND,
        "execution_plan_kind": NATIVE_PLAN_KIND,
        "execution_plan_hash": (
            plan.get("execution_plan_hash")
            or response.get("execution_plan_hash")
            or plan.get("request", {}).get("execution_plan_hash")
        ),
        "execution_input_hash": plan.get("request", {}).get("execution_input_hash"),
        "partition_strategy": plan["partition_strategy"],
        "partition_mode": "output_tile" if plan["partition_strategy"] == "output" else "contracted_partial_sum",
        "quantization_mode": plan["quantization_mode"],
        "numeric_mode": response.get("numeric_mode"),
        "numeric_arithmetic": response.get("numeric_arithmetic"),
        "numeric_transport": response.get("numeric_transport"),
        "requantization_scope": response.get("requantization_scope"),
        "packed_int8_transfer": response.get("packed_int8_transfer"),
        "host_quantization": response.get("host_quantization"),
        "host_quantization_time_s": plan.get("request", {}).get(
            "host_quantization_time_s"
        ),
        "dpu_intermediate_requantization": response.get(
            "dpu_intermediate_requantization"
        ),
        "requested_dpu_count": plan["requested_dpu_count"],
        "allocated_dpu_count": response.get("allocated_dpu_count"),
        "observed_rank_count": response.get("observed_rank_count"),
        "tasklets_per_dpu": plan["tasklets_per_dpu"],
        "scaling_kind": plan.get("request", {}).get("scaling_kind", plan.get("scaling_kind", "strong_scaling")),
        "output_elements": plan.get("request", {}).get("output_elements"),
        "contracted_elements": plan.get("request", {}).get("contracted_elements"),
        "mac_count": plan.get("request", {}).get("mac_count"),
        "transport": plan.get("request", {}).get("transport", "float32_mram"),
        "timing_scope": plan.get("request", {}).get("timing_scope"),
        "repeat_id": repeat_id,
        "warmup": False,
        "persistent_session_intent": True,
        "persistent_session_reused": response.get("persistent_session_reused"),
        "dispatch_mode": response.get("dispatch_mode"),
        "kernel_launch_api_calls": response.get("kernel_launch_api_calls"),
        "explicit_sync_api_calls": response.get("explicit_sync_api_calls"),
        "synchronize_count": response.get("synchronize_count"),
        "hardware_allocation_verified": response.get("hardware_allocation_verified"),
        "native_kernel_executed": response.get("native_kernel_executed"),
        "hardware_kernel_executed": response.get("hardware_kernel_executed"),
        "hardware_release_verified": response.get("hardware_release_verified"),
        "allocation_provider": response.get("allocation_provider"),
        "simplepim_role": response.get("simplepim_role"),
        "kernel_provider": response.get("kernel_provider"),
        "transfer_provider": response.get("transfer_provider"),
        "collective_provider": response.get("collective_provider"),
        "reconstruction_provider": response.get("reconstruction_provider"),
        "timing": dict(timing),
        "per_repeat_timing": dict(timing),
        "transfers": dict(per_repeat_transfers),
        "per_repeat_transfers": dict(per_repeat_transfers),
        **_application_visible_transfer_fields(per_repeat_transfers),
        "run_metadata": {
            "transfers": dict(run_global_transfers),
            "timing": dict(response.get("timing", {})) if isinstance(response.get("timing"), Mapping) else {},
        },
        "run_global_transfers": dict(run_global_transfers),
        "load_balance": response.get("load_balance", response.get("load_balance_metrics", {})),
        "validation": response.get("validation", {}),
        "policy_reference_validation": policy_validation,
        "policy_reference_status": "passed",
        "exact_integer_validation": exact_integer_validation,
        "exact_integer_validation_status": exact_integer_validation.get("status"),
        "exact_integer_passed": exact_integer_validation.get("passed"),
        "exact_integer_match": exact_integer_validation.get("exact_match"),
        "exact_integer_mismatch_count": exact_integer_validation.get("mismatch_count"),
        "exact_integer_reference_sha256": exact_integer_validation.get(
            "reference_sha256"
        ),
        "full_precision_accuracy": full_precision_accuracy,
        "full_precision_accuracy_status": full_precision_accuracy["status"],
        "quantization_error_vs_float32": quantization_error_vs_float32,
        "scientific_validation_status": scientific_status,
        "validation_status": response.get("validation_status", "native_completion_verified"),
        "reference_error": response.get("reference_error", response.get("validation", {}).get("reference_error")),
        "output_error": response.get("output_error", response.get("validation", {}).get("output_error")),
        "allocation": native_evidence.get("allocation", {}),
        "launch_attempted": native_evidence.get("launch_attempted", False),
        "launch_count": native_evidence.get("launch_count", 0),
        "output_sha256": response.get("output_sha256"),
        "raw_int32_output_sha256": response.get("raw_int32_output_sha256"),
        "cpu_fallback_used": native_evidence.get("cpu_fallback_used", False),
        "simulator_kernel_executed": native_evidence.get("simulator_kernel_executed", False),
        "fallback_used": native_evidence.get("fallback_used", False),
        "claims": _false_claims(),
        "native_output_artifacts": response.get("native_output_artifacts"),
        "native_response": _compact_native_response(response),
        **_identity_hashes(plan.get("selection", {}), plan.get("request", {})),
        **_request_hashes(plan.get("request", {})),
        **hardware_environment_metadata(environment),
    }
    transfers = row["transfers"]
    if isinstance(transfers, Mapping):
        for name in ("h2d_bytes", "d2h_bytes", "actual_h2d_bytes", "actual_d2h_bytes", "actual_transfer_bytes"):
            if transfers.get(name) is not None:
                row[name] = transfers[name]
    validation = row["validation"]
    if isinstance(validation, Mapping):
        for name in ("max_abs_error", "l2_error", "reference_error", "output_error"):
            if validation.get(name) is not None:
                row[name] = validation[name]
    row["timing_s"] = timing.get("total_time_s", timing.get("elapsed_s"))
    row["steady_state_execution_time_s"] = row["timing_s"]
    row["launch_sync_time_s"] = timing.get("launch_sync_time_s")
    row["host_dequantization_time_s"] = timing.get("host_dequantization_time_s")
    row["reset_h2d_bytes"] = _first_transfer_number(
        per_repeat_transfers, "reset_h2d_bytes"
    )
    row["completion_d2h_bytes"] = _first_transfer_number(
        per_repeat_transfers, "completion_d2h_bytes"
    )
    row["final_d2h_bytes"] = _first_transfer_number(
        per_repeat_transfers, "final_d2h_bytes", "output_d2h_bytes"
    )
    per_dpu = timing.get("per_dpu")
    if isinstance(per_dpu, list):
        cycles = [
            int(item["runtime_cycles"])
            for item in per_dpu
            if isinstance(item, Mapping)
            and isinstance(item.get("runtime_cycles"), int)
            and not isinstance(item.get("runtime_cycles"), bool)
            and int(item["runtime_cycles"]) >= 0
        ]
        row["max_dpu_cycles"] = max(cycles) if cycles else None
        row["total_dpu_cycles"] = sum(cycles) if cycles else None
        work = [
            int(item["work_elements"])
            for item in per_dpu
            if isinstance(item, Mapping)
            and isinstance(item.get("work_elements"), int)
            and not isinstance(item.get("work_elements"), bool)
            and int(item["work_elements"]) >= 0
        ]
        row["total_assigned_work_elements"] = sum(work) if work else None
    else:
        row["max_dpu_cycles"] = None
        row["total_dpu_cycles"] = None
        row["total_assigned_work_elements"] = None
    row["run_operand_h2d_bytes"] = _first_transfer_number(
        run_global_transfers, "operand_h2d_bytes"
    )
    return row


def _validation_payload(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {"passed": False, "status": "failed"}


def _reference_binding(request: Mapping[str, Any], name: str) -> dict[str, Any]:
    reference = request.get(name)
    if not isinstance(reference, Mapping):
        raise ValueError(f"native_v3_request_invalid: {name} is missing")
    path = reference.get("path")
    sha256 = reference.get("sha256")
    tolerance = reference.get("max_abs_tolerance")
    if not isinstance(path, str) or not path:
        raise ValueError(f"native_v3_request_invalid: {name} path is missing")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"native_v3_request_invalid: {name} SHA-256 is invalid")
    if not _is_finite_nonnegative(tolerance):
        raise ValueError(f"native_v3_request_invalid: {name} tolerance is invalid")
    return {"path": path, "sha256": sha256, "max_abs_tolerance": float(tolerance)}


def _integer_reference_binding(request: Mapping[str, Any]) -> dict[str, str] | None:
    reference = request.get("integer_reference")
    if not isinstance(reference, Mapping):
        return None
    required = reference.get("required") is True
    path = reference.get("path")
    sha256 = reference.get("sha256")
    if not required:
        if path is not None or sha256 is not None:
            raise ValueError(
                "native_v3_request_invalid: optional integer reference must be empty"
            )
        return None
    if not isinstance(path, str) or not path:
        raise ValueError("native_v3_request_invalid: integer reference path is missing")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("native_v3_request_invalid: integer reference SHA-256 is invalid")
    return {"path": path, "sha256": sha256}


def _full_precision_accuracy(request: Mapping[str, Any]) -> dict[str, Any]:
    reference = _reference_binding(request, "full_precision_reference")
    output_path = Path(str(request.get("output_path", "")))
    reference_path = Path(reference["path"])
    if not output_path.is_file() or not reference_path.is_file():
        raise RuntimeError("full_precision_accuracy_failed: output or reference file is missing")
    output = np.fromfile(output_path, dtype="<f4")
    expected = np.fromfile(reference_path, dtype="<f4")
    if output.size != expected.size or not np.all(np.isfinite(output)):
        raise RuntimeError("full_precision_accuracy_failed: output does not match full-precision reference shape")
    metrics = conversion_error_metrics(expected, output)
    passed = metrics.max_abs_error <= reference["max_abs_tolerance"]
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "required": request.get("quantization_mode") == "none",
        "reference_kind": "cpu_full_precision_float32_reference",
        "reference_path": str(reference_path),
        "reference_sha256": reference["sha256"],
        "tolerance": reference["max_abs_tolerance"],
        "max_abs_error": metrics.max_abs_error,
        "l2_error": metrics.l2_error,
        "relative_l2_error": metrics.relative_l2_error,
    }


def _is_finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _full_precision_required(request: Mapping[str, Any], response: Mapping[str, Any]) -> bool:
    accuracy = response.get("full_precision_accuracy")
    if isinstance(accuracy, Mapping) and isinstance(accuracy.get("required"), bool):
        return accuracy["required"]
    reference = request.get("full_precision_reference")
    if isinstance(reference, Mapping) and isinstance(reference.get("required"), bool):
        return reference["required"]
    return request.get("quantization_mode") == "none"


def _scientific_validation_fields(
    request: Mapping[str, Any], response: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    required = _full_precision_required(request, response)
    raw = response.get("full_precision_accuracy")
    accuracy = dict(raw) if isinstance(raw, Mapping) else {"passed": False, "status": "not_run"}
    accuracy["required"] = required
    if not _is_finite_nonnegative(accuracy.get("max_abs_error")):
        accuracy["passed"] = False
        accuracy["status"] = "failed" if required else "not_run"
        accuracy.setdefault("max_abs_error", None)
    if required:
        accuracy["interpretation"] = "mandatory_full_precision_accuracy"
        status = "passed" if accuracy.get("passed") is True else "failed"
        accuracy["status"] = status
        return accuracy, None, status
    accuracy["interpretation"] = "descriptive_quantization_difference"
    accuracy["status"] = "descriptive"
    quantization_error_vs_float32 = {
        key: accuracy.get(key)
        for key in (
            "reference_kind", "reference_path", "reference_sha256", "max_abs_error",
            "l2_error", "relative_l2_error", "tolerance",
        )
    }
    quantization_error_vs_float32.update({
        "status": "descriptive",
        "interpretation": "descriptive_quantization_difference",
    })
    return accuracy, quantization_error_vs_float32, "passed_with_descriptive_quantization_difference"


def _failure_record(
    row: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    provider_kind: str = "default_native",
) -> dict[str, Any]:
    native_response = row.get("native_response")
    native_evidence = _native_evidence(native_response)
    if isinstance(native_response, Mapping):
        policy_validation = _validation_payload(native_response.get("policy_reference_validation"))
        full_precision_accuracy, quantization_error_vs_float32, scientific_status = _scientific_validation_fields(
            row.get("request", {}), native_response
        )
        policy_status = "passed" if policy_validation.get("passed") is True else "failed"
    else:
        policy_validation = None
        policy_status = "failed"
        full_precision_accuracy = None
        quantization_error_vs_float32 = None
        scientific_status = "failed"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": row.get("status", "failed"),
        "suite_id": SUITE_ID,
        "case_id": row.get("case_id"),
        "workload_id": row.get("workload_id"),
        "benchmark_role": row.get("benchmark_role"),
        "quantum_case": row.get("quantum_case"),
        "route_id": ROUTE_ID,
        "backend_id": BACKEND_ID,
        "native_route_id": native_response.get("route_id") if isinstance(native_response, Mapping) else None,
        "native_backend_id": native_response.get("backend_id") if isinstance(native_response, Mapping) else None,
        "native_hardware_profile_version": (
            native_response.get("hardware_profile_version")
            if isinstance(native_response, Mapping)
            else None
        ),
        "output_checksum_policy": (
            native_response.get("output_checksum_policy")
            if isinstance(native_response, Mapping)
            else None
        ),
        "execution_plan_kind": NATIVE_PLAN_KIND,
        "partition_strategy": row.get("partition_strategy"),
        "partition_mode": "output_tile" if row.get("partition_strategy") == "output" else "contracted_partial_sum" if row.get("partition_strategy") == "contracted" else row.get("partition_strategy"),
        "quantization_mode": row.get("quantization_mode"),
        "numeric_mode": "float32" if row.get("quantization_mode") == "none" else row.get("quantization_mode"),
        "scaling_kind": row.get("scaling_kind", "strong_scaling"),
        "requested_dpu_count": row.get("requested_dpu_count"),
        "allocated_dpu_count": native_evidence.get("allocated_dpu_count", 0),
        "observed_rank_count": native_evidence.get("observed_rank_count"),
        "tasklets_per_dpu": row.get("tasklets_per_dpu"),
        "failure_stage": row.get("failure_stage", "unknown"),
        "reason": _bounded_text(row.get("reason", "unknown failure"), ERROR_TEXT_LIMIT_BYTES),
        "validation_status": "failed",
        "policy_reference_validation": policy_validation,
        "policy_reference_status": policy_status,
        "full_precision_accuracy": full_precision_accuracy,
        "full_precision_accuracy_status": (
            full_precision_accuracy["status"] if full_precision_accuracy is not None else "failed"
        ),
        "quantization_error_vs_float32": quantization_error_vs_float32,
        "scientific_validation_status": scientific_status,
        "reference_error": None,
        "output_error": None,
        "transfers": {},
        "per_repeat_transfers": {},
        "run_metadata": {"transfers": {}, "timing": {}},
        "run_global_transfers": {},
        "load_balance": {},
        "hardware_functionality_evidence": False,
        "native_provider_kind": provider_kind,
        "execution_class": native_response.get("execution_class") if isinstance(native_response, Mapping) else None,
        "kernel_strategy": native_response.get("kernel_strategy") if isinstance(native_response, Mapping) else None,
        "requested_rank_path": native_response.get("requested_rank_path") if isinstance(native_response, Mapping) else None,
        "rank_count": native_response.get("rank_count") if isinstance(native_response, Mapping) else None,
        "one_rank": native_response.get("one_rank") if isinstance(native_response, Mapping) else None,
        "single_rank": native_response.get("single_rank") if isinstance(native_response, Mapping) else None,
        "application_visible_h2d_bytes": None,
        "application_visible_d2h_bytes": None,
        "application_visible_transfer_bytes": None,
        "hardware_allocation_verified": native_evidence.get("hardware_allocation_verified"),
        "native_kernel_executed": native_evidence.get("native_kernel_executed"),
        "hardware_kernel_executed": native_evidence.get("hardware_kernel_executed"),
        "hardware_release_verified": native_evidence.get("hardware_release_verified"),
        "allocation": native_evidence.get("allocation", {}),
        "launch_attempted": native_evidence.get("launch_attempted", False),
        "launch_count": native_evidence.get("launch_count", 0),
        "cpu_fallback_used": native_evidence.get("cpu_fallback_used", False),
        "simulator_kernel_executed": native_evidence.get("simulator_kernel_executed", False),
        "fallback_used": native_evidence.get("fallback_used", False),
        "claims": _false_claims(),
        "native_output_artifacts": row.get("native_output_artifacts"),
        "native_response": _compact_native_response(native_response),
        **_identity_hashes(row.get("selection", {}), row.get("request", {})),
        **_request_hashes(row.get("request", {})),
        **hardware_environment_metadata(environment),
    }


def _base_row(case: Mapping[str, Any], dpu_count: int, tasklets: int, mode: str, strategy: str) -> dict[str, Any]:
    quantum_case = str(case.get("quantum_case", case.get("kind", "real_circuit")))
    return {
        "case_id": str(case["case_id"]),
        "workload_id": str(case.get("workload_id", case["case_id"])),
        "benchmark_role": str(case.get("benchmark_role", "m5_distributed_hardware")),
        "quantum_case": quantum_case,
        "requested_dpu_count": dpu_count,
        "tasklets_per_dpu": tasklets,
        "quantization_mode": mode,
        "partition_strategy": strategy,
        "scaling_kind": _canonical_scaling_kind(case),
    }


def _strategies_for(
    case: Mapping[str, Any], defaults: Sequence[str] = PARTITION_STRATEGIES
) -> tuple[str, ...]:
    return _parse_suite_options(
        case.get("partition_strategies", defaults),
        "partition_strategies",
        defaults,
    )


def _parse_suite_options(
    value: Any, name: str, allowed: Sequence[str]
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"suite_invalid: {name} must be a list")
    result = tuple(str(item) for item in value)
    if not result or len(set(result)) != len(result) or any(item not in allowed for item in result):
        allowed_text = ", ".join(allowed)
        raise ValueError(f"suite_invalid: {name} must contain unique values from {allowed_text}")
    return result


def _canonical_scaling_kind(case: Mapping[str, Any]) -> str:
    return "weak_scaling" if str(case.get("diagnostic")) in {"weak", "weak_scaling"} else "strong_scaling"


def _identity_hashes(*values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    aliases = {
        "circuit_semantics_hash": ("circuit_semantics_hash", "circuit_hash"),
        "tensor_network_hash": ("tensor_network_hash", "network_hash"),
        "contraction_plan_hash": ("contraction_plan_hash", "task_graph_hash"),
        "contraction_path_structure_hash": (
            "contraction_path_structure_hash", "path_hash", "contraction_path_hash"
        ),
        "task_hash": ("task_hash", "selected_task_hash"),
    }
    for name, keys in aliases.items():
        for value in values:
            if isinstance(value, Mapping):
                for key in keys:
                    if value.get(key) is not None:
                        result[name] = value[key]
                        break
            if name in result:
                break
    return result


def _diagnostic_hashes(case: Mapping[str, Any]) -> dict[str, str]:
    encoded = json.dumps(dict(case), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return {
        "circuit_semantics_hash": f"non_quantum:{digest}",
        "tensor_network_hash": f"non_quantum:{digest}",
        "contraction_plan_hash": f"non_quantum:{digest}",
        "contraction_path_structure_hash": f"non_quantum:{digest}",
        "task_hash": f"non_quantum:{digest}",
    }


def _identity_hashes_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _identity_hashes(plan.get("selection", {}), plan.get("request", {}), plan)


def _plan_key(row: Mapping[str, Any]) -> str:
    return ":".join(str(row[key]) for key in ("case_id", "requested_dpu_count", "tasklets_per_dpu", "quantization_mode", "partition_strategy"))


def _false_claims() -> dict[str, bool]:
    return {
        "speedup": False,
        "cpu_speedup": False,
        "gpu_speedup": False,
        "planner_superiority": False,
        "multi_rank": False,
        "energy": False,
        "same_route_dpu_scaling_ratios": True,
        "scaling": False,
        "performance": False,
    }


def _request_hashes(request: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: request[key]
        for key in (
            "package_sha256", "operation_sha256", "sidecar_sha256", "task_hash",
            "package_circuit_semantics_hash", "package_tensor_network_hash",
            "package_contraction_plan_hash",
            "host_binary_hash", "dpu_binary_hash", "host_binary_sha256", "dpu_binary_sha256",
            "initialization_binary", "initialization_binary_sha256",
            "simplepim_role", "collective_provider", "selected_rank_path", "rank_path",
        )
        if request.get(key) is not None
    }
    for name in ("policy_reference", "integer_reference", "full_precision_reference"):
        reference = request.get(name)
        if isinstance(reference, Mapping):
            result[f"{name}_path"] = reference.get("path")
            result[f"{name}_sha256"] = reference.get("sha256")
            if reference.get("max_abs_tolerance") is not None:
                result[f"{name}_tolerance"] = reference.get("max_abs_tolerance")
    return result


def _native_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allocation = value.get("allocation")
    return {
        "target_observed": value.get("target_observed"),
        "allocated_dpu_count": value.get("allocated_dpu_count"),
        "observed_rank_count": value.get("observed_rank_count"),
        "hardware_allocation_verified": value.get("hardware_allocation_verified"),
        "native_kernel_executed": value.get("native_kernel_executed"),
        "hardware_kernel_executed": value.get("hardware_kernel_executed"),
        "hardware_release_verified": value.get("hardware_release_verified"),
        "cpu_fallback_used": value.get("cpu_fallback_used", False),
        "simulator_kernel_executed": value.get("simulator_kernel_executed", False),
        "fallback_used": value.get("fallback_used", False),
        "allocation": dict(allocation) if isinstance(allocation, Mapping) else {},
        "launch_attempted": value.get("launch_attempted"),
        "launch_count": value.get("launch_count", 0),
    }


def _physical_functionality_evidence(
    response: Mapping[str, Any], *, provider_kind: str = "default_native"
) -> bool:
    """Admit functionality evidence only from an already validated physical response."""

    return provider_kind == "default_native" and (
        response.get("status") == "completed"
        and response.get("target_observed") == "physical_hardware"
        and response.get("hardware_allocation_verified") is True
        and response.get("native_kernel_executed") is True
        and response.get("hardware_kernel_executed") is True
        and response.get("hardware_release_verified") is True
        and response.get("cpu_fallback_used", False) is False
        and response.get("simulator_kernel_executed", False) is False
        and response.get("fallback_used", False) is False
    )


def _has_completed_physical_rows(records: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        row.get("status") == "completed"
        and row.get("target_observed") == "physical_hardware"
        and row.get("hardware_functionality_evidence") is True
        for row in records
    )


def _application_visible_transfer_fields(transfers: Mapping[str, Any]) -> dict[str, Any]:
    """Expose per-repeat application-visible bytes without mixing run totals."""

    h2d = _first_transfer_number(
        transfers,
        "application_visible_h2d_bytes",
        "actual_h2d_bytes",
        "h2d_bytes",
    )
    d2h = _first_transfer_number(
        transfers,
        "application_visible_d2h_bytes",
        "actual_d2h_bytes",
        "d2h_bytes",
    )
    total = _first_transfer_number(
        transfers,
        "application_visible_transfer_bytes",
        "actual_transfer_bytes",
        "total_bytes",
    )
    if total is None and h2d is not None and d2h is not None:
        total = h2d + d2h
    if (
        total is not None
        and h2d is not None
        and d2h is not None
        and total != h2d + d2h
    ):
        raise ValueError("transfer_evidence_invalid: total does not equal h2d+d2h")
    return {
        "application_visible_h2d_bytes": h2d,
        "application_visible_d2h_bytes": d2h,
        "application_visible_transfer_bytes": total,
    }


def _first_transfer_number(transfers: Mapping[str, Any], *names: str) -> int | float | None:
    for name in names:
        if name not in transfers:
            continue
        value = transfers[name]
        if not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
        ):
            raise ValueError(f"transfer_evidence_invalid: {name} is not finite and nonnegative")
        return value
    return None


def _upmem_sdk_provenance(environment: Mapping[str, str]) -> dict[str, Any]:
    tools: dict[str, dict[str, Any]] = {}
    for name in CORE_UPMEM_SDK_TOOLS:
        path = shutil.which(name, path=environment.get("PATH"))
        if path is None:
            tools[name] = {
                "path": None,
                "version_output": None,
                "version_probe_status": "not_found",
            }
            continue
        try:
            completed = subprocess.run(
                [path, "--version"],
                env=dict(environment),
                capture_output=True,
                text=True,
                check=False,
                timeout=SDK_PROBE_TIMEOUT_S,
            )
            output = completed.stdout.strip() or completed.stderr.strip()
            version_output = output.splitlines()[0][:SDK_VERSION_OUTPUT_LIMIT_BYTES] if output else None
            tools[name] = {
                "path": path,
                "version_output": version_output,
                "version_probe_status": "passed" if completed.returncode == 0 else "failed",
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            tools[name] = {
                "path": path,
                "version_output": None,
                "version_probe_status": "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed",
            }
    rank_path = environment.get("UPMEM_HW_RANK_PATH")
    return {
        "requested_rank_path": rank_path,
        "effective_profile": hardware_environment_metadata(environment)["upmem_sdk_profile_effective"],
        "tools": tools,
    }


def _native_attempt_flags(records: Sequence[Mapping[str, Any]]) -> tuple[bool, bool]:
    allocation_attempted = False
    launch_attempted = False
    for record in records:
        evidence = _native_evidence(record.get("native_response"))
        allocation = evidence.get("allocation", {})
        allocation_attempted = allocation_attempted or allocation.get("attempted") is True
        if isinstance(evidence.get("launch_attempted"), bool):
            launch_attempted = launch_attempted or evidence["launch_attempted"]
            continue
        launch_count = evidence.get("launch_count")
        launch_attempted = launch_attempted or (
            isinstance(launch_count, (int, float))
            and not isinstance(launch_count, bool)
            and launch_count > 0
        )
    return allocation_attempted, launch_attempted


def _require_physical_environment(environment: Mapping[str, str]) -> None:
    if environment.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("hardware_opt_in_missing: UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    if not str(environment.get("UPMEM_HW_RANK_PATH", "")).strip():
        raise ValueError("hardware_rank_path_missing: UPMEM_HW_RANK_PATH is required")
    for key in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE"):
        value = str(environment.get(key, "")).lower()
        if value in {"simulator", "sdk_simulator", "cpu", "mock"}:
            raise ValueError(f"hardware_profile_violation: {key} selects {value}, physical hardware is required")


def _parse_positive_ints(values: Sequence[int] | str) -> tuple[int, ...]:
    raw = values.split(",") if isinstance(values, str) else values
    result = tuple(int(value) for value in raw)
    if not result or any(value < 1 for value in result):
        raise ValueError("dpu_counts_invalid: values must be positive integers")
    if len(set(result)) != len(result):
        raise ValueError("dpu_counts_invalid: values must be unique")
    return result


def _validate_tasklets(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("tasklets_invalid: tasklets must be an integer")
    if not MIN_TASKLETS <= value <= MAX_TASKLETS:
        raise ValueError(f"tasklets_invalid: tasklets must be in {MIN_TASKLETS}..{MAX_TASKLETS}")


def _invoke_flexible(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if accepts_kwargs:
        return function(*args, **kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **filtered)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _unique_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / "latest"
    if not candidate.exists():
        candidate.mkdir()
        return candidate
    index = 1
    while True:
        candidate = parent / f"run_{index:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        index += 1


def _failure_stage(exc: BaseException) -> str:
    explicit = getattr(exc, "failure_stage", None) or getattr(exc, "stage", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    return str(exc).split(":", 1)[0] or "runner"


def _write_native_response(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2), encoding="utf-8")


def _bounded_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[-limit:].decode("utf-8", errors="replace")


def _bounded_output(value: Any) -> str:
    return _bounded_text(value, NATIVE_OUTPUT_LIMIT_BYTES)


def _compact_native_response(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    response = dict(value)
    if response.get("error") is not None:
        response["error"] = _bounded_text(response["error"], ERROR_TEXT_LIMIT_BYTES)
    return response


def _artifact_metadata(path: Path, *, relative_path: str) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": relative_path,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_native_output_artifacts(
    response_path: Path, stdout: Any, stderr: Any
) -> dict[str, dict[str, Any]]:
    response_path = response_path.resolve()
    plan_dir = response_path.parent
    plans_dir = plan_dir.parent
    if plans_dir.name != "plans" or response_path.name != "response.json":
        raise RuntimeError(
            "native_output_artifact_path_invalid: response must be under "
            "run_dir/plans/<plan_id>/response.json"
        )
    run_dir = plans_dir.parent
    plan_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = response_path.parent / "native_stdout.txt"
    stderr_path = response_path.parent / "native_stderr.txt"
    stdout_path.write_text(_bounded_output(stdout), encoding="utf-8")
    stderr_path.write_text(_bounded_output(stderr), encoding="utf-8")
    return {
        "stdout": _artifact_metadata(
            stdout_path, relative_path=stdout_path.relative_to(run_dir).as_posix()
        ),
        "stderr": _artifact_metadata(
            stderr_path, relative_path=stderr_path.relative_to(run_dir).as_posix()
        ),
    }


def _write_build_failure_artifact(build_dir: Path, exc: BaseException) -> dict[str, dict[str, Any]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    path = build_dir / "build_failure.log"
    path.write_text(
        _bounded_text(f"{_failure_stage(exc)}: {exc}\n", ERROR_TEXT_LIMIT_BYTES),
        encoding="utf-8",
    )
    metadata = _artifact_metadata(path, relative_path="native_build/build_failure.log")
    metadata["path"] = "native_build/build_failure.log"
    return {"build_failure": metadata}


def _read_native_response(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _validate_timeout(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        or float(value) > MAX_TIMEOUT_S
    ):
        raise ValueError(
            f"timeout_invalid: timeout_s must be finite, positive, and <= {MAX_TIMEOUT_S}"
        )
    return float(value)


__all__ = [
    "BACKEND_ID", "DEFAULT_DPU_COUNTS", "DEFAULT_TASKLETS", "DEFAULT_TIMEOUT_S",
    "M5StudyConfig",
    "M5NativeTarget", "NATIVE_BACKEND_ID", "NATIVE_HARDWARE_PROFILE_VERSION",
    "NATIVE_PLAN_KIND", "NATIVE_ROUTE_ID", "PARTITION_STRATEGIES", "QUANTIZATION_MODES",
    "REPEATS", "ROUTE_ID", "SCHEMA_VERSION", "SUITE_ID", "WARMUPS", "execute",
    "load_m5_suite", "prepare",
]
