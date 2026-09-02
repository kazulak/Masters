"""Analyze a matched request-record-template diagnostic.

The analyzer consumes two already verified canonical evidence directories.  It
does not execute a route and it does not change the generic report or evidence
schemas.  Measurements are paired by circuit, route, and block ID; warmups
are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any, Mapping, Sequence


EXPECTED_CASES = (
    "quantization_stress_18q_l2",
    "hs_18q_d1",
    "ghz_chain_18q",
)
EXPECTED_ROUTES = (
    "upmem_float32_1dpu_t8",
    "upmem_float32_4dpu_t8",
)
MEASUREMENT_BLOCKS = (1, 2, 3, 4, 5)
COMPLEX_LANE_COUNT = 4
ACCOUNTING_EPSILON_S = 1e-6
COMPONENTS = (
    "total_wall_s",
    "kernel_s",
    "request_wave_wall_s",
    "request_build_s",
    "payload_record_staging_s",
    "payload_record_construction_s",
    "payload_materialization_s",
    "payload_file_write_s",
    "payload_hashing_s",
    "manifest_sidecar_staging_s",
    "session_open_s",
    "session_close_s",
)
COUNTERS = (
    "payload_record_count",
    "payload_files_created",
    "payload_bytes_staged",
    "payload_bytes_hashed",
)
RECONCILIATION_COMPONENTS = (
    "payload_record_staging_residual_s",
    "payload_record_construction_s",
    "request_build_s",
    "request_wave_wall_s",
    "total_wall_s",
    "session_open_s",
    "session_close_s",
    "attempt_elapsed_s",
)
IDENTITY_FIELDS = (
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "physical_plan_id",
    "executable_id",
    "validation_policy_id",
)
BINARY_HASH_FIELDS = (
    "host_binary_sha256",
    "dpu_binary_sha256",
)
ENVIRONMENT_FIELDS = (
    "host",
    "platform",
    "python",
    "numpy_version",
    "upmem_sdk_version",
    "affinity",
    "selected_cpu_ids",
    "observed_cpu_governors",
    "observed_numa_nodes",
    "requested_rank_paths",
    "blas",
    "thread_environment",
    "collection_machine_policy",
)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _binary_identity(session: Mapping[str, Any]) -> dict[str, str]:
    facts = _mapping(session.get("terminal_backend_facts"), "terminal backend facts")
    identity: dict[str, str] = {}
    for field in BINARY_HASH_FIELDS:
        value = _required_string(facts.get(field), f"terminal_backend_facts.{field}")
        if len(value) != 64:
            raise ValueError(f"terminal_backend_facts.{field} must be SHA-256")
        identity[field] = value
    return identity


def _sample_identity(
    sample: Mapping[str, Any], facts: Mapping[str, Any]
) -> dict[str, Any]:
    identities = _mapping(sample.get("identities"), "sample identities")
    numeric_facts = _mapping(sample.get("numeric_facts"), "sample numeric_facts")
    identity = {
        field: _required_string(identities.get(field), f"identities.{field}")
        for field in IDENTITY_FIELDS
    }
    identity["plan_id"] = _required_string(sample.get("plan_id"), "sample.plan_id")
    identity.update(
        {
            "numeric_policy": _required_string(
                numeric_facts.get("numeric_policy"), "numeric_facts.numeric_policy"
            ),
            "kernel_implementation_id": _required_string(
                facts.get("kernel_implementation_id"),
                "backend_facts.kernel_implementation_id",
            ),
            "kernel_policy": _required_string(
                facts.get("kernel_policy"), "backend_facts.kernel_policy"
            ),
            "intermediate_policy": _required_string(
                facts.get("intermediate_policy"),
                "backend_facts.intermediate_policy",
            ),
            "rank_count": facts.get("rank_count"),
            "requested_dpus": facts.get("requested_dpus"),
            "allocated_dpus": facts.get("allocated_dpus"),
            "active_dpus": facts.get("active_dpus"),
            "tasklets_per_dpu": facts.get("tasklets_per_dpu"),
        }
    )
    for field in (
        "rank_count",
        "requested_dpus",
        "allocated_dpus",
        "active_dpus",
        "tasklets_per_dpu",
    ):
        value = identity[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"backend_facts.{field} must be a positive integer")
    return identity


def _environment_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    configuration = _mapping(manifest.get("configuration"), "manifest.configuration")
    environment = _mapping(
        configuration.get("environment"), "manifest.configuration.environment"
    )
    missing = [field for field in ENVIRONMENT_FIELDS if field not in environment]
    if missing:
        raise ValueError(f"environment is missing {missing[0]}")
    return {field: environment[field] for field in ENVIRONMENT_FIELDS}


def _experiment_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    configuration = _mapping(manifest.get("configuration"), "manifest.configuration")
    experiment = _mapping(
        configuration.get("experiment"), "manifest.configuration.experiment"
    )
    collection = _mapping(experiment.get("collection"), "experiment.collection")
    cases = experiment.get("cases")
    matrix = experiment.get("matrix")
    plans = experiment.get("plans")
    routes = _mapping(experiment.get("routes"), "experiment.routes")
    if not isinstance(cases, Mapping) or not isinstance(matrix, list):
        raise ValueError("experiment contract is missing cases or matrix")
    normalized_routes: dict[str, Any] = {}
    for route_id, raw_route in routes.items():
        route = _mapping(raw_route, f"experiment.routes.{route_id}")
        options = _mapping(route.get("options"), f"experiment.routes.{route_id}.options")
        normalized_routes[str(route_id)] = {
            "executor": route.get("executor"),
            "numeric_policy": route.get("numeric_policy"),
            "dpu_count": options.get("dpu_count"),
            "rank_count": options.get("rank_count"),
            "tasklets_per_dpu": options.get("tasklets_per_dpu"),
            "rank_paths": options.get("rank_paths"),
        }
    return {
        "cases": cases,
        "matrix": matrix,
        "plans": plans,
        "routes": normalized_routes,
        "collection": {
            field: collection.get(field)
            for field in (
                "base_seed",
                "claim_policy",
                "measurement_blocks",
                "warmup_blocks",
                "session_policy",
            )
        },
    }


def _validate_diagnostic_report(root: Path, manifest: Mapping[str, Any]) -> None:
    configuration = _mapping(manifest.get("configuration"), "manifest.configuration")
    experiment = _mapping(
        configuration.get("experiment"), "manifest.configuration.experiment"
    )
    collection = _mapping(experiment.get("collection"), "experiment.collection")
    if collection.get("claim_policy") != "diagnostic_v1":
        raise ValueError("A/B evidence must use diagnostic_v1")
    report = _json(root / "report" / "report.json")
    if report.get("status") != "completed" or report.get("artifact_status") != "completed":
        raise ValueError("A/B report is not completed")
    if report.get("experiment_id") != manifest.get("experiment_id"):
        raise ValueError("A/B report experiment ID does not match the manifest")
    if report.get("run_id") != manifest.get("run_id"):
        raise ValueError("A/B report run ID does not match the manifest")
    qualification = _mapping(report.get("qualification"), "report.qualification")
    if qualification.get("claim_eligible_aggregate_count") != 0:
        raise ValueError("diagnostic report contains claim-eligible aggregates")
    if report.get("speedup_count") != 0:
        raise ValueError("diagnostic report contains generic speedups")
    rejections = _mapping(report.get("speedup_rejections"), "report.speedup_rejections")
    reason_count = rejections.get("candidate_diagnostic_claim_policy")
    if isinstance(reason_count, bool) or not isinstance(reason_count, int) or reason_count <= 0:
        raise ValueError("diagnostic report lacks an explicit claim-ineligibility reason")


def _operation_facts(
    sample: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    facts = sample.get("backend_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("sample is missing backend_facts")
    operations = facts.get("operation_facts")
    if not isinstance(operations, list) or not operations:
        raise ValueError("A/B evidence must contain operation facts")
    timings: list[Mapping[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError("sample operation fact is not an object")
        timing = operation.get("timing")
        if not isinstance(timing, Mapping):
            raise ValueError("sample operation fact is missing timing")
        timings.append(timing)
    return facts, tuple(operations), tuple(timings)


def _validate_physical_sample(sample: Mapping[str, Any]) -> None:
    if sample.get("status") != "success":
        raise ValueError("A/B evidence contains a non-success sample")
    facts, operations, operation_timings = _operation_facts(sample)
    required = {
        "target_observed": "physical_hardware",
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "requested_dpus": facts.get("allocated_dpus"),
        "allocated_dpus": facts.get("active_dpus"),
    }
    if any(value is None for value in required.values()):
        raise ValueError("physical sample lacks resource or target facts")
    if facts.get("target_observed") != required["target_observed"]:
        raise ValueError("sample did not execute on physical hardware")
    for field in (
        "simulator_kernel_executed",
        "cpu_fallback_used",
    ):
        if facts.get(field) is not required[field]:
            raise ValueError(f"sample has invalid {field}")
    if facts.get("hardware_kernel_executed") is False:
        raise ValueError("sample has invalid hardware_kernel_executed")
    for operation in operations:
        if operation.get("target_observed") != "physical_hardware":
            raise ValueError("an operation did not execute on physical hardware")
        if operation.get("simulator_kernel_executed") is not False:
            raise ValueError("an operation used the simulator kernel")
        if operation.get("cpu_fallback_used") is not False:
            raise ValueError("an operation used CPU fallback")
        if operation.get("hardware_kernel_executed") is not True:
            raise ValueError("an operation did not execute the hardware kernel")
    if facts.get("requested_dpus") != facts.get("allocated_dpus"):
        raise ValueError("requested and allocated DPU counts differ")
    if facts.get("allocated_dpus") != facts.get("active_dpus"):
        raise ValueError("allocated and active DPU counts differ")
    if sample.get("measurement", {}).get("scope_id") != "steady_execution_v1":
        raise ValueError("A/B sample does not use steady_execution_v1")
    for operation_timing in operation_timings:
        if operation_timing.get("timing_scope") not in {
            None,
            "sum_of_per_request_max_rank_response_counters_v1",
        }:
            raise ValueError("unexpected operation timing scope")


def _load_arm(root: Path) -> dict[str, Any]:
    evidence = root / "evidence"
    manifest = _json(evidence / "manifest.json")
    _validate_diagnostic_report(root, manifest)
    samples = _jsonl(evidence / "samples.jsonl")
    sessions = _jsonl(evidence / "sessions.jsonl")
    source_commit = _required_string(manifest.get("source_commit"), "source_commit")
    if len(source_commit) != 40:
        raise ValueError("source_commit must be a full commit SHA")
    if manifest.get("status") != "completed":
        raise ValueError(f"evidence is not completed: {root}")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError("A/B evidence must bind to a clean source worktree")
    if len(samples) != 36 or len(sessions) != 36:
        raise ValueError("each A/B arm must contain 36 samples and sessions")
    sessions_by_id = {str(row["session_instance_id"]): row for row in sessions}
    if len(sessions_by_id) != 36:
        raise ValueError("session IDs are not unique")
    for session in sessions_by_id.values():
        if session.get("status") != "success" or session.get("release_verified") is not True:
            raise ValueError("A/B evidence contains an unsuccessful physical session")

    measurements: dict[tuple[str, str, int], dict[str, Any]] = {}
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    binary_identities: dict[tuple[str, str], dict[str, str]] = {}
    warmups: set[tuple[str, str, int]] = set()
    sample_session_ids: set[str] = set()
    counts = {"warmup": 0, "measurement": 0}
    for sample in samples:
        kind = sample.get("attempt_kind")
        if kind not in counts:
            raise ValueError(f"unexpected attempt kind: {kind!r}")
        counts[kind] += 1
        case_id = sample.get("case_id")
        route_id = sample.get("route_id")
        block_id = sample.get("block_id")
        session_id = str(sample.get("session_instance_id"))
        if session_id == "None" or session_id in sample_session_ids:
            raise ValueError("sample session IDs must be unique and present")
        session = sessions_by_id.get(session_id)
        if session is None:
            raise ValueError("sample does not have a corresponding session")
        if sample.get("experiment_id") != manifest.get("experiment_id"):
            raise ValueError("sample experiment ID does not match the manifest")
        if sample.get("run_id") != manifest.get("run_id"):
            raise ValueError("sample run ID does not match the manifest")
        if session.get("experiment_id") != manifest.get("experiment_id"):
            raise ValueError("session experiment ID does not match the manifest")
        if session.get("run_id") != manifest.get("run_id"):
            raise ValueError("session run ID does not match the manifest")
        if session.get("case_id") != case_id or session.get("route_id") != route_id:
            raise ValueError("session does not match sample case and route")
        sample_session_ids.add(session_id)
        if case_id not in EXPECTED_CASES or route_id not in EXPECTED_ROUTES:
            raise ValueError("unexpected A/B case or route")
        _validate_physical_sample(sample)
        sample_identity = _sample_identity(sample, _operation_facts(sample)[0])
        binary_identity = _binary_identity(session)
        cell = (case_id, route_id)
        if cell in identities and identities[cell] != sample_identity:
            raise ValueError("sample identity changes within an A/B cell")
        if cell in binary_identities and binary_identities[cell] != binary_identity:
            raise ValueError("binary identity changes within an A/B cell")
        identities[cell] = sample_identity
        binary_identities[cell] = binary_identity
        validation = _mapping(sample.get("validation"), "sample.validation")
        for field in ("accuracy_qualified", "full_precision_passed", "policy_reference_passed"):
            if validation.get(field) is not True:
                raise ValueError(f"sample validation did not pass: {field}")
        if kind == "warmup":
            key = (case_id, route_id, block_id)
            if block_id != 0 or key in warmups:
                raise ValueError("each A/B cell must contain one warmup block 0")
            warmups.add(key)
            continue
        if block_id not in MEASUREMENT_BLOCKS:
            raise ValueError("measurement blocks must be exactly 1..5")
        key = (case_id, route_id, block_id)
        if key in measurements:
            raise ValueError(f"duplicate measurement key: {key}")
        if session.get("status") != "success":
            raise ValueError("sample does not have a successful session")
        measurement = sample.get("measurement")
        if not isinstance(measurement, Mapping):
            raise ValueError("sample is missing measurement")
        _, operations, operation_timings = _operation_facts(sample)
        if any(
            operation.get("lane_pass_count") != COMPLEX_LANE_COUNT
            for operation in operations
        ):
            raise ValueError("request-template evidence must record four complex lanes")

        def sum_timing(field: str) -> float:
            return sum(
                _number(timing.get(field), field) for timing in operation_timings
            )

        def sum_counter(field: str) -> int:
            return sum(
                _integer(timing.get(field), field) for timing in operation_timings
            )

        payload_record_count = sum_counter("request_payload_record_count")
        payload_record_staging_s = sum_timing(
            "request_payload_record_staging_sum_s"
        )
        payload_children_s = sum(
            sum_timing(field)
            for field in (
                "request_payload_materialization_sum_s",
                "request_payload_file_write_sum_s",
                "request_payload_hashing_sum_s",
                "request_payload_record_construction_sum_s",
            )
        )
        payload_residual_s = payload_record_staging_s - payload_children_s
        if payload_residual_s < -ACCOUNTING_EPSILON_S:
            raise ValueError("payload record staging accounting is materially negative")

        values = {
            "total_wall_s": _number(measurement.get("total_wall_s"), "total_wall_s"),
            "kernel_s": _number(measurement.get("kernel_s"), "kernel_s"),
            "request_wave_wall_s": sum_timing("request_wave_wall_sum_s"),
            "request_build_s": sum_timing("request_build_sum_s"),
            "payload_record_staging_s": payload_record_staging_s,
            "payload_record_staging_residual_s": max(0.0, payload_residual_s),
            "payload_record_construction_s": sum_timing(
                "request_payload_record_construction_sum_s"
            ),
            "payload_materialization_s": sum_timing(
                "request_payload_materialization_sum_s"
            ),
            "payload_file_write_s": sum_timing("request_payload_file_write_sum_s"),
            "payload_hashing_s": sum_timing("request_payload_hashing_sum_s"),
            "manifest_sidecar_staging_s": sum_timing(
                "request_manifest_sidecar_staging_sum_s"
            ),
            "session_open_s": _number(session.get("open_s"), "session.open_s"),
            "session_close_s": _number(
                session.get("session_close_s"), "session.session_close_s"
            ),
            "attempt_elapsed_s": (
                _number(session.get("open_s"), "session.open_s")
                + _number(measurement.get("total_wall_s"), "total_wall_s")
                + _number(session.get("session_close_s"), "session.session_close_s")
            ),
            "payload_record_count": payload_record_count,
            "payload_files_created": sum_counter("request_payload_files_created"),
            "payload_bytes_staged": sum_counter("request_payload_bytes_staged"),
            "payload_bytes_hashed": sum_counter("request_payload_bytes_hashed"),
        }
        measurements[key] = values
    if counts != {"warmup": 6, "measurement": 30}:
        raise ValueError(f"expected 6 warmups and 30 measurements, got {counts}")
    if len(warmups) != 6:
        raise ValueError("A/B arm does not contain one warmup per cell")
    if len(measurements) != 30:
        raise ValueError("A/B arm does not contain one measurement per cell/block")
    if sample_session_ids != set(sessions_by_id):
        raise ValueError("A/B sample/session mapping is not bijective")
    return {
        "source_commit": source_commit,
        "experiment_id": manifest.get("experiment_id"),
        "run_id": manifest.get("run_id"),
        "environment_identity": _environment_identity(manifest),
        "experiment_contract": _experiment_contract(manifest),
        "identities": identities,
        "binary_identities": binary_identities,
        "measurements": measurements,
    }


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    center = median(values)
    return {
        "n": len(values),
        "median": center,
        "raw_mad": median(abs(value - center) for value in values),
        "min": min(values),
        "max": max(values),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _paired_summary(
    baseline: Sequence[float], optimized: Sequence[float], *, seed: int
) -> dict[str, float | int | bool]:
    baseline_median = median(baseline)
    optimized_median = median(optimized)
    if optimized_median <= 0:
        raise ValueError("optimized median must be positive")
    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(10_000):
        indexes = [rng.randrange(len(baseline)) for _ in baseline]
        bootstrap.append(
            median(baseline[index] for index in indexes)
            / median(optimized[index] for index in indexes)
        )
    return {
        "baseline_median": baseline_median,
        "optimized_median": optimized_median,
        "descriptive_speedup": baseline_median / optimized_median,
        "optimized_change_fraction": optimized_median / baseline_median - 1.0,
        "paired_bootstrap_low": _percentile(bootstrap, 0.025),
        "paired_bootstrap_high": _percentile(bootstrap, 0.975),
        "bootstrap_resamples": len(bootstrap),
        "diagnostic_only": True,
    }


def _paired_delta_summary(
    baseline: Sequence[float], optimized: Sequence[float]
) -> dict[str, float | int]:
    """Summarize optimized-minus-baseline deltas for matched block IDs."""

    if len(baseline) != len(optimized) or not baseline:
        raise ValueError("paired A/B values must have equal non-zero length")
    return _stats(
        [optimized_value - baseline_value for baseline_value, optimized_value in zip(
            baseline, optimized, strict=True
        )]
    )


def _reconciliation_rows(
    baseline: Mapping[str, Any], optimized: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in EXPECTED_CASES:
        for route_id in EXPECTED_ROUTES:
            keys = [(case_id, route_id, block) for block in MEASUREMENT_BLOCKS]
            base_values = [baseline["measurements"][key] for key in keys]
            optimized_values = [optimized["measurements"][key] for key in keys]
            row: dict[str, Any] = {
                "case_id": case_id,
                "route_id": route_id,
                "diagnostic_only": True,
            }
            for component in RECONCILIATION_COMPONENTS:
                base_stats = _stats(
                    [float(value[component]) for value in base_values]
                )
                optimized_stats = _stats(
                    [float(value[component]) for value in optimized_values]
                )
                delta = _paired_delta_summary(
                    [float(value[component]) for value in base_values],
                    [float(value[component]) for value in optimized_values],
                )
                row[f"baseline_{component}_median"] = base_stats["median"]
                row[f"baseline_{component}_raw_mad"] = base_stats["raw_mad"]
                row[f"optimized_{component}_median"] = optimized_stats["median"]
                row[f"optimized_{component}_raw_mad"] = optimized_stats["raw_mad"]
                row[f"delta_{component}_median"] = delta["median"]
                row[f"delta_{component}_raw_mad"] = delta["raw_mad"]

            baseline_record = [
                float(value["payload_record_construction_s"])
                for value in base_values
            ]
            optimized_record = [
                float(value["payload_record_construction_s"])
                for value in optimized_values
            ]
            baseline_steady = [float(value["total_wall_s"]) for value in base_values]
            optimized_steady = [
                float(value["total_wall_s"]) for value in optimized_values
            ]
            baseline_attempt = [
                float(value["attempt_elapsed_s"]) for value in base_values
            ]
            optimized_attempt = [
                float(value["attempt_elapsed_s"]) for value in optimized_values
            ]
            saved_record = _stats(
                [base - candidate for base, candidate in zip(
                    baseline_record, optimized_record, strict=True
                )]
            )["median"]
            saved_steady = _stats(
                [base - candidate for base, candidate in zip(
                    baseline_steady, optimized_steady, strict=True
                )]
            )["median"]
            saved_attempt = _stats(
                [base - candidate for base, candidate in zip(
                    baseline_attempt, optimized_attempt, strict=True
                )]
            )["median"]
            row["median_record_time_saved_s"] = saved_record
            row["median_steady_total_time_saved_s"] = saved_steady
            row["median_session_inclusive_time_saved_s"] = saved_attempt
            row["steady_propagation_ratio"] = (
                saved_steady / saved_record if saved_record > 0 else None
            )
            row["session_inclusive_propagation_ratio"] = (
                saved_attempt / saved_record if saved_record > 0 else None
            )

            rows.append(row)
    return rows


def _validate_shared_identity(
    baseline: Mapping[str, Any], optimized: Mapping[str, Any]
) -> None:
    if baseline["environment_identity"] != optimized["environment_identity"]:
        raise ValueError("A/B arms do not share the controlled environment")
    if baseline["experiment_contract"] != optimized["experiment_contract"]:
        raise ValueError("A/B arms do not share the experiment contract")
    if baseline["identities"] != optimized["identities"]:
        raise ValueError("A/B arms do not share logical or physical identities")
    if baseline["binary_identities"] != optimized["binary_identities"]:
        raise ValueError("A/B arms do not share binary identities")


def analyze(baseline_root: Path, optimized_root: Path) -> dict[str, Any]:
    baseline = _load_arm(baseline_root)
    optimized = _load_arm(optimized_root)
    if baseline["source_commit"] == optimized["source_commit"]:
        raise ValueError("A/B arms must bind different source commits")
    _validate_shared_identity(baseline, optimized)
    rows: list[dict[str, Any]] = []
    for case_id in EXPECTED_CASES:
        for route_id in EXPECTED_ROUTES:
            for component_index, component in enumerate(COMPONENTS):
                base_values = [
                    baseline["measurements"][(case_id, route_id, block)][component]
                    for block in MEASUREMENT_BLOCKS
                ]
                optimized_values = [
                    optimized["measurements"][(case_id, route_id, block)][component]
                    for block in MEASUREMENT_BLOCKS
                ]
                base_stats = _stats(base_values)
                optimized_stats = _stats(optimized_values)
                comparison = _paired_summary(
                    base_values,
                    optimized_values,
                    seed=20260831 + component_index,
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "route_id": route_id,
                        "component": component,
                        "baseline_median": base_stats["median"],
                        "baseline_raw_mad": base_stats["raw_mad"],
                        "optimized_median": optimized_stats["median"],
                        "optimized_raw_mad": optimized_stats["raw_mad"],
                        "baseline_min": base_stats["min"],
                        "baseline_max": base_stats["max"],
                        "optimized_min": optimized_stats["min"],
                        "optimized_max": optimized_stats["max"],
                        **comparison,
                    }
                )
            for counter in COUNTERS:
                base_values = [
                    float(baseline["measurements"][(case_id, route_id, block)][counter])
                    for block in MEASUREMENT_BLOCKS
                ]
                optimized_values = [
                    float(optimized["measurements"][(case_id, route_id, block)][counter])
                    for block in MEASUREMENT_BLOCKS
                ]
                rows.append(
                    {
                        "case_id": case_id,
                        "route_id": route_id,
                        "component": counter,
                        "baseline_median": median(base_values),
                        "baseline_raw_mad": _stats(base_values)["raw_mad"],
                        "optimized_median": median(optimized_values),
                        "optimized_raw_mad": _stats(optimized_values)["raw_mad"],
                        "baseline_min": min(base_values),
                        "baseline_max": max(base_values),
                        "optimized_min": min(optimized_values),
                        "optimized_max": max(optimized_values),
                        "diagnostic_only": True,
                    }
                )
    return {
        "analysis_version": "request_template_ab_v1",
        "baseline": {
            "source_commit": baseline["source_commit"],
            "experiment_id": baseline["experiment_id"],
            "run_id": baseline["run_id"],
            "sample_count": 36,
            "session_count": 36,
        },
        "optimized": {
            "source_commit": optimized["source_commit"],
            "experiment_id": optimized["experiment_id"],
            "run_id": optimized["run_id"],
            "sample_count": 36,
            "session_count": 36,
        },
        "cases": list(EXPECTED_CASES),
        "routes": list(EXPECTED_ROUTES),
        "measurement_blocks": list(MEASUREMENT_BLOCKS),
        "rows": rows,
        "reconciliation_rows": _reconciliation_rows(baseline, optimized),
        "claim_boundary": "diagnostic_only_no_optimized_performance_claim",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "case_id",
        "route_id",
        "component",
        "baseline_median",
        "baseline_raw_mad",
        "optimized_median",
        "optimized_raw_mad",
        "baseline_min",
        "baseline_max",
        "optimized_min",
        "optimized_max",
        "descriptive_speedup",
        "optimized_change_fraction",
        "paired_bootstrap_low",
        "paired_bootstrap_high",
        "bootstrap_resamples",
        "diagnostic_only",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_reconciliation_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    if not rows:
        raise ValueError("cannot write an empty reconciliation")
    fields = tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _reconciliation_document(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "analysis_version": result["analysis_version"],
        "baseline": result["baseline"],
        "optimized": result["optimized"],
        "cases": result["cases"],
        "routes": result["routes"],
        "measurement_blocks": result["measurement_blocks"],
        "rows": result["reconciliation_rows"],
        "timing_semantics": {
            "steady_total_wall_s": (
                "measurement.total_wall_s under steady_execution_v1"
            ),
            "attempt_elapsed_s": (
                "session.open_s + measurement.total_wall_s "
                "+ session.session_close_s"
            ),
            "paired_delta": "optimized minus baseline for the same block ID",
            "propagation_ratio": (
                "saved steady or session-inclusive time divided by saved "
                "payload record construction time"
            ),
        },
        "template_semantics": {
            "lifetime": (
                "operation-local inside steady_execution_v1; discarded after "
                "the operation"
            ),
            "cardinality": (
                "not inferred: dense ABI record counts include zero-work "
                "records and do not identify cached templates"
            ),
        },
        "accounting_semantics": {
            "payload_record_staging_residual_s": (
                "payload_record_staging_s minus materialization, file-write, "
                "hashing, and record-construction children, computed per sample"
            ),
            "negative_residual_tolerance_s": ACCOUNTING_EPSILON_S,
        },
        "claim_boundary": result["claim_boundary"],
    }


def _write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Request-template A/B diagnostic",
        "",
        "The comparison pairs measurement blocks by circuit and route. Warmups "
        "are excluded. Values are descriptive and are not claim-eligible "
        "performance results.",
        "",
        "| Circuit | Route | Component | Baseline median | Optimized median | Descriptive ratio | Change |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        ratio = row.get("descriptive_speedup")
        change = row.get("optimized_change_fraction")
        ratio_text = "" if ratio is None else f"{ratio:.6g}"
        change_text = "" if change is None else f"{change:+.2%}"
        lines.append(
            f"| {row['case_id']} | {row['route_id']} | {row['component']} | "
            f"{row['baseline_median']:.6g} | {row['optimized_median']:.6g} | "
            f"{ratio_text} | {change_text} |"
        )
    lines.extend(
        [
            "",
            "Claim boundary: `diagnostic_only_no_optimized_performance_claim`.",
            "The session-open and session-close rows are reported separately from "
            "steady execution; moving work outside the steady timer is not treated "
            "as an optimization.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.baseline, args.optimized)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "request_template_ab_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "request_template_ab_analysis.csv", result["rows"])
    reconciliation = _reconciliation_document(result)
    (args.output_dir / "request_template_ab_reconciliation.json").write_text(
        json.dumps(reconciliation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_reconciliation_csv(
        args.output_dir / "request_template_ab_reconciliation.csv",
        reconciliation["rows"],
    )
    _write_markdown(args.output_dir / "request_template_ab_analysis.md", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
