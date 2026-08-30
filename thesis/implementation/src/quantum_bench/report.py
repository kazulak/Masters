"""Pure reporting for finalized canonical experiment evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import csv
import hashlib
import math
import os
from pathlib import Path
import random
from statistics import median
import tempfile
from typing import Any

from quantum_bench.evidence import canonical_json, load_artifacts


_COMPONENT_FIELDS = (
    "lowering_s",
    "planning_s",
    "slicing_s",
    "mapping_s",
    "session_open_s",
    "encode_s",
    "preparation_s",
    "h2d_s",
    "kernel_s",
    "host_reduce_s",
    "d2h_s",
    "decode_s",
    "rank_work_s",
    "energy_j",
)
_RESOURCE_FACTS = (
    "requested_dpus",
    "allocated_dpus",
    "active_dpus",
    "rank_count",
    "tasklets_per_dpu",
    "dominant_wave_useful_slots",
    "dominant_wave_allocated_slots",
    "dominant_wave_utilization",
    "fully_populated_wave_count",
)
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
_TERMINAL_AUTHORITY_FIELDS = frozenset(
    {
        "target_observed",
        "physical_target_verified",
        "hardware_kernel_executed",
        "simulator_kernel_executed",
        "cpu_fallback_used",
        "requested_dpu_count",
        "allocated_dpu_count",
        "tasklets_per_dpu",
    }
)
_AGGREGATE_COLUMNS = (
    "experiment_id",
    "collection_policy_id",
    "claim_policy",
    "case_id",
    "plan_id",
    "route_id",
    "scope_id",
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "physical_plan_id",
    "validation_policy_id",
    "kernel_policy",
    "numeric_policy",
    "sample_count",
    "planned_measurement_count",
    "successful_measurement_count",
    "failed_measurement_count",
    "unsupported_measurement_count",
    "complete_measurement_count",
    "planned_warmup_count",
    "successful_warmup_count",
    "failed_warmup_count",
    "unsupported_warmup_count",
    "policy_reference_qualified",
    "accuracy_qualified",
    "claim_eligible",
    "claim_ineligibility_reason",
    "median_total_wall_s",
    "mad_total_wall_s",
    "median_total_wall_ci_low_s",
    "median_total_wall_ci_high_s",
    "min_total_wall_s",
    "max_total_wall_s",
    *tuple(f"median_{field}" for field in _COMPONENT_FIELDS),
    "median_h2d_bytes",
    "median_d2h_bytes",
    "median_max_abs_error",
    "median_relative_l2_error",
    "median_norm_drift",
    "median_phase_aligned_max_abs_error",
    *tuple(f"median_{field}" for field in _RESOURCE_FACTS),
)
_SPEEDUP_COLUMNS = (
    "experiment_id",
    "collection_policy_id",
    "case_id",
    "plan_id",
    "baseline_route_id",
    "candidate_route_id",
    "scope_id",
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "numeric_policy",
    "baseline_median_total_wall_s",
    "candidate_median_total_wall_s",
    "complete_pair_count",
    "bootstrap_method",
    "bootstrap_seed",
    "speedup",
    "speedup_ci_low",
    "speedup_ci_high",
)
_SCALING_COLUMNS = (
    "experiment_id",
    "collection_policy_id",
    "claim_policy",
    "comparison_kind",
    "comparison_role",
    "case_id",
    "plan_id",
    "scope_id",
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "numeric_policy",
    "kernel_policy",
    "validation_policy_id",
    "baseline_route_id",
    "candidate_route_id",
    "baseline_physical_plan_id",
    "candidate_physical_plan_id",
    "baseline_executable_id",
    "candidate_executable_id",
    "baseline_dpu_count",
    "candidate_dpu_count",
    "baseline_tasklet_count",
    "candidate_tasklet_count",
    "resource_ratio",
    "planned_pair_count",
    "complete_pair_count",
    "speedup",
    "speedup_ci_low",
    "speedup_ci_high",
    "parallel_efficiency",
    "claim_eligible",
    "claim_ineligibility_reason",
    "baseline_dominant_work_wave",
    "candidate_dominant_work_wave",
    "baseline_dominant_work_wave_arithmetic_work",
    "candidate_dominant_work_wave_arithmetic_work",
    "baseline_dominant_work_wave_populated_dpu_slots",
    "candidate_dominant_work_wave_populated_dpu_slots",
    "baseline_dominant_work_wave_allocated_dpu_slots",
    "candidate_dominant_work_wave_allocated_dpu_slots",
    "baseline_dominant_work_wave_utilization",
    "candidate_dominant_work_wave_utilization",
    "baseline_arithmetic_weighted_dpu_slot_utilization",
    "candidate_arithmetic_weighted_dpu_slot_utilization",
    "baseline_arithmetic_weighted_tasklet_utilization",
    "candidate_arithmetic_weighted_tasklet_utilization",
    "baseline_fully_populated_wave_count",
    "candidate_fully_populated_wave_count",
    "bootstrap_method",
    "bootstrap_seed",
)


def verify_artifacts(input_dir: str | os.PathLike[str]) -> dict[str, object]:
    """Return a narrow verification summary for one finalized evidence directory."""

    manifest, samples, sessions = load_artifacts(input_dir)
    return _verification_summary(manifest, samples, sessions)


def report_artifacts(
    input_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]
) -> dict[str, object]:
    """Write deterministic tables and plots from finalized evidence only."""

    manifest, samples, sessions = load_artifacts(input_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("report output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    aggregates = _aggregate_measurements(manifest, samples, sessions)
    if manifest["status"] != "completed":
        speedups, rejections = [], Counter({"artifact_not_completed": 1})
    elif manifest["source_worktree_dirty"] is True:
        speedups, rejections = [], Counter({"source_worktree_dirty": 1})
    else:
        speedups, rejections = _admit_speedups(
            aggregates, _collection_base_seed(manifest)
        )
        scaling = _admit_scaling(aggregates, _collection_base_seed(manifest))
    if manifest["status"] != "completed" or manifest["source_worktree_dirty"] is True:
        scaling = []
    verification = _verification_summary(manifest, samples, sessions)
    simulator_present = any(
        _all_fact_values(aggregate, "target_observed", "sdk_simulator")
        or _any_fact_true(aggregate, "simulator_kernel_executed")
        for aggregate in aggregates
    )
    report = {
        "schema_version": "evidence_report_v5",
        "status": "completed",
        "run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "artifact_status": manifest["status"],
        "verification": verification,
        "aggregate_count": len(aggregates),
        "speedup_count": len(speedups),
        "scaling_count": len(scaling),
        "speedup_rejections": dict(sorted(rejections.items())),
        "failed_count": verification["failed_count"],
        "unsupported_count": verification["unsupported_count"],
        "session_count": len(sessions),
        "statistics": {
            "summary": "median_raw_mad_v1",
            "confidence_interval": "percentile_bootstrap_95_v1",
            "confidence_level": _BOOTSTRAP_CONFIDENCE_LEVEL,
            "resample_count": _BOOTSTRAP_RESAMPLES,
            "speedup_method": "block_paired_median_ratio_bootstrap_v1",
            "outlier_policy": "no_post_hoc_exclusion_v1",
        },
        "qualification": {
            "accuracy_qualified_aggregate_count": sum(
                aggregate["accuracy_qualified"] is True for aggregate in aggregates
            ),
            "accuracy_unqualified_aggregate_count": sum(
                aggregate["accuracy_qualified"] is False for aggregate in aggregates
            ),
            "claim_eligible_aggregate_count": sum(
                aggregate["claim_eligible"] is True for aggregate in aggregates
            ),
        },
        "simulator_timing": {
            "present": simulator_present,
            "diagnostic_only": simulator_present,
            "prohibited_claims": (
                ["timing", "scaling", "speedup", "energy"]
                if simulator_present
                else []
            ),
        },
        "energy": {
            "measurement_count": sum(
                1
                for aggregate in aggregates
                if aggregate.get("median_energy_j") is not None
            ),
            "energy_efficiency_claim_generated": False,
        },
    }
    _write_json(output / "report.json", report)
    _write_csv(output / "aggregate.csv", _AGGREGATE_COLUMNS, aggregates)
    _write_csv(output / "speedups.csv", _SPEEDUP_COLUMNS, speedups)
    _write_csv(output / "scaling.csv", _SCALING_COLUMNS, scaling)
    _write_plots(output / "plots", aggregates, speedups)
    return report


def _aggregate_measurements(
    manifest: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
    sessions: Iterable[Mapping[str, Any]],
) -> list[dict[str, object]]:
    terminal_facts_by_session = {
        str(session["session_instance_id"]): session["terminal_backend_facts"]
        for session in sessions
        if isinstance(session["terminal_backend_facts"], Mapping)
    }
    sample_rows = tuple(samples)
    attempts_by_route: dict[
        tuple[object, ...], list[Mapping[str, Any]]
    ] = {}
    for sample in sample_rows:
        attempts_by_route.setdefault(_attempt_route_key(sample), []).append(sample)

    grouped: dict[
        tuple[object, ...], list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = {}
    for sample in sample_rows:
        if sample["status"] != "success" or sample["attempt_kind"] != "measurement":
            continue
        measurement = sample["measurement"]
        if not isinstance(measurement, Mapping):  # validated by load_artifacts
            raise ValueError("successful measurement sample lacks a measurement")
        identities = sample["identities"]
        if not isinstance(identities, Mapping):  # validated by load_artifacts
            raise ValueError("sample lacks identity mapping")
        key = (
            sample["case_id"],
            sample["plan_id"],
            sample["route_id"],
            measurement["scope_id"],
            identities["problem_id"],
            identities["tensor_network_structure_id"],
            identities["logical_plan_id"],
            identities["physical_plan_id"],
            _numeric_policy(sample),
        )
        grouped.setdefault(key, []).append(
            (sample, _joined_backend_facts(sample, terminal_facts_by_session))
        )

    planned_measurements = _planned_measurements_per_route(manifest)
    planned_warmups = _planned_warmups_per_route(manifest)
    bootstrap_base_seed = _collection_base_seed(manifest)
    claim_policy = _claim_policy(manifest)
    machine_preflight_passed = _machine_preflight_passed(manifest)
    aggregates = [
        _make_aggregate(
            key,
            rows,
            attempts_by_route.get(_attempt_route_key(rows[0][0]), ()),
            planned_measurements,
            planned_warmups,
            bootstrap_base_seed,
            manifest,
            claim_policy,
            machine_preflight_passed,
        )
        for key, rows in grouped.items()
    ]
    return sorted(aggregates, key=_aggregate_sort_key)


def _attempt_route_key(sample: Mapping[str, Any]) -> tuple[object, ...]:
    identities = sample["identities"]
    if not isinstance(identities, Mapping):  # validated by load_artifacts
        raise ValueError("sample lacks identity mapping")
    return (
        sample["case_id"],
        sample["plan_id"],
        sample["route_id"],
        identities["problem_id"],
        identities["tensor_network_structure_id"],
        identities["logical_plan_id"],
        identities["physical_plan_id"],
        _numeric_policy(sample),
    )


def _planned_measurements_per_route(manifest: Mapping[str, Any]) -> int | None:
    configuration = manifest["configuration"]
    if not isinstance(configuration, Mapping):  # validated by load_artifacts
        raise ValueError("manifest configuration must be a mapping")
    experiment = configuration.get("experiment")
    if not isinstance(experiment, Mapping):
        return None
    collection = experiment.get("collection")
    if not isinstance(collection, Mapping):
        return None
    count = collection.get("measurement_blocks")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    return None


def _planned_warmups_per_route(manifest: Mapping[str, Any]) -> int | None:
    collection = _manifest_collection(manifest)
    if collection is None:
        return None
    count = collection.get("warmup_blocks")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    return None


def _manifest_collection(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    configuration = manifest["configuration"]
    if not isinstance(configuration, Mapping):  # validated by load_artifacts
        raise ValueError("manifest configuration must be a mapping")
    experiment = configuration.get("experiment")
    if not isinstance(experiment, Mapping):
        return None
    collection = experiment.get("collection")
    return collection if isinstance(collection, Mapping) else None


def _claim_policy(manifest: Mapping[str, Any]) -> str:
    collection = _manifest_collection(manifest)
    if collection is None:
        return "diagnostic_v1"
    policy = collection.get("claim_policy")
    return policy if isinstance(policy, str) else "diagnostic_v1"


def _machine_preflight_passed(manifest: Mapping[str, Any]) -> bool:
    configuration = manifest["configuration"]
    if not isinstance(configuration, Mapping):  # validated by load_artifacts
        return False
    environment = configuration.get("environment")
    if not isinstance(environment, Mapping):
        return False
    preflight = environment.get("machine_preflight")
    return isinstance(preflight, Mapping) and preflight.get(
        "machine_preflight_passed"
    ) is True


def _collection_base_seed(manifest: Mapping[str, Any]) -> int:
    collection = _manifest_collection(manifest)
    if collection is None:
        return 0
    seed = collection.get("base_seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    return 0


def _verification_summary(
    manifest: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
    sessions: Iterable[Mapping[str, Any]],
) -> dict[str, object]:
    sample_rows = tuple(samples)
    session_rows = tuple(sessions)
    statuses = Counter(str(sample["status"]) for sample in sample_rows)
    scopes = sorted(
        {
            str(sample["measurement"]["scope_id"])
            for sample in sample_rows
            if sample["status"] == "success" and sample["measurement"] is not None
        }
    )
    validated_rows = [
        sample["validation"]
        for sample in sample_rows
        if isinstance(sample["validation"], Mapping)
    ]
    successful_measurements = [
        sample
        for sample in sample_rows
        if sample["attempt_kind"] == "measurement"
        and sample["status"] == "success"
    ]
    applicable_policy_validations = [
        validation
        for validation in validated_rows
        if validation["policy_reference_applicable"]
    ]
    qualified_count = sum(
        validation["accuracy_qualified"] is True
        for validation in validated_rows
    )
    unqualified_count = sum(
        validation["accuracy_qualified"] is False
        for validation in validated_rows
    )
    policy_failure_count = sum(
        validation["policy_reference_passed"] is not True
        for validation in applicable_policy_validations
    )
    measured_policy_validations = [
        sample["validation"]
        for sample in successful_measurements
        if isinstance(sample["validation"], Mapping)
        and sample["validation"]["policy_reference_applicable"]
    ]
    return {
        "status": manifest["status"],
        "run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "sample_count": len(sample_rows),
        "session_count": len(session_rows),
        "success_count": statuses["success"],
        "failed_count": statuses["failed"],
        "unsupported_count": statuses["unsupported"],
        "timing_scopes": scopes,
        "policy_reference_applicable_count": len(applicable_policy_validations),
        "policy_reference_failure_count": policy_failure_count,
        "policy_reference_qualified": (
            None
            if not measured_policy_validations
            else all(
                isinstance(sample["validation"], Mapping)
                and (
                    not sample["validation"]["policy_reference_applicable"]
                    or sample["validation"]["policy_reference_passed"] is True
                )
                for sample in successful_measurements
            )
        ),
        "accuracy_qualified_count": qualified_count,
        "accuracy_unqualified_count": unqualified_count,
        "accuracy_qualified": bool(successful_measurements)
        and all(
            isinstance(sample["validation"], Mapping)
            and sample["validation"]["accuracy_qualified"] is True
            for sample in successful_measurements
        ),
    }


def _make_aggregate(
    key: tuple[object, ...],
    rows_with_facts: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    attempted_rows: Iterable[Mapping[str, Any]],
    planned_measurements: int | None,
    planned_warmups: int | None,
    bootstrap_base_seed: int,
    manifest: Mapping[str, Any],
    claim_policy: str,
    machine_preflight_passed: bool,
) -> dict[str, object]:
    (
        case_id,
        plan_id,
        route_id,
        scope_id,
        problem_id,
        tensor_network_structure_id,
        logical_plan_id,
        physical_plan_id,
        numeric_policy,
    ) = key
    rows = [row for row, _ in rows_with_facts]
    measurements = [row["measurement"] for row in rows]
    if not all(isinstance(measurement, Mapping) for measurement in measurements):
        raise ValueError("aggregate rows must have measurements")
    typed_measurements = [
        measurement for measurement in measurements if isinstance(measurement, Mapping)
    ]
    totals = sorted(
        float(measurement["total_wall_s"]) for measurement in typed_measurements
    )
    attempted = tuple(attempted_rows)
    measured_attempts = tuple(
        row for row in attempted if row["attempt_kind"] == "measurement"
    )
    warmup_attempts = tuple(
        row for row in attempted if row["attempt_kind"] == "warmup"
    )
    statuses = Counter(str(row["status"]) for row in measured_attempts)
    warmup_statuses = Counter(str(row["status"]) for row in warmup_attempts)
    successful_count = statuses["success"]
    expected_count = (
        planned_measurements if planned_measurements is not None else len(attempted)
    )
    policy_qualified = _policy_reference_qualified(rows)
    accuracy_qualified = _all_accuracy_qualified_from_rows(rows)
    claim_eligible, ineligibility_reason = _aggregate_claim_eligibility(
        rows=rows,
        planned_measurements=expected_count,
        successful_count=successful_count,
        failed_count=statuses["failed"],
        unsupported_count=statuses["unsupported"],
        planned_warmups=planned_warmups,
        successful_warmup_count=warmup_statuses["success"],
        failed_warmup_count=warmup_statuses["failed"],
        unsupported_warmup_count=warmup_statuses["unsupported"],
        policy_reference_qualified=policy_qualified,
        accuracy_qualified=accuracy_qualified,
        claim_policy=claim_policy,
        machine_preflight_passed=machine_preflight_passed,
    )
    first_sample = rows[0]
    first_identities = first_sample["identities"]
    if not isinstance(first_identities, Mapping):  # validated by load_artifacts
        raise ValueError("aggregate sample lacks identities")
    interval = _median_bootstrap_interval(
        totals,
        _statistics_seed(bootstrap_base_seed, "aggregate", key),
    )
    aggregate: dict[str, object] = {
        "experiment_id": manifest["experiment_id"],
        "collection_policy_id": manifest["collection_policy_id"],
        "claim_policy": claim_policy,
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "scope_id": scope_id,
        "problem_id": problem_id,
        "tensor_network_structure_id": tensor_network_structure_id,
        "logical_plan_id": logical_plan_id,
        "physical_plan_id": physical_plan_id,
        "executable_id": first_identities["executable_id"],
        "validation_policy_id": first_identities["validation_policy_id"],
        "kernel_policy": _common_fact_value(
            [facts for _, facts in rows_with_facts], "kernel_policy"
        ),
        "numeric_policy": numeric_policy,
        "sample_count": len(rows),
        "planned_measurement_count": expected_count,
        "successful_measurement_count": successful_count,
        "failed_measurement_count": statuses["failed"],
        "unsupported_measurement_count": statuses["unsupported"],
        "complete_measurement_count": successful_count,
        "planned_warmup_count": planned_warmups,
        "successful_warmup_count": warmup_statuses["success"],
        "failed_warmup_count": warmup_statuses["failed"],
        "unsupported_warmup_count": warmup_statuses["unsupported"],
        "policy_reference_qualified": policy_qualified,
        "accuracy_qualified": accuracy_qualified,
        "claim_eligible": claim_eligible,
        "claim_ineligibility_reason": ineligibility_reason,
        "median_total_wall_s": median(totals),
        "mad_total_wall_s": _raw_mad(totals),
        "median_total_wall_ci_low_s": interval[0],
        "median_total_wall_ci_high_s": interval[1],
        "min_total_wall_s": min(totals),
        "max_total_wall_s": max(totals),
        "_samples": rows,
        "_joined_backend_facts": [facts for _, facts in rows_with_facts],
    }
    for field in _COMPONENT_FIELDS:
        values = _non_null_measurements(typed_measurements, field)
        if values:
            aggregate[f"median_{field}"] = median(values)
        else:
            aggregate[f"median_{field}"] = None
    for field in ("h2d_bytes", "d2h_bytes"):
        values = _non_null_measurements(typed_measurements, field)
        aggregate[f"median_{field}"] = median(values) if values else None
    for field in (
        "max_abs_error",
        "relative_l2_error",
        "norm_drift",
        "phase_aligned_max_abs_error",
    ):
        values = _validation_values(rows, field)
        aggregate[f"median_{field}"] = median(values) if values else None
    for field in _RESOURCE_FACTS:
        values = _resource_values(rows_with_facts, field)
        aggregate[f"median_{field}"] = median(values) if values else None
    return aggregate


def _raw_mad(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    center = median(ordered)
    return float(median(abs(value - center) for value in ordered))


def _statistics_seed(base_seed: int, kind: str, key: object) -> int:
    payload = canonical_json({"kind": kind, "key": key})
    digest = hashlib.sha256(
        b"quantum_bench.bootstrap_seed.v1\0"
        + str(base_seed).encode("ascii")
        + b"\0"
        + payload.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile from no values")
    index = int(quantile * (len(ordered) - 1))
    return ordered[index]


def _median_bootstrap_interval(values: Iterable[float], seed: int) -> tuple[float, float]:
    observations = sorted(float(value) for value in values)
    if not observations:
        raise ValueError("cannot bootstrap no observations")
    generator = random.Random(seed)
    population_size = len(observations)
    medians = sorted(
        float(
            median(
                observations[generator.randrange(population_size)]
                for _ in range(population_size)
            )
        )
        for _ in range(_BOOTSTRAP_RESAMPLES)
    )
    tail = (1.0 - _BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    return _percentile(medians, tail), _percentile(medians, 1.0 - tail)


def _policy_reference_qualified(rows: Iterable[Mapping[str, Any]]) -> bool | None:
    validations = [
        row["validation"]
        for row in rows
        if isinstance(row["validation"], Mapping)
        and row["validation"].get("policy_reference_applicable") is True
    ]
    if not validations:
        return None
    return all(validation.get("policy_reference_passed") is True for validation in validations)


def _all_accuracy_qualified_from_rows(rows: Iterable[Mapping[str, Any]]) -> bool:
    row_values = tuple(rows)
    return bool(row_values) and all(
        isinstance(row["validation"], Mapping)
        and row["validation"].get("accuracy_qualified") is True
        for row in row_values
    )


def _aggregate_claim_eligibility(
    *,
    rows: Iterable[Mapping[str, Any]],
    planned_measurements: int,
    successful_count: int,
    failed_count: int,
    unsupported_count: int,
    planned_warmups: int | None,
    successful_warmup_count: int,
    failed_warmup_count: int,
    unsupported_warmup_count: int,
    policy_reference_qualified: bool | None,
    accuracy_qualified: bool,
    claim_policy: str,
    machine_preflight_passed: bool,
) -> tuple[bool, str | None]:
    row_values = tuple(rows)
    if claim_policy != "physical_performance_v1":
        return False, "diagnostic_claim_policy"
    if planned_measurements != 30:
        return False, "physical_campaign_requires_30_measurements"
    if planned_warmups != 2:
        return False, "physical_campaign_requires_2_warmups"
    if not machine_preflight_passed:
        return False, "machine_preflight_failed"
    if successful_count != planned_measurements:
        return False, "incomplete_measurements"
    if failed_count or unsupported_count:
        return False, "non_successful_measurement"
    if successful_warmup_count != planned_warmups:
        return False, "incomplete_warmups"
    if failed_warmup_count or unsupported_warmup_count:
        return False, "non_successful_warmup"
    if not row_values:
        return False, "no_successful_measurements"
    if policy_reference_qualified is False:
        return False, "policy_reference_failed"
    if not accuracy_qualified:
        return False, "accuracy_unqualified"
    return True, None


def _non_null_measurements(
    measurements: Iterable[Mapping[str, Any]], field: str
) -> list[int | float]:
    values: list[int | float] = []
    for measurement in measurements:
        value = measurement[field]
        if value is not None:
            values.append(value)
    return values


def _validation_values(rows: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        validation = row["validation"]
        if isinstance(validation, Mapping) and validation[field] is not None:
            values.append(float(validation[field]))
    return values


def _resource_values(
    rows: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]], field: str
) -> list[int | float]:
    """Collect resource facts, including floating-point utilization facts."""

    values: list[int | float] = []
    for _, facts in rows:
        value = facts.get(field)
        if field == "dominant_wave_utilization" and value is None:
            value = facts.get("dominant_work_wave_utilization")
        if value is None:
            allocation = facts.get("allocation")
            if isinstance(allocation, Mapping):
                value = allocation.get(field)
        if (
            type(value) in (int, float)
            and math.isfinite(float(value))
            and value >= 0
        ):
            values.append(value)
    return values


def _common_fact_value(
    facts: Iterable[Mapping[str, Any]], field: str
) -> object | None:
    values = [facts_row.get(field) for facts_row in facts]
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _joined_backend_facts(
    sample: Mapping[str, Any],
    terminal_facts_by_session: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Use terminal facts only to fill facts absent from a linked sample."""

    facts = sample["backend_facts"]
    if not isinstance(facts, Mapping):
        raise ValueError("sample backend_facts must be a mapping")
    joined = dict(facts)
    session_instance_id = sample["session_instance_id"]
    if isinstance(session_instance_id, str):
        terminal = terminal_facts_by_session.get(session_instance_id)
        if terminal is not None:
            conflicts: list[str] = []
            for key, value in terminal.items():
                if (
                    key in _TERMINAL_AUTHORITY_FIELDS
                    and key in joined
                    and joined[key] != value
                ):
                    conflicts.append(key)
                joined.setdefault(key, value)
            if conflicts:
                joined["terminal_fact_conflicts"] = sorted(conflicts)
    return joined


def _numeric_policy(sample: Mapping[str, Any]) -> str | None:
    facts = sample["numeric_facts"]
    if not isinstance(facts, Mapping):
        return None
    value = facts.get("numeric_policy")
    return value if isinstance(value, str) and value else None


def _aggregate_sort_key(aggregate: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        "" if aggregate[field] is None else str(aggregate[field])
        for field in (
            "experiment_id",
            "collection_policy_id",
            "case_id",
            "plan_id",
            "route_id",
            "scope_id",
            "problem_id",
            "tensor_network_structure_id",
            "logical_plan_id",
            "physical_plan_id",
            "numeric_policy",
        )
    )


def _admit_speedups(
    aggregates: Iterable[Mapping[str, object]], bootstrap_base_seed: int
) -> tuple[list[dict[str, object]], Counter[str]]:
    rows = list(aggregates)
    baselines = [
        row for row in rows if _all_fact_values(row, "backend_id", "numpy_cpu_v1")
    ]
    speedups: list[dict[str, object]] = []
    rejections: Counter[str] = Counter()
    for candidate in rows:
        if _all_fact_values(candidate, "backend_id", "numpy_cpu_v1"):
            continue
        if not _is_upmem_candidate(candidate):
            continue
        reason = _speedup_rejection(candidate, baselines)
        if reason is not None:
            rejections[reason] += 1
            continue
        baseline = _matching_baseline(candidate, baselines)
        if baseline is None:
            rejections[_missing_baseline_reason(candidate, baselines)] += 1
            continue
        baseline_time = float(baseline["median_total_wall_s"])
        candidate_time = float(candidate["median_total_wall_s"])
        pairs = _paired_measurements(baseline, candidate)
        if pairs is None:
            rejections["incomplete_paired_blocks"] += 1
            continue
        comparison_key = {
            "experiment_id": candidate["experiment_id"],
            "collection_policy_id": candidate["collection_policy_id"],
            "case_id": candidate["case_id"],
            "plan_id": candidate["plan_id"],
            "baseline_route_id": baseline["route_id"],
            "candidate_route_id": candidate["route_id"],
            "scope_id": candidate["scope_id"],
            "numeric_policy": candidate["numeric_policy"],
        }
        bootstrap_seed = _statistics_seed(
            bootstrap_base_seed, "block_paired_speedup", comparison_key
        )
        speedup_ci_low, speedup_ci_high = _paired_speedup_bootstrap_interval(
            pairs, bootstrap_seed
        )
        speedups.append(
            {
                "experiment_id": candidate["experiment_id"],
                "collection_policy_id": candidate["collection_policy_id"],
                "case_id": candidate["case_id"],
                "plan_id": candidate["plan_id"],
                "baseline_route_id": baseline["route_id"],
                "candidate_route_id": candidate["route_id"],
                "scope_id": candidate["scope_id"],
                "problem_id": candidate["problem_id"],
                "tensor_network_structure_id": candidate["tensor_network_structure_id"],
                "logical_plan_id": candidate["logical_plan_id"],
                "numeric_policy": candidate["numeric_policy"],
                "baseline_median_total_wall_s": baseline_time,
                "candidate_median_total_wall_s": candidate_time,
                "complete_pair_count": len(pairs),
                "bootstrap_method": "block_paired_median_ratio_bootstrap_v1",
                "bootstrap_seed": bootstrap_seed,
                "speedup": baseline_time / candidate_time,
                "speedup_ci_low": speedup_ci_low,
                "speedup_ci_high": speedup_ci_high,
            }
        )
    return sorted(
        speedups, key=lambda row: tuple(str(row[key]) for key in _SPEEDUP_COLUMNS)
    ), rejections


def _admit_scaling(
    aggregates: Iterable[Mapping[str, object]], bootstrap_base_seed: int
) -> list[dict[str, object]]:
    """Compare physical UPMEM topology variants without using CPU speedup rows."""

    physical_rows = [
        row
        for row in aggregates
        if _is_upmem_candidate(row)
        and _all_fact_values(row, "target_observed", "physical_hardware")
    ]
    rows: list[dict[str, object]] = []
    for baseline in physical_rows:
        for candidate in physical_rows:
            kind = _scaling_kind(baseline, candidate)
            if kind is None or not _same_scaling_dimensions(baseline, candidate):
                continue
            baseline_dpus = _common_nonnegative_int_fact(baseline, "requested_dpus")
            candidate_dpus = _common_nonnegative_int_fact(candidate, "requested_dpus")
            baseline_tasklets = _common_nonnegative_int_fact(
                baseline, "tasklets_per_dpu"
            )
            candidate_tasklets = _common_nonnegative_int_fact(
                candidate, "tasklets_per_dpu"
            )
            if (
                baseline_dpus is None
                or candidate_dpus is None
                or baseline_tasklets is None
                or candidate_tasklets is None
            ):
                continue
            resource_ratio = (
                candidate_dpus / baseline_dpus
                if kind == "dpu_scaling"
                else candidate_tasklets / baseline_tasklets
            )
            comparison_role = _scaling_comparison_role(
                kind,
                baseline_dpus,
                baseline_tasklets,
            )
            pairs = _paired_measurements(baseline, candidate)
            rejection = _scaling_rejection(baseline, candidate, pairs)
            row: dict[str, object] = {
                "experiment_id": candidate["experiment_id"],
                "collection_policy_id": candidate["collection_policy_id"],
                "claim_policy": candidate["claim_policy"],
                "comparison_kind": kind,
                "comparison_role": comparison_role,
                "case_id": candidate["case_id"],
                "plan_id": candidate["plan_id"],
                "scope_id": candidate["scope_id"],
                "problem_id": candidate["problem_id"],
                "tensor_network_structure_id": candidate[
                    "tensor_network_structure_id"
                ],
                "logical_plan_id": candidate["logical_plan_id"],
                "numeric_policy": candidate["numeric_policy"],
                "kernel_policy": candidate["kernel_policy"],
                "validation_policy_id": candidate["validation_policy_id"],
                "baseline_route_id": baseline["route_id"],
                "candidate_route_id": candidate["route_id"],
                "baseline_physical_plan_id": baseline["physical_plan_id"],
                "candidate_physical_plan_id": candidate["physical_plan_id"],
                "baseline_executable_id": baseline["executable_id"],
                "candidate_executable_id": candidate["executable_id"],
                "baseline_dpu_count": baseline_dpus,
                "candidate_dpu_count": candidate_dpus,
                "baseline_tasklet_count": baseline_tasklets,
                "candidate_tasklet_count": candidate_tasklets,
                "resource_ratio": resource_ratio,
                "planned_pair_count": baseline["planned_measurement_count"],
                "complete_pair_count": 0 if pairs is None else len(pairs),
                "speedup": None,
                "speedup_ci_low": None,
                "speedup_ci_high": None,
                "parallel_efficiency": None,
                "claim_eligible": rejection is None,
                "claim_ineligibility_reason": rejection,
                "baseline_dominant_work_wave": _common_nonnegative_int_fact(
                    baseline, "dominant_work_wave"
                ),
                "candidate_dominant_work_wave": _common_nonnegative_int_fact(
                    candidate, "dominant_work_wave"
                ),
                "baseline_dominant_work_wave_arithmetic_work": _common_nonnegative_int_fact(
                    baseline, "dominant_work_wave_arithmetic_work"
                ),
                "candidate_dominant_work_wave_arithmetic_work": _common_nonnegative_int_fact(
                    candidate, "dominant_work_wave_arithmetic_work"
                ),
                "baseline_dominant_work_wave_populated_dpu_slots": _common_nonnegative_int_fact(
                    baseline, "dominant_work_wave_populated_dpu_slots"
                ),
                "candidate_dominant_work_wave_populated_dpu_slots": _common_nonnegative_int_fact(
                    candidate, "dominant_work_wave_populated_dpu_slots"
                ),
                "baseline_dominant_work_wave_allocated_dpu_slots": _common_nonnegative_int_fact(
                    baseline, "dominant_work_wave_allocated_dpu_slots"
                ),
                "candidate_dominant_work_wave_allocated_dpu_slots": _common_nonnegative_int_fact(
                    candidate, "dominant_work_wave_allocated_dpu_slots"
                ),
                "baseline_dominant_work_wave_utilization": _common_nonnegative_number_fact(
                    baseline, "dominant_work_wave_utilization"
                ),
                "candidate_dominant_work_wave_utilization": _common_nonnegative_number_fact(
                    candidate, "dominant_work_wave_utilization"
                ),
                "baseline_arithmetic_weighted_dpu_slot_utilization": _common_nonnegative_number_fact(
                    baseline, "arithmetic_weighted_dpu_slot_utilization"
                ),
                "candidate_arithmetic_weighted_dpu_slot_utilization": _common_nonnegative_number_fact(
                    candidate, "arithmetic_weighted_dpu_slot_utilization"
                ),
                "baseline_arithmetic_weighted_tasklet_utilization": _common_nonnegative_number_fact(
                    baseline, "arithmetic_weighted_tasklet_utilization"
                ),
                "candidate_arithmetic_weighted_tasklet_utilization": _common_nonnegative_number_fact(
                    candidate, "arithmetic_weighted_tasklet_utilization"
                ),
                "baseline_fully_populated_wave_count": _common_nonnegative_int_fact(
                    baseline, "fully_populated_wave_count"
                ),
                "candidate_fully_populated_wave_count": _common_nonnegative_int_fact(
                    candidate, "fully_populated_wave_count"
                ),
                "bootstrap_method": "block_paired_median_ratio_bootstrap_v1",
                "bootstrap_seed": None,
            }
            if pairs is not None and rejection is None:
                comparison_key = {
                    "experiment_id": candidate["experiment_id"],
                    "comparison_kind": kind,
                    "baseline_route_id": baseline["route_id"],
                    "candidate_route_id": candidate["route_id"],
                    "scope_id": candidate["scope_id"],
                    "numeric_policy": candidate["numeric_policy"],
                }
                bootstrap_seed = _statistics_seed(
                    bootstrap_base_seed, "block_paired_scaling", comparison_key
                )
                speedup = float(baseline["median_total_wall_s"]) / float(
                    candidate["median_total_wall_s"]
                )
                low, high = _paired_speedup_bootstrap_interval(pairs, bootstrap_seed)
                row.update(
                    {
                        "speedup": speedup,
                        "speedup_ci_low": low,
                        "speedup_ci_high": high,
                        "parallel_efficiency": speedup / resource_ratio,
                        "bootstrap_seed": bootstrap_seed,
                    }
                )
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: tuple(str(row[field]) for field in _SCALING_COLUMNS),
    )


def _scaling_comparison_role(
    kind: str,
    baseline_dpus: int,
    baseline_tasklets: int,
) -> str:
    """Label only one-resource baselines as primary scaling comparisons."""

    if kind == "tasklet_scaling" and baseline_tasklets == 1:
        return "primary"
    if kind == "dpu_scaling" and baseline_dpus == 1:
        return "primary"
    return "secondary"


def _scaling_kind(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> str | None:
    baseline_dpus = _common_nonnegative_int_fact(baseline, "requested_dpus")
    candidate_dpus = _common_nonnegative_int_fact(candidate, "requested_dpus")
    baseline_tasklets = _common_nonnegative_int_fact(
        baseline, "tasklets_per_dpu"
    )
    candidate_tasklets = _common_nonnegative_int_fact(
        candidate, "tasklets_per_dpu"
    )
    if None in {
        baseline_dpus,
        candidate_dpus,
        baseline_tasklets,
        candidate_tasklets,
    }:
        return None
    if baseline_dpus == candidate_dpus and baseline_tasklets < candidate_tasklets:
        return "tasklet_scaling"
    if baseline_tasklets == candidate_tasklets and baseline_dpus < candidate_dpus:
        return "dpu_scaling"
    return None


def _same_scaling_dimensions(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    return all(
        left[field] == right[field]
        for field in (
            "experiment_id",
            "collection_policy_id",
            "case_id",
            "plan_id",
            "scope_id",
            "problem_id",
            "tensor_network_structure_id",
            "logical_plan_id",
            "numeric_policy",
            "kernel_policy",
            "validation_policy_id",
            "claim_policy",
        )
    )


def _scaling_rejection(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    pairs: list[tuple[float, float]] | None,
) -> str | None:
    if baseline["claim_eligible"] is not True:
        return "baseline_" + str(
            baseline.get("claim_ineligibility_reason") or "not_eligible"
        )
    if candidate["claim_eligible"] is not True:
        return "candidate_" + str(
            candidate.get("claim_ineligibility_reason") or "not_eligible"
        )
    for role, aggregate in (("baseline", baseline), ("candidate", candidate)):
        if not _all_fact_values(aggregate, "physical_target_verified", True):
            return f"{role}_physical_target_not_verified"
        if not _all_fact_values(aggregate, "cpu_fallback_used", False):
            return f"{role}_cpu_fallback_used"
        if not _all_fact_values(aggregate, "simulator_kernel_executed", False):
            return f"{role}_simulator_kernel_executed"
        if not _all_fact_values(aggregate, "startup_resource_admission_passed", True):
            return f"{role}_startup_resource_admission_failed"
        if not _all_fact_values(aggregate, "execution_resource_admission_passed", True):
            return f"{role}_execution_resource_admission_failed"
    if baseline["physical_plan_id"] == candidate["physical_plan_id"]:
        return "physical_plan_not_distinct"
    if pairs is None:
        return "incomplete_paired_blocks"
    return None


def _common_nonnegative_int_fact(
    aggregate: Mapping[str, object], field: str
) -> int | None:
    value = _common_fact_value_from_aggregate(aggregate, field)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _common_nonnegative_number_fact(
    aggregate: Mapping[str, object], field: str
) -> float | None:
    value = _common_fact_value_from_aggregate(aggregate, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _common_fact_value_from_aggregate(
    aggregate: Mapping[str, object], field: str
) -> object | None:
    facts = aggregate.get("_joined_backend_facts")
    if not isinstance(facts, list) or not all(isinstance(row, Mapping) for row in facts):
        return None
    return _common_fact_value(
        [row for row in facts if isinstance(row, Mapping)], field
    )


def _is_upmem_candidate(aggregate: Mapping[str, object]) -> bool:
    facts = aggregate["_joined_backend_facts"]
    return isinstance(facts, list) and bool(facts) and all(
        isinstance(sample_facts, Mapping)
        and (
            str(sample_facts.get("backend_id") or "").startswith("upmem_")
            or sample_facts.get("execution_class")
            in {"sdk_simulator", "physical_hardware"}
            or sample_facts.get("target_observed")
            in {"sdk_simulator", "physical_hardware"}
        )
        for sample_facts in facts
    )


def _speedup_rejection(
    candidate: Mapping[str, object], baselines: Iterable[Mapping[str, object]]
) -> str | None:
    if _all_fact_values(
        candidate, "target_observed", "sdk_simulator"
    ) or _any_fact_true(candidate, "simulator_kernel_executed"):
        return "simulator_execution"
    if candidate["claim_eligible"] is not True:
        reason = candidate.get("claim_ineligibility_reason")
        return f"candidate_{reason}" if isinstance(reason, str) else "candidate_not_eligible"
    if not _all_policy_references_applicable_and_passed(candidate):
        return "candidate_validation_failed"
    if not _all_accuracy_qualified(candidate):
        return "full_precision_threshold_not_passed"
    if _any_terminal_fact_conflict(candidate):
        return "terminal_fact_conflict"
    if not _all_samples_linked_to_sessions(candidate):
        return "candidate_session_not_linked"
    if not _all_fact_values(candidate, "target_observed", "physical_hardware"):
        return "candidate_not_physical_hardware"
    if not _all_fact_values(candidate, "physical_target_verified", True):
        return "physical_target_not_verified"
    if not _all_fact_values(candidate, "cpu_fallback_used", False):
        return "cpu_fallback_used"
    if not _all_fact_values(candidate, "simulator_kernel_executed", False):
        return "simulator_kernel_executed"
    if not _all_fact_values(candidate, "startup_resource_admission_passed", True):
        return "startup_resource_admission_failed"
    if not _all_fact_values(candidate, "execution_resource_admission_passed", True):
        return "execution_resource_admission_failed"
    if not any(
        _same_speedup_dimensions(candidate, baseline)
        and _baseline_is_eligible(baseline)
        for baseline in baselines
    ):
        return _missing_baseline_reason(candidate, baselines)
    return None


def _missing_baseline_reason(
    candidate: Mapping[str, object], baselines: Iterable[Mapping[str, object]]
) -> str:
    baseline_rows = tuple(baselines)
    if any(_same_speedup_dimensions(candidate, baseline) for baseline in baseline_rows):
        return "baseline_not_admissible"
    dimensions_without_scope = (
        "case_id",
        "plan_id",
        "problem_id",
        "tensor_network_structure_id",
        "logical_plan_id",
        "numeric_policy",
    )
    if any(
        all(candidate[field] == baseline[field] for field in dimensions_without_scope)
        for baseline in baseline_rows
    ):
        return "timing_scope_mismatch"
    return "no_matching_numpy_baseline"


def _matching_baseline(
    candidate: Mapping[str, object], baselines: Iterable[Mapping[str, object]]
) -> Mapping[str, object] | None:
    for baseline in baselines:
        if _same_speedup_dimensions(candidate, baseline) and _baseline_is_eligible(
            baseline
        ):
            return baseline
    return None


def _baseline_is_eligible(aggregate: Mapping[str, object]) -> bool:
    return (
        aggregate["claim_eligible"] is True
        and _all_policy_references_passed(aggregate)
        and _all_accuracy_qualified(aggregate)
    )


def _all_measurements_completed(aggregate: Mapping[str, object]) -> bool:
    return (
        aggregate.get("successful_measurement_count")
        == aggregate.get("planned_measurement_count")
        and aggregate.get("failed_measurement_count") == 0
        and aggregate.get("unsupported_measurement_count") == 0
    )


def _paired_measurements(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> list[tuple[float, float]] | None:
    if not _all_measurements_completed(baseline) or not _all_measurements_completed(
        candidate
    ):
        return None
    baseline_rows = baseline["_samples"]
    candidate_rows = candidate["_samples"]
    if not isinstance(baseline_rows, list) or not isinstance(candidate_rows, list):
        return None
    baseline_by_block = _measurement_by_block(baseline_rows)
    candidate_by_block = _measurement_by_block(candidate_rows)
    if baseline_by_block is None or candidate_by_block is None:
        return None
    if set(baseline_by_block) != set(candidate_by_block):
        return None
    return [
        (baseline_by_block[block_id], candidate_by_block[block_id])
        for block_id in sorted(baseline_by_block)
    ]


def _measurement_by_block(
    rows: Iterable[Mapping[str, Any]],
) -> dict[int, float] | None:
    values: dict[int, float] = {}
    for row in rows:
        block_id = row.get("block_id")
        measurement = row.get("measurement")
        if (
            isinstance(block_id, bool)
            or not isinstance(block_id, int)
            or not isinstance(measurement, Mapping)
            or block_id in values
        ):
            return None
        values[block_id] = float(measurement["total_wall_s"])
    return values


def _paired_speedup_bootstrap_interval(
    pairs: Iterable[tuple[float, float]], seed: int
) -> tuple[float, float]:
    observations = sorted((float(base), float(candidate)) for base, candidate in pairs)
    if not observations or any(candidate <= 0.0 for _, candidate in observations):
        raise ValueError("paired speedup observations must be nonempty and positive")
    generator = random.Random(seed)
    population_size = len(observations)
    ratios = sorted(
        float(
            median(sample[0] for sample in selected)
            / median(sample[1] for sample in selected)
        )
        for selected in (
            [observations[generator.randrange(population_size)] for _ in range(population_size)]
            for _ in range(_BOOTSTRAP_RESAMPLES)
        )
    )
    tail = (1.0 - _BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    return _percentile(ratios, tail), _percentile(ratios, 1.0 - tail)


def _same_speedup_dimensions(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    return all(
        left[field] == right[field]
        for field in (
            "case_id",
            "plan_id",
            "scope_id",
            "problem_id",
            "tensor_network_structure_id",
            "logical_plan_id",
            "numeric_policy",
            "validation_policy_id",
        )
    )


def _all_policy_references_passed(aggregate: Mapping[str, object]) -> bool:
    samples = aggregate["_samples"]
    return (
        isinstance(samples, list)
        and bool(samples)
        and all(
            isinstance(sample["validation"], Mapping)
            and (
                not sample["validation"].get("policy_reference_applicable")
                or sample["validation"].get("policy_reference_passed") is True
            )
            for sample in samples
        )
    )


def _all_policy_references_applicable_and_passed(
    aggregate: Mapping[str, object],
) -> bool:
    samples = aggregate["_samples"]
    return (
        isinstance(samples, list)
        and bool(samples)
        and all(
            isinstance(sample["validation"], Mapping)
            and sample["validation"].get("policy_reference_applicable") is True
            and sample["validation"].get("policy_reference_passed") is True
            for sample in samples
        )
    )


def _all_accuracy_qualified(aggregate: Mapping[str, object]) -> bool:
    samples = aggregate["_samples"]
    return (
        isinstance(samples, list)
        and bool(samples)
        and all(
            isinstance(sample["validation"], Mapping)
            and sample["validation"].get("accuracy_qualified") is True
            for sample in samples
        )
    )


def _all_samples_linked_to_sessions(aggregate: Mapping[str, object]) -> bool:
    samples = aggregate["_samples"]
    return (
        isinstance(samples, list)
        and bool(samples)
        and all(
            isinstance(sample.get("session_instance_id"), str) for sample in samples
        )
    )


def _any_terminal_fact_conflict(aggregate: Mapping[str, object]) -> bool:
    facts = aggregate["_joined_backend_facts"]
    return isinstance(facts, list) and any(
        isinstance(sample_facts, Mapping)
        and bool(sample_facts.get("terminal_fact_conflicts"))
        for sample_facts in facts
    )


def _all_fact_values(
    aggregate: Mapping[str, object], field: str, expected: object
) -> bool:
    facts = aggregate["_joined_backend_facts"]
    return (
        isinstance(facts, list)
        and bool(facts)
        and all(
            isinstance(sample_facts, Mapping) and sample_facts.get(field) == expected
            for sample_facts in facts
        )
    )


def _any_fact_true(aggregate: Mapping[str, object], field: str) -> bool:
    facts = aggregate["_joined_backend_facts"]
    return isinstance(facts, list) and any(
        isinstance(sample_facts, Mapping) and sample_facts.get(field) is True
        for sample_facts in facts
    )


def _write_plots(
    directory: Path,
    aggregates: Iterable[Mapping[str, object]],
    speedups: Iterable[Mapping[str, object]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = list(aggregates)
    for scope_id in sorted({str(row["scope_id"]) for row in rows}):
        points = [
            _point(
                figure_id=f"runtime_{scope_id}",
                facet_id=scope_id,
                row=row,
                value=float(row["median_total_wall_s"]),
                ci_low=_nullable_float(row["median_total_wall_ci_low_s"]),
                ci_high=_nullable_float(row["median_total_wall_ci_high_s"]),
            )
            for row in rows
            if row["scope_id"] == scope_id
        ]
        _plot_grouped_bars(
            directory / f"runtime_{scope_id}.png",
            points,
            title=f"Median Runtime ({_humanize(scope_id)})",
            ylabel="Median total wall time (s)",
        )

    error_points = [
        _point(
            figure_id="numeric_error_by_case",
            facet_id="all",
                row=row,
                value=float(row["median_max_abs_error"]),
        )
        for row in rows
        if row["median_max_abs_error"] is not None
    ]
    if error_points:
        _plot_grouped_bars(
            directory / "numeric_error_by_case.png",
            error_points,
            title="Median Numeric Error by Case",
            ylabel="Median maximum absolute error",
        )

    transfer_points = []
    for row in rows:
        h2d = row["median_h2d_bytes"]
        d2h = row["median_d2h_bytes"]
        if h2d is None and d2h is None:
            continue
        transfer_points.append(
            _point(
                figure_id="transfer_bytes_by_case",
                facet_id="all",
                row=row,
                value=float(h2d or 0) + float(d2h or 0),
            )
        )
    if transfer_points:
        _plot_grouped_bars(
            directory / "transfer_bytes_by_case.png",
            transfer_points,
            title="Median Host-DPU Transfer by Case",
            ylabel="Median H2D + D2H bytes",
        )

    speedup_rows = list(speedups)
    if speedup_rows:
        points = [
            {
                "figure_id": "physical_speedup_by_case",
                "facet_id": str(row["scope_id"]),
                "series_id": "|".join(
                    str(row[field])
                    for field in (
                        "plan_id",
                        "candidate_route_id",
                        "numeric_policy",
                    )
                ),
                "series_label": _plan_label(row["plan_id"])
                + " | "
                + _humanize(str(row["candidate_route_id"]))
                + " | "
                + _numeric_label(row["numeric_policy"]),
                "x_value": str(row["case_id"]),
                "x_label": _humanize(str(row["case_id"])),
                "value": float(row["speedup"]),
                "ci_low": float(row["speedup_ci_low"]),
                "ci_high": float(row["speedup_ci_high"]),
                "accuracy_qualified": True,
            }
            for row in speedup_rows
        ]
        _plot_grouped_bars(
            directory / "physical_speedup_by_case.png",
            points,
            title="Physical UPMEM Speedup by Case",
            ylabel="NumPy median time / UPMEM median time",
            reference_line=1.0,
        )


def _point(
    *,
    figure_id: str,
    facet_id: str,
    row: Mapping[str, object],
    value: float,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> dict[str, object]:
    # A series identifies the route intervention. Plan hashes vary by case and
    # must not manufacture a separate legend entry for every x-axis value.
    qualified = row.get("accuracy_qualified") is True
    qualification_label = "accuracy-qualified" if qualified else "accuracy-unqualified"
    series_id = "|".join(
        "" if row[field] is None else str(row[field])
        for field in ("plan_id", "route_id", "numeric_policy")
    ) + "|" + qualification_label
    return {
        "figure_id": figure_id,
        "facet_id": facet_id,
        "series_id": series_id,
        "series_label": _series_label(row) + " | " + qualification_label,
        "x_value": str(row["case_id"]),
        "x_label": _humanize(str(row["case_id"])),
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "accuracy_qualified": qualified,
    }


def _plot_grouped_bars(
    path: Path,
    points: list[Mapping[str, object]],
    *,
    title: str,
    ylabel: str,
    reference_line: float | None = None,
) -> None:
    _assert_unique_plot_points(points)
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    x_values = sorted({str(point["x_value"]) for point in points})
    series_ids = sorted({str(point["series_id"]) for point in points})
    x_labels = {str(point["x_value"]): str(point["x_label"]) for point in points}
    series_labels = {
        str(point["series_id"]): str(point["series_label"]) for point in points
    }
    values = {
        (str(point["x_value"]), str(point["series_id"])): float(point["value"])
        for point in points
    }
    intervals = {
        (str(point["x_value"]), str(point["series_id"])): (
            _nullable_float(point.get("ci_low")),
            _nullable_float(point.get("ci_high")),
        )
        for point in points
    }
    qualified = {
        (str(point["x_value"]), str(point["series_id"])): point.get(
            "accuracy_qualified"
        ) is True
        for point in points
    }
    figure, axis = plt.subplots(figsize=(max(6.0, 1.4 * len(x_values)), 4.8))
    width = 0.8 / max(1, len(series_ids))
    centers = list(range(len(x_values)))
    for offset, series_id in enumerate(series_ids):
        positions = [center - 0.4 + width / 2 + offset * width for center in centers]
        heights = [
            values.get((x_value, series_id), float("nan")) for x_value in x_values
        ]
        yerr_low: list[float] = []
        yerr_high: list[float] = []
        for x_value, height in zip(x_values, heights, strict=True):
            lower, upper = intervals.get((x_value, series_id), (None, None))
            yerr_low.append(max(0.0, height - lower) if lower is not None else 0.0)
            yerr_high.append(max(0.0, upper - height) if upper is not None else 0.0)
        bars = axis.bar(
            positions,
            heights,
            width=width,
            label=series_labels[series_id],
            yerr=[yerr_low, yerr_high],
            capsize=3.0 if any(yerr_low) or any(yerr_high) else 0.0,
        )
        for bar, x_value in zip(bars, x_values, strict=True):
            if not qualified.get((x_value, series_id), False):
                bar.set_hatch("//")
    axis.set_xticks(centers, [x_labels[value] for value in x_values])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if reference_line is not None:
        axis.axhline(reference_line, color="black", linewidth=1.0)
    if len(series_ids) > 1:
        longest_label = max(len(series_labels[series_id]) for series_id in series_ids)
        columns = 1 if longest_label > 45 else min(2, len(series_ids))
        rows = (len(series_ids) + columns - 1) // columns
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=columns,
            fontsize=8,
        )
        figure.tight_layout(rect=(0.0, min(0.22, 0.08 + 0.04 * rows), 1.0, 1.0))
    else:
        figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _assert_unique_plot_points(points: Iterable[Mapping[str, object]]) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    for point in points:
        key = tuple(
            str(point[field])
            for field in ("figure_id", "facet_id", "series_id", "x_value")
        )
        if key in seen:
            raise ValueError(f"duplicate plot point: {key}")
        seen.add(key)


def _humanize(value: str) -> str:
    known = {
        "numpy_cpu_v1": "NumPy CPU",
        "split_complex_float32_v1": "Complex float32",
        "split_complex_int8_shared_scale_v1": "Complex int8 shared-scale",
        "simulation_end_to_end_v1": "Simulation end-to-end",
        "steady_execution_v1": "Steady execution",
    }
    return known.get(value, value.replace("_", " ").replace("-", " ").title())


def _nullable_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _humanize_nullable(value: object) -> str:
    return _humanize(value) if isinstance(value, str) else "Unspecified numeric policy"


def _series_label(row: Mapping[str, object]) -> str:
    parts = [
        _plan_label(row["plan_id"]),
        _humanize(str(row["route_id"])),
        _numeric_label(row["numeric_policy"]),
    ]
    active_dpus = row.get("median_active_dpus")
    tasklets = row.get("median_tasklets_per_dpu")
    if isinstance(active_dpus, (int, float)):
        count = int(active_dpus)
        parts.append(f"{count} active DPU" + ("s" if count != 1 else ""))
    if isinstance(tasklets, (int, float)):
        count = int(tasklets)
        parts.append(f"{count} tasklet" + ("s" if count != 1 else "") + "/DPU")
    return " | ".join(parts)


def _plan_label(value: object) -> str:
    return _humanize(value) if isinstance(value, str) else "Route-owned plan"


def _numeric_label(value: object) -> str:
    known = {
        "split_complex_float32_v1": "float32",
        "split_complex_int8_shared_scale_v1": "int8 shared-scale",
    }
    return known.get(value, _humanize_nullable(value))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(path, (canonical_json(value) + "\n").encode("utf-8"))


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]
) -> None:
    lines: list[str] = []
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    lines.append(stream.getvalue())
    _atomic_write(path, "".join(lines).encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = ["report_artifacts", "verify_artifacts"]
