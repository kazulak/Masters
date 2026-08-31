#!/usr/bin/env python3
"""Attribute native-host boundary costs from canonical UPMEM evidence."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from quantum_bench.evidence import load_artifacts
from quantum_bench.report import verify_artifacts


ANALYSIS_VERSION = "native_host_boundary_attribution_v1"
EPSILON_S = 1e-6

_DIRECT_TIMINGS = {
    "request_wave_wall_s": "request_wave_wall_sum_s",
    "native_route_s": "rank_response_total_route_max_sum_s",
    "native_h2d_s": "rank_response_h2d_max_sum_s",
    "native_kernel_s": "rank_response_kernel_max_sum_s",
    "native_d2h_s": "rank_response_d2h_max_sum_s",
    "request_build_s": "request_build_sum_s",
    "request_work_unit_materialization_s": (
        "request_work_unit_materialization_sum_s"
    ),
    "request_artifact_build_s": "request_artifact_build_sum_s",
    "payload_record_staging_s": "request_payload_record_staging_sum_s",
    "manifest_sidecar_staging_s": "request_manifest_sidecar_staging_sum_s",
    "payload_materialization_s": "request_payload_materialization_sum_s",
    "payload_file_write_s": "request_payload_file_write_sum_s",
    "payload_hashing_s": "request_payload_hashing_sum_s",
    "payload_record_construction_s": (
        "request_payload_record_construction_sum_s"
    ),
    "rank_submit_parallel_s": "rank_submit_parallel_wall_sum_s",
    "rank_submit_total_s": "rank_submit_total_max_sum_s",
    "rank_submit_artifact_validation_s": (
        "rank_submit_artifact_validation_max_sum_s"
    ),
    "rank_submit_protocol_write_s": "rank_submit_protocol_write_max_sum_s",
    "rank_submit_response_wait_s": "rank_submit_response_wait_max_sum_s",
    "rank_submit_response_validation_s": (
        "rank_submit_response_validation_max_sum_s"
    ),
    "coordinator_response_processing_s": "coordinator_response_processing_sum_s",
}

_REQUIRED_SAMPLE_TIMING = ("total_wall_s",)
_COUNTER_FIELDS = (
    "request_payload_record_count",
    "request_payload_files_created",
    "request_payload_bytes_staged",
    "request_payload_bytes_hashed",
)
_CSV_FIELDS = (
    "case_id",
    "plan_id",
    "route_id",
    "measurement_count",
    "total_operation_count",
    "median_operations_per_measurement",
    "median_request_count",
    "median_record_count",
    "median_payload_files_created",
    "median_payload_bytes_staged",
    "median_payload_bytes_hashed",
    "median_session_open_s",
    "median_session_close_s",
    "median_attempt_elapsed_s",
    "median_total_wall_s",
    "median_request_wave_wall_s",
    "median_native_route_s",
    "median_host_request_overhead_s",
    "median_native_request_overhead_s",
    "median_request_wave_residual_s",
    "confirmed_costs",
    "plausible_costs",
    "unknown_costs",
)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _seconds(value: object, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_integer(timing: Mapping[str, Any], field: str) -> int | None:
    if field not in timing or timing[field] is None:
        return None
    return _integer(timing[field], f"operation timing {field}")


def _difference(value: float, field: str) -> float:
    if value < -EPSILON_S:
        raise ValueError(f"{field} is materially negative")
    return max(0.0, value)


def _raw_mad(values: Sequence[float]) -> float:
    center = median(values)
    return float(median(abs(value - center) for value in values))


def _stats(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    return {
        "median_s": float(median(numeric)),
        "raw_mad_s": _raw_mad(numeric),
        "min_s": min(numeric),
        "max_s": max(numeric),
    }


def _optional_seconds(timing: Mapping[str, Any], field: str) -> float | None:
    if field not in timing or timing[field] is None:
        return None
    return _seconds(timing[field], f"operation timing {field}")


def _operation_attribution(operation: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one operation without adding inclusive timers together."""

    timing = _mapping(operation.get("timing"), "operation timing")
    values = {
        name: _optional_seconds(timing, field)
        for name, field in _DIRECT_TIMINGS.items()
    }
    values["operation_total_s"] = _optional_seconds(timing, "total_wall_s")

    costs: dict[str, dict[str, Any]] = {}
    for name, value in values.items():
        if value is not None:
            costs[name] = {
                "classification": "confirmed",
                "value_s": value,
                "source": (
                    "operation timing total_wall_s"
                    if name == "operation_total_s"
                    else f"operation timing {_DIRECT_TIMINGS[name]}"
                ),
            }

    def residual(name: str, parent: str, children: Sequence[str]) -> None:
        parent_value = values.get(parent)
        child_values = [values.get(child) for child in children]
        if parent_value is None or any(value is None for value in child_values):
            return
        costs[name] = {
            "classification": "plausible",
            "value_s": _difference(
                parent_value - sum(value for value in child_values if value is not None),
                name,
            ),
            "source": f"{parent} minus {', '.join(children)}",
        }

    counters = {
        field: _optional_integer(timing, field) for field in _COUNTER_FIELDS
    }

    residual(
        "host_request_overhead_s",
        "request_wave_wall_s",
        ("native_route_s",),
    )
    residual(
        "native_request_overhead_s",
        "native_route_s",
        ("native_h2d_s", "native_kernel_s", "native_d2h_s"),
    )
    residual(
        "operation_outside_request_wave_s",
        "operation_total_s",
        ("request_wave_wall_s",),
    )
    residual(
        "request_wave_residual_s",
        "request_wave_wall_s",
        (
            "request_build_s",
            "rank_submit_parallel_s",
            "coordinator_response_processing_s",
        ),
    )
    residual(
        "request_build_residual_s",
        "request_build_s",
        ("request_work_unit_materialization_s", "request_artifact_build_s"),
    )
    residual(
        "artifact_build_residual_s",
        "request_artifact_build_s",
        ("payload_record_staging_s", "manifest_sidecar_staging_s"),
    )
    residual(
        "payload_record_residual_s",
        "payload_record_staging_s",
        (
            "payload_materialization_s",
            "payload_file_write_s",
            "payload_hashing_s",
            "payload_record_construction_s",
        ),
    )
    residual(
        "rank_submit_internal_residual_s",
        "rank_submit_total_s",
        (
            "rank_submit_artifact_validation_s",
            "rank_submit_protocol_write_s",
            "rank_submit_response_wait_s",
            "rank_submit_response_validation_s",
        ),
    )
    residual(
        "rank_submit_parallel_residual_s",
        "rank_submit_parallel_s",
        ("rank_submit_total_s",),
    )

    return {
        "operation_count": 1,
        "rank_count": operation.get("rank_count"),
        "costs": costs,
        "counters": counters,
    }


def _sample_attribution(
    sample: Mapping[str, Any], session: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    facts = _mapping(sample.get("backend_facts"), "sample backend_facts")
    raw_operations = facts.get("operation_facts")
    if raw_operations is None:
        return None
    if not isinstance(raw_operations, Sequence) or isinstance(
        raw_operations, (str, bytes)
    ):
        raise ValueError("operation_facts must be a sequence")
    if not raw_operations:
        raise ValueError("operation_facts must not be empty")
    measurement = _mapping(sample.get("measurement"), "sample measurement")
    total_wall_s = _seconds(measurement.get("total_wall_s"), "total_wall_s")

    operation_results = [
        _operation_attribution(
            _mapping(operation, f"operation_facts[{index}]")
        )
        for index, operation in enumerate(raw_operations)
    ]
    cost_names = set(_DIRECT_TIMINGS) | {
        "operation_total_s",
        "host_request_overhead_s",
        "native_request_overhead_s",
        "operation_outside_request_wave_s",
        "request_wave_residual_s",
        "request_build_residual_s",
        "artifact_build_residual_s",
        "payload_record_residual_s",
        "rank_submit_internal_residual_s",
        "rank_submit_parallel_residual_s",
    }
    costs: dict[str, float | None] = {}
    for name in sorted(cost_names):
        operation_costs = [result["costs"].get(name) for result in operation_results]
        if all(cost is not None for cost in operation_costs):
            costs[name] = sum(cost["value_s"] for cost in operation_costs if cost)
        else:
            costs[name] = None

    counters: dict[str, int | None] = {}
    for field in _COUNTER_FIELDS:
        values_for_field = [
            result["counters"].get(field) for result in operation_results
        ]
        if all(value is not None for value in values_for_field):
            counters[field] = sum(
                int(value) for value in values_for_field if value is not None
            )
        else:
            counters[field] = None

    backend_facts = facts
    total_wave_count = backend_facts.get("total_wave_count")
    request_count = None
    if type(total_wave_count) is int and total_wave_count >= 0:
        request_count = total_wave_count * 4

    host_reduce = measurement.get("host_reduce_s")
    host_reduce_s = (
        None if host_reduce is None else _seconds(host_reduce, "host_reduce_s")
    )
    operation_total_s = costs.get("operation_total_s")
    if operation_total_s is not None and host_reduce_s is not None:
        costs["coordinator_other_s"] = _difference(
            total_wall_s - operation_total_s - host_reduce_s,
            "coordinator_other_s",
        )
    else:
        costs["coordinator_other_s"] = None

    session_costs: dict[str, float | None] = {}
    if session is not None:
        if session.get("status") != "success":
            raise ValueError("measurement session is not successful")
        if (
            session.get("case_id") != sample.get("case_id")
            or session.get("route_id") != sample.get("route_id")
        ):
            raise ValueError("measurement session does not match sample route")
        session_open = session.get("open_s")
        session_close = session.get("session_close_s")
        if session_open is None or session_close is None:
            raise ValueError("successful measurement session lacks open/close timing")
        session_costs["session_open_s"] = _seconds(session_open, "session open_s")
        session_costs["session_close_s"] = _seconds(
            session_close, "session_close_s"
        )
        session_costs["attempt_elapsed_s"] = (
            session_costs["session_open_s"]
            + total_wall_s
            + session_costs["session_close_s"]
        )

    return {
        "case_id": sample.get("case_id"),
        "plan_id": sample.get("plan_id"),
        "route_id": sample.get("route_id"),
        "block_id": sample.get("block_id"),
        "total_wall_s": total_wall_s,
        "operation_count": len(operation_results),
        "costs": {"total_wall_s": total_wall_s, **costs, **session_costs},
        "counters": {
            "request_count": request_count,
            **counters,
        },
    }


def _cost_summary(
    values: Sequence[Mapping[str, Any]],
    name: str,
    parent: str | None = None,
) -> dict[str, Any]:
    observed = [value["costs"].get(name) for value in values]
    numeric = [float(value) for value in observed if value is not None]
    if len(numeric) != len(values):
        return {
            "classification": "unknown",
            "source": "timing fact unavailable for one or more measurements",
            "observed_measurement_count": len(numeric),
        }
    result: dict[str, Any] = {
        "classification": (
            "confirmed"
            if name in _DIRECT_TIMINGS or name in {"total_wall_s", "session_open_s", "session_close_s"}
            else "plausible"
        ),
        **_stats(numeric),
        "median_total_share": float(
            median(
                value / sample["total_wall_s"] if sample["total_wall_s"] else 0.0
                for value, sample in zip(numeric, values, strict=True)
            )
        ),
    }
    if parent is not None:
        parent_values = [sample["costs"].get(parent) for sample in values]
        if all(value is not None and value != 0.0 for value in parent_values):
            result["median_parent_share"] = float(
                median(
                    value / parent_value
                    for value, parent_value in zip(
                        numeric, parent_values, strict=True
                    )
                )
            )
        else:
            result["median_parent_share"] = None
    return result


def _classify_costs(costs: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    result = {name: [] for name in ("confirmed", "plausible", "unknown")}
    for name in sorted(costs):
        result[costs[name]["classification"]].append(name)
    return result


def _summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {name for value in values for name in value["costs"]}
    )
    costs = {
        name: _cost_summary(
            values,
            name,
            parent={
                "native_h2d_s": "native_route_s",
                "native_kernel_s": "native_route_s",
                "native_d2h_s": "native_route_s",
                "native_request_overhead_s": "native_route_s",
                "host_request_overhead_s": "request_wave_wall_s",
                "request_build_s": "request_wave_wall_s",
                "rank_submit_parallel_s": "request_wave_wall_s",
                "coordinator_response_processing_s": "request_wave_wall_s",
                "request_wave_residual_s": "request_wave_wall_s",
                "request_work_unit_materialization_s": "request_build_s",
                "request_artifact_build_s": "request_build_s",
                "request_build_residual_s": "request_build_s",
                "payload_record_staging_s": "request_artifact_build_s",
                "manifest_sidecar_staging_s": "request_artifact_build_s",
                "artifact_build_residual_s": "request_artifact_build_s",
                "payload_materialization_s": "payload_record_staging_s",
                "payload_file_write_s": "payload_record_staging_s",
                "payload_hashing_s": "payload_record_staging_s",
                "payload_record_construction_s": "payload_record_staging_s",
                "payload_record_residual_s": "payload_record_staging_s",
            }.get(name),
        )
        for name in names
    }
    classification = _classify_costs(costs)
    operation_counts = [int(value["operation_count"]) for value in values]
    counter_summary: dict[str, dict[str, float] | None] = {}
    counter_names = ("request_count", *_COUNTER_FIELDS)
    for name in counter_names:
        counter_values = [value["counters"].get(name) for value in values]
        if any(value is None for value in counter_values):
            counter_summary[name] = None
        else:
            numeric = [float(value) for value in counter_values if value is not None]
            counter_summary[name] = {
                "median": float(median(numeric)),
                "raw_mad": _raw_mad(numeric),
                "min": min(numeric),
                "max": max(numeric),
            }
    return {
        "measurement_count": len(values),
        "total_operation_count": sum(operation_counts),
        "operation_count": {
            "median": float(median(operation_counts)),
            "raw_mad": _raw_mad([float(count) for count in operation_counts]),
            "min": min(operation_counts),
            "max": max(operation_counts),
        },
        "counters": counter_summary,
        "costs": costs,
        "confirmed_costs": classification["confirmed"],
        "plausible_costs": classification["plausible"],
        "unknown_costs": classification["unknown"],
    }


def _row(
    case_id: str, route_id: str, values: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    plan_ids = {value["plan_id"] for value in values}
    if len(plan_ids) != 1:
        raise ValueError("native host boundary plan changed across measurements")
    summary = _summary(values)
    costs = summary["costs"]

    def median_cost(name: str) -> float | None:
        value = costs.get(name)
        return None if value is None or value["classification"] == "unknown" else value["median_s"]

    return {
        "case_id": case_id,
        "plan_id": next(iter(plan_ids)),
        "route_id": route_id,
        "measurement_count": summary["measurement_count"],
        "total_operation_count": summary["total_operation_count"],
        "median_operations_per_measurement": summary["operation_count"]["median"],
        "median_request_count": (
            None if summary["counters"]["request_count"] is None
            else summary["counters"]["request_count"]["median"]
        ),
        "median_record_count": (
            None if summary["counters"]["request_payload_record_count"] is None
            else summary["counters"]["request_payload_record_count"]["median"]
        ),
        "median_payload_files_created": (
            None if summary["counters"]["request_payload_files_created"] is None
            else summary["counters"]["request_payload_files_created"]["median"]
        ),
        "median_payload_bytes_staged": (
            None if summary["counters"]["request_payload_bytes_staged"] is None
            else summary["counters"]["request_payload_bytes_staged"]["median"]
        ),
        "median_payload_bytes_hashed": (
            None if summary["counters"]["request_payload_bytes_hashed"] is None
            else summary["counters"]["request_payload_bytes_hashed"]["median"]
        ),
        "median_session_open_s": median_cost("session_open_s"),
        "median_session_close_s": median_cost("session_close_s"),
        "median_attempt_elapsed_s": median_cost("attempt_elapsed_s"),
        "median_total_wall_s": median_cost("total_wall_s"),
        "median_request_wave_wall_s": median_cost("request_wave_wall_s"),
        "median_native_route_s": median_cost("native_route_s"),
        "median_host_request_overhead_s": median_cost("host_request_overhead_s"),
        "median_native_request_overhead_s": median_cost("native_request_overhead_s"),
        "median_request_wave_residual_s": median_cost("request_wave_residual_s"),
        "confirmed_costs": ";".join(summary["confirmed_costs"]),
        "plausible_costs": ";".join(summary["plausible_costs"]),
        "unknown_costs": ";".join(summary["unknown_costs"]),
        "summary": summary,
    }


def derive_attribution(
    manifest: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive boundary facts from successful measurement samples only."""

    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("manifest source_commit is invalid")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("manifest experiment_id is invalid")

    sessions_by_id: dict[str, Mapping[str, Any]] = {}
    for session in sessions or ():
        session_id = session.get("session_instance_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_instance_id is required")
        if session_id in sessions_by_id:
            raise ValueError(f"duplicate session_instance_id: {session_id}")
        sessions_by_id[session_id] = session

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        if sample.get("status") != "success" or sample.get("attempt_kind") != "measurement":
            continue
        session: Mapping[str, Any] | None = None
        if sessions is not None:
            session_id = sample.get("session_instance_id")
            if not isinstance(session_id, str) or session_id not in sessions_by_id:
                raise ValueError("measurement has no matching session")
            session = sessions_by_id[session_id]
        attribution = _sample_attribution(sample, session)
        if attribution is None:
            continue
        case_id = attribution.get("case_id")
        route_id = attribution.get("route_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("sample case_id is invalid")
        if not isinstance(route_id, str) or not route_id:
            raise ValueError("sample route_id is invalid")
        grouped.setdefault((case_id, route_id), []).append(attribution)
    if not grouped:
        raise ValueError("evidence contains no successful native operation measurements")

    rows = [
        _row(case_id, route_id, grouped[(case_id, route_id)])
        for case_id, route_id in sorted(grouped)
    ]
    return {
        "analysis_version": ANALYSIS_VERSION,
        "source_commit": source_commit,
        "experiment_id": experiment_id,
        "measurement_cells": rows,
        "classification_policy": {
            "confirmed": "persisted timing or counter fact",
            "plausible": "non-negative residual from nested persisted facts",
            "unknown": "required fact absent from at least one measurement",
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Native Host Boundary Attribution",
        "",
        f"Source: `{result['source_commit']}`  ",
        f"Experiment: `{result['experiment_id']}`  ",
        "Measurements only; warmups and unsuccessful attempts are excluded. "
        "Inclusive timers are shown as nested diagnostics and are never summed "
        "with their parents.",
        "",
        "| Case | Route | Measurements | Operations | Request wave (s) | Native route (s) | Host overhead (s) | Native overhead (s) | Wave residual (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["measurement_cells"]:
        def display(name: str) -> str:
            value = row[name]
            return "unknown" if value is None else f"{value:.6f}"

        lines.append(
            f"| {row['case_id']} | {row['route_id']} | {row['measurement_count']} | "
            f"{row['total_operation_count']} | {display('median_request_wave_wall_s')} | "
            f"{display('median_native_route_s')} | {display('median_host_request_overhead_s')} | "
            f"{display('median_native_request_overhead_s')} | "
            f"{display('median_request_wave_residual_s')} |"
        )
    lines.extend(
        [
            "",
            "Classification: `confirmed` is persisted evidence; `plausible` is a "
            "non-negative subtraction within an inclusive parent; `unknown` means "
            "the required timing fact was not recorded.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"analysis output directory must be empty: {output_dir}")
    verification = verify_artifacts(input_dir)
    manifest, samples, sessions = load_artifacts(input_dir)
    result = derive_attribution(manifest, samples, sessions)
    result["verification"] = verification
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "native_host_boundary_attribution.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "native_host_boundary_attribution.csv", result["measurement_cells"])
    _write_markdown(output_dir / "native_host_boundary_attribution.md", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.input, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
