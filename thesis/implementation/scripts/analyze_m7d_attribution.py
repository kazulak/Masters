#!/usr/bin/env python3
"""Derive disjoint UPMEM request-lifecycle timing attribution from evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from quantum_bench.evidence import load_artifacts
from quantum_bench.report import verify_artifacts


_ANALYSIS_VERSION = "m7g_request_build_attribution_v1"
_EPSILON_S = 1e-6
_OPERATION_TIMING_FIELDS = (
    "total_wall_s",
    "preparation_s",
    "encode_s",
    "rank_response_h2d_max_sum_s",
    "rank_response_kernel_max_sum_s",
    "rank_response_d2h_max_sum_s",
    "rank_response_total_route_max_sum_s",
    "request_wave_wall_sum_s",
    "request_build_sum_s",
    "rank_submit_parallel_wall_sum_s",
    "rank_submit_total_max_sum_s",
    "rank_submit_artifact_validation_max_sum_s",
    "rank_submit_protocol_write_max_sum_s",
    "rank_submit_response_wait_max_sum_s",
    "rank_submit_response_validation_max_sum_s",
    "coordinator_response_processing_sum_s",
    "assembly_s",
    "decode_s",
)
_COMPONENT_FIELDS = (
    "preparation_s",
    "encode_s",
    "host_request_overhead_s",
    "native_request_overhead_s",
    "h2d_s",
    "kernel_s",
    "d2h_s",
    "assembly_s",
    "decode_s",
    "operation_other_s",
    "host_reduce_s",
    "coordinator_other_s",
)
_REQUEST_BUILD_TIMING_FIELDS = (
    "request_work_unit_materialization_sum_s",
    "request_artifact_build_sum_s",
    "request_payload_record_staging_sum_s",
    "request_manifest_sidecar_staging_sum_s",
)
_REQUEST_BUILD_CHILD_FIELDS = (
    "work_unit_materialization_s",
    "payload_record_staging_s",
    "manifest_sidecar_staging_s",
    "artifact_build_residual_s",
    "request_build_residual_s",
)


def _seconds(value: object, *, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _difference(value: float, *, field: str) -> float:
    if value < -_EPSILON_S:
        raise ValueError(f"{field} is materially negative")
    return max(0.0, value)


def _request_build_details(
    timing: Mapping[str, Any], values: Mapping[str, float]
) -> dict[str, float] | None:
    present = [field in timing for field in _REQUEST_BUILD_TIMING_FIELDS]
    if not any(present):
        return None
    if not all(present):
        missing = next(
            field
            for field in _REQUEST_BUILD_TIMING_FIELDS
            if field not in timing
        )
        raise ValueError(f"request-build timing is missing {missing}")
    request_values = {
        field: _seconds(timing[field], field=f"operation timing {field}")
        for field in _REQUEST_BUILD_TIMING_FIELDS
    }
    artifact_build_residual_s = _difference(
        request_values["request_artifact_build_sum_s"]
        - request_values["request_payload_record_staging_sum_s"]
        - request_values["request_manifest_sidecar_staging_sum_s"],
        field="request artifact build residual",
    )
    request_build_residual_s = _difference(
        values["request_build_sum_s"]
        - request_values["request_work_unit_materialization_sum_s"]
        - request_values["request_artifact_build_sum_s"],
        field="request build residual",
    )
    return {
        "work_unit_materialization_s": request_values[
            "request_work_unit_materialization_sum_s"
        ],
        "payload_record_staging_s": request_values[
            "request_payload_record_staging_sum_s"
        ],
        "manifest_sidecar_staging_s": request_values[
            "request_manifest_sidecar_staging_sum_s"
        ],
        "artifact_build_residual_s": artifact_build_residual_s,
        "request_build_residual_s": request_build_residual_s,
    }


def _operation_components(
    operation: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float] | None, dict[str, float]]:
    if operation.get("rank_count") != 1:
        raise ValueError("M7F host request attribution requires one rank")
    timing = _mapping(operation.get("timing"), field="operation timing")
    missing = [field for field in _OPERATION_TIMING_FIELDS if field not in timing]
    if missing:
        raise ValueError(f"operation timing is missing {missing[0]}")
    values = {
        field: _seconds(timing[field], field=f"operation timing {field}")
        for field in _OPERATION_TIMING_FIELDS
    }
    native_request_overhead_s = _difference(
        values["rank_response_total_route_max_sum_s"]
        - values["rank_response_h2d_max_sum_s"]
        - values["rank_response_kernel_max_sum_s"]
        - values["rank_response_d2h_max_sum_s"],
        field="native request overhead",
    )
    host_request_overhead_s = _difference(
        values["request_wave_wall_sum_s"]
        - values["rank_response_total_route_max_sum_s"],
        field="host request overhead",
    )
    rank_submit_internal_residual_s = _difference(
        values["rank_submit_total_max_sum_s"]
        - values["rank_submit_artifact_validation_max_sum_s"]
        - values["rank_submit_protocol_write_max_sum_s"]
        - values["rank_submit_response_wait_max_sum_s"]
        - values["rank_submit_response_validation_max_sum_s"],
        field="rank submit internal residual",
    )
    rank_submit_parallel_residual_s = _difference(
        values["rank_submit_parallel_wall_sum_s"]
        - values["rank_submit_total_max_sum_s"],
        field="rank submit parallel residual",
    )
    request_wave_residual_s = _difference(
        values["request_wave_wall_sum_s"]
        - values["request_build_sum_s"]
        - values["rank_submit_parallel_wall_sum_s"]
        - values["coordinator_response_processing_sum_s"],
        field="request wave residual",
    )
    operation_other_s = _difference(
        values["total_wall_s"]
        - values["preparation_s"]
        - values["encode_s"]
        - values["request_wave_wall_sum_s"]
        - values["assembly_s"]
        - values["decode_s"],
        field="operation other",
    )
    components = {
        "preparation_s": values["preparation_s"],
        "encode_s": values["encode_s"],
        "host_request_overhead_s": host_request_overhead_s,
        "native_request_overhead_s": native_request_overhead_s,
        "h2d_s": values["rank_response_h2d_max_sum_s"],
        "kernel_s": values["rank_response_kernel_max_sum_s"],
        "d2h_s": values["rank_response_d2h_max_sum_s"],
        "assembly_s": values["assembly_s"],
        "decode_s": values["decode_s"],
        "operation_other_s": operation_other_s,
    }
    components["operation_accounting_residual_s"] = _difference(
        values["total_wall_s"] - sum(components.values()),
        field="operation accounting residual",
    )
    nested = {
        "request_wave_wall_sum_s": values["request_wave_wall_sum_s"],
        "rank_response_total_route_max_sum_s": values[
            "rank_response_total_route_max_sum_s"
        ],
        "rank_submit_parallel_wall_sum_s": values[
            "rank_submit_parallel_wall_sum_s"
        ],
        "rank_submit_total_max_sum_s": values["rank_submit_total_max_sum_s"],
        "rank_submit_artifact_validation_max_sum_s": values[
            "rank_submit_artifact_validation_max_sum_s"
        ],
        "rank_submit_protocol_write_max_sum_s": values[
            "rank_submit_protocol_write_max_sum_s"
        ],
        "rank_submit_response_wait_max_sum_s": values[
            "rank_submit_response_wait_max_sum_s"
        ],
        "rank_submit_response_validation_max_sum_s": values[
            "rank_submit_response_validation_max_sum_s"
        ],
        "rank_submit_internal_residual_s": rank_submit_internal_residual_s,
        "rank_submit_parallel_residual_s": rank_submit_parallel_residual_s,
        "coordinator_response_processing_sum_s": values[
            "coordinator_response_processing_sum_s"
        ],
        "request_wave_residual_s": request_wave_residual_s,
        "request_build_parent_s": values["request_build_sum_s"],
    }
    return components, _request_build_details(timing, values), nested


def _sample_components(sample: Mapping[str, Any]) -> dict[str, float] | None:
    facts = _mapping(sample.get("backend_facts"), field="sample backend_facts")
    operations = facts.get("operation_facts")
    if operations is None:
        return None
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("operation_facts must be a sequence")
    if not operations:
        raise ValueError("operation_facts must not be empty")
    measurement = _mapping(sample.get("measurement"), field="sample measurement")
    total_wall_s = _seconds(measurement.get("total_wall_s"), field="total_wall_s")
    host_reduce_value = measurement.get("host_reduce_s")
    host_reduce_s = (
        0.0
        if host_reduce_value is None
        else _seconds(host_reduce_value, field="host_reduce_s")
    )
    result = {field: 0.0 for field in _COMPONENT_FIELDS}
    nested = {
        "request_wave_wall_sum_s": 0.0,
        "rank_response_total_route_max_sum_s": 0.0,
        "rank_submit_parallel_wall_sum_s": 0.0,
        "rank_submit_total_max_sum_s": 0.0,
        "rank_submit_artifact_validation_max_sum_s": 0.0,
        "rank_submit_protocol_write_max_sum_s": 0.0,
        "rank_submit_response_wait_max_sum_s": 0.0,
        "rank_submit_response_validation_max_sum_s": 0.0,
        "rank_submit_internal_residual_s": 0.0,
        "rank_submit_parallel_residual_s": 0.0,
        "coordinator_response_processing_sum_s": 0.0,
        "request_wave_residual_s": 0.0,
        "request_build_parent_s": 0.0,
    }
    request_build_children = {field: 0.0 for field in _REQUEST_BUILD_CHILD_FIELDS}
    request_build_available: bool | None = None
    operation_total_s = 0.0
    for index, operation in enumerate(operations):
        components, request_build_details, operation_nested = _operation_components(
            _mapping(operation, field=f"operation_facts[{index}]")
        )
        operation_total_s += _seconds(
            _mapping(operation.get("timing"), field="operation timing")["total_wall_s"],
            field="operation total_wall_s",
        )
        for field in _COMPONENT_FIELDS:
            if field in components:
                result[field] += components[field]
        for field, value in operation_nested.items():
            nested[field] += value
        available = request_build_details is not None
        if request_build_available is None:
            request_build_available = available
        elif request_build_available != available:
            raise ValueError("request-build timing must be present for every operation")
        if request_build_details is not None:
            for field, value in request_build_details.items():
                request_build_children[field] += value
    result["host_reduce_s"] = host_reduce_s
    result["coordinator_other_s"] = _difference(
        total_wall_s - operation_total_s - host_reduce_s,
        field="coordinator other",
    )
    result["total_wall_s"] = total_wall_s
    result.update(nested)
    result.update(request_build_children)
    result["request_build_breakdown_available"] = bool(request_build_available)
    result["accounting_residual_s"] = _difference(
        total_wall_s - sum(result[field] for field in _COMPONENT_FIELDS),
        field="sample accounting residual",
    )
    result["unresolved_boundary_s"] = (
        result["operation_other_s"] + result["coordinator_other_s"]
    )
    return result


def _raw_mad(values: Sequence[float]) -> float:
    center = median(values)
    return float(median(abs(value - center) for value in values))


def _route_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    totals = [value["total_wall_s"] for value in values]
    median_total = float(median(totals))
    components = {
        field: {
            "median_s": float(median([value[field] for value in values])),
            "median_share": float(
                median(value[field] / value["total_wall_s"] for value in values)
            ),
        }
        for field in _COMPONENT_FIELDS
    }
    request_build_breakdown: dict[str, object] | None = None
    if all(value["request_build_breakdown_available"] for value in values):
        request_build_breakdown = {
            "median_parent_s": float(
                median(value["request_build_parent_s"] for value in values)
            ),
            "median_parent_share": float(
                median(
                    value["request_build_parent_s"] / value["total_wall_s"]
                    for value in values
                )
            ),
            "children": {
                field: {
                    "median_s": float(median(value[field] for value in values)),
                    "median_parent_share": float(
                        median(
                            value[field] / value["request_build_parent_s"]
                            for value in values
                        )
                    ),
                    "median_total_share": float(
                        median(value[field] / value["total_wall_s"] for value in values)
                    ),
                }
                for field in _REQUEST_BUILD_CHILD_FIELDS
            },
        }
    return {
        "measurement_count": len(values),
        "median_total_wall_s": median_total,
        "raw_mad_total_wall_s": _raw_mad(totals),
        "components": components,
        "nested_request_timing_medians_s": {
            field: float(median(value[field] for value in values))
            for field in (
                "request_wave_wall_sum_s",
                "rank_response_total_route_max_sum_s",
                "rank_submit_parallel_wall_sum_s",
                "rank_submit_total_max_sum_s",
                "rank_submit_artifact_validation_max_sum_s",
                "rank_submit_protocol_write_max_sum_s",
                "rank_submit_response_wait_max_sum_s",
                "rank_submit_response_validation_max_sum_s",
                "rank_submit_internal_residual_s",
                "rank_submit_parallel_residual_s",
                "coordinator_response_processing_sum_s",
                "request_wave_residual_s",
            )
        },
        "request_build_breakdown": request_build_breakdown,
        "median_unresolved_boundary_s": float(
            median(value["unresolved_boundary_s"] for value in values)
        ),
        "median_accounting_residual_s": float(
            median(value["accounting_residual_s"] for value in values)
        ),
    }


def derive_attribution(
    manifest: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, object]:
    """Return attribution statistics from successful UPMEM measurements."""

    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("manifest source_commit is invalid")
    route_samples: dict[str, list[dict[str, float]]] = {}
    for sample in samples:
        if sample.get("status") != "success" or sample.get("attempt_kind") != "measurement":
            continue
        result = _sample_components(sample)
        if result is None:
            continue
        route_id = sample.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            raise ValueError("UPMEM sample route_id is invalid")
        route_samples.setdefault(route_id, []).append(result)
    if not route_samples:
        raise ValueError("evidence contains no successful UPMEM measurements")
    return {
        "analysis_version": _ANALYSIS_VERSION,
        "source_commit": source_commit,
        "routes": {
            route_id: _route_summary(route_samples[route_id])
            for route_id in sorted(route_samples)
        },
    }


def analyze(input_dir: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"attribution output must be absent: {output}")
    verification = verify_artifacts(input_dir)
    manifest, samples, _ = load_artifacts(input_dir)
    result = derive_attribution(manifest, samples)
    result["verification"] = verification
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.input, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
