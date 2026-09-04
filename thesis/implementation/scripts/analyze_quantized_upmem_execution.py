#!/usr/bin/env python3
"""Derive the fixed quantized-UPMEM diagnostic tables from canonical evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_bench.circuits import builtin_circuit  # noqa: E402
from quantum_bench.evidence import load_artifacts  # noqa: E402
from quantum_bench.lowering import build_contraction_dag, lower_tensor_network  # noqa: E402
from quantum_bench.model import make_simulation_job  # noqa: E402
from quantum_bench.planning import plan_opt_einsum  # noqa: E402
from quantum_bench.quantized_contraction import replay_quantized_dag  # noqa: E402
from quantum_bench.report import verify_artifacts  # noqa: E402


POLICY = "complex_int8_shared_scale_v1"
FLOAT32 = "split_complex_float32_v1"
SCHEMA = "quantized_upmem_execution_analysis_v2"
OUTPUTS = (
    "quantized_upmem_summary.json",
    "quantized_upmem_routes.csv",
    "quantized_upmem_components.csv",
    "quantized_upmem_error_runtime.csv",
    "quantized_upmem_resource_optima.csv",
    "quantized_upmem_analysis.md",
    "quantized_upmem_sample_components.csv",
    "quantized_upmem_timing_envelopes.csv",
    "quantized_upmem_fixed_route_comparisons.csv",
)
RESIDUAL_TOLERANCE_S = 1e-6
MEASUREMENT_FIELDS = (
    "total_wall_s",
    "encode_s",
    "preparation_s",
    "h2d_s",
    "kernel_s",
    "host_reduce_s",
    "d2h_s",
    "decode_s",
    "h2d_bytes",
    "d2h_bytes",
)

# These are the raw operation timing fields required for one-rank attribution.
# The names are evidence fields, so preserve them exactly when aggregating.
OPERATION_FIELDS = (
    "total_wall_s",
    "preparation_s",
    "encode_s",
    "rank_response_h2d_max_sum_s",
    "rank_response_kernel_max_sum_s",
    "rank_response_d2h_max_sum_s",
    "rank_response_total_route_max_sum_s",
    "request_wave_wall_sum_s",
    "request_build_sum_s",
    "request_work_unit_materialization_sum_s",
    "request_artifact_build_sum_s",
    "request_payload_record_staging_sum_s",
    "request_manifest_sidecar_staging_sum_s",
    "request_payload_materialization_sum_s",
    "request_payload_file_write_sum_s",
    "request_payload_hashing_sum_s",
    "request_payload_record_construction_sum_s",
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
ENVELOPE_FIELDS = (
    "request_wave_wall_sum_s",
    "request_build_sum_s",
    "request_work_unit_materialization_sum_s",
    "request_artifact_build_sum_s",
    "request_payload_record_staging_sum_s",
    "request_manifest_sidecar_staging_sum_s",
    "request_payload_materialization_sum_s",
    "request_payload_file_write_sum_s",
    "request_payload_hashing_sum_s",
    "request_payload_record_construction_sum_s",
    "rank_submit_parallel_wall_sum_s",
    "rank_submit_total_max_sum_s",
    "rank_submit_artifact_validation_max_sum_s",
    "rank_submit_protocol_write_max_sum_s",
    "rank_submit_response_wait_max_sum_s",
    "rank_submit_response_validation_max_sum_s",
    "rank_response_total_route_max_sum_s",
    "rank_response_h2d_max_sum_s",
    "rank_response_kernel_max_sum_s",
    "rank_response_d2h_max_sum_s",
    "coordinator_response_processing_sum_s",
)
RAW_OPERATION_SUM_FIELDS = ("operation_total_s", "assembly_s", *ENVELOPE_FIELDS)
OPERATION_COMPONENT_FIELDS = (
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
)
DISJOINT_COMPONENT_FIELDS = (*OPERATION_COMPONENT_FIELDS, "host_reduce_s", "coordinator_other_s")
HOST_COMPONENT_FIELDS = (
    "preparation_s",
    "encode_s",
    "host_request_overhead_s",
    "assembly_s",
    "decode_s",
    "operation_other_s",
    "host_reduce_s",
    "coordinator_other_s",
)
SAMPLE_COMPONENT_OUTPUT_FIELDS = (
    "case_id",
    "route_id",
    "numeric_policy",
    "dpu_count",
    "tasklets_per_dpu",
    "block_id",
    "session_open_s",
    "session_close_s",
    "session_inclusive_s",
    "total_wall_s",
    "operation_total_s",
    *DISJOINT_COMPONENT_FIELDS,
    "accounting_residual_s",
    "operation_count",
    *ENVELOPE_FIELDS,
)
CASE_SPECS = {
    "ghz18": ("ghz_chain", {"n_qubits": 18}),
    "hs18": ("hs", {"n_qubits": 18, "depth": 1}),
    "stress18": ("quantization_stress", {"n_qubits": 18, "repeat_layers": 2}),
}


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} is not finite and nonnegative")
    return result


def _difference(value: float, field: str) -> float:
    """Return a non-negative residual, rejecting real accounting failures."""

    if not math.isfinite(value):
        raise ValueError(f"{field} is not finite")
    if value < -RESIDUAL_TOLERANCE_S:
        raise ValueError(f"{field} is materially negative")
    return max(0.0, float(value))


def _mad(values: Sequence[float]) -> float:
    center = float(median(values))
    return float(median(abs(value - center) for value in values))


def _stats(values: Sequence[float]) -> tuple[float | None, float | None]:
    return (float(median(values)), _mad(values)) if values else (None, None)


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _session_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get("session_instance_id"))
        if not key or key in result:
            raise ValueError("session IDs must be present and unique")
        result[key] = row
    return result


def _operation_attribution(
    operation: Mapping[str, Any], *, index: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Validate one operation and derive its disjoint timing components."""

    if operation.get("rank_count") != 1:
        raise ValueError("quantized timing attribution requires one rank")
    timing = operation.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError(f"operation_facts[{index}] timing is missing")
    missing = [field for field in OPERATION_FIELDS if field not in timing]
    if missing:
        raise ValueError(
            f"operation_facts[{index}] timing is missing {missing[0]}"
        )
    values = {
        field: _number(timing[field], f"operation timing {field}")
        for field in OPERATION_FIELDS
    }
    native_request_overhead_s = _difference(
        values["rank_response_total_route_max_sum_s"]
        - values["rank_response_h2d_max_sum_s"]
        - values["rank_response_kernel_max_sum_s"]
        - values["rank_response_d2h_max_sum_s"],
        "native request overhead",
    )
    host_request_overhead_s = _difference(
        values["request_wave_wall_sum_s"]
        - values["rank_response_total_route_max_sum_s"],
        "host request overhead",
    )
    operation_other_s = _difference(
        values["total_wall_s"]
        - values["preparation_s"]
        - values["encode_s"]
        - values["request_wave_wall_sum_s"]
        - values["assembly_s"]
        - values["decode_s"],
        "operation other",
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
    return values, components


def _operation_sums(
    facts: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float], int]:
    """Return raw operation sums, disjoint operation components, and count."""

    operations = facts.get("operation_facts")
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("measurement lacks operation_facts")
    if not operations:
        raise ValueError("operation_facts must not be empty")
    totals = {field: 0.0 for field in RAW_OPERATION_SUM_FIELDS}
    components = {field: 0.0 for field in OPERATION_COMPONENT_FIELDS}
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            raise ValueError("operation fact is not a mapping")
        values, operation_components = _operation_attribution(
            operation, index=index
        )
        totals["operation_total_s"] += values["total_wall_s"]
        totals["assembly_s"] += values["assembly_s"]
        for field in ENVELOPE_FIELDS:
            totals[field] += values[field]
        for field in OPERATION_COMPONENT_FIELDS:
            components[field] += operation_components[field]
    return totals, components, len(operations)


def _logical_facts(numeric: Mapping[str, Any]) -> dict[str, int]:
    records = numeric.get("operand_records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("numeric facts lack operand records")
    values = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("operand record is not a mapping")
        shape = record.get("shape")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise ValueError("operand shape is missing")
        values += _product(int(item) for item in shape)
    return {
        "quantized_complex_values": values,
        "scalar_components_quantized": 2 * values,
        "scale_reduction_count": len(records),
        "scale_metadata_bytes": 8 * len(records),
        "logical_int8_operand_bytes": 2 * values + 8 * len(records),
        "logical_float32_operand_bytes": 8 * values,
    }


def _host_reduce_s(measurement: Mapping[str, Any], case_id: str) -> float:
    """Read host reduction, using zero only for the known no-reduce DAGs."""

    if "host_reduce_s" not in measurement:
        raise ValueError("measurement is missing host_reduce_s")
    value = measurement["host_reduce_s"]
    if value is None:
        if case_id not in CASE_SPECS:
            raise ValueError(
                "host_reduce_s is unavailable and the DAG has not been asserted "
                "to have no ReduceNode"
            )
        return 0.0
    return _number(value, "host_reduce_s")


def _sample_row(
    sample: Mapping[str, Any],
    session: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    measurement = sample.get("measurement")
    facts = sample.get("backend_facts")
    numeric = sample.get("numeric_facts")
    validation = sample.get("validation")
    if not all(isinstance(value, Mapping) for value in (measurement, facts, numeric, validation)):
        raise ValueError("successful sample lacks measurement/facts/validation")
    assert isinstance(measurement, Mapping)
    assert isinstance(facts, Mapping)
    assert isinstance(numeric, Mapping)
    assert isinstance(validation, Mapping)
    case_id = str(sample["case_id"])
    open_s = _number(session.get("open_s"), "session.open_s")
    close_s = _number(session.get("session_close_s"), "session.session_close_s")
    total_s = _number(measurement.get("total_wall_s"), "total_wall_s")
    options = route["options"]
    logical = _logical_facts(numeric)
    operation_sums, operation_components, operation_count = _operation_sums(facts)
    host_reduce_s = _host_reduce_s(measurement, case_id)
    coordinator_other_s = _difference(
        total_s - operation_sums["operation_total_s"] - host_reduce_s,
        "coordinator other",
    )
    disjoint = {
        **operation_components,
        "host_reduce_s": host_reduce_s,
        "coordinator_other_s": coordinator_other_s,
    }
    accounting_residual_s = _difference(
        total_s - sum(disjoint.values()),
        "sample accounting residual",
    )
    row: dict[str, Any] = {
        "case_id": case_id,
        "route_id": str(sample["route_id"]),
        "numeric_policy": str(route["numeric_policy"]),
        "dpu_count": int(options["dpu_count"]),
        "tasklets_per_dpu": int(options["tasklets_per_dpu"]),
        "block_id": int(sample["block_id"]),
        "session_open_s": open_s,
        "session_close_s": close_s,
        "session_inclusive_s": open_s + total_s + close_s,
        "operation_count": operation_count,
        "accounting_residual_s": accounting_residual_s,
        "operation_total_s": operation_sums["operation_total_s"],
        "max_abs_error_vs_complex128": _number(
            validation.get("max_abs_error"), "validation.max_abs_error"
        ),
        "relative_l2_vs_complex128": _number(
            validation.get("relative_l2_error"), "validation.relative_l2_error"
        ),
        "norm_drift_vs_complex128": _number(
            validation.get("norm_drift"), "validation.norm_drift"
        ),
        "policy_reference_passed": validation.get("policy_reference_passed") is True,
        "saturation_real": int(numeric.get("saturation_real", 0)),
        "saturation_imag": int(numeric.get("saturation_imag", 0)),
        **logical,
    }
    for field in DISJOINT_COMPONENT_FIELDS:
        row[f"component_{field}"] = disjoint[field]
    row.update(
        {
            field: value
            for field, value in operation_sums.items()
            if field != "operation_total_s"
        }
    )
    for field in MEASUREMENT_FIELDS:
        value = measurement.get(field)
        if field == "host_reduce_s":
            row[field] = host_reduce_s
        elif value is None:
            raise ValueError(f"measurement is missing {field}")
        else:
            row[field] = _number(value, field)
    for field in (
        "arithmetic_weighted_tasklet_utilization",
        "arithmetic_weighted_dpu_slot_utilization",
        "dominant_work_wave_utilization",
    ):
        value = facts.get(field)
        row[field] = None if value is None else _number(value, field)
    return row


def _software_metrics(case_id: str) -> dict[str, Any]:
    name, parameters = CASE_SPECS[case_id]
    network, inputs = lower_tensor_network(
        make_simulation_job(builtin_circuit(name, parameters))
    )
    path, _ = plan_opt_einsum(network, optimize="greedy")
    replay = replay_quantized_dag(build_contraction_dag(network, path), inputs)
    return {
        "int8_max_abs_error_vs_float32_same_dag": replay.max_abs_error_vs_float32_same_dag,
        "int8_relative_l2_vs_float32_same_dag": replay.relative_l2_vs_float32_same_dag,
        "int8_norm_drift_vs_float32_same_dag": replay.norm_drift_vs_float32_same_dag,
        "int8_max_abs_error_vs_complex128": replay.max_abs_error_vs_complex128,
        "int8_relative_l2_vs_complex128": replay.relative_l2_vs_complex128,
        "int8_norm_drift_vs_complex128": replay.norm_drift_vs_complex128,
        "float32_max_abs_error_vs_complex128": replay.max_abs_error_float32_vs_complex128,
        "float32_relative_l2_vs_complex128": replay.relative_l2_float32_vs_complex128,
        "float32_norm_drift_vs_complex128": replay.norm_drift_float32_vs_complex128,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    result = {
        key: first[key]
        for key in ("case_id", "route_id", "numeric_policy", "dpu_count", "tasklets_per_dpu")
    }
    result["measurement_count"] = len(rows)
    statistic_fields = (
        *MEASUREMENT_FIELDS,
        "session_open_s",
        "session_close_s",
        "session_inclusive_s",
        *RAW_OPERATION_SUM_FIELDS,
    )
    for field in statistic_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        center, spread = _stats(values)
        result[f"median_{field}"] = center
        result[f"raw_mad_{field}"] = spread
    for field in DISJOINT_COMPONENT_FIELDS:
        values = [float(row[f"component_{field}"]) for row in rows]
        center, spread = _stats(values)
        result[f"component_median_{field}"] = center
        result[f"component_raw_mad_{field}"] = spread
    for field in (
        "max_abs_error_vs_complex128",
        "relative_l2_vs_complex128",
        "norm_drift_vs_complex128",
        "saturation_real",
        "saturation_imag",
        "quantized_complex_values",
        "scalar_components_quantized",
        "scale_reduction_count",
        "scale_metadata_bytes",
        "logical_int8_operand_bytes",
        "logical_float32_operand_bytes",
        "arithmetic_weighted_tasklet_utilization",
        "arithmetic_weighted_dpu_slot_utilization",
        "dominant_work_wave_utilization",
    ):
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = float(median(values)) if values else None
    result["same_policy_replay_passed"] = all(
        bool(row["policy_reference_passed"]) for row in rows
    )
    float_bytes = result["logical_float32_operand_bytes"]
    int8_bytes = result["logical_int8_operand_bytes"]
    result["nominal_logical_compression_ratio"] = (
        float(float_bytes / int8_bytes) if int8_bytes else None
    )
    host_candidates = {
        name: result.get(f"component_median_{name}")
        for name in HOST_COMPONENT_FIELDS
        if result.get(f"component_median_{name}") is not None
    }
    result["dominant_host_component"] = (
        max(host_candidates, key=lambda name: float(host_candidates[name]))
        if host_candidates
        else None
    )
    return result


def _ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None
    return float(numerator_value / denominator_value)


def _base_cell(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "route_id": row["route_id"],
        "numeric_policy": row["numeric_policy"],
        "dpu_count": row["dpu_count"],
        "tasklets_per_dpu": row["tasklets_per_dpu"],
        "measurement_count": row["measurement_count"],
    }


def _component_rows(routes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in routes:
        item = _base_cell(row)
        for field in ("total_wall_s", "session_inclusive_s"):
            item[f"median_{field}"] = row[f"median_{field}"]
            item[f"raw_mad_{field}"] = row[f"raw_mad_{field}"]
        for field in DISJOINT_COMPONENT_FIELDS:
            item[f"median_{field}"] = row[f"component_median_{field}"]
            item[f"raw_mad_{field}"] = row[f"component_raw_mad_{field}"]
        item["dominant_host_component"] = row["dominant_host_component"]
        result.append(item)
    return result


def _timing_envelope_rows(routes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in routes:
        item = _base_cell(row)
        item["envelope_semantics"] = "inclusive_non_additive"
        for field in ("operation_total_s", *ENVELOPE_FIELDS):
            item[f"median_{field}"] = row[f"median_{field}"]
            item[f"raw_mad_{field}"] = row[f"raw_mad_{field}"]
        result.append(item)
    return result


def _fixed_route_comparisons(
    routes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_topology = {
        (row["case_id"], row["numeric_policy"], row["dpu_count"], row["tasklets_per_dpu"]): row
        for row in routes
    }
    topologies = sorted(
        {
            (row["case_id"], row["dpu_count"], row["tasklets_per_dpu"])
            for row in routes
        }
    )
    result: list[dict[str, Any]] = []
    for case_id, dpu_count, tasklets_per_dpu in topologies:
        baseline = by_topology[(case_id, FLOAT32, dpu_count, tasklets_per_dpu)]
        candidate = by_topology[(case_id, POLICY, dpu_count, tasklets_per_dpu)]
        item = {
            "case_id": case_id,
            "dpu_count": dpu_count,
            "tasklets_per_dpu": tasklets_per_dpu,
            "float32_route_id": baseline["route_id"],
            "int8_route_id": candidate["route_id"],
            "float32_measurement_count": baseline["measurement_count"],
            "int8_measurement_count": candidate["measurement_count"],
            "comparison_scope": "fixed_route_same_topology",
        }
        for field in (
            "kernel_s",
            "total_wall_s",
            "session_inclusive_s",
            "h2d_s",
            "d2h_s",
        ):
            item[f"float32_median_{field}"] = baseline[f"median_{field}"]
            item[f"float32_raw_mad_{field}"] = baseline[f"raw_mad_{field}"]
            item[f"int8_median_{field}"] = candidate[f"median_{field}"]
            item[f"int8_raw_mad_{field}"] = candidate[f"raw_mad_{field}"]
            item[f"float32_over_int8_{field}"] = _ratio(
                baseline[f"median_{field}"], candidate[f"median_{field}"]
            )
        item["float32_median_h2d_bytes"] = baseline["median_h2d_bytes"]
        item["float32_raw_mad_h2d_bytes"] = baseline["raw_mad_h2d_bytes"]
        item["int8_median_h2d_bytes"] = candidate["median_h2d_bytes"]
        item["int8_raw_mad_h2d_bytes"] = candidate["raw_mad_h2d_bytes"]
        item["actual_h2d_byte_reduction_ratio"] = _ratio(
            baseline["median_h2d_bytes"], candidate["median_h2d_bytes"]
        )
        result.append(item)
    return result


def _best_observed_comparisons(
    routes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case_id in CASE_SPECS:
        candidates = {
            policy: [
                row
                for row in routes
                if row["case_id"] == case_id and row["numeric_policy"] == policy
            ]
            for policy in (FLOAT32, POLICY)
        }
        baseline = min(
            candidates[FLOAT32], key=lambda row: float(row["median_total_wall_s"])
        )
        candidate = min(
            candidates[POLICY], key=lambda row: float(row["median_total_wall_s"])
        )
        result.append(
            {
                "case_id": case_id,
                "selection_scope": "best_observed_within_tested_route_grid",
                "selection_metric": "median_total_wall_s",
                "grid_route_count": len(candidates[FLOAT32]),
                "float32_route_id": baseline["route_id"],
                "int8_route_id": candidate["route_id"],
                "float32_median_total_wall_s": baseline["median_total_wall_s"],
                "float32_raw_mad_total_wall_s": baseline["raw_mad_total_wall_s"],
                "int8_median_total_wall_s": candidate["median_total_wall_s"],
                "int8_raw_mad_total_wall_s": candidate["raw_mad_total_wall_s"],
                "float32_median_session_inclusive_s": baseline[
                    "median_session_inclusive_s"
                ],
                "float32_raw_mad_session_inclusive_s": baseline[
                    "raw_mad_session_inclusive_s"
                ],
                "int8_median_session_inclusive_s": candidate[
                    "median_session_inclusive_s"
                ],
                "int8_raw_mad_session_inclusive_s": candidate[
                    "raw_mad_session_inclusive_s"
                ],
                "best_observed_float32_over_int8_total_wall_s": _ratio(
                    baseline["median_total_wall_s"],
                    candidate["median_total_wall_s"],
                ),
                "best_observed_float32_over_int8_session_inclusive_s": _ratio(
                    baseline["median_session_inclusive_s"],
                    candidate["median_session_inclusive_s"],
                ),
            }
        )
    return result


def derive(
    manifest: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    config = manifest["configuration"]["experiment"]
    identity = config.get("experiment_identity_payload")
    label = identity.get("label") if isinstance(identity, Mapping) else None
    if label != "quantized-upmem-physical-diagnostic-v1":
        raise ValueError("analysis requires the fixed physical diagnostic")
    if len(samples) != 180 or len(sessions) != 180:
        raise ValueError("diagnostic must contain exactly 180 samples and sessions")
    if any(sample.get("status") != "success" for sample in samples):
        raise ValueError("diagnostic contains a non-successful attempt")
    session_by_id = _session_map(sessions)
    measurement_rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("attempt_kind") != "measurement":
            continue
        session = session_by_id.get(str(sample.get("session_instance_id")))
        if session is None:
            raise ValueError("sample has no matching session")
        route = config["routes"][sample["route_id"]]
        measurement_rows.append(_sample_row(sample, session, route))
    if len(measurement_rows) != 150:
        raise ValueError("diagnostic must contain exactly 150 measurements")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in measurement_rows:
        grouped[(row["case_id"], row["route_id"])].append(row)
    if len(grouped) != 30 or any(len(rows) != 5 for rows in grouped.values()):
        raise ValueError("diagnostic measurement cells are incomplete")
    routes = [_aggregate(grouped[key]) for key in sorted(grouped)]
    software = {case_id: _software_metrics(case_id) for case_id in CASE_SPECS}

    by_topology = {
        (row["case_id"], row["numeric_policy"], row["dpu_count"], row["tasklets_per_dpu"]): row
        for row in routes
    }
    for row in routes:
        key = (row["case_id"], row["dpu_count"], row["tasklets_per_dpu"])
        baseline = by_topology[(key[0], FLOAT32, key[1], key[2])]
        candidate = by_topology[(key[0], POLICY, key[1], key[2])]
        for name in ("kernel_s", "total_wall_s", "session_inclusive_s", "h2d_s", "d2h_s"):
            row[f"float32_over_int8_{name}"] = _ratio(
                baseline.get(f"median_{name}"), candidate.get(f"median_{name}")
            )
        left_bytes = baseline.get("median_h2d_bytes")
        right_bytes = candidate.get("median_h2d_bytes")
        row["actual_h2d_byte_reduction_ratio"] = _ratio(
            left_bytes, right_bytes
        )
        row.update(software[row["case_id"]])

    best_observed: list[dict[str, Any]] = []
    for case_id in CASE_SPECS:
        for policy in (FLOAT32, POLICY):
            candidates = [
                row for row in routes if row["case_id"] == case_id and row["numeric_policy"] == policy
            ]
            best = min(candidates, key=lambda row: float(row["median_total_wall_s"]))
            best_observed.append(
                {
                    "case_id": case_id,
                    "numeric_policy": policy,
                    "route_id": best["route_id"],
                    "dpu_count": best["dpu_count"],
                    "tasklets_per_dpu": best["tasklets_per_dpu"],
                    "selection_scope": "best_observed_within_tested_route_grid",
                    "selection_metric": "median_total_wall_s",
                    "grid_route_count": len(candidates),
                    "median_total_wall_s": best["median_total_wall_s"],
                    "raw_mad_total_wall_s": best["raw_mad_total_wall_s"],
                    "median_session_inclusive_s": best["median_session_inclusive_s"],
                    "raw_mad_session_inclusive_s": best[
                        "raw_mad_session_inclusive_s"
                    ],
                    "claim_status": "best_observed_within_tested_route_grid",
                }
            )

    fixed_route_comparisons = _fixed_route_comparisons(routes)
    best_observed_comparisons = _best_observed_comparisons(routes)

    error_runtime = [
        {
            "case_id": row["case_id"],
            "route_id": row["route_id"],
            "numeric_policy": row["numeric_policy"],
            "dpu_count": row["dpu_count"],
            "tasklets_per_dpu": row["tasklets_per_dpu"],
            "comparison_scope": "fixed_route_same_topology",
            "median_total_wall_s": row["median_total_wall_s"],
            "raw_mad_total_wall_s": row["raw_mad_total_wall_s"],
            "median_session_inclusive_s": row["median_session_inclusive_s"],
            "raw_mad_session_inclusive_s": row["raw_mad_session_inclusive_s"],
            "median_kernel_s": row["median_kernel_s"],
            "raw_mad_kernel_s": row["raw_mad_kernel_s"],
            "median_h2d_s": row["median_h2d_s"],
            "raw_mad_h2d_s": row["raw_mad_h2d_s"],
            "median_d2h_s": row["median_d2h_s"],
            "raw_mad_d2h_s": row["raw_mad_d2h_s"],
            "float32_median_total_wall_s": by_topology[
                (row["case_id"], FLOAT32, row["dpu_count"], row["tasklets_per_dpu"])
            ]["median_total_wall_s"],
            "float32_raw_mad_total_wall_s": by_topology[
                (row["case_id"], FLOAT32, row["dpu_count"], row["tasklets_per_dpu"])
            ]["raw_mad_total_wall_s"],
            "float32_over_int8_total_wall_s": row["float32_over_int8_total_wall_s"],
            "relative_l2_vs_float32_same_dag": row[
                "int8_relative_l2_vs_float32_same_dag"
            ],
            "relative_l2_vs_complex128": row["relative_l2_vs_complex128"],
            "nominal_logical_compression_ratio": row["nominal_logical_compression_ratio"],
            "actual_h2d_byte_reduction_ratio": row["actual_h2d_byte_reduction_ratio"],
            "claim_status": "diagnostic_descriptive_accuracy_unqualified",
        }
        for row in routes
        if row["numeric_policy"] == POLICY
    ]
    components = _component_rows(routes)
    timing_envelopes = _timing_envelope_rows(routes)
    sample_components = [
        {
            field: (
                row[field]
                if field not in DISJOINT_COMPONENT_FIELDS
                else row[f"component_{field}"]
            )
            for field in SAMPLE_COMPONENT_OUTPUT_FIELDS
        }
        for row in measurement_rows
    ]
    return {
        "summary": {
            "schema_version": SCHEMA,
            "source_commit": manifest["source_commit"],
            "experiment_id": manifest["experiment_id"],
            "sample_count": len(samples),
            "session_count": len(sessions),
            "measurement_count": len(measurement_rows),
            "same_policy_replay_passed": all(
                row["same_policy_replay_passed"] for row in routes
            ),
            "software_metrics": software,
            "claim_status": "diagnostic_descriptive_accuracy_unqualified",
            "accumulator_contract": "DPU int32 lanes; conservative 2*K*127^2 preflight; host int64 combination",
            "timing_note": (
                "simulator timings excluded; CPU integer matmul is not a DPU predictor; "
                "request lifecycle fields are inclusive envelopes and are not additive"
            ),
            "timing_attribution": {
                "schema_version": "quantized_upmem_timing_attribution_v2",
                "residual_tolerance_s": RESIDUAL_TOLERANCE_S,
                "disjoint_components": list(DISJOINT_COMPONENT_FIELDS),
                "raw_envelopes": list(ENVELOPE_FIELDS),
                "median_rule": "derive residuals per sample, then summarize median and raw MAD",
            },
            "fixed_route_comparison_count": len(fixed_route_comparisons),
            "best_observed_comparison_count": len(best_observed_comparisons),
            "best_observed_comparisons": best_observed_comparisons,
        },
        "routes": routes,
        "components": components,
        "timing_envelopes": timing_envelopes,
        "sample_components": sample_components,
        "fixed_route_comparisons": fixed_route_comparisons,
        "error_runtime": error_runtime,
        # Keep the result key stable for existing consumers; rows explicitly
        # describe a best observed route within the finite tested grid.
        "optima": best_observed,
    }


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(result: Mapping[str, Any]) -> str:
    best_observed = result["optima"]
    fixed = result["fixed_route_comparisons"]
    lines = [
        "# Quantized UPMEM Execution Diagnostic",
        "",
        "## Source-derived facts",
        "",
        "The frozen policy uses one float64 scale per complex operand, ties-to-even int8 encoding in [-127,127], four real products, int32 DPU lanes, and checked host int64 combination. ABI-v4 and packed-operation transport are unchanged.",
        "",
        "The physical route is hybrid: host complex values are encoded to compact int8 operands, the DPU performs integer real-product lanes, and the host reconstructs a complex64 intermediate before later requantization. It is not an integer-resident end-to-end tensor-network execution.",
        "",
        "## Measured facts",
        "",
        f"All {result['summary']['measurement_count']} measurements and {result['summary']['session_count']} fresh sessions are retained; same-policy replay passed: `{result['summary']['same_policy_replay_passed']}`.",
        "",
        "The raw request-wave, rank-submit, response-wait, and native-route timers are inclusive envelopes. They are retained in `quantized_upmem_timing_envelopes.csv` and are not summed as peer components. The disjoint attribution in `quantized_upmem_components.csv` derives residuals per sample with a 1e-6-second tolerance before calculating medians and raw MADs.",
        "",
        "## Fixed-route comparisons",
        "",
        "The following ratios hold circuit, contraction path, resource topology, and timing scope fixed. They are not route-selection results.",
        "",
        "| Circuit | Topology | kernel F32/int8 | steady F32/int8 | session F32/int8 | H2D-byte ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in fixed:
        lines.append(
            "| {case_id} | {dpu_count} DPU × T{tasklets_per_dpu} | "
            "{float32_over_int8_kernel_s:.4g} | {float32_over_int8_total_wall_s:.4g} | "
            "{float32_over_int8_session_inclusive_s:.4g} | "
            "{actual_h2d_byte_reduction_ratio:.4g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Best observed routes",
            "",
            "These selections minimize median steady wall within the five-route grid tested for each circuit and policy. They are best-observed diagnostic routes, not global resource optima.",
            "",
            "| Circuit | Policy | Best observed route | Median steady (s) | Raw MAD (s) |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in best_observed:
        lines.append(
            f"| {row['case_id']} | {row['numeric_policy']} | {row['route_id']} | "
            f"{row['median_total_wall_s']:.6g} | {row['raw_mad_total_wall_s']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Best-observed policy-conditioned comparison (the selected routes may differ):",
            "",
            "| Circuit | Float32 route | Int8 route | steady F32/int8 | session F32/int8 |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in result["summary"]["best_observed_comparisons"]:
        lines.append(
            f"| {row['case_id']} | {row['float32_route_id']} | {row['int8_route_id']} | "
            f"{row['best_observed_float32_over_int8_total_wall_s']:.4g} | "
            f"{row['best_observed_float32_over_int8_session_inclusive_s']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and claim boundary",
            "",
            "Physical outputs match the CPU same-policy replay exactly. Error against float32 and complex128 is circuit dependent and remains accuracy-unqualified. Actual H2D/D2H movement, kernel time, steady wall, and session-inclusive wall are reported separately; logical compression is not treated as measured transfer compression. The best observed topology is reported per circuit and policy within the tested five-route grid, never as universal. The dominant host component is selected only from disjoint host-side components in `quantized_upmem_components.csv`.",
            "",
            "This diagnostic may describe fixed-route float32/int8 ratios and best-observed policy-conditioned route comparisons. It does not establish universal speedup, CPU/GPU competitiveness, an accuracy threshold, path/quantization co-optimization, energy efficiency, or final thesis performance.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(result: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / OUTPUTS[0]).write_text(
        json.dumps(result["summary"], sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    _csv(output / OUTPUTS[1], result["routes"])
    _csv(output / OUTPUTS[2], result["components"])
    _csv(output / OUTPUTS[3], result["error_runtime"])
    _csv(output / OUTPUTS[4], result["optima"])
    (output / OUTPUTS[5]).write_text(_markdown(result), encoding="utf-8")
    _csv(output / OUTPUTS[6], result["sample_components"])
    _csv(output / OUTPUTS[7], result["timing_envelopes"])
    _csv(output / OUTPUTS[8], result["fixed_route_comparisons"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verification = verify_artifacts(args.input.resolve())
        if verification.get("status") != "completed":
            raise ValueError("canonical evidence is not completed")
        manifest, samples, sessions = load_artifacts(args.input.resolve())
        result = derive(manifest, samples, sessions)
        write_outputs(result, args.output.resolve())
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "completed", "outputs": list(OUTPUTS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
