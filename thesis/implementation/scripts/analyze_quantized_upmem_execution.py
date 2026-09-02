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
SCHEMA = "quantized_upmem_execution_analysis_v1"
OUTPUTS = (
    "quantized_upmem_summary.json",
    "quantized_upmem_routes.csv",
    "quantized_upmem_components.csv",
    "quantized_upmem_error_runtime.csv",
    "quantized_upmem_resource_optima.csv",
    "quantized_upmem_analysis.md",
)
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
OPERATION_FIELDS = (
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
    "coordinator_response_processing_sum_s",
    "assembly_s",
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


def _operation_sums(facts: Mapping[str, Any]) -> dict[str, float]:
    operations = facts.get("operation_facts")
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("measurement lacks operation_facts")
    totals = {field: 0.0 for field in OPERATION_FIELDS}
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError("operation fact is not a mapping")
        timing = operation.get("timing")
        if not isinstance(timing, Mapping):
            raise ValueError("operation timing is missing")
        for field in OPERATION_FIELDS:
            value = timing.get(field)
            if value is not None:
                totals[field] += _number(value, field)
    return totals


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
    open_s = _number(session.get("open_s"), "session.open_s")
    close_s = _number(session.get("session_close_s"), "session.session_close_s")
    total_s = _number(measurement.get("total_wall_s"), "total_wall_s")
    options = route["options"]
    logical = _logical_facts(numeric)
    row: dict[str, Any] = {
        "case_id": str(sample["case_id"]),
        "route_id": str(sample["route_id"]),
        "numeric_policy": str(route["numeric_policy"]),
        "dpu_count": int(options["dpu_count"]),
        "tasklets_per_dpu": int(options["tasklets_per_dpu"]),
        "block_id": int(sample["block_id"]),
        "session_open_s": open_s,
        "session_close_s": close_s,
        "session_inclusive_s": open_s + total_s + close_s,
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
        **_operation_sums(facts),
    }
    for field in MEASUREMENT_FIELDS:
        value = measurement.get(field)
        row[field] = None if value is None else _number(value, field)
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
        *OPERATION_FIELDS,
    )
    for field in statistic_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        center, spread = _stats(values)
        result[f"median_{field}"] = center
        result[f"raw_mad_{field}"] = spread
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
        name: result.get(f"median_{name}")
        for name in (
            "encode_s",
            "preparation_s",
            "request_build_sum_s",
            "host_reduce_s",
            "decode_s",
        )
        if result.get(f"median_{name}") is not None
    }
    result["dominant_host_component"] = (
        max(host_candidates, key=lambda name: float(host_candidates[name]))
        if host_candidates
        else None
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
            left = baseline.get(f"median_{name}")
            right = candidate.get(f"median_{name}")
            speedup = float(left / right) if left and right else None
            row[f"float32_over_int8_{name}"] = speedup
        left_bytes = baseline.get("median_h2d_bytes")
        right_bytes = candidate.get("median_h2d_bytes")
        row["actual_h2d_byte_reduction_ratio"] = (
            float(left_bytes / right_bytes) if left_bytes and right_bytes else None
        )
        row.update(software[row["case_id"]])

    optima: list[dict[str, Any]] = []
    for case_id in CASE_SPECS:
        for policy in (FLOAT32, POLICY):
            candidates = [
                row for row in routes if row["case_id"] == case_id and row["numeric_policy"] == policy
            ]
            best = min(candidates, key=lambda row: float(row["median_total_wall_s"]))
            optima.append(
                {
                    "case_id": case_id,
                    "numeric_policy": policy,
                    "route_id": best["route_id"],
                    "dpu_count": best["dpu_count"],
                    "tasklets_per_dpu": best["tasklets_per_dpu"],
                    "median_total_wall_s": best["median_total_wall_s"],
                    "median_session_inclusive_s": best["median_session_inclusive_s"],
                }
            )

    error_runtime = [
        {
            "case_id": row["case_id"],
            "route_id": row["route_id"],
            "numeric_policy": row["numeric_policy"],
            "dpu_count": row["dpu_count"],
            "tasklets_per_dpu": row["tasklets_per_dpu"],
            "median_total_wall_s": row["median_total_wall_s"],
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
    components = [
        {
            "case_id": row["case_id"],
            "route_id": row["route_id"],
            "numeric_policy": row["numeric_policy"],
            "dpu_count": row["dpu_count"],
            "tasklets_per_dpu": row["tasklets_per_dpu"],
            **{
                f"median_{field}": row.get(f"median_{field}")
                for field in (*MEASUREMENT_FIELDS, "session_open_s", "session_close_s", "session_inclusive_s", *OPERATION_FIELDS)
            },
            "dominant_host_component": row["dominant_host_component"],
        }
        for row in routes
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
            "timing_note": "simulator timings excluded; CPU integer matmul is not a DPU predictor",
        },
        "routes": routes,
        "components": components,
        "error_runtime": error_runtime,
        "optima": optima,
    }


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(result: Mapping[str, Any]) -> str:
    routes = result["routes"]
    optima = result["optima"]
    int8 = [row for row in routes if row["numeric_policy"] == POLICY]
    lines = [
        "# Quantized UPMEM Execution Diagnostic",
        "",
        "## Source-derived facts",
        "",
        "The frozen policy uses one float64 scale per complex operand, ties-to-even int8 encoding in [-127,127], four real products, int32 DPU lanes, and checked host int64 combination. ABI-v4 and packed-operation transport are unchanged.",
        "",
        "## Measured facts",
        "",
        f"All {result['summary']['measurement_count']} measurements and {result['summary']['session_count']} fresh sessions are retained; same-policy replay passed: `{result['summary']['same_policy_replay_passed']}`.",
        "",
        "## Calculated comparisons",
        "",
        "| Circuit | Route | kernel F32/int8 | steady F32/int8 | session F32/int8 | H2D-byte ratio | int8 relative L2 vs F32 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in int8:
        lines.append(
            "| {case_id} | {route_id} | {float32_over_int8_kernel_s:.4g} | "
            "{float32_over_int8_total_wall_s:.4g} | {float32_over_int8_session_inclusive_s:.4g} | "
            "{actual_h2d_byte_reduction_ratio:.4g} | {int8_relative_l2_vs_float32_same_dag:.4g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Resource optima (minimum median steady wall):",
            "",
            "| Circuit | Policy | Best route | Median steady (s) |",
            "|---|---|---|---:|",
        ]
    )
    for row in optima:
        lines.append(
            f"| {row['case_id']} | {row['numeric_policy']} | {row['route_id']} | {row['median_total_wall_s']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and claim boundary",
            "",
            "Physical outputs match the CPU same-policy replay exactly. Error against float32 and complex128 is circuit dependent and remains accuracy-unqualified. Actual H2D/D2H movement, kernel time, steady wall, and session-inclusive wall are reported separately; logical compression is not treated as measured transfer compression. The fastest observed topology is reported per circuit and policy, never as universal. The dominant host component is listed per route in `quantized_upmem_components.csv`.",
            "",
            "This diagnostic may describe observed float32/int8 ratios and resource optima. It does not establish universal speedup, CPU/GPU competitiveness, an accuracy threshold, path/quantization co-optimization, energy efficiency, or final thesis performance.",
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
