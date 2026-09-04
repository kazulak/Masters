#!/usr/bin/env python3
"""Write deterministic CPU-only shared-scale complex-int8 analysis artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_bench.circuits import builtin_circuit  # noqa: E402
from quantum_bench.cpu import run_cpu_once  # noqa: E402
from quantum_bench.lowering import (  # noqa: E402
    build_contraction_dag,
    contraction_dag_hash,
    lower_tensor_network,
)
from quantum_bench.model import ContractionDAG, make_simulation_job  # noqa: E402
from quantum_bench.planning import plan_opt_einsum  # noqa: E402
from quantum_bench.quantized_contraction import (  # noqa: E402
    POLICY_ID,
    ContractionTrace,
    accumulator_bounds,
    replay_quantized_dag,
)


FLOAT32_POLICY_ID = "split_complex_float32_v1"
SCHEMA_VERSION = "quantized_contraction_policy_analysis_v1"


@dataclass(frozen=True, slots=True)
class CircuitCase:
    """One deterministic built-in circuit in the qualification suite."""

    circuit_id: str
    circuit_name: str
    parameters: Mapping[str, int]


FIXED_SUITE: tuple[CircuitCase, ...] = (
    CircuitCase("bell2", "bell_2q", {}),
    CircuitCase(
        "stress4_l2",
        "quantization_stress",
        {"n_qubits": 4, "repeat_layers": 2},
    ),
    CircuitCase("ghz18", "ghz_chain", {"n_qubits": 18}),
    CircuitCase("hs18_d1", "hs", {"n_qubits": 18, "depth": 1}),
    CircuitCase(
        "stress18_l2",
        "quantization_stress",
        {"n_qubits": 18, "repeat_layers": 2},
    ),
)


NODE_COLUMNS: tuple[str, ...] = (
    "circuit_id",
    "logical_plan_id",
    "node_index",
    "node_id",
    "operand_a_shape",
    "operand_b_shape",
    "output_shape",
    "B",
    "M",
    "N",
    "K",
    "operand_a_scale",
    "operand_b_scale",
    "operand_a_max_abs_component",
    "operand_b_max_abs_component",
    "operand_a_encoded_element_count",
    "operand_b_encoded_element_count",
    "operand_a_q_real_min",
    "operand_a_q_real_max",
    "operand_a_q_imag_min",
    "operand_a_q_imag_max",
    "operand_b_q_real_min",
    "operand_b_q_real_max",
    "operand_b_q_imag_min",
    "operand_b_q_imag_max",
    "operand_a_zero_count",
    "operand_b_zero_count",
    "operand_a_clipping_count",
    "operand_b_clipping_count",
    "operand_a_saturation_at_boundary_count",
    "operand_b_saturation_at_boundary_count",
    "int32_theoretical_accumulator_bound",
    "int32_accumulator_safe",
    "local_output_max_abs_error_vs_same_node_float32",
    "local_output_relative_l2_vs_same_node_float32",
    "local_output_norm_drift_vs_same_node_float32",
    "cumulative_output_max_abs_error_vs_same_node_float32",
    "cumulative_output_relative_l2_vs_same_node_float32",
    "cumulative_output_norm_drift_vs_same_node_float32",
    "theoretical_local_error_bound",
    "observed_local_error",
    "rounding_bound_applicable",
    "logical_encoded_bytes",
    "corresponding_float32_complex_bytes",
    "nominal_compression_ratio",
    "scale_metadata_bytes",
    "quantized_complex_values",
    "scalar_components_quantized",
    "scale_computation_count",
    "quantization_event_count",
    "requantization_event_count",
    "dequantized_output_elements",
    "four_real_products_per_complex_product",
    "integer_multiply_accumulate_count",
)


CIRCUIT_COLUMNS: tuple[str, ...] = (
    "circuit_id",
    "circuit_name",
    "parameters",
    "logical_plan_id",
    "contraction_count",
    "max_abs_error_vs_float32_same_dag",
    "relative_l2_vs_float32_same_dag",
    "norm_drift_vs_float32_same_dag",
    "max_abs_error_vs_complex128",
    "relative_l2_vs_complex128",
    "norm_drift_vs_complex128",
    "max_abs_error_float32_vs_complex128",
    "relative_l2_float32_vs_complex128",
    "norm_drift_float32_vs_complex128",
    "maximum_node_local_error",
    "node_id_of_max_local_error",
    "maximum_cumulative_intermediate_error",
    "node_id_of_max_cumulative_error",
    "minimum_scale",
    "maximum_scale",
    "total_quantized_complex_values",
    "total_scale_computations",
    "logical_float32_operand_bytes",
    "logical_int8_operand_bytes",
    "nominal_compression_ratio",
    "number_of_int32_unsafe_whole_K_contractions",
    "maximum_theoretical_accumulator_bound",
)


CLAIM_BOUNDARY: dict[str, Any] = {
    "accuracy_status": "accuracy_unqualified",
    "valid_claim": (
        "A deterministic per-contraction shared-scale complex-int8 numerical "
        "policy was implemented and characterized in CPU software."
    ),
    "forbidden_claims": (
        "int8 accelerates UPMEM",
        "int8 is faster than float32",
        "int8 reduces measured H2D or MRAM traffic by 4x",
        "int8 improves tasklet or DPU scaling",
        "int8 is accuracy-sufficient for all circuits",
        "int8 should be the default UPMEM policy",
    ),
}


def int32_accumulator_bound(k: int) -> int:
    """Return the full component bound ``2*K*127**2``."""

    return accumulator_bounds(k).component_bound


def int32_accumulator_safe(k: int) -> bool:
    """Return whether the full component bound fits signed int32."""

    return accumulator_bounds(k).int32_safe


def _trace_row(index: int, trace: ContractionTrace) -> dict[str, Any]:
    quantized_values = (
        trace.operand_a_encoded_element_count
        + trace.operand_b_encoded_element_count
    )
    return {
        "node_index": index,
        "node_id": trace.node_id,
        "operand_a_shape": trace.operand_a_shape,
        "operand_b_shape": trace.operand_b_shape,
        "output_shape": trace.output_shape,
        "B": trace.B,
        "M": trace.M,
        "N": trace.N,
        "K": trace.K,
        "operand_a_scale": trace.operand_a_scale,
        "operand_b_scale": trace.operand_b_scale,
        "operand_a_max_abs_component": trace.operand_a_max_abs_component,
        "operand_b_max_abs_component": trace.operand_b_max_abs_component,
        "operand_a_encoded_element_count": trace.operand_a_encoded_element_count,
        "operand_b_encoded_element_count": trace.operand_b_encoded_element_count,
        "operand_a_q_real_min": trace.operand_a_q_real_min,
        "operand_a_q_real_max": trace.operand_a_q_real_max,
        "operand_a_q_imag_min": trace.operand_a_q_imag_min,
        "operand_a_q_imag_max": trace.operand_a_q_imag_max,
        "operand_b_q_real_min": trace.operand_b_q_real_min,
        "operand_b_q_real_max": trace.operand_b_q_real_max,
        "operand_b_q_imag_min": trace.operand_b_q_imag_min,
        "operand_b_q_imag_max": trace.operand_b_q_imag_max,
        "operand_a_zero_count": trace.operand_a_zero_count,
        "operand_b_zero_count": trace.operand_b_zero_count,
        "operand_a_clipping_count": trace.operand_a_clipping_count,
        "operand_b_clipping_count": trace.operand_b_clipping_count,
        "operand_a_saturation_at_boundary_count": trace.operand_a_boundary_saturation_count,
        "operand_b_saturation_at_boundary_count": trace.operand_b_boundary_saturation_count,
        "int32_theoretical_accumulator_bound": trace.int32_theoretical_accumulator_bound,
        "int32_accumulator_safe": trace.int32_accumulator_safe,
        "local_output_max_abs_error_vs_same_node_float32": trace.local_max_abs_error_vs_same_node_float32,
        "local_output_relative_l2_vs_same_node_float32": trace.local_relative_l2_vs_same_node_float32,
        "local_output_norm_drift_vs_same_node_float32": trace.local_norm_drift_vs_same_node_float32,
        "cumulative_output_max_abs_error_vs_same_node_float32": trace.cumulative_max_abs_error_vs_same_node_float32,
        "cumulative_output_relative_l2_vs_same_node_float32": trace.cumulative_relative_l2_vs_same_node_float32,
        "cumulative_output_norm_drift_vs_same_node_float32": trace.cumulative_norm_drift_vs_same_node_float32,
        "theoretical_local_error_bound": trace.theoretical_local_error_bound,
        "observed_local_error": trace.observed_local_error,
        "rounding_bound_applicable": trace.rounding_bound_applicable,
        "logical_encoded_bytes": trace.logical_encoded_bytes,
        "corresponding_float32_complex_bytes": trace.logical_float32_complex_bytes,
        "nominal_compression_ratio": trace.nominal_operand_compression_ratio,
        "scale_metadata_bytes": trace.scale_metadata_bytes,
        "quantized_complex_values": quantized_values,
        "scalar_components_quantized": 2 * quantized_values,
        "scale_computation_count": trace.scale_computation_count,
        "quantization_event_count": trace.quantization_event_count,
        "requantization_event_count": trace.requantization_event_count,
        "dequantized_output_elements": trace.B * trace.M * trace.N,
        "four_real_products_per_complex_product": 4,
        "integer_multiply_accumulate_count": trace.integer_multiply_accumulate_count,
    }


def analyze_dag(
    circuit_id: str,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    *,
    circuit_name: str | None = None,
    parameters: Mapping[str, int] | None = None,
    logical_plan_id: str | None = None,
) -> dict[str, Any]:
    """Analyze one existing DAG with the fixed policy and accepted references."""

    if not isinstance(circuit_id, str) or not circuit_id:
        raise ValueError("circuit_id must be a nonempty string")
    replay = replay_quantized_dag(dag, inputs)
    accepted_float32 = run_cpu_once(dag, inputs, FLOAT32_POLICY_ID).output
    if not np.array_equal(replay.float32_output, accepted_float32):
        raise ValueError("analysis float32 replay differs from the accepted route")
    metrics = {
        "max_abs_error_vs_float32_same_dag": replay.max_abs_error_vs_float32_same_dag,
        "relative_l2_vs_float32_same_dag": replay.relative_l2_vs_float32_same_dag,
        "norm_drift_vs_float32_same_dag": replay.norm_drift_vs_float32_same_dag,
        "max_abs_error_vs_complex128": replay.max_abs_error_vs_complex128,
        "relative_l2_vs_complex128": replay.relative_l2_vs_complex128,
        "norm_drift_vs_complex128": replay.norm_drift_vs_complex128,
        "float32_vs_complex128_max_abs_error": replay.max_abs_error_float32_vs_complex128,
        "float32_vs_complex128_relative_l2": replay.relative_l2_float32_vs_complex128,
        "float32_vs_complex128_norm_drift": replay.norm_drift_float32_vs_complex128,
    }
    return {
        "circuit_id": circuit_id,
        "circuit_name": circuit_name or circuit_id,
        "parameters": dict(sorted((parameters or {}).items())),
        "logical_plan_id": logical_plan_id or contraction_dag_hash(dag),
        "contraction_count": len(replay.traces),
        "metrics": metrics,
        "nodes": [_trace_row(index, trace) for index, trace in enumerate(replay.traces)],
    }


def analyze_case(case: CircuitCase) -> dict[str, Any]:
    """Lower one fixed circuit with its deterministic greedy path."""

    circuit = builtin_circuit(case.circuit_name, dict(case.parameters))
    network, inputs = lower_tensor_network(make_simulation_job(circuit))
    path, _ = plan_opt_einsum(network, optimize="greedy")
    dag = build_contraction_dag(network, path)
    return analyze_dag(
        case.circuit_id,
        dag,
        inputs,
        circuit_name=case.circuit_name,
        parameters=case.parameters,
        logical_plan_id=contraction_dag_hash(dag),
    )


def _maximum_node_value(
    rows: Sequence[Mapping[str, Any]], field: str
) -> tuple[float | None, str | None]:
    if not rows:
        return None, None
    selected = max(rows, key=lambda row: float(row[field]))
    return float(selected[field]), str(selected["node_id"])


def _circuit_summary(analysis: Mapping[str, Any]) -> dict[str, Any]:
    nodes = tuple(analysis["nodes"])
    local_max, local_node = _maximum_node_value(
        nodes, "local_output_max_abs_error_vs_same_node_float32"
    )
    cumulative_max, cumulative_node = _maximum_node_value(
        nodes, "cumulative_output_max_abs_error_vs_same_node_float32"
    )
    scales = [
        float(row[field])
        for row in nodes
        for field in ("operand_a_scale", "operand_b_scale")
    ]
    float_bytes = sum(int(row["corresponding_float32_complex_bytes"]) for row in nodes)
    int8_bytes = sum(int(row["logical_encoded_bytes"]) for row in nodes)
    return {
        "circuit_id": analysis["circuit_id"],
        "circuit_name": analysis["circuit_name"],
        "parameters": dict(sorted(analysis["parameters"].items())),
        "logical_plan_id": analysis["logical_plan_id"],
        "contraction_count": int(analysis["contraction_count"]),
        "final_metrics": dict(analysis["metrics"]),
        "maximum_node_local_error": local_max,
        "node_id_of_max_local_error": local_node,
        "maximum_cumulative_intermediate_error": cumulative_max,
        "node_id_of_max_cumulative_error": cumulative_node,
        "minimum_scale": min(scales) if scales else None,
        "maximum_scale": max(scales) if scales else None,
        "total_quantized_complex_values": sum(int(row["quantized_complex_values"]) for row in nodes),
        "total_scale_computations": sum(int(row["scale_computation_count"]) for row in nodes),
        "logical_float32_operand_bytes": float_bytes,
        "logical_int8_operand_bytes": int8_bytes,
        "nominal_compression_ratio": float(float_bytes / int8_bytes) if int8_bytes else None,
        "number_of_int32_unsafe_whole_K_contractions": sum(
            not bool(row["int32_accumulator_safe"]) for row in nodes
        ),
        "maximum_theoretical_accumulator_bound": max(
            (int(row["int32_theoretical_accumulator_bound"]) for row in nodes),
            default=0,
        ),
    }


def _circuit_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary["final_metrics"]
    return {
        "circuit_id": summary["circuit_id"],
        "circuit_name": summary["circuit_name"],
        "parameters": summary["parameters"],
        "logical_plan_id": summary["logical_plan_id"],
        "contraction_count": summary["contraction_count"],
        "max_abs_error_vs_float32_same_dag": metrics["max_abs_error_vs_float32_same_dag"],
        "relative_l2_vs_float32_same_dag": metrics["relative_l2_vs_float32_same_dag"],
        "norm_drift_vs_float32_same_dag": metrics["norm_drift_vs_float32_same_dag"],
        "max_abs_error_vs_complex128": metrics["max_abs_error_vs_complex128"],
        "relative_l2_vs_complex128": metrics["relative_l2_vs_complex128"],
        "norm_drift_vs_complex128": metrics["norm_drift_vs_complex128"],
        "max_abs_error_float32_vs_complex128": metrics["float32_vs_complex128_max_abs_error"],
        "relative_l2_float32_vs_complex128": metrics["float32_vs_complex128_relative_l2"],
        "norm_drift_float32_vs_complex128": metrics["float32_vs_complex128_norm_drift"],
        **{field: summary[field] for field in CIRCUIT_COLUMNS[14:]},
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def analyze_suite(cases: Sequence[CircuitCase] = FIXED_SUITE) -> dict[str, Any]:
    """Analyze deterministic cases without writing files."""

    analyses = tuple(analyze_case(case) for case in cases)
    summaries = tuple(_circuit_summary(analysis) for analysis in analyses)
    node_rows = [
        {
            "circuit_id": analysis["circuit_id"],
            "logical_plan_id": analysis["logical_plan_id"],
            **node,
        }
        for analysis in analyses
        for node in analysis["nodes"]
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "accuracy_status": "accuracy_unqualified",
        "mathematical_policy": {
            "scale": "max(max(abs(real)),max(abs(imag)))/127; zero tensor uses 1",
            "rounding": "round_to_nearest_even",
            "encoded_range": [-127, 127],
            "scale_dtype": "float64",
            "accumulator_dtype": "int64_software_reference",
            "component_bound": "2*K*127^2",
        },
        "execution": {
            "scope": "cpu_only_software_analysis",
            "planner_engine": "opt_einsum",
            "planner_mode": "greedy",
            "host_timing_recorded": False,
            "physical_execution": False,
            "simulator_execution": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "suite": {
            "fixed_deterministic": cases is FIXED_SUITE,
            "case_count": len(summaries),
            "circuit_ids": [item["circuit_id"] for item in summaries],
        },
        "artifacts": {
            "summary": "quantization_summary.json",
            "nodes": "quantization_nodes.csv",
            "circuits": "quantization_circuits.csv",
            "node_row_count": len(node_rows),
            "node_columns": list(NODE_COLUMNS),
            "circuit_columns": list(CIRCUIT_COLUMNS),
        },
        "circuits": list(summaries),
    }
    return {
        "summary": _jsonable(summary),
        "circuit_rows": _jsonable([_circuit_row(item) for item in summaries]),
        "node_rows": _jsonable(node_rows),
    }


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=True, separators=(",", ":"))
    return str(value)


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_cell(row.get(column)) for column in columns})


def write_outputs(
    report: Mapping[str, Any], output: Path
) -> tuple[Path, Path, Path]:
    """Write the fixed summary, node, and circuit artifacts."""

    output = Path(output)
    if output.exists() and not output.is_dir():
        raise ValueError(f"analysis output must be a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "quantization_summary.json"
    nodes_path = output / "quantization_nodes.csv"
    circuits_path = output / "quantization_circuits.csv"
    summary_path.write_text(
        json.dumps(_jsonable(report["summary"]), ensure_ascii=True, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(nodes_path, NODE_COLUMNS, report["node_rows"])
    _write_csv(circuits_path, CIRCUIT_COLUMNS, report["circuit_rows"])
    return summary_path, nodes_path, circuits_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = write_outputs(analyze_suite(), args.output.resolve())
    except (OSError, TypeError, ValueError, FloatingPointError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {"status": "written", "files": [str(path) for path in paths]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
