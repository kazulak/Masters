from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_bench.bench.result_artifacts import load_result_records  # noqa: E402
from quantum_bench.core.records import to_jsonable  # noqa: E402


SCHEMA_VERSION = "thesis_report_v1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantum_bench_mplconfig")

JsonDict = dict[str, Any]

FULL_STATE_FIELDS = [
    "schema_version",
    "case_id",
    "case_family",
    "n_qubits",
    "repeat_id",
    "cpu_route_id",
    "gpu_route_id",
    "cpu_simulation_compute_time_s",
    "gpu_simulation_compute_time_s",
    "compute_speedup_cpu_over_gpu",
    "cpu_total_wall_time_s",
    "gpu_total_wall_time_s",
    "wall_time_ratio_cpu_over_gpu",
    "validation_method",
    "state_output_mode",
    "performance_tier",
    "gpu_device_name",
    "gpu_backend_verified",
    "gpu_program_executed",
]

FULL_STATE_SUMMARY_FIELDS = [
    "schema_version",
    "case_family",
    "n_qubits",
    "matched_repeat_count",
    "cpu_simulation_compute_time_median_s",
    "gpu_simulation_compute_time_median_s",
    "compute_speedup_median_cpu_over_gpu",
    "cpu_total_wall_time_median_s",
    "gpu_total_wall_time_median_s",
    "wall_time_ratio_median_cpu_over_gpu",
    "validation_method",
    "state_output_mode",
    "performance_tier",
    "gpu_device_name",
]

TN_PATH_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "n_qubits",
    "repeat_id",
    "route_id",
    "thesis_route_label",
    "benchmark_role",
    "contraction_execution_target",
    "accelerator_kind",
    "parallelism_mode",
    "path_replay_execution",
    "path_strategy",
    "path_planner_engine",
    "quantization_mode",
    "per_contraction_quantization",
    "input_dtype",
    "accumulator_dtype",
    "validation_status",
    "simulation_compute_time_s",
    "total_wall_time_s",
    "slicing_enabled",
    "slice_count",
    "slicing_flop_ratio",
    "total_quantization_time_s",
    "total_dequantization_time_s",
    "quantization_max_abs_error",
    "quantization_l2_error",
    "max_abs_error",
    "l2_error",
]

TN_PATH_SUMMARY_FIELDS = [
    "schema_version",
    "case_family",
    "n_qubits",
    "route_id",
    "thesis_route_label",
    "benchmark_role",
    "contraction_execution_target",
    "accelerator_kind",
    "parallelism_mode",
    "path_strategy",
    "quantization_mode",
    "repeat_count",
    "simulation_compute_time_median_s",
    "total_wall_time_median_s",
    "slice_count_median",
    "slicing_flop_ratio_median",
    "validation_pass_count",
]

TN_QUANT_FIELDS = [
    "schema_version",
    "case_id",
    "case_family",
    "n_qubits",
    "repeat_id",
    "path_strategy",
    "unquantized_route_id",
    "quantized_route_id",
    "comparison_scope",
    "contraction_execution_target",
    "accelerator_kind",
    "unquantized_input_dtype",
    "unquantized_accumulator_dtype",
    "quantized_input_dtype",
    "quantized_accumulator_dtype",
    "unquantized_simulation_compute_time_s",
    "quantized_simulation_compute_time_s",
    "compute_ratio_unquantized_over_quantized",
    "compute_slowdown_quantized_over_unquantized",
    "unquantized_total_wall_time_s",
    "quantized_total_wall_time_s",
    "wall_ratio_unquantized_over_quantized",
    "wall_slowdown_quantized_over_unquantized",
    "quantization_max_abs_error",
    "quantization_l2_error",
    "max_abs_error_vs_reference",
    "l2_error_vs_reference",
    "validation_status",
    "quantized_replay_numeric_contract",
    "interpretation",
]

TN_QUANT_SPEEDUP_FIELDS = [
    "schema_version",
    "case_family",
    "n_qubits",
    "path_strategy",
    "matched_repeat_count",
    "unquantized_simulation_compute_time_median_s",
    "quantized_simulation_compute_time_median_s",
    "compute_ratio_median_unquantized_over_quantized",
    "compute_slowdown_median_quantized_over_unquantized",
    "unquantized_total_wall_time_median_s",
    "quantized_total_wall_time_median_s",
    "wall_ratio_median_unquantized_over_quantized",
    "wall_slowdown_median_quantized_over_unquantized",
    "comparison_scope",
]

TN_QUANT_ERROR_FIELDS = [
    "schema_version",
    "case_family",
    "n_qubits",
    "path_strategy",
    "matched_repeat_count",
    "quantization_max_abs_error_median",
    "quantization_l2_error_median",
    "max_abs_error_vs_reference_median",
    "l2_error_vs_reference_median",
]

UPMEM_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "n_qubits",
    "repeat_id",
    "route_id",
    "thesis_route_label",
    "quantization_mode",
    "policy",
    "status",
    "validation_status",
    "contraction_execution_target",
    "backend_family",
    "accelerator_kind",
    "upmem_execution_mode",
    "execution_backend",
    "cpu_fallback_used",
    "cpu_fallback_task_count",
    "task_count",
    "upmem_task_count",
    "upmem_program_executed",
    "dpu_program_invocations",
    "native_sdk_control_path",
    "simplepim_api_used",
    "hardware_execution",
    "hardware_timing_available",
    "hardware_speedup_applicable",
    "simulation_compute_time_s",
    "total_simulator_time_s",
    "total_wall_time_s",
    "input_dtype_on_dpu",
    "accumulator_dtype_on_dpu",
    "scaling_applied",
    "actual_h2d_bytes",
    "actual_d2h_bytes",
    "actual_transfer_bytes",
    "max_abs_error",
    "l2_error",
    "resource_skip_reason",
]

UPMEM_ACCURACY_FIELDS = [
    "schema_version",
    "case_family",
    "n_qubits",
    "route_id",
    "thesis_route_label",
    "quantization_mode",
    "row_count",
    "supported_count",
    "unsupported_count",
    "max_abs_error_median",
    "l2_error_median",
    "cpu_fallback_rows",
    "hardware_speedup_applicable_rows",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate thesis benchmark tables from explicit evidence runs.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Evidence run directories or normalized artifact paths.")
    parser.add_argument("--out", type=Path, required=True, help="Derived report output directory under runs/comparisons.")
    parser.add_argument("--title", default="Thesis Benchmark Report")
    args = parser.parse_args(argv)

    records = load_result_records(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)

    cpu_gpu_rows = full_state_cpu_gpu_rows(records)
    tn_path_rows = tn_path_rows_from_records(records)
    tn_quant_rows = tn_quantization_rows(records)
    upmem_rows = upmem_boundary_rows(records)
    cpu_gpu_summary_rows = full_state_cpu_gpu_summary_rows(cpu_gpu_rows)
    tn_path_summary_rows = tn_path_runtime_summary_rows(tn_path_rows)
    tn_quant_speedup_rows = tn_quantization_speedup_summary_rows(tn_quant_rows)
    tn_quant_error_rows = tn_quantization_error_summary_rows(tn_quant_rows)
    upmem_accuracy_rows = upmem_accuracy_summary_rows(upmem_rows)

    _write_csv(args.out / "full_state_cpu_gpu_by_circuit.csv", cpu_gpu_rows, FULL_STATE_FIELDS)
    _write_csv(args.out / "full_state_cpu_gpu_speedup_by_circuit_size.csv", cpu_gpu_summary_rows, FULL_STATE_SUMMARY_FIELDS)
    _write_csv(args.out / "tn_path_comparison_by_circuit.csv", tn_path_rows, TN_PATH_FIELDS)
    _write_csv(args.out / "tn_path_runtime_by_circuit_size.csv", tn_path_summary_rows, TN_PATH_SUMMARY_FIELDS)
    _write_csv(args.out / "tn_quantization_comparison.csv", tn_quant_rows, TN_QUANT_FIELDS)
    _write_csv(args.out / "tn_quantization_speedup_by_circuit_size.csv", tn_quant_speedup_rows, TN_QUANT_SPEEDUP_FIELDS)
    _write_csv(args.out / "tn_quantization_error_by_circuit_size.csv", tn_quant_error_rows, TN_QUANT_ERROR_FIELDS)
    _write_csv(args.out / "upmem_boundary_quantization.csv", upmem_rows, UPMEM_FIELDS)
    _write_csv(args.out / "upmem_accuracy_by_circuit_size.csv", upmem_accuracy_rows, UPMEM_ACCURACY_FIELDS)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "thesis_comparison_report",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_count": len(args.inputs),
        "record_count": len(records),
        "inputs": [path.as_posix() for path in args.inputs],
        "outputs": [
            "full_state_cpu_gpu_by_circuit.csv",
            "full_state_cpu_gpu_speedup_by_circuit_size.csv",
            "tn_path_comparison_by_circuit.csv",
            "tn_path_runtime_by_circuit_size.csv",
            "tn_quantization_comparison.csv",
            "tn_quantization_speedup_by_circuit_size.csv",
            "tn_quantization_error_by_circuit_size.csv",
            "upmem_boundary_quantization.csv",
            "upmem_accuracy_by_circuit_size.csv",
            "benchmark_summary.md",
            "plot_manifest.json",
        ],
        "claims": {
            "gpu_scope": "QuEST full-state GPU only; not GPU tensor-network evidence.",
            "tn_quantization_scope": "CPU diagnostic TN path replay uses int8 operand quantization, immediate dequantization, and complex128 CPU einsum.",
            "upmem_scope": "UPMEM SDK simulator evidence is not hardware timing or hardware speedup.",
        },
    }
    _write_json(args.out / "thesis_report_manifest.json", manifest)
    plot_manifest = write_plots(
        args.out,
        cpu_gpu_summary_rows,
        tn_path_summary_rows,
        tn_quant_speedup_rows,
        tn_quant_error_rows,
        upmem_rows,
        upmem_accuracy_rows,
    )
    _write_json(args.out / "plot_manifest.json", plot_manifest)
    (args.out / "benchmark_summary.md").write_text(
        benchmark_summary(
            args.title,
            records,
            cpu_gpu_rows,
            cpu_gpu_summary_rows,
            tn_path_rows,
            tn_path_summary_rows,
            tn_quant_rows,
            tn_quant_speedup_rows,
            tn_quant_error_rows,
            upmem_rows,
            upmem_accuracy_rows,
            plot_manifest,
        ),
        encoding="utf-8",
    )
    print(args.out)
    return 0


def full_state_cpu_gpu_rows(records: list[JsonDict]) -> list[JsonDict]:
    cpu = _records_by_case_repeat(records, "quest_cpu_full_state_exact")
    gpu = _records_by_case_repeat(records, "quest_gpu_full_state_exact")
    rows: list[JsonDict] = []
    for key in sorted(set(cpu) & set(gpu)):
        cpu_row = cpu[key]
        gpu_row = gpu[key]
        if not _valid_cpu_gpu_pair(cpu_row, gpu_row):
            continue
        cpu_compute = _positive(cpu_row.get("simulation_compute_time_s"))
        gpu_compute = _positive(gpu_row.get("simulation_compute_time_s"))
        cpu_wall = _positive(cpu_row.get("total_wall_time_s"))
        gpu_wall = _positive(gpu_row.get("total_wall_time_s"))
        if None in {cpu_compute, gpu_compute, cpu_wall, gpu_wall}:
            continue
        family, qubits = _family_and_qubits(cpu_row)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": key[0],
                "case_family": family,
                "n_qubits": qubits,
                "repeat_id": key[1],
                "cpu_route_id": "quest_cpu_full_state_exact",
                "gpu_route_id": "quest_gpu_full_state_exact",
                "cpu_simulation_compute_time_s": cpu_compute,
                "gpu_simulation_compute_time_s": gpu_compute,
                "compute_speedup_cpu_over_gpu": cpu_compute / gpu_compute,
                "cpu_total_wall_time_s": cpu_wall,
                "gpu_total_wall_time_s": gpu_wall,
                "wall_time_ratio_cpu_over_gpu": cpu_wall / gpu_wall,
                "validation_method": cpu_row.get("validation_method"),
                "state_output_mode": cpu_row.get("state_output_mode"),
                "performance_tier": bool(cpu_row.get("performance_tier", False)),
                "gpu_device_name": gpu_row.get("gpu_device_name"),
                "gpu_backend_verified": bool(gpu_row.get("gpu_backend_verified", False)),
                "gpu_program_executed": bool(gpu_row.get("gpu_program_executed", False)),
            }
        )
    return rows


def full_state_cpu_gpu_summary_rows(rows: list[JsonDict]) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for (family, qubits), group in _group_rows(rows, "case_family", "n_qubits").items():
        cpu_compute = _numbers(group, "cpu_simulation_compute_time_s")
        gpu_compute = _numbers(group, "gpu_simulation_compute_time_s")
        cpu_wall = _numbers(group, "cpu_total_wall_time_s")
        gpu_wall = _numbers(group, "gpu_total_wall_time_s")
        speedups = _numbers(group, "compute_speedup_cpu_over_gpu")
        wall_ratios = _numbers(group, "wall_time_ratio_cpu_over_gpu")
        if not (cpu_compute and gpu_compute and speedups):
            continue
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_family": family,
                "n_qubits": qubits,
                "matched_repeat_count": len(group),
                "cpu_simulation_compute_time_median_s": statistics.median(cpu_compute),
                "gpu_simulation_compute_time_median_s": statistics.median(gpu_compute),
                "compute_speedup_median_cpu_over_gpu": statistics.median(speedups),
                "cpu_total_wall_time_median_s": statistics.median(cpu_wall) if cpu_wall else None,
                "gpu_total_wall_time_median_s": statistics.median(gpu_wall) if gpu_wall else None,
                "wall_time_ratio_median_cpu_over_gpu": statistics.median(wall_ratios) if wall_ratios else None,
                "validation_method": _first_present(group, "validation_method"),
                "state_output_mode": _first_present(group, "state_output_mode"),
                "performance_tier": bool(_first_present(group, "performance_tier")),
                "gpu_device_name": _first_present(group, "gpu_device_name"),
            }
        )
    return sorted(summaries, key=_family_qubits_sort_key)


def tn_path_rows_from_records(records: list[JsonDict]) -> list[JsonDict]:
    selected_routes = {
        "quimb_tn_exact",
        "quimb_tn_sliced_exact",
        "cpu_tn_path_replay_float64",
        "cpu_tn_path_replay_int8_quantized",
    }
    rows: list[JsonDict] = []
    for record in records:
        if record.get("route_id") not in selected_routes:
            continue
        family, qubits = _family_and_qubits(record)
        errors = _validation_errors(record)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": record.get("suite_id"),
                "case_id": record.get("case_id"),
                "case_family": family,
                "n_qubits": qubits,
                "repeat_id": record.get("repeat_id"),
                "route_id": record.get("route_id"),
                "thesis_route_label": _thesis_route_label(record),
                "benchmark_role": record.get("benchmark_role"),
                "contraction_execution_target": record.get("contraction_execution_target"),
                "accelerator_kind": record.get("accelerator_kind"),
                "parallelism_mode": record.get("parallelism_mode"),
                "path_replay_execution": bool(record.get("path_replay_execution", False)),
                "path_strategy": record.get("path_strategy"),
                "path_planner_engine": record.get("path_planner_engine"),
                "quantization_mode": record.get("quantization_mode"),
                "per_contraction_quantization": bool(record.get("per_contraction_quantization", False)),
                "input_dtype": record.get("input_dtype"),
                "accumulator_dtype": record.get("accumulator_dtype"),
                "validation_status": record.get("validation_status"),
                "simulation_compute_time_s": record.get("simulation_compute_time_s"),
                "total_wall_time_s": record.get("total_wall_time_s"),
                "slicing_enabled": bool(record.get("slicing_enabled", False)),
                "slice_count": record.get("slice_count"),
                "slicing_flop_ratio": record.get("slicing_flop_ratio"),
                "total_quantization_time_s": record.get("total_quantization_time_s"),
                "total_dequantization_time_s": record.get("total_dequantization_time_s"),
                "quantization_max_abs_error": record.get("quantization_max_abs_error"),
                "quantization_l2_error": record.get("quantization_l2_error"),
                "max_abs_error": errors.get("max_abs_error"),
                "l2_error": errors.get("l2_error"),
            }
        )
    return rows


def tn_path_runtime_summary_rows(rows: list[JsonDict]) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for (family, qubits, route_id), group in _group_rows(rows, "case_family", "n_qubits", "route_id").items():
        compute = _numbers(group, "simulation_compute_time_s")
        wall = _numbers(group, "total_wall_time_s")
        if not compute:
            continue
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_family": family,
                "n_qubits": qubits,
                "route_id": route_id,
                "thesis_route_label": _first_present(group, "thesis_route_label"),
                "benchmark_role": _first_present(group, "benchmark_role"),
                "contraction_execution_target": _first_present(group, "contraction_execution_target"),
                "accelerator_kind": _first_present(group, "accelerator_kind"),
                "parallelism_mode": _first_present(group, "parallelism_mode"),
                "path_strategy": _first_present(group, "path_strategy"),
                "quantization_mode": _first_present(group, "quantization_mode"),
                "repeat_count": len(group),
                "simulation_compute_time_median_s": statistics.median(compute),
                "total_wall_time_median_s": statistics.median(wall) if wall else None,
                "slice_count_median": _median_or_none(_numbers(group, "slice_count")),
                "slicing_flop_ratio_median": _median_or_none(_numbers(group, "slicing_flop_ratio")),
                "validation_pass_count": sum(1 for row in group if _validation_ok(row)),
            }
        )
    return sorted(summaries, key=lambda row: (_family_qubits_sort_key(row), str(row.get("route_id"))))


def tn_quantization_rows(records: list[JsonDict]) -> list[JsonDict]:
    baseline = _records_by_case_repeat(records, "cpu_tn_path_replay_float64")
    quantized = _records_by_case_repeat(records, "cpu_tn_path_replay_int8_quantized")
    rows: list[JsonDict] = []
    for key in sorted(set(baseline) & set(quantized)):
        base = baseline[key]
        quant = quantized[key]
        if not (_validation_ok(base) and _validation_ok(quant)):
            continue
        base_compute = _positive(base.get("simulation_compute_time_s"))
        quant_compute = _positive(quant.get("simulation_compute_time_s"))
        base_wall = _positive(base.get("total_wall_time_s"))
        quant_wall = _positive(quant.get("total_wall_time_s"))
        if None in {base_compute, quant_compute, base_wall, quant_wall}:
            continue
        family, qubits = _family_and_qubits(base)
        errors = _validation_errors(quant)
        comparison_scope = "same_route_family_cpu_diagnostic_path_replay"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": key[0],
                "case_family": family,
                "n_qubits": qubits,
                "repeat_id": key[1],
                "path_strategy": base.get("path_strategy"),
                "unquantized_route_id": "cpu_tn_path_replay_float64",
                "quantized_route_id": "cpu_tn_path_replay_int8_quantized",
                "comparison_scope": comparison_scope,
                "contraction_execution_target": quant.get("contraction_execution_target") or base.get("contraction_execution_target"),
                "accelerator_kind": quant.get("accelerator_kind") or base.get("accelerator_kind"),
                "unquantized_input_dtype": base.get("input_dtype"),
                "unquantized_accumulator_dtype": base.get("accumulator_dtype"),
                "quantized_input_dtype": quant.get("input_dtype"),
                "quantized_accumulator_dtype": quant.get("accumulator_dtype"),
                "unquantized_simulation_compute_time_s": base_compute,
                "quantized_simulation_compute_time_s": quant_compute,
                "compute_ratio_unquantized_over_quantized": base_compute / quant_compute,
                "compute_slowdown_quantized_over_unquantized": quant_compute / base_compute,
                "unquantized_total_wall_time_s": base_wall,
                "quantized_total_wall_time_s": quant_wall,
                "wall_ratio_unquantized_over_quantized": base_wall / quant_wall,
                "wall_slowdown_quantized_over_unquantized": quant_wall / base_wall,
                "quantization_max_abs_error": quant.get("quantization_max_abs_error"),
                "quantization_l2_error": quant.get("quantization_l2_error"),
                "max_abs_error_vs_reference": errors.get("max_abs_error"),
                "l2_error_vs_reference": errors.get("l2_error"),
                "validation_status": quant.get("validation_status"),
                "quantized_replay_numeric_contract": quant.get("quantized_replay_numeric_contract"),
                "interpretation": "CPU diagnostic replay; int8 operands are dequantized before complex128 einsum.",
            }
        )
    return rows


def tn_quantization_speedup_summary_rows(rows: list[JsonDict]) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for (family, qubits, path_strategy), group in _group_rows(rows, "case_family", "n_qubits", "path_strategy").items():
        unquant_compute = _numbers(group, "unquantized_simulation_compute_time_s")
        quant_compute = _numbers(group, "quantized_simulation_compute_time_s")
        compute_ratios = _numbers(group, "compute_ratio_unquantized_over_quantized")
        unquant_wall = _numbers(group, "unquantized_total_wall_time_s")
        quant_wall = _numbers(group, "quantized_total_wall_time_s")
        wall_ratios = _numbers(group, "wall_ratio_unquantized_over_quantized")
        compute_slowdowns = _numbers(group, "compute_slowdown_quantized_over_unquantized")
        wall_slowdowns = _numbers(group, "wall_slowdown_quantized_over_unquantized")
        if not (unquant_compute and quant_compute and compute_ratios):
            continue
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_family": family,
                "n_qubits": qubits,
                "path_strategy": path_strategy,
                "matched_repeat_count": len(group),
                "unquantized_simulation_compute_time_median_s": statistics.median(unquant_compute),
                "quantized_simulation_compute_time_median_s": statistics.median(quant_compute),
                "compute_ratio_median_unquantized_over_quantized": statistics.median(compute_ratios),
                "compute_slowdown_median_quantized_over_unquantized": statistics.median(compute_slowdowns) if compute_slowdowns else None,
                "unquantized_total_wall_time_median_s": statistics.median(unquant_wall) if unquant_wall else None,
                "quantized_total_wall_time_median_s": statistics.median(quant_wall) if quant_wall else None,
                "wall_ratio_median_unquantized_over_quantized": statistics.median(wall_ratios) if wall_ratios else None,
                "wall_slowdown_median_quantized_over_unquantized": statistics.median(wall_slowdowns) if wall_slowdowns else None,
                "comparison_scope": _first_present(group, "comparison_scope"),
            }
        )
    return sorted(summaries, key=lambda row: (_family_qubits_sort_key(row), str(row.get("path_strategy"))))


def tn_quantization_error_summary_rows(rows: list[JsonDict]) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for (family, qubits, path_strategy), group in _group_rows(rows, "case_family", "n_qubits", "path_strategy").items():
        max_errors = _numbers(group, "max_abs_error_vs_reference")
        l2_errors = _numbers(group, "l2_error_vs_reference")
        quant_max = _numbers(group, "quantization_max_abs_error")
        quant_l2 = _numbers(group, "quantization_l2_error")
        if not (max_errors or l2_errors or quant_max or quant_l2):
            continue
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_family": family,
                "n_qubits": qubits,
                "path_strategy": path_strategy,
                "matched_repeat_count": len(group),
                "quantization_max_abs_error_median": _median_or_none(quant_max),
                "quantization_l2_error_median": _median_or_none(quant_l2),
                "max_abs_error_vs_reference_median": _median_or_none(max_errors),
                "l2_error_vs_reference_median": _median_or_none(l2_errors),
            }
        )
    return sorted(summaries, key=lambda row: (_family_qubits_sort_key(row), str(row.get("path_strategy"))))


def upmem_boundary_rows(records: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for record in records:
        if record.get("route_id") != "upmem_tn_sdk_simulator_quantized" and record.get("contraction_execution_target") != "upmem":
            continue
        family, qubits = _family_and_qubits(record)
        errors = _validation_errors(record)
        task_count = _first_number(record, "task_count", "tn_task_count", "dpu_program_invocations")
        upmem_task_count = _first_number(record, "upmem_task_count", "dpu_program_invocations")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": record.get("suite_id"),
                "case_id": record.get("case_id"),
                "case_family": family,
                "n_qubits": qubits,
                "repeat_id": record.get("repeat_id"),
                "route_id": record.get("route_id"),
                "thesis_route_label": _upmem_thesis_label(record),
                "quantization_mode": record.get("quantization_mode"),
                "policy": record.get("policy") or "generic-only",
                "status": record.get("status"),
                "validation_status": record.get("validation_status"),
                "contraction_execution_target": record.get("contraction_execution_target"),
                "backend_family": record.get("execution_backend") or record.get("backend_family") or "upmem_sdk",
                "accelerator_kind": record.get("accelerator_kind") or "upmem",
                "upmem_execution_mode": record.get("upmem_execution_mode"),
                "execution_backend": record.get("execution_backend"),
                "cpu_fallback_used": bool(record.get("cpu_fallback_used", False)),
                "cpu_fallback_task_count": _first_number(record, "cpu_fallback_task_count") or 0,
                "task_count": task_count,
                "upmem_task_count": upmem_task_count,
                "upmem_program_executed": bool(record.get("upmem_program_executed", False)),
                "dpu_program_invocations": record.get("dpu_program_invocations"),
                "native_sdk_control_path": bool(record.get("native_sdk_control_path", False)),
                "simplepim_api_used": bool(record.get("simplepim_api_used", False)),
                "hardware_execution": bool(record.get("hardware_execution", False)),
                "hardware_timing_available": bool(record.get("hardware_timing_available", False)),
                "hardware_speedup_applicable": bool(record.get("hardware_speedup_applicable", False)),
                "simulation_compute_time_s": record.get("simulation_compute_time_s"),
                "total_simulator_time_s": record.get("total_simulator_time_s") or record.get("simulation_compute_time_s"),
                "total_wall_time_s": record.get("total_wall_time_s"),
                "input_dtype_on_dpu": record.get("input_dtype_on_dpu") or record.get("input_dtype"),
                "accumulator_dtype_on_dpu": record.get("accumulator_dtype_on_dpu") or record.get("accumulator_dtype"),
                "scaling_applied": record.get("scaling_applied"),
                "actual_h2d_bytes": record.get("actual_h2d_bytes"),
                "actual_d2h_bytes": record.get("actual_d2h_bytes"),
                "actual_transfer_bytes": record.get("actual_transfer_bytes"),
                "max_abs_error": errors.get("max_abs_error"),
                "l2_error": errors.get("l2_error"),
                "resource_skip_reason": record.get("resource_skip_reason"),
            }
        )
    return rows


def upmem_accuracy_summary_rows(rows: list[JsonDict]) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for (family, qubits, route_id, quantization_mode), group in _group_rows(rows, "case_family", "n_qubits", "route_id", "quantization_mode").items():
        max_errors = _numbers(group, "max_abs_error")
        l2_errors = _numbers(group, "l2_error")
        supported = [row for row in group if _validation_ok(row)]
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_family": family,
                "n_qubits": qubits,
                "route_id": route_id,
                "thesis_route_label": _first_present(group, "thesis_route_label"),
                "quantization_mode": quantization_mode,
                "row_count": len(group),
                "supported_count": len(supported),
                "unsupported_count": len(group) - len(supported),
                "max_abs_error_median": _median_or_none(max_errors),
                "l2_error_median": _median_or_none(l2_errors),
                "cpu_fallback_rows": sum(1 for row in group if bool(row.get("cpu_fallback_used", False))),
                "hardware_speedup_applicable_rows": sum(1 for row in group if bool(row.get("hardware_speedup_applicable", False))),
            }
        )
    return sorted(summaries, key=lambda row: (_family_qubits_sort_key(row), str(row.get("quantization_mode"))))


def write_plots(
    out_dir: Path,
    cpu_gpu_summary_rows: list[JsonDict],
    tn_path_summary_rows: list[JsonDict],
    tn_quant_speedup_rows: list[JsonDict],
    tn_quant_error_rows: list[JsonDict],
    upmem_rows: list[JsonDict],
    upmem_accuracy_rows: list[JsonDict],
) -> JsonDict:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"schema_version": SCHEMA_VERSION, "status": "skipped", "reason": "matplotlib_unavailable", "error": str(exc), "plots": []}
    entries = [
        _plot_entry(
            plt,
            plots_dir / "full_state_cpu_gpu_runtime_by_circuit_size.png",
            "Full-state CPU/GPU runtime",
            "full_state_cpu_gpu_speedup_by_circuit_size.csv",
            lambda path: _plot_full_state_runtime(plt, path, cpu_gpu_summary_rows),
        ),
        _plot_entry(
            plt,
            plots_dir / "full_state_cpu_gpu_speedup_by_circuit_size.png",
            "Full-state CPU/GPU speedup",
            "full_state_cpu_gpu_speedup_by_circuit_size.csv",
            lambda path: _plot_speedup(plt, path, cpu_gpu_summary_rows),
        ),
        _plot_entry(
            plt,
            plots_dir / "tn_path_runtime_by_circuit_size.png",
            "TN route/path runtime",
            "tn_path_runtime_by_circuit_size.csv",
            lambda path: _plot_tn_runtime(plt, path, tn_path_summary_rows),
        ),
        _plot_entry(
            plt,
            plots_dir / "tn_quantization_runtime_by_circuit_size.png",
            "CPU diagnostic TN replay runtime",
            "tn_quantization_speedup_by_circuit_size.csv",
            lambda path: _plot_tn_quant_runtime(plt, path, tn_quant_speedup_rows),
        ),
        _plot_entry(
            plt,
            plots_dir / "tn_quantization_error_by_circuit_size.png",
            "CPU diagnostic TN replay quantization error",
            "tn_quantization_error_by_circuit_size.csv",
            lambda path: _plot_tn_quant_error(plt, path, tn_quant_error_rows),
        ),
        _plot_entry(
            plt,
            plots_dir / "upmem_boundary_status.png",
            "UPMEM SDK simulator boundary",
            "upmem_boundary_quantization.csv",
            lambda path: _plot_upmem_boundary(plt, path, upmem_rows),
        ),
        _plot_entry(
            plt,
            plots_dir / "upmem_accuracy_error_by_circuit_size.png",
            "UPMEM SDK simulator accuracy",
            "upmem_accuracy_by_circuit_size.csv",
            lambda path: _plot_upmem_accuracy(plt, path, upmem_accuracy_rows),
        ),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "plots": entries,
        "generated": [entry["plot"] for entry in entries if entry["status"] == "generated"],
        "skipped": [entry for entry in entries if entry["status"] == "skipped"],
    }


def benchmark_summary(
    title: str,
    records: list[JsonDict],
    cpu_gpu_rows: list[JsonDict],
    cpu_gpu_summary_rows: list[JsonDict],
    tn_path_rows: list[JsonDict],
    tn_path_summary_rows: list[JsonDict],
    tn_quant_rows: list[JsonDict],
    tn_quant_speedup_rows: list[JsonDict],
    tn_quant_error_rows: list[JsonDict],
    upmem_rows: list[JsonDict],
    upmem_accuracy_rows: list[JsonDict],
    plot_manifest: JsonDict,
) -> str:
    routes = sorted({str(record.get("route_id")) for record in records})
    lines = [
        f"# {title}",
        "",
        "This report is derived from explicit evidence inputs. It does not run benchmarks.",
        "",
        "## Evidence Inputs",
        "",
        f"- Normalized records loaded: {len(records)}",
        f"- Routes present: {', '.join(routes) if routes else 'none'}",
        "",
        "## Tables",
        "",
        f"- Full-state CPU/GPU matched rows: {len(cpu_gpu_rows)}",
        f"- Full-state CPU/GPU per-circuit-size rows: {len(cpu_gpu_summary_rows)}",
        f"- TN route/path rows: {len(tn_path_rows)}",
        f"- TN route/path per-circuit-size rows: {len(tn_path_summary_rows)}",
        f"- TN quantization matched rows: {len(tn_quant_rows)}",
        f"- TN quantization speedup per-circuit-size rows: {len(tn_quant_speedup_rows)}",
        f"- TN quantization error per-circuit-size rows: {len(tn_quant_error_rows)}",
        f"- UPMEM boundary rows: {len(upmem_rows)}",
        f"- UPMEM accuracy per-circuit-size rows: {len(upmem_accuracy_rows)}",
        "",
        "## Required Thesis Outputs",
        "",
        "- Full-state CPU/GPU runtime and speedup by circuit family and qubit size.",
        "- Tensor-network route/path runtime by circuit family and qubit size.",
        "- CPU diagnostic tensor-network replay runtime/error for float64 versus int8-dequantized complex128 replay.",
        "- UPMEM SDK simulator supported/unsupported boundary and accuracy rows.",
        "",
        "## Claims Allowed",
        "",
        "- QuEST CPU vs QuEST GPU rows are direct full-state route comparisons when GPU rows are verified.",
        "- Quimb rows are serious CPU tensor-network evidence.",
        "- CPU path replay rows are diagnostic path and quantization attribution evidence only.",
        "- CPU quantized replay slowdown/speedup rows describe CPU diagnostic replay, not native int8 hardware.",
        "- UPMEM SDK simulator rows are strict code-path/boundary evidence only.",
        "",
        "## Claims Not Allowed",
        "",
        "- QuEST full-state GPU only: these rows are not GPU tensor-network evidence.",
        "- QuEST GPU full-state rows are not GPU tensor-network evidence.",
        "- CPU path replay rows are not serious external TN baselines.",
        "- CPU quantized replay rows are not UPMEM or native int8 kernel performance evidence.",
        "- UPMEM SDK simulator timing is not hardware timing or hardware speedup.",
        "- No energy claim is made unless energy rows contain measured sensor data.",
        "",
        "## Plot Inventory",
        "",
    ]
    for entry in plot_manifest.get("plots", []):
        lines.append(f"- {entry['plot']}: {entry['status']}" + (f" ({entry['reason']})" if entry.get("reason") else ""))
    lines.append("")
    return "\n".join(lines)


def _records_by_case_repeat(records: list[JsonDict], route_id: str) -> dict[tuple[str, int], JsonDict]:
    result: dict[tuple[str, int], JsonDict] = {}
    for record in records:
        if record.get("route_id") != route_id:
            continue
        case_id = str(record.get("case_id") or "")
        repeat_id = _int_or_none(record.get("repeat_id"))
        if case_id and repeat_id is not None:
            result[(case_id, repeat_id)] = record
    return result


def _valid_cpu_gpu_pair(cpu: JsonDict, gpu: JsonDict) -> bool:
    return (
        _validation_ok(cpu)
        and _validation_ok(gpu)
        and gpu.get("gpu_backend_verified") is True
        and gpu.get("gpu_program_executed") is True
        and cpu.get("state_output_mode") == gpu.get("state_output_mode")
        and cpu.get("validation_method") == gpu.get("validation_method")
        and bool(cpu.get("performance_tier", False)) == bool(gpu.get("performance_tier", False))
    )


def _validation_ok(record: JsonDict) -> bool:
    return str(record.get("validation_status")) in {"passed", "passed_native_status", "passed_runtime_only"}


def _family_and_qubits(record: JsonDict) -> tuple[str, int | None]:
    case_id = str(record.get("case_id") or "")
    family = case_id
    for marker in ("_6q", "_8q", "_10q", "_12q", "_14q", "_16q", "_18q", "_20q", "_3q", "_4q", "_5q", "_7q"):
        if marker in case_id:
            family = case_id.split(marker, 1)[0]
            break
    qubits = _int_or_none(record.get("actual_n_qubits") or record.get("benchmark_n_qubits") or record.get("n_qubits"))
    if qubits is None:
        import re

        match = re.search(r"_(\d+)q(?:_|$)", case_id)
        if match:
            qubits = int(match.group(1))
    return family, qubits


def _validation_errors(record: JsonDict) -> JsonDict:
    payload = record.get("validation_error_metrics")
    if isinstance(payload, dict):
        result = dict(payload)
        if result.get("max_abs_error") is None and record.get("max_abs_error") is not None:
            result["max_abs_error"] = record.get("max_abs_error")
        if result.get("l2_error") is None and record.get("l2_error") is not None:
            result["l2_error"] = record.get("l2_error")
        return result
    if isinstance(payload, str) and payload:
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                if parsed.get("max_abs_error") is None and record.get("max_abs_error") is not None:
                    parsed["max_abs_error"] = record.get("max_abs_error")
                if parsed.get("l2_error") is None and record.get("l2_error") is not None:
                    parsed["l2_error"] = record.get("l2_error")
                return parsed
        except json.JSONDecodeError:
            pass
    return {
        key: record.get(key)
        for key in ("max_abs_error", "l2_error")
        if record.get(key) is not None
    }


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_number(record: JsonDict, *fields: str) -> int | float | None:
    for field in fields:
        value = record.get(field)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        return int(number) if number.is_integer() else number
    return None


def _thesis_route_label(record: JsonDict) -> str:
    route_id = str(record.get("route_id") or "")
    if route_id == "quimb_tn_exact":
        return "Quimb TN exact"
    if route_id == "quimb_tn_sliced_exact":
        return "Quimb/cotengra sliced TN exact"
    if route_id == "cpu_tn_path_replay_float64":
        return "CPU diagnostic TN path replay float64"
    if route_id == "cpu_tn_path_replay_int8_quantized":
        return "CPU diagnostic TN path replay int8-dequantized"
    if str(record.get("contraction_execution_target")) == "upmem":
        return _upmem_thesis_label(record)
    return route_id


def _upmem_thesis_label(record: JsonDict) -> str:
    quantization_mode = str(record.get("quantization_mode") or "not_applicable")
    if quantization_mode == "none":
        return "UPMEM SDK simulator generic float32/no quantization"
    if quantization_mode == "per_task_input_quantize":
        return "UPMEM SDK simulator generic int8 per-task quantized"
    return f"UPMEM SDK simulator generic {quantization_mode}"


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(to_jsonable(rows))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plot_entry(plt: Any, path: Path, title: str, source_csv: str, plotter: Any) -> JsonDict:
    reason = plotter(path)
    if reason:
        _write_todo_plot(plt, path, title, reason)
        return {
            "plot": path.name,
            "title": title,
            "status": "generated_todo_missing_data",
            "reason": reason,
            "source_csv": source_csv,
            "size_bytes": path.stat().st_size,
        }
    return {"plot": path.name, "title": title, "status": "generated", "reason": None, "source_csv": source_csv, "size_bytes": path.stat().st_size}


def _write_todo_plot(plt: Any, path: Path, title: str, reason: str) -> None:
    """Keep the expected report surface without inventing a missing value."""
    fig, axis = plt.subplots(figsize=(8, 4.5), dpi=160)
    axis.set_title(title)
    axis.set_xlabel("Evidence input")
    axis.set_ylabel("Value")
    axis.text(
        0.5,
        0.5,
        f"TODO\n{reason}",
        ha="center",
        va="center",
        wrap=True,
        transform=axis.transAxes,
        color="#b45309",
        fontsize=14,
        weight="bold",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_speedup(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_matched_cpu_gpu_rows"
    grouped = _points_by_family(rows, "compute_speedup_median_cpu_over_gpu")
    return _line_plot(plt, path, grouped, "Full-state CPU/GPU compute speedup", "CPU/GPU speedup")


def _plot_full_state_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_matched_cpu_gpu_rows"
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        family = str(row.get("case_family"))
        qubits = _int_or_none(row.get("n_qubits"))
        cpu = _positive(row.get("cpu_simulation_compute_time_median_s"))
        gpu = _positive(row.get("gpu_simulation_compute_time_median_s"))
        if qubits is None:
            continue
        if cpu is not None:
            grouped.setdefault(f"{family} CPU", []).append((qubits, cpu))
        if gpu is not None:
            grouped.setdefault(f"{family} GPU", []).append((qubits, gpu))
    return _line_plot(plt, path, grouped, "Full-state CPU/GPU compute runtime", "Median compute time (s)", log_y=True)


def _plot_tn_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if _positive(row.get("simulation_compute_time_median_s")) is not None]
    if not selected:
        return "no_tn_runtime_rows"
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in selected:
        qubits = _int_or_none(row.get("n_qubits"))
        if qubits is None:
            continue
        label = f"{row.get('case_family')} {row.get('route_id')}"
        grouped.setdefault(label, []).append((qubits, float(row["simulation_compute_time_median_s"])))
    return _line_plot(plt, path, grouped, "Tensor-network route runtime", "Median compute time (s)", log_y=True)


def _plot_tn_quant_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_tn_quantization_rows"
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        qubits = _int_or_none(row.get("n_qubits"))
        unquant = _positive(row.get("unquantized_simulation_compute_time_median_s"))
        quant = _positive(row.get("quantized_simulation_compute_time_median_s"))
        family = str(row.get("case_family"))
        if qubits is None:
            continue
        if unquant is not None:
            grouped.setdefault(f"{family} CPU float64 replay", []).append((qubits, unquant))
        if quant is not None:
            grouped.setdefault(f"{family} CPU int8-dequantized replay", []).append((qubits, quant))
    return _line_plot(plt, path, grouped, "CPU diagnostic TN replay runtime", "Median CPU compute time (s)", log_y=True)


def _plot_tn_quant_error(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_tn_quantization_rows"
    grouped = _points_by_family(rows, "max_abs_error_vs_reference_median")
    return _line_plot(plt, path, grouped, "CPU diagnostic TN replay quantization error", "Max abs error", log_y=True)


def _plot_upmem_boundary(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_upmem_rows"
    counts: dict[str, int] = {}
    for row in rows:
        status = "supported" if row.get("validation_status") in {"passed", "passed_native_status", "passed_runtime_only"} else "unsupported"
        counts[status] = counts.get(status, 0) + 1
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.bar(list(counts), list(counts.values()), color=["#2563eb", "#dc2626"][: len(counts)])
    ax.set_title("UPMEM SDK simulator boundary")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return None


def _plot_upmem_accuracy(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if _positive(row.get("max_abs_error_median")) is not None]
    if not selected:
        return "no_upmem_accuracy_rows"
    grouped = _points_by_family(selected, "max_abs_error_median")
    return _line_plot(plt, path, grouped, "UPMEM SDK simulator max absolute error", "Max abs error", log_y=True)


def _points_by_family(rows: list[JsonDict], field: str) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        qubits = _int_or_none(row.get("n_qubits"))
        value = _positive(row.get(field))
        if qubits is None or value is None:
            continue
        grouped.setdefault(str(row.get("case_family")), []).append((qubits, value))
    return grouped


def _group_rows(rows: list[JsonDict], *fields: str) -> dict[tuple[Any, ...], list[JsonDict]]:
    grouped: dict[tuple[Any, ...], list[JsonDict]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        grouped.setdefault(key, []).append(row)
    return grouped


def _numbers(rows: list[JsonDict], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _positive(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _first_present(rows: list[JsonDict], field: str) -> Any:
    for row in rows:
        value = row.get(field)
        if value is not None and value != "":
            return value
    return None


def _family_qubits_sort_key(row: JsonDict | tuple[Any, ...]) -> tuple[str, int, str]:
    if isinstance(row, tuple):
        family = str(row[0])
        qubits = _int_or_none(row[1]) if len(row) > 1 else None
        extra = str(row[2]) if len(row) > 2 else ""
    else:
        family = str(row.get("case_family"))
        qubits = _int_or_none(row.get("n_qubits"))
        extra = str(row.get("route_id") or row.get("path_strategy") or row.get("quantization_mode") or "")
    return (family, qubits if qubits is not None else -1, extra)


def _median_by_family_qubits(rows: list[JsonDict], field: str) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        qubits = _int_or_none(row.get("n_qubits"))
        value = _positive(row.get(field))
        if qubits is None or value is None:
            continue
        grouped.setdefault((str(row.get("case_family")), qubits), []).append(value)
    result: dict[str, list[tuple[int, float]]] = {}
    for (family, qubits), values in grouped.items():
        result.setdefault(family, []).append((qubits, statistics.median(values)))
    return result


def _line_plot(plt: Any, path: Path, grouped: dict[str, list[tuple[int, float]]], title: str, ylabel: str, *, log_y: bool = False) -> str | None:
    if not grouped:
        return "no_plot_rows"
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for label, points in sorted(grouped.items()):
        ordered = sorted(points)
        ax.plot([x for x, _ in ordered], [y for _, y in ordered], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Qubits")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
