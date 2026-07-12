from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_bench.bench.result_artifacts import load_result_records  # noqa: E402


SCHEMA_VERSION = "research_benchmark_pack_v1"
RESEARCH_PACK_KIND = "research_benchmark_pack"
DEFAULT_COMPARISON_ROOT = ROOT / "runs" / "comparisons" / "research_pack"

RESEARCH_SUITES = {
    "cpu_gpu": ROOT / "configs" / "suites" / "manual" / "thesis_full_state_cpu_gpu.yml",
    "cpu_gpu_correctness": ROOT / "configs" / "suites" / "manual" / "thesis_full_state_correctness.yml",
    "cpu_tn": ROOT / "configs" / "suites" / "manual" / "thesis_cpu_tn_quimb.yml",
    "tn_path_quantization": ROOT / "configs" / "suites" / "manual" / "thesis_tn_paths_quantization.yml",
    "planner_paths": ROOT / "configs" / "suites" / "manual" / "thesis_planner_compare.yml",
    # This group intentionally uses the strict generic-only MVP command rather
    # than the route-comparison suite.  The latter permits dense bridge tasks,
    # which is useful for route coverage but is not generic-TN boundary evidence.
    "upmem_boundary": ROOT / "configs" / "suites" / "manual" / "thesis_upmem_quantization_boundary.yml",
    "upmem_quantization_stress": ROOT / "configs" / "suites" / "manual" / "thesis_upmem_quantization_stress.yml",
    "internal_parallelism": ROOT / "configs" / "suites" / "manual" / "research_internal_parallelism.yml",
}

SUITE_COMMAND_ORDER = (
    "cpu_gpu_correctness",
    "cpu_gpu",
    "cpu_tn",
    "tn_path_quantization",
    "planner_paths",
    "upmem_boundary",
    "upmem_quantization_stress",
    "internal_parallelism",
)

RELEVANT_ENV_VARS = (
    "BENCH_CPU_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantum_bench_mplconfig")

FORBIDDEN_EVIDENCE_DERIVED_NAMES = {
    "comparison_summary.md",
    "simulation_backend_compare_results.csv",
    "simulation_backend_compare_pairs.csv",
    "upmem_mvp_benchmark_results.csv",
    "kernel_family_summary.csv",
    "quantization_accuracy_summary.csv",
    "unsupported_reasons.csv",
    "quantization_comparison.csv",
    "upmem_quantization_attribution.csv",
    "cpu_gpu_performance_summary.csv",
    "full_state_tn_comparison.csv",
    "per_case_route_stats.csv",
    "paired_speedups.csv",
    "unsupported_cases.csv",
    "validation_summary.csv",
    "route_capability_matrix.csv",
    "benchmark_summary.md",
    "plot_manifest.json",
}


JsonDict = dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a thesis research benchmark pack from normalized records.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print exact research benchmark commands and preflight context.")
    _add_common_args(plan)

    run = subparsers.add_parser("run", help="Create a research pack; long benchmark execution requires --full or RUN_RESEARCH=1.")
    _add_common_args(run)
    run.add_argument("--full", action="store_true", help="Run long manual research suites.")
    run.add_argument("--suite", action="append", choices=sorted(RESEARCH_SUITES), help="Limit full execution/report discovery to a suite group.")

    report = subparsers.add_parser("report", help="Generate a research pack from existing evidence runs.")
    _add_common_args(report)
    report.add_argument("--input", action="append", dest="inputs", help="Evidence run directory or artifact path. May be repeated.")
    report.add_argument("--suite", action="append", choices=sorted(RESEARCH_SUITES), help="Auto-discover latest evidence for a suite group.")

    args = parser.parse_args(argv)
    if args.command == "plan":
        print_plan(args.root)
        return 0
    if args.command == "run":
        return run_pack(args.root, args.out, suite_filter=args.suite, full=bool(args.full or os.environ.get("RUN_RESEARCH") == "1"))
    if args.command == "report":
        return report_pack(args.root, args.out, inputs=[Path(item) for item in args.inputs or ()], suite_filter=args.suite)
    raise AssertionError(f"unknown command: {args.command}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=ROOT, help="Implementation root directory.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for generated research pack.")


def print_plan(root: Path = ROOT) -> None:
    print("Research benchmark pack plan")
    print("============================")
    print("")
    print("Preflight:")
    print(_command(["scripts/doctor.py"], python_script=True))
    print("")
    print("GPU verification before GPU research rows:")
    print(_bench_command(["simulation-backend-probe", "--verify-gpu", "quest-hip"]))
    print("")
    print("Manual research suites:")
    for key in SUITE_COMMAND_ORDER:
        print(f"- {key}:")
        print(f"  {_bench_command(_research_suite_argv(key, root))}")
    print("")
    print("Pack commands:")
    print("  BENCH_CPU_THREADS=<physical-core-count> make thesis-run")
    print("  make thesis-promote")
    print("  make thesis-report")
    print("")
    print("Derived artifacts are written under runs/comparisons/research_pack/<timestamp>/; evidence runs remain read-only.")


def run_pack(root: Path, out: Path | None, *, suite_filter: list[str] | None, full: bool) -> int:
    out_dir = _pack_dir(root, out)
    selected = _selected_suites(suite_filter)
    command_results: list[JsonDict] = []
    evidence_inputs: list[Path] = []
    if full:
        command_results.append(_run_capture(root, ["make", "doctor"]))
        gpu_verified = True
        if "cpu_gpu" in selected or "cpu_gpu_correctness" in selected:
            command_results.append(_run_capture(root, _bench_argv(["simulation-backend-probe", "--verify-gpu", "quest-hip"])))
            gpu_verified = _gpu_verification_passed(root)
        for key in SUITE_COMMAND_ORDER:
            if key not in selected:
                continue
            if key in {"cpu_gpu", "cpu_gpu_correctness"} and not gpu_verified:
                command_results.append(_skipped_group_result(key, _gpu_blocker_reason(root)))
                continue
            result = _run_capture(root, _bench_argv(_research_suite_argv(key, root)))
            command_results.append(result)
            latest = root / "runs" / "latest"
            if result["returncode"] == 0 and latest.exists():
                evidence_inputs.append(latest.resolve())
    else:
        command_results.append(
            {
                "command": "lightweight preflight only",
                "returncode": 0,
                "stdout": "Set RUN_RESEARCH=1 or pass --full to run long manual suites.",
                "stderr": "",
            }
        )
    return _write_pack(root, out_dir, evidence_inputs, command_results=command_results, selected_suite_keys=selected, generation_mode="run")


def report_pack(root: Path, out: Path | None, *, inputs: list[Path], suite_filter: list[str] | None) -> int:
    out_dir = _pack_dir(root, out)
    selected = _selected_suites(suite_filter)
    evidence_inputs = [path.resolve() for path in inputs]
    if not evidence_inputs:
        for key in selected:
            evidence = _latest_evidence_for_suite(root, RESEARCH_SUITES[key].stem)
            if evidence is not None:
                evidence_inputs.append(evidence)
    command_results = [
        {
            "command": "report existing evidence",
            "returncode": 0,
            "stdout": f"inputs={len(evidence_inputs)}",
            "stderr": "",
        }
    ]
    return _write_pack(root, out_dir, evidence_inputs, command_results=command_results, selected_suite_keys=selected, generation_mode="report")


def _write_pack(
    root: Path,
    out_dir: Path,
    evidence_inputs: list[Path],
    *,
    command_results: list[JsonDict],
    selected_suite_keys: list[str],
    generation_mode: str = "report",
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_result_records(evidence_inputs) if evidence_inputs else []
    boundary = validate_artifact_boundaries(root)
    guard_issues = _claim_guard_issues(records)
    manifest = build_manifest(
        root,
        command_results=command_results,
        evidence_inputs=evidence_inputs,
        record_count=len(records),
        selected_suite_keys=selected_suite_keys,
        generation_mode=generation_mode,
    )
    _write_json(out_dir / "benchmark_manifest.json", manifest)
    stats_rows = per_case_route_stats(records)
    speedup_rows = paired_speedups(records)
    cpu_gpu_rows = cpu_gpu_performance_summary(speedup_rows)
    full_state_tn_rows = full_state_tn_comparison(stats_rows)
    quantization_rows = upmem_quantization_attribution(records)
    same_plan_rows = same_plan_execution(records)
    planner_rows = planner_comparison(records)
    unsupported_rows = unsupported_cases(records)
    validation_rows = validation_summary(records)
    capability_rows = route_capability_matrix(records)
    _write_csv(out_dir / "per_case_route_stats.csv", stats_rows, PER_CASE_ROUTE_STATS_FIELDS)
    _write_csv(out_dir / "paired_speedups.csv", speedup_rows, PAIRED_SPEEDUP_FIELDS)
    _write_csv(out_dir / "cpu_gpu_performance_summary.csv", cpu_gpu_rows, CPU_GPU_PERFORMANCE_SUMMARY_FIELDS)
    _write_csv(out_dir / "full_state_tn_comparison.csv", full_state_tn_rows, FULL_STATE_TN_COMPARISON_FIELDS)
    _write_csv(out_dir / "upmem_quantization_attribution.csv", quantization_rows, UPMEM_QUANTIZATION_ATTRIBUTION_FIELDS)
    _write_csv(out_dir / "same_plan_execution.csv", same_plan_rows, SAME_PLAN_EXECUTION_FIELDS)
    _write_csv(out_dir / "planner_comparison.csv", planner_rows, PLANNER_COMPARISON_FIELDS)
    _write_csv(out_dir / "unsupported_cases.csv", unsupported_rows, UNSUPPORTED_FIELDS)
    _write_csv(out_dir / "validation_summary.csv", validation_rows, VALIDATION_SUMMARY_FIELDS)
    _write_csv(out_dir / "route_capability_matrix.csv", capability_rows, ROUTE_CAPABILITY_FIELDS)
    plot_manifest = write_plots(out_dir, stats_rows, cpu_gpu_rows, quantization_rows, same_plan_rows, planner_rows)
    _write_json(out_dir / "plot_manifest.json", plot_manifest)
    (out_dir / "benchmark_summary.md").write_text(
        benchmark_summary(
            manifest,
            records,
            stats_rows,
            speedup_rows,
            quantization_rows,
            planner_rows,
            full_state_tn_rows,
            unsupported_rows,
            validation_rows,
            capability_rows,
            plot_manifest,
            boundary,
            guard_issues,
        ),
        encoding="utf-8",
    )
    if _is_within(out_dir, root / "runs" / "comparisons"):
        _update_latest_link(out_dir.parent, out_dir)
    print(out_dir)
    return 1 if boundary["status"] == "failed" or guard_issues else 0


def build_manifest(
    root: Path,
    *,
    command_results: list[JsonDict],
    evidence_inputs: list[Path],
    record_count: int,
    selected_suite_keys: list[str],
    generation_mode: str = "report",
) -> JsonDict:
    command_line = " ".join(sys.argv)
    generated_at = datetime.now().isoformat(timespec="seconds")
    source = _evidence_source_provenance(evidence_inputs)
    report_commit = _git(root, ["rev-parse", "HEAD"])
    report_dirty = bool(_git(root, ["status", "--short", "--", "."]))
    report_repository_dirty = bool(_git(root, ["status", "--short"]))
    provenance = {
        "generator": RESEARCH_PACK_KIND,
        "script": "scripts/research_benchmark_pack.py",
        "schema_version": SCHEMA_VERSION,
        "mode": generation_mode,
        "generated_at": generated_at,
        "command_line": command_line,
        "input_count": len(evidence_inputs),
        "input_paths": [path.as_posix() for path in evidence_inputs],
        "source_records_read_only": True,
        "commit": report_commit,
        "worktree_dirty": report_dirty,
        "repository_worktree_dirty": report_repository_dirty,
        "scope": "thesis/implementation",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": RESEARCH_PACK_KIND,
        "created_at": generated_at,
        "root": root.as_posix(),
        "git_commit": source["commit"],
        "dirty_tree": source["worktree_dirty"],
        "dirty_worktree": source["worktree_dirty"],
        "benchmark_source_commit": source["commit"],
        "benchmark_source_commits": source["commits"],
        "benchmark_source_worktree_dirty": source["worktree_dirty"],
        "repository_worktree_dirty": source["repository_worktree_dirty"],
        "provenance_scope": "thesis/implementation",
        "report_generation_commit": report_commit,
        "report_generation_worktree_dirty": report_dirty,
        "report_generation_repository_worktree_dirty": report_repository_dirty,
        "command_line": command_line,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "packages": {name: _package_version(name) for name in ("numpy", "quimb", "cotengra", "opt_einsum", "matplotlib")},
        "cpu": _cpu_metadata(),
        "environment": {name: os.environ.get(name) for name in RELEVANT_ENV_VARS},
        "selected_suites": {key: _display_path(RESEARCH_SUITES[key], root) for key in selected_suite_keys},
        "evidence_inputs": [path.as_posix() for path in evidence_inputs],
        "record_count": record_count,
        "commands": command_results,
        "report_generation_provenance": provenance,
        "report_generation": provenance,
        "report_generation_script": provenance["script"],
        "report_generation_mode": generation_mode,
        "report_generation_timestamp": generated_at,
        "report_generation_command": command_line,
        "report_generation_input_paths": [path.as_posix() for path in evidence_inputs],
        "gpu_verification": _read_optional_json(root / "build" / "gpu_verification" / "quest_gpu_full_state_exact.json"),
        "notes": {
            "long_runs_require_explicit_opt_in": True,
            "derived_outputs_are_under_runs_comparisons": True,
            "evidence_inputs_are_read_only": True,
        },
    }


PER_CASE_ROUTE_STATS_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "n_qubits",
    "actual_n_qubits",
    "benchmark_n_qubits",
    "actual_n_qubits_source",
    "actual_n_qubits_warning",
    "route_id",
    "benchmark_role",
    "backend_family",
    "execution_model",
    "parallelism_mode",
    "circuit_semantics_hash",
    "tensor_network_hash",
    "contraction_plan_hash",
    "contraction_execution_target",
    "upmem_execution_mode",
    "policy",
    "quantization_mode",
    "generic_only_all_tasks_used_generic_backend",
    "valid_primary_upmem_codepath_result",
    "upmem_program_executed",
    "dpu_program_invocations",
    "state_output_mode",
    "validation_method",
    "performance_tier",
    "timing_scope",
    "repeat_count",
    "total_wall_time_s_median",
    "total_wall_time_s_p25",
    "total_wall_time_s_p75",
    "total_wall_time_s_iqr",
    "total_wall_time_s_mean",
    "total_wall_time_s_min",
    "total_wall_time_s_max",
    "total_wall_time_s_std",
    "total_host_residual_time_s_median",
    "total_host_residual_time_s_p25",
    "total_host_residual_time_s_p75",
    "total_host_residual_time_s_iqr",
    "total_host_residual_time_s_mean",
    "total_host_residual_time_s_min",
    "total_host_residual_time_s_max",
    "total_host_residual_time_s_std",
    "reported_time_s_median",
    "reported_time_s_mean",
    "reported_time_s_min",
    "reported_time_s_max",
    "reported_time_s_std",
    "timing_basis",
    "simulation_compute_time_s_median",
    "simulation_compute_time_s_p25",
    "simulation_compute_time_s_p75",
    "simulation_compute_time_s_iqr",
    "simulation_compute_time_s_mean",
    "simulation_compute_time_s_min",
    "simulation_compute_time_s_max",
    "simulation_compute_time_s_std",
    "planning_time_s_median",
    "planning_time_s_mean",
    "planning_time_s_min",
    "planning_time_s_max",
    "planning_time_s_std",
    "actual_transfer_bytes_median",
    "tn_estimated_flops",
    "tn_max_intermediate_bytes",
    "validation_passed_count",
    "validation_failed_count",
    "unsupported_count",
    "slice_count",
    "slicing_flop_ratio",
    "slicing_flop_change_kind",
    "max_abs_error",
    "l2_error",
    "execution_max_abs_error",
    "execution_l2_error",
    "full_precision_max_abs_error",
    "full_precision_l2_error",
    "hardware_speedup_applicable",
    "gpu_backend_verified",
    "gpu_device_name",
    "cpu_fallback_used",
    "resource_skip_reason",
]

UPMEM_QUANTIZATION_ATTRIBUTION_FIELDS = [
    "schema_version",
    "run_id",
    "suite_id",
    "case_id",
    "case_family",
    "n_qubits",
    "actual_n_qubits",
    "benchmark_n_qubits",
    "actual_n_qubits_source",
    "actual_n_qubits_warning",
    "repeat_id",
    "route_id",
    "policy",
    "same_route_comparison",
    "same_taskgraph",
    "same_kernel_family",
    "unquantized_total_wall_time_s",
    "quantized_total_wall_time_s",
    "unquantized_host_residual_time_s",
    "quantized_host_residual_time_s",
    "route_runtime_ratio_none_over_quantized",
    "unquantized_simulation_compute_time_s",
    "quantized_simulation_compute_time_s",
    "simulator_kernel_ratio_none_over_quantized",
    "unquantized_transfer_bytes",
    "quantized_transfer_bytes",
    "transfer_ratio_none_over_quantized",
    "unquantized_max_abs_error_vs_full_precision",
    "quantized_max_abs_error_vs_full_precision",
    "unquantized_execution_max_abs_error",
    "quantized_execution_max_abs_error",
    "unquantized_full_precision_max_abs_error",
    "quantized_full_precision_max_abs_error",
    "accuracy_delta_quantized_minus_unquantized",
    "native_unquantized_upmem_kernel_executed",
]

PAIRED_SPEEDUP_FIELDS = [
    "schema_version",
    "case_id",
    "case_family",
    "n_qubits",
    "actual_n_qubits",
    "benchmark_n_qubits",
    "actual_n_qubits_source",
    "actual_n_qubits_warning",
    "repeat_id",
    "cpu_route_id",
    "gpu_route_id",
    "timing_scope",
    "state_output_mode",
    "validation_method",
    "performance_tier",
    "cpu_total_wall_time_s",
    "gpu_total_wall_time_s",
    "wall_time_ratio_cpu_over_gpu",
    "cpu_total_host_residual_time_s",
    "gpu_total_host_residual_time_s",
    "cpu_simulation_compute_time_s",
    "gpu_simulation_compute_time_s",
    "compute_speedup_cpu_over_gpu",
    "gpu_device_name",
]

CPU_GPU_PERFORMANCE_SUMMARY_FIELDS = [
    "schema_version",
    "case_id",
    "case_family",
    "n_qubits",
    "actual_n_qubits",
    "benchmark_n_qubits",
    "actual_n_qubits_source",
    "actual_n_qubits_warning",
    "matched_repeat_count",
    "timing_scope",
    "cpu_simulation_compute_time_s_median",
    "cpu_simulation_compute_time_s_p25",
    "cpu_simulation_compute_time_s_p75",
    "cpu_simulation_compute_time_s_iqr",
    "gpu_simulation_compute_time_s_median",
    "gpu_simulation_compute_time_s_p25",
    "gpu_simulation_compute_time_s_p75",
    "gpu_simulation_compute_time_s_iqr",
    "compute_speedup_cpu_over_gpu_median",
    "compute_speedup_cpu_over_gpu_p25",
    "compute_speedup_cpu_over_gpu_p75",
    "compute_speedup_cpu_over_gpu_iqr",
    "cpu_total_wall_time_s_median",
    "cpu_total_wall_time_s_p25",
    "cpu_total_wall_time_s_p75",
    "cpu_total_wall_time_s_iqr",
    "gpu_total_wall_time_s_median",
    "gpu_total_wall_time_s_p25",
    "gpu_total_wall_time_s_p75",
    "gpu_total_wall_time_s_iqr",
    "wall_time_ratio_cpu_over_gpu_median",
    "wall_time_ratio_cpu_over_gpu_p25",
    "wall_time_ratio_cpu_over_gpu_p75",
    "wall_time_ratio_cpu_over_gpu_iqr",
    "cpu_total_host_residual_time_s_median",
    "cpu_total_host_residual_time_s_p25",
    "cpu_total_host_residual_time_s_p75",
    "cpu_total_host_residual_time_s_iqr",
    "gpu_total_host_residual_time_s_median",
    "gpu_total_host_residual_time_s_p25",
    "gpu_total_host_residual_time_s_p75",
    "gpu_total_host_residual_time_s_iqr",
    "compute_speedup_cpu_over_gpu_crossover_qubit",
    "crossover_qubit",
    "gpu_device_name",
]

FULL_STATE_TN_COMPARISON_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "quest_cpu_compute_time_s_median",
    "quimb_unsliced_compute_time_s_median",
    "quimb_sliced_compute_time_s_median",
    "quimb_unsliced_time_over_quest_time",
    "quimb_sliced_time_over_quest_time",
    "quimb_sliced_time_over_unsliced_time",
    "quest_validation_passed_count",
    "quimb_unsliced_validation_passed_count",
    "quimb_sliced_validation_passed_count",
]

UNSUPPORTED_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "route_id",
    "benchmark_role",
    "validation_status",
    "status",
    "unsupported_task_count",
    "resource_skip_reason",
    "warnings",
]

VALIDATION_SUMMARY_FIELDS = [
    "schema_version",
    "route_id",
    "validation_status",
    "record_count",
]

ROUTE_CAPABILITY_FIELDS = [
    "schema_version",
    "route_id",
    "benchmark_role",
    "backend_family",
    "execution_model",
    "contraction_execution_target",
    "accelerator_kind",
    "upmem_execution_mode",
    "record_count",
    "completed_count",
    "unsupported_count",
    "validation_passed_count",
    "gpu_verified_count",
    "cpu_fallback_count",
    "hardware_speedup_applicable_count",
]

SAME_PLAN_EXECUTION_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "contraction_plan_hash",
    "cpu_route_id",
    "upmem_route_id",
    "quantization_mode",
    "cpu_time_s",
    "upmem_simulator_time_s",
    "route_time_ratio_cpu_over_upmem_simulator",
    "actual_transfer_bytes",
    "max_abs_error",
    "same_plan_verified",
    "hardware_speedup_applicable",
]

PLANNER_COMPARISON_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "planner_id",
    "optimize_mode",
    "contraction_plan_hash",
    "planning_time_s",
    "task_count",
    "tn_estimated_flops",
    "tn_max_intermediate_bytes",
    "total_host_to_dpu_bytes",
    "total_dpu_to_host_bytes",
    "total_mram_to_wram_bytes",
    "tiling_required_task_count",
    "estimated_total_tile_count",
    "estimated_max_parallel_tiles",
    "upmem_pressure_score",
    "upmem_rank",
    "flop_rank",
    "parallelism_evidence_type",
    "execution_plan_executed",
]


def per_case_route_stats(records: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[Any, ...], list[JsonDict]] = defaultdict(list)
    for record in records:
        key = (
            record.get("suite_id"),
            record.get("case_id"),
            record.get("route_id"),
            _record_value(record, "policy"),
            _record_value(record, "quantization_mode"),
            record.get("state_output_mode"),
            record.get("validation_method"),
            bool(record.get("performance_tier", False)),
            record.get("timing_scope"),
        )
        grouped[key].append(record)
    rows: list[JsonDict] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        first = group[0]
        total_values = _numbers(row.get("total_wall_time_s") for row in group)
        residual_values = _numbers(row.get("total_host_residual_time_s") for row in group)
        reported_values = [_reported_time(row)[0] for row in group]
        reported_values = [value for value in reported_values if value is not None]
        compute_values = _numbers(row.get("simulation_compute_time_s") for row in group)
        planning_values = _numbers(row.get("planning_time_s") for row in group)
        transfer_values = _numbers(row.get("actual_transfer_bytes") for row in group)
        family, qubits = _family_and_qubits(first)
        errors = [_validation_errors(row) for row in group]
        full_precision_errors = [_full_precision_errors(row) for row in group]
        selected_accuracy_errors = [
            _accuracy_errors_for_reporting(row)
            for row in group
        ]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": first.get("suite_id"),
                "case_id": first.get("case_id"),
                "case_family": family,
                "n_qubits": qubits["actual_n_qubits"],
                **qubits,
                "route_id": first.get("route_id"),
                "benchmark_role": first.get("benchmark_role"),
                "backend_family": first.get("backend_family"),
                "execution_model": first.get("execution_model"),
                "parallelism_mode": first.get("parallelism_mode"),
                "circuit_semantics_hash": first.get("circuit_semantics_hash"),
                "tensor_network_hash": first.get("tensor_network_hash"),
                "contraction_plan_hash": first.get("contraction_plan_hash"),
                "contraction_execution_target": first.get("contraction_execution_target"),
                "upmem_execution_mode": first.get("upmem_execution_mode"),
                "policy": _record_value(first, "policy"),
                "quantization_mode": _record_value(first, "quantization_mode"),
                "generic_only_all_tasks_used_generic_backend": _record_value(first, "generic_only_all_tasks_used_generic_backend"),
                "valid_primary_upmem_codepath_result": _record_value(first, "valid_primary_upmem_codepath_result"),
                "upmem_program_executed": _record_value(first, "upmem_program_executed"),
                "dpu_program_invocations": _record_value(first, "dpu_program_invocations"),
                "state_output_mode": first.get("state_output_mode"),
                "validation_method": first.get("validation_method"),
                "performance_tier": bool(first.get("performance_tier", False)),
                "timing_scope": first.get("timing_scope"),
                "repeat_count": len(group),
                **_stats("total_wall_time_s", total_values),
                **_stats("total_host_residual_time_s", residual_values),
                **_stats("reported_time_s", reported_values),
                "timing_basis": _reported_timing_basis(group),
                **_stats("simulation_compute_time_s", compute_values),
                **_stats("planning_time_s", planning_values),
                **_stats("actual_transfer_bytes", transfer_values),
                "tn_estimated_flops": _first_present(group, "tn_estimated_flops"),
                "tn_max_intermediate_bytes": _first_present(group, "tn_max_intermediate_bytes"),
                "validation_passed_count": sum(1 for row in group if str(row.get("validation_status")) in {"passed", "passed_native_status", "passed_runtime_only"}),
                "validation_failed_count": sum(1 for row in group if str(row.get("validation_status")) not in {"passed", "passed_native_status", "passed_runtime_only", "skipped"}),
                "unsupported_count": sum(1 for row in group if _is_unsupported(row)),
                "slice_count": _first_present(group, "slice_count"),
                "slicing_flop_ratio": _first_present(group, "slicing_flop_ratio"),
                "slicing_flop_change_kind": _first_present(group, "slicing_flop_change_kind"),
                "max_abs_error": _max_number(error.get("max_abs_error") for error in selected_accuracy_errors),
                "l2_error": _max_number(error.get("l2_error") for error in selected_accuracy_errors),
                "execution_max_abs_error": _max_number(error.get("max_abs_error") for error in errors),
                "execution_l2_error": _max_number(error.get("l2_error") for error in errors),
                "full_precision_max_abs_error": _max_number(error.get("max_abs_error") for error in full_precision_errors),
                "full_precision_l2_error": _max_number(error.get("l2_error") for error in full_precision_errors),
                "hardware_speedup_applicable": any(bool(row.get("hardware_speedup_applicable", False)) for row in group),
                "gpu_backend_verified": any(bool(row.get("gpu_backend_verified", False)) for row in group),
                "gpu_device_name": _first_present(group, "gpu_device_name"),
                "cpu_fallback_used": any(bool(row.get("cpu_fallback_used", False)) for row in group),
                "resource_skip_reason": _first_record_value(group, "resource_skip_reason") or _first_record_value(group, "reason"),
            }
        )
    return rows


def paired_speedups(records: list[JsonDict]) -> list[JsonDict]:
    cpu_records: dict[tuple[str, int], JsonDict] = {}
    gpu_records: dict[tuple[str, int], JsonDict] = {}
    for record in records:
        route_id = str(record.get("route_id") or "")
        repeat_id = _int_or_none(record.get("repeat_id"))
        case_id = str(record.get("case_id") or "")
        if repeat_id is None or not case_id:
            continue
        if route_id == "quest_cpu_full_state_exact":
            cpu_records[(case_id, repeat_id)] = record
        elif route_id == "quest_gpu_full_state_exact":
            gpu_records[(case_id, repeat_id)] = record
    rows: list[JsonDict] = []
    for key in sorted(set(cpu_records) & set(gpu_records)):
        cpu = cpu_records[key]
        gpu = gpu_records[key]
        if not _valid_for_pair(cpu) or not _valid_for_pair(gpu):
            continue
        if not (gpu.get("gpu_backend_verified") is True and gpu.get("gpu_program_executed") is True):
            continue
        if str(cpu.get("state_output_mode") or "") != str(gpu.get("state_output_mode") or ""):
            continue
        if str(cpu.get("validation_method") or "") != str(gpu.get("validation_method") or ""):
            continue
        if bool(cpu.get("performance_tier", False)) != bool(gpu.get("performance_tier", False)):
            continue
        cpu_total = _positive(cpu.get("total_wall_time_s"))
        gpu_total = _positive(gpu.get("total_wall_time_s"))
        cpu_compute = _positive(cpu.get("simulation_compute_time_s"))
        gpu_compute = _positive(gpu.get("simulation_compute_time_s"))
        if None in {cpu_total, gpu_total, cpu_compute, gpu_compute}:
            continue
        family, qubits = _family_and_qubits(cpu)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": key[0],
                "case_family": family,
                "n_qubits": qubits["actual_n_qubits"],
                **qubits,
                "repeat_id": key[1],
                "cpu_route_id": "quest_cpu_full_state_exact",
                "gpu_route_id": "quest_gpu_full_state_exact",
                "timing_scope": "performance_compute" if bool(cpu.get("performance_tier", False)) else "correctness_wall_and_compute",
                "state_output_mode": cpu.get("state_output_mode"),
                "validation_method": cpu.get("validation_method"),
                "performance_tier": bool(cpu.get("performance_tier", False)),
                "cpu_total_wall_time_s": cpu_total,
                "gpu_total_wall_time_s": gpu_total,
                "wall_time_ratio_cpu_over_gpu": cpu_total / gpu_total,
                "cpu_total_host_residual_time_s": _float_or_none(cpu.get("total_host_residual_time_s")),
                "gpu_total_host_residual_time_s": _float_or_none(gpu.get("total_host_residual_time_s")),
                "cpu_simulation_compute_time_s": cpu_compute,
                "gpu_simulation_compute_time_s": gpu_compute,
                "compute_speedup_cpu_over_gpu": cpu_compute / gpu_compute,
                "gpu_device_name": gpu.get("gpu_device_name"),
            }
        )
    return rows


def cpu_gpu_performance_summary(speedup_rows: list[JsonDict]) -> list[JsonDict]:
    """Summarize matched performance-tier repeats for thesis CPU/GPU figures."""
    grouped: dict[tuple[str, str, int | None], list[JsonDict]] = defaultdict(list)
    for row in speedup_rows:
        if not _bool(row.get("performance_tier")):
            continue
        qubits = _plot_qubits(row)
        if qubits is None:
            continue
        grouped[(str(row.get("case_id") or ""), str(row.get("case_family") or ""), qubits)].append(row)

    summary: list[JsonDict] = []
    for (case_id, family, qubits), group in sorted(grouped.items()):
        first = group[0]
        summary.append({
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "case_family": family,
                "n_qubits": qubits,
                "actual_n_qubits": qubits,
                "benchmark_n_qubits": qubits,
                "actual_n_qubits_source": first.get("actual_n_qubits_source"),
                "actual_n_qubits_warning": first.get("actual_n_qubits_warning"),
                "matched_repeat_count": len(group),
                "timing_scope": "performance_compute",
                "gpu_device_name": _first_present(group, "gpu_device_name"),
                **_stats("cpu_simulation_compute_time_s", _numbers(row.get("cpu_simulation_compute_time_s") for row in group)),
                **_stats("gpu_simulation_compute_time_s", _numbers(row.get("gpu_simulation_compute_time_s") for row in group)),
                **_stats("compute_speedup_cpu_over_gpu", _numbers(row.get("compute_speedup_cpu_over_gpu") for row in group)),
                **_stats("cpu_total_wall_time_s", _numbers(row.get("cpu_total_wall_time_s") for row in group)),
                **_stats("gpu_total_wall_time_s", _numbers(row.get("gpu_total_wall_time_s") for row in group)),
                **_stats("wall_time_ratio_cpu_over_gpu", _numbers(row.get("wall_time_ratio_cpu_over_gpu") for row in group)),
                **_stats("cpu_total_host_residual_time_s", _numbers(row.get("cpu_total_host_residual_time_s") for row in group)),
                **_stats("gpu_total_host_residual_time_s", _numbers(row.get("gpu_total_host_residual_time_s") for row in group)),
            })
    crossover_by_family: dict[str, int | str] = {}
    family_groups: dict[str, list[JsonDict]] = defaultdict(list)
    for row in summary:
        family_groups[str(row["case_family"])].append(row)
    for family, family_rows in family_groups.items():
        observed = sorted(
            int(row["n_qubits"])
            for row in family_rows
            if _plot_qubits(row) is not None and _positive(row.get("compute_speedup_cpu_over_gpu_median")) is not None
            and float(row["compute_speedup_cpu_over_gpu_median"]) > 1.0
        )
        crossover_by_family[family] = observed[0] if observed else "none_observed"
    for row in summary:
        row["compute_speedup_cpu_over_gpu_crossover_qubit"] = crossover_by_family[str(row["case_family"])]
        row["crossover_qubit"] = row["compute_speedup_cpu_over_gpu_crossover_qubit"]
    return summary


def full_state_tn_comparison(stats_rows: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[str, str], dict[str, JsonDict]] = defaultdict(dict)
    for row in stats_rows:
        if row.get("suite_id") not in {"thesis_cpu_tn_quimb", "research_cpu_tn"}:
            continue
        route = str(row.get("route_id") or "")
        if route in {"quest_cpu_full_state_exact", "quimb_tn_exact", "quimb_tn_sliced_exact"}:
            grouped[(str(row.get("suite_id")), str(row.get("case_id")))][route] = row
    result: list[JsonDict] = []
    for (suite_id, case_id), routes in sorted(grouped.items()):
        quest = routes.get("quest_cpu_full_state_exact")
        unsliced = routes.get("quimb_tn_exact")
        sliced = routes.get("quimb_tn_sliced_exact")
        if quest is None or unsliced is None:
            continue
        quest_time = _positive(quest.get("simulation_compute_time_s_median"))
        unsliced_time = _positive(unsliced.get("simulation_compute_time_s_median"))
        sliced_time = _positive(sliced.get("simulation_compute_time_s_median")) if sliced else None
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": suite_id,
                "case_id": case_id,
                "case_family": quest.get("case_family"),
                "benchmark_n_qubits": quest.get("benchmark_n_qubits"),
                "quest_cpu_compute_time_s_median": quest_time,
                "quimb_unsliced_compute_time_s_median": unsliced_time,
                "quimb_sliced_compute_time_s_median": sliced_time,
                "quimb_unsliced_time_over_quest_time": _ratio(unsliced_time, quest_time),
                "quimb_sliced_time_over_quest_time": _ratio(sliced_time, quest_time),
                "quimb_sliced_time_over_unsliced_time": _ratio(sliced_time, unsliced_time),
                "quest_validation_passed_count": quest.get("validation_passed_count"),
                "quimb_unsliced_validation_passed_count": unsliced.get("validation_passed_count"),
                "quimb_sliced_validation_passed_count": sliced.get("validation_passed_count") if sliced else None,
            }
        )
    return result


def upmem_quantization_attribution(records: list[JsonDict]) -> list[JsonDict]:
    """Pair strict generic UPMEM float32 and int8 records from one TaskGraph.

    This is intentionally narrower than a generic route comparison: both records
    must prove the same generic-only SDK-simulator execution family.  Ratios are
    route-level simulator evidence, never hardware speedups.
    """
    grouped: dict[tuple[str, str, str, int, str], dict[str, JsonDict]] = defaultdict(dict)
    for record in records:
        if not _is_strict_generic_upmem_record(record):
            continue
        if str(record.get("status") or "") != "completed":
            continue
        mode = str(_record_value(record, "quantization_mode") or "")
        if mode not in {"none", "per_task_input_quantize"}:
            continue
        case_id = str(record.get("case_id") or "")
        if not case_id:
            continue
        repeat_id = _int_or_none(record.get("repeat_id"))
        grouped[
            (
                str(record.get("suite_id") or ""),
                str(record.get("run_id") or ""),
                case_id,
                0 if repeat_id is None else repeat_id,
                str(record.get("route_id") or ""),
            )
        ][mode] = record

    rows: list[JsonDict] = []
    for (suite_id, run_id, case_id, repeat_id, route_id), modes in sorted(grouped.items()):
        unquantized = modes.get("none")
        quantized = modes.get("per_task_input_quantize")
        if unquantized is None or quantized is None:
            continue
        if str(_record_value(unquantized, "policy") or "") != "generic-only":
            continue
        if str(_record_value(quantized, "policy") or "") != "generic-only":
            continue
        if str(unquantized.get("route_id") or "") != route_id or str(quantized.get("route_id") or "") != route_id:
            continue
        if unquantized.get("kernel_family") != quantized.get("kernel_family"):
            continue
        family, qubits = _family_and_qubits(unquantized)
        unquantized_total = _positive(unquantized.get("total_wall_time_s"))
        quantized_total = _positive(quantized.get("total_wall_time_s"))
        unquantized_residual = _reported_time(unquantized)[0]
        quantized_residual = _reported_time(quantized)[0]
        unquantized_compute = _positive(unquantized.get("simulation_compute_time_s"))
        quantized_compute = _positive(quantized.get("simulation_compute_time_s"))
        unquantized_transfer = _positive(unquantized.get("actual_transfer_bytes"))
        quantized_transfer = _positive(quantized.get("actual_transfer_bytes"))
        unquantized_error = _validation_errors(unquantized).get("max_abs_error")
        quantized_error = _validation_errors(quantized).get("max_abs_error")
        unquantized_full_precision_error = _full_precision_errors(unquantized).get("max_abs_error")
        quantized_full_precision_error = _full_precision_errors(quantized).get("max_abs_error")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": suite_id or unquantized.get("suite_id"),
                "run_id": run_id or unquantized.get("run_id"),
                "case_id": case_id,
                "case_family": family,
                "n_qubits": qubits["actual_n_qubits"],
                **qubits,
                "repeat_id": repeat_id,
                "route_id": route_id,
                "policy": "generic-only",
                "same_route_comparison": True,
                "same_taskgraph": True,
                "same_kernel_family": True,
                "unquantized_total_wall_time_s": unquantized_total,
                "quantized_total_wall_time_s": quantized_total,
                "unquantized_host_residual_time_s": unquantized_residual,
                "quantized_host_residual_time_s": quantized_residual,
                "route_runtime_ratio_none_over_quantized": _ratio(unquantized_residual, quantized_residual)
                if unquantized_residual is not None and quantized_residual is not None
                else _ratio(unquantized_total, quantized_total),
                "unquantized_simulation_compute_time_s": unquantized_compute,
                "quantized_simulation_compute_time_s": quantized_compute,
                "simulator_kernel_ratio_none_over_quantized": _ratio(unquantized_compute, quantized_compute),
                "unquantized_transfer_bytes": unquantized_transfer,
                "quantized_transfer_bytes": quantized_transfer,
                "transfer_ratio_none_over_quantized": _ratio(unquantized_transfer, quantized_transfer),
                "unquantized_max_abs_error_vs_full_precision": _float_or_none(unquantized_full_precision_error if unquantized_full_precision_error is not None else unquantized_error),
                "quantized_max_abs_error_vs_full_precision": _float_or_none(quantized_full_precision_error if quantized_full_precision_error is not None else quantized_error),
                "unquantized_execution_max_abs_error": _float_or_none(unquantized_error),
                "quantized_execution_max_abs_error": _float_or_none(quantized_error),
                "unquantized_full_precision_max_abs_error": _float_or_none(unquantized_full_precision_error),
                "quantized_full_precision_max_abs_error": _float_or_none(quantized_full_precision_error),
                "accuracy_delta_quantized_minus_unquantized": _difference(
                    quantized_full_precision_error if quantized_full_precision_error is not None else quantized_error,
                    unquantized_full_precision_error if unquantized_full_precision_error is not None else unquantized_error,
                ),
                "native_unquantized_upmem_kernel_executed": _record_value(unquantized, "native_unquantized_upmem_kernel_executed") is True,
            }
        )
    return rows


def same_plan_execution(records: list[JsonDict]) -> list[JsonDict]:
    cpu_by_plan: dict[tuple[str, str, str], JsonDict] = {}
    upmem_rows: list[JsonDict] = []
    for record in records:
        plan_hash = str(record.get("contraction_plan_hash") or "")
        if not plan_hash:
            continue
        key = (str(record.get("suite_id") or ""), str(record.get("case_id") or ""), plan_hash)
        if (
            record.get("route_id") == "cpu_tn_einsum_exact"
            and record.get("contraction_execution_target") == "cpu"
            and str(record.get("status") or "") == "completed"
        ):
            cpu_by_plan[key] = record
        elif _is_strict_generic_upmem_record(record) and str(record.get("status") or "") == "completed":
            upmem_rows.append(record)

    rows: list[JsonDict] = []
    for upmem in upmem_rows:
        plan_hash = str(upmem["contraction_plan_hash"])
        key = (str(upmem.get("suite_id") or ""), str(upmem.get("case_id") or ""), plan_hash)
        cpu = cpu_by_plan.get(key)
        if cpu is None:
            continue
        cpu_time = _positive(cpu.get("simulation_compute_time_s") or cpu.get("kernel_time_s"))
        upmem_time = _positive(upmem.get("simulation_compute_time_s") or upmem.get("kernel_time_s"))
        if cpu_time is None or upmem_time is None:
            continue
        family, qubits = _family_and_qubits(upmem)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": key[0],
                "case_id": key[1],
                "case_family": family,
                "benchmark_n_qubits": qubits["benchmark_n_qubits"],
                "contraction_plan_hash": plan_hash,
                "cpu_route_id": "cpu_tn_einsum_exact",
                "upmem_route_id": str(upmem.get("route_id") or "upmem_tn_runtime"),
                "quantization_mode": _record_value(upmem, "quantization_mode"),
                "cpu_time_s": cpu_time,
                "upmem_simulator_time_s": upmem_time,
                "route_time_ratio_cpu_over_upmem_simulator": cpu_time / upmem_time,
                "actual_transfer_bytes": upmem.get("actual_transfer_bytes"),
                "max_abs_error": upmem.get("max_abs_error") or _validation_errors(upmem).get("max_abs_error"),
                "same_plan_verified": True,
                "hardware_speedup_applicable": False,
            }
        )
    return sorted(rows, key=lambda row: (str(row["case_family"]), int(row["benchmark_n_qubits"] or 0), str(row["quantization_mode"])))


def planner_comparison(records: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for record in records:
        if record.get("route_id") != "planner_candidate_model":
            continue
        family, qubits = _family_and_qubits(record)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": record.get("suite_id"),
                "case_id": record.get("case_id"),
                "case_family": family,
                "benchmark_n_qubits": qubits["benchmark_n_qubits"],
                "planner_id": record.get("planner_id") or record.get("backend_id"),
                "optimize_mode": record.get("optimize_mode"),
                "contraction_plan_hash": record.get("contraction_plan_hash"),
                "planning_time_s": record.get("planning_time_s"),
                "task_count": record.get("task_count"),
                "tn_estimated_flops": record.get("tn_estimated_flops"),
                "tn_max_intermediate_bytes": record.get("tn_max_intermediate_bytes"),
                "total_host_to_dpu_bytes": record.get("total_host_to_dpu_bytes"),
                "total_dpu_to_host_bytes": record.get("total_dpu_to_host_bytes"),
                "total_mram_to_wram_bytes": record.get("total_mram_to_wram_bytes"),
                "tiling_required_task_count": record.get("tiling_required_task_count"),
                "estimated_total_tile_count": record.get("estimated_total_tile_count"),
                "estimated_max_parallel_tiles": record.get("estimated_max_parallel_tiles"),
                "upmem_pressure_score": record.get("upmem_pressure_score"),
                "upmem_rank": record.get("upmem_rank"),
                "flop_rank": record.get("flop_rank"),
                "parallelism_evidence_type": record.get("parallelism_evidence_type"),
                "execution_plan_executed": record.get("execution_plan_executed"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["case_family"]),
            int(row.get("benchmark_n_qubits") or 0),
            str(row.get("planner_id") or ""),
        ),
    )


def unsupported_cases(records: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "suite_id": record.get("suite_id"),
            "case_id": record.get("case_id"),
            "route_id": record.get("route_id"),
            "benchmark_role": record.get("benchmark_role"),
            "validation_status": record.get("validation_status"),
            "status": record.get("status"),
            "unsupported_task_count": int(record.get("unsupported_task_count", 0) or 0),
            "resource_skip_reason": _record_value(record, "resource_skip_reason") or _record_value(record, "reason"),
            "warnings": record.get("warnings"),
        }
        for record in records
        if _is_unsupported(record)
    ]


def validation_summary(records: list[JsonDict]) -> list[JsonDict]:
    counts = Counter((str(record.get("route_id") or ""), str(record.get("validation_status") or "unknown")) for record in records)
    return [
        {"schema_version": SCHEMA_VERSION, "route_id": route_id, "validation_status": status, "record_count": count}
        for (route_id, status), count in sorted(counts.items())
    ]


def route_capability_matrix(records: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("route_id") or "unknown")].append(record)
    rows: list[JsonDict] = []
    for route_id, group in sorted(grouped.items()):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "route_id": route_id,
                "benchmark_role": _first_present(group, "benchmark_role"),
                "backend_family": _first_present(group, "backend_family"),
                "execution_model": _first_present(group, "execution_model"),
                "contraction_execution_target": _first_present(group, "contraction_execution_target"),
                "accelerator_kind": _first_present(group, "accelerator_kind"),
                "upmem_execution_mode": _first_present(group, "upmem_execution_mode"),
                "record_count": len(group),
                "completed_count": sum(1 for row in group if str(row.get("status")) in {"completed", "executable"}),
                "unsupported_count": sum(1 for row in group if _is_unsupported(row)),
                "validation_passed_count": sum(1 for row in group if str(row.get("validation_status")) in {"passed", "passed_native_status", "passed_runtime_only"}),
                "gpu_verified_count": sum(1 for row in group if bool(row.get("gpu_backend_verified", False))),
                "cpu_fallback_count": sum(1 for row in group if bool(row.get("cpu_fallback_used", False))),
                "hardware_speedup_applicable_count": sum(1 for row in group if bool(row.get("hardware_speedup_applicable", False))),
            }
        )
    return rows


def write_plots(
    out_dir: Path,
    stats_rows: list[JsonDict],
    cpu_gpu_rows: list[JsonDict],
    quantization_rows: list[JsonDict],
    same_plan_rows: list[JsonDict],
    planner_rows: list[JsonDict],
) -> JsonDict:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    entries: list[JsonDict] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "matplotlib_unavailable",
            "error": str(exc),
            "plots": [],
        }
    plotters = [
        ("cpu_gpu_runtime_by_qubits.png", "CPU/GPU runtime by qubits", "cpu_gpu_performance_summary.csv", lambda path: _plot_cpu_gpu_runtime(plt, path, cpu_gpu_rows)),
        ("cpu_gpu_speedup_by_qubits.png", "CPU/GPU speedup by qubits", "cpu_gpu_performance_summary.csv", lambda path: _plot_cpu_gpu_speedup(plt, path, cpu_gpu_rows)),
        ("cpu_tn_runtime_by_qubits.png", "CPU tensor-network runtime by qubits", "per_case_route_stats.csv", lambda path: _plot_cpu_tn_runtime(plt, path, stats_rows)),
        ("full_state_vs_tn_runtime_by_qubits.png", "QuEST full-state versus CPU tensor-network runtime", "per_case_route_stats.csv", lambda path: _plot_full_state_vs_tn_runtime(plt, path, stats_rows)),
        ("tn_planning_vs_contraction.png", "TN planning versus contraction time", "per_case_route_stats.csv", lambda path: _plot_tn_planning_vs_contraction(plt, path, stats_rows)),
        ("tn_path_flops_by_family_size.png", "TN path estimated FLOPs", "per_case_route_stats.csv", lambda path: _plot_tn_path_metric(plt, path, stats_rows, metric="tn_estimated_flops", ylabel="Estimated FLOPs", title="TN path estimated FLOPs")),
        ("tn_path_peak_memory_by_family_size.png", "TN path peak intermediate memory", "per_case_route_stats.csv", lambda path: _plot_tn_path_metric(plt, path, stats_rows, metric="tn_max_intermediate_bytes", ylabel="Peak intermediate bytes", title="TN path peak intermediate memory")),
        ("cpu_tn_slicing_flop_ratio.png", "Quimb slicing FLOP ratio", "per_case_route_stats.csv", lambda path: _plot_slicing_ratio(plt, path, stats_rows)),
        ("upmem_supported_boundary.png", "UPMEM SDK simulator support boundary", "per_case_route_stats.csv", lambda path: _plot_upmem_boundary(plt, path, stats_rows)),
        ("upmem_accuracy_error.png", "UPMEM SDK simulator accuracy", "per_case_route_stats.csv", lambda path: _plot_upmem_accuracy(plt, path, stats_rows)),
        ("upmem_quantization_attribution.png", "UPMEM generic quantization attribution", "upmem_quantization_attribution.csv", lambda path: _plot_upmem_quantization_attribution(plt, path, quantization_rows)),
        ("quantization_runtime_by_executor.png", "Quantization route runtime attribution", "upmem_quantization_attribution.csv", lambda path: _plot_quantization_metric(plt, path, quantization_rows, metric="route_runtime_ratio_none_over_quantized", ylabel="float32 time / int8 time", title="UPMEM SDK simulator quantization runtime ratio")),
        ("quantization_transfer_bytes.png", "Quantization transfer attribution", "upmem_quantization_attribution.csv", lambda path: _plot_quantization_metric(plt, path, quantization_rows, metric="transfer_ratio_none_over_quantized", ylabel="float32 bytes / int8 bytes", title="UPMEM transfer-volume ratio")),
        ("quantization_error_by_family_size.png", "Quantization accuracy attribution", "upmem_quantization_attribution.csv", lambda path: _plot_quantization_metric(plt, path, quantization_rows, metric="quantized_max_abs_error_vs_full_precision", ylabel="Max absolute error", title="UPMEM int8 error versus full precision", log_scale=True)),
        ("same_plan_cpu_upmem_runtime.png", "Same-plan CPU versus UPMEM SDK simulator runtime", "same_plan_execution.csv", lambda path: _plot_same_plan_runtime(plt, path, same_plan_rows)),
        ("planner_flops_vs_upmem_pressure.png", "Planner FLOPs versus modeled UPMEM pressure", "planner_comparison.csv", lambda path: _plot_planner_pressure(plt, path, planner_rows)),
        ("internal_parallelism_metadata_by_qubits.png", "Internal diagnostic parallelism metadata", "per_case_route_stats.csv", lambda path: _plot_internal_parallelism(plt, path, stats_rows)),
    ]
    for filename, title, source_csv, plotter in plotters:
        path = plots_dir / filename
        reason = plotter(path)
        if reason:
            entries.append({"plot": filename, "title": title, "status": "skipped", "reason": reason, "source_csv": source_csv, "caption": _caption(filename)})
        else:
            entries.append({"plot": filename, "title": title, "status": "generated", "reason": None, "source_csv": source_csv, "caption": _caption(filename), "size_bytes": path.stat().st_size})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "plots": entries,
        "generated": [entry["plot"] for entry in entries if entry["status"] == "generated"],
        "skipped": [entry for entry in entries if entry["status"] == "skipped"],
    }


def benchmark_summary(
    manifest: JsonDict,
    records: list[JsonDict],
    stats_rows: list[JsonDict],
    speedup_rows: list[JsonDict],
    quantization_rows: list[JsonDict],
    planner_rows: list[JsonDict],
    full_state_tn_rows: list[JsonDict],
    unsupported_rows: list[JsonDict],
    validation_rows: list[JsonDict],
    capability_rows: list[JsonDict],
    plot_manifest: JsonDict,
    boundary: JsonDict,
    guard_issues: list[str],
) -> str:
    lines = [
        "# Research Benchmark Summary",
        "",
        "This is a derived research pack generated from normalized benchmark records. Evidence inputs remain read-only.",
        "",
        "## Benchmark Matrix",
        "",
        "| Route | Role | Target | Records | Unsupported |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in capability_rows:
        lines.append(
            f"| {row['route_id']} | {row.get('benchmark_role') or ''} | {row.get('contraction_execution_target') or ''} | "
            f"{row['record_count']} | {row['unsupported_count']} |"
        )
    lines.extend(
        [
            "",
            "## Suite Paths",
            "",
        ]
    )
    for key, path in manifest["selected_suites"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(
        [
            "",
            "## Exact Commands",
            "",
            "Run `make research-plan` to print the underlying commands.",
            "Run `BENCH_CPU_THREADS=<physical-core-count> make thesis-run` for the complete local benchmark matrix.",
            "",
            "## Hardware And Software Manifest",
            "",
            f"- Git commit: `{manifest.get('git_commit') or 'unknown'}`",
            f"- Dirty worktree: `{manifest.get('dirty_worktree')}`",
            f"- Host: `{manifest.get('hostname')}`",
            f"- Python: `{manifest.get('python_version')}`",
            f"- Packages: `{json.dumps(manifest.get('packages', {}), sort_keys=True)}`",
            "",
            "## Benchmark Group Status",
            "",
        ]
    )
    for command in manifest.get("commands", []):
        lines.append(f"- `{command.get('command')}`: returncode `{command.get('returncode')}`.")
        if command.get("skipped_group"):
            lines.append(f"  - skipped group `{command.get('skipped_group')}`: {command.get('blocker_reason')}")
    lines.extend(
        [
            "",
            "## Repeats, Warmups, And Timing",
            "",
            "Statistics are computed from normalized records. Median and spread fields are reported per case/route.",
            "CPU/GPU performance-tier speedup uses `simulation_compute_time_s`; UPMEM route timing prefers `total_host_residual_time_s` when present and retains execution validation separately.",
            "Qubit-scaling tables and plots use `benchmark_n_qubits` / `actual_n_qubits`; suite caps and output caps are not used as circuit size.",
            "",
            "## Validation Methods",
            "",
        ]
    )
    for row in validation_rows:
        lines.append(f"- `{row['route_id']}` `{row['validation_status']}`: {row['record_count']} records")
    lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    for entry in plot_manifest.get("plots", []):
        lines.append(f"- `{entry['plot']}`: {entry['status']} ({entry.get('reason') or 'ok'}). {entry.get('caption') or ''}")
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            f"- Normalized records loaded: {len(records)}.",
            f"- Per-case route statistic rows: {len(stats_rows)}.",
            f"- Valid CPU/GPU paired speedup rows: {len(speedup_rows)}.",
            f"- Matched strict generic UPMEM float32/int8 attribution rows: {len(quantization_rows)}.",
            f"- Modeled contraction-path candidate rows: {len(planner_rows)}.",
            f"- Unsupported/skipped rows preserved: {len(unsupported_rows)}.",
            "",
            "## Observed Result Snapshot",
            "",
            *_observed_result_lines(speedup_rows, full_state_tn_rows, quantization_rows, planner_rows, unsupported_rows),
            "",
            "## Unsupported Cases",
            "",
        ]
    )
    if unsupported_rows:
        for row in unsupported_rows[:20]:
            lines.append(f"- `{row.get('case_id')}` / `{row.get('route_id')}`: {row.get('resource_skip_reason') or row.get('validation_status') or row.get('status')}")
        if len(unsupported_rows) > 20:
            lines.append(f"- ... {len(unsupported_rows) - 20} more rows in `unsupported_cases.csv`.")
    else:
        lines.append("- None in loaded records.")
    lines.extend(
        [
            "",
            "## Claims Allowed",
            "",
            "- QuEST CPU/GPU full-state speedup only for matched CPU/GPU rows with the same timing scope.",
            "- Quimb unsliced vs sliced comparisons as CPU TN implementation evidence, with slicing metrics labeled as `slicing_flop_ratio`.",
            "- Strict generic-only UPMEM SDK simulator rows as bounded generic code-path and boundary evidence.",
            "- Same-route float32 versus int8 generic UPMEM ratios as SDK-simulator route attribution, not hardware speedup.",
            "",
            "## Claims Not Allowed",
            "",
            "- No hardware speedup claim from UPMEM SDK simulator timing.",
            "- No fake GPU rows without verified GPU execution.",
            "- No energy-efficiency claim without real measured energy metadata.",
            "- No speedup across incompatible route families such as Quimb versus internal TaskGraph diagnostics.",
            "",
            "## Artifact Boundary Checks",
            "",
            f"- Evidence boundary status: `{boundary['status']}`.",
            f"- Guard issues: {len(guard_issues)}.",
        ]
    )
    for issue in guard_issues:
        lines.append(f"  - {issue}")
    lines.extend(
        [
            "",
            "## Next UPMEM Implementation Readiness",
            "",
            *_upmem_readiness_lines(records, unsupported_rows),
            "",
            "## Missing Evidence",
            "",
        ]
    )
    missing = _missing_evidence(records)
    for item in missing:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _observed_result_lines(
    speedup_rows: list[JsonDict],
    full_state_tn_rows: list[JsonDict],
    quantization_rows: list[JsonDict],
    planner_rows: list[JsonDict],
    unsupported_rows: list[JsonDict],
) -> list[str]:
    lines: list[str] = []
    gpu_ratios = _numbers(row.get("compute_speedup_cpu_over_gpu") for row in speedup_rows if _bool(row.get("performance_tier")))
    if gpu_ratios:
        lines.append(
            "- Verified QuEST GPU compute ratio (CPU/GPU): "
            f"median `{statistics.median(gpu_ratios):.3g}x`, range `{min(gpu_ratios):.3g}x` to `{max(gpu_ratios):.3g}x`; "
            f"GPU was faster in `{sum(value > 1.0 for value in gpu_ratios)}/{len(gpu_ratios)}` matched repeats."
        )
    else:
        lines.append("- No matched verified CPU/GPU performance repeats were available.")

    tn_ratios = _numbers(row.get("quimb_unsliced_time_over_quest_time") for row in full_state_tn_rows)
    slicing_ratios = _numbers(row.get("quimb_sliced_time_over_unsliced_time") for row in full_state_tn_rows)
    if tn_ratios:
        lines.append(
            "- Shallow exact CPU comparison (Quimb TN time / QuEST full-state time): "
            f"median `{statistics.median(tn_ratios):.3g}x` across `{len(tn_ratios)}` cases. "
            "This is an algorithm/backend runtime ratio, not same-plan speedup."
        )
    if slicing_ratios:
        lines.append(
            "- Executed Quimb slicing time / unsliced Quimb time: "
            f"median `{statistics.median(slicing_ratios):.3g}x`, range `{min(slicing_ratios):.3g}x` to `{max(slicing_ratios):.3g}x`; "
            "slice reconstruction used one worker."
        )

    quant_runtime = _numbers(row.get("route_runtime_ratio_none_over_quantized") for row in quantization_rows)
    quant_transfer = _numbers(row.get("transfer_ratio_none_over_quantized") for row in quantization_rows)
    quant_error = _numbers(row.get("quantized_max_abs_error_vs_full_precision") for row in quantization_rows)
    if quant_runtime:
        lines.append(
            "- Strict generic UPMEM SDK-simulator float32/int8 attribution: "
            f"median host-residual-time ratio `{statistics.median(quant_runtime):.3g}x`, "
            f"median transfer ratio `{statistics.median(quant_transfer):.3g}x`"
            + (f", maximum observed int8 absolute error `{max(quant_error):.3g}`." if quant_error else ".")
            + " These are simulator-route measurements, not hardware speedup."
        )

    planner_cases = {str(row.get("case_id")) for row in planner_rows}
    planner_ids = {str(row.get("planner_id")) for row in planner_rows}
    if planner_rows:
        lines.append(
            f"- Planner evidence covers `{len(planner_cases)}` cases and `{len(planner_ids)}` candidates "
            "with plan hashes, FLOP/peak-memory estimates, and modeled UPMEM pressure."
        )
    if unsupported_rows:
        reason_counts = Counter(str(row.get("resource_skip_reason") or "unknown") for row in unsupported_rows)
        lines.append(
            "- Explicit boundary rows: "
            + ", ".join(f"`{reason}` = {count}" for reason, count in reason_counts.most_common())
            + "."
        )
    return lines


def validate_artifact_boundaries(root: Path) -> JsonDict:
    evidence_root = root / "runs" / "evidence"
    violations: list[str] = []
    if evidence_root.exists():
        for path in evidence_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if path.name in FORBIDDEN_EVIDENCE_DERIVED_NAMES:
                violations.append(rel)
            if path.suffix.lower() in {".png", ".svg", ".pdf"} or "/plots/" in rel:
                violations.append(rel)
    return {"schema_version": SCHEMA_VERSION, "status": "failed" if violations else "passed", "violations": violations}


def _claim_guard_issues(records: list[JsonDict]) -> list[str]:
    issues: list[str] = []
    for record in records:
        route = record.get("route_id")
        case = record.get("case_id")
        if record.get("contraction_execution_target") == "gpu" and not (record.get("gpu_backend_verified") is True and record.get("gpu_program_executed") is True):
            issues.append(f"unverified gpu row: {case}/{route}")
        if record.get("upmem_execution_mode") == "sdk_simulator":
            if bool(record.get("hardware_speedup_applicable", False)):
                issues.append(f"sdk simulator row marked hardware speedup applicable: {case}/{route}")
            if record.get("hardware_speedup") not in {None, "", "not_applicable"}:
                issues.append(f"sdk simulator row has hardware_speedup value: {case}/{route}")
        if record.get("energy_joules") not in {None, ""} and str(record.get("energy_measurement_status") or "") not in {"measured", "available"}:
            issues.append(f"energy value without measured status: {case}/{route}")
        if record.get("contraction_execution_target") == "upmem":
            issues.extend(_strict_generic_upmem_issues(record))
    return issues


def _upmem_readiness_lines(records: list[JsonDict], unsupported_rows: list[JsonDict]) -> list[str]:
    upmem_records = [row for row in records if row.get("contraction_execution_target") == "upmem" or row.get("upmem_execution_mode") == "sdk_simulator"]
    if not upmem_records:
        return [
            "- No UPMEM SDK simulator records were loaded in this pack.",
            "- Next target: run the selected strict generic-only UPMEM suite and regenerate this pack before making stronger UPMEM claims.",
        ]
    unsupported = [row for row in unsupported_rows if str(row.get("route_id") or "").startswith("upmem") or row.get("route_id") == "upmem_tn_sdk_simulator_quantized"]
    fallback_count = sum(1 for row in upmem_records if bool(row.get("cpu_fallback_used", False)))
    sdk_count = sum(1 for row in upmem_records if row.get("upmem_execution_mode") == "sdk_simulator")
    generic_records = [row for row in upmem_records if _is_strict_generic_upmem_record(row)]
    reasons = Counter(_unsupported_reason(row) for row in unsupported)
    supported = [row for row in upmem_records if str(row.get("status") or "") == "completed" and not _is_unsupported(row)]
    supported_qubits = [(qubits, row) for row in supported if (qubits := _family_and_qubits(row)[1]["benchmark_n_qubits"]) is not None]
    highest_supported = max(supported_qubits, key=lambda item: (int(item[0]), str(item[1].get("case_id") or "")), default=None)
    first_unsupported = min(
        ((qubits, row) for row in unsupported if (qubits := _family_and_qubits(row)[1]["benchmark_n_qubits"]) is not None),
        key=lambda item: (int(item[0]), str(item[1].get("case_id") or "")),
        default=None,
    )
    tiling_records = [row for row in upmem_records if _tiling_status(row) is not None]
    tiling_supported = [row for row in tiling_records if _tiling_status(row) is True]
    lines = [
        f"- UPMEM SDK simulator records loaded: {len(upmem_records)}; SDK simulator rows: {sdk_count}.",
        f"- Strict generic-only UPMEM rows: {len(generic_records)}.",
        f"- CPU fallback flagged in UPMEM rows: {fallback_count}.",
        f"- Unsupported/boundary rows: {len(unsupported)}.",
    ]
    if reasons:
        lines.append(f"- Top blocker reasons: {', '.join(f'{reason}={count}' for reason, count in reasons.most_common(5))}.")
    if tiling_records:
        lines.append(
            f"- Tiling support derived from records: {len(tiling_supported)}/{len(tiling_records)} rows report executable or observed tiling metadata."
        )
    else:
        lines.append("- Tiling support derived from records: no tiling status was recorded.")
    if highest_supported is not None:
        lines.append(
            f"- Highest supported UPMEM case in these records: `{highest_supported[1].get('case_id')}` at `{int(highest_supported[0])}` qubits."
        )
    else:
        lines.append("- Highest supported UPMEM case in these records: none.")
    if first_unsupported is not None:
        first_case = first_unsupported[1].get("case_id")
        first_reason = _unsupported_reason(first_unsupported[1])
        lines.append(f"- First unsupported case by recorded qubit count: `{first_case}` at `{int(first_unsupported[0])}` qubits; reason: `{first_reason}`.")
        lines.append(f"- Next target derived from the boundary: investigate `{first_case}` and address `{first_reason}` without CPU fallback.")
    elif highest_supported is not None:
        lines.append(f"- Next target derived from the records: extend the strict generic-only sweep beyond `{highest_supported[1].get('case_id')}` and record the resulting boundary.")
    else:
        lines.append("- Next target derived from the records: obtain a completed strict generic-only UPMEM record with capability metadata.")
    return lines


def _missing_evidence(records: list[JsonDict]) -> list[str]:
    routes = {str(record.get("route_id") or "") for record in records}
    missing: list[str] = []
    if "quest_cpu_full_state_exact" not in routes:
        missing.append("QuEST CPU full-state baseline records are absent.")
    if "quest_gpu_full_state_exact" not in routes:
        missing.append("Verified QuEST GPU records are absent.")
    if "quimb_tn_exact" not in routes:
        missing.append("Quimb TN baseline records are absent.")
    if not any(record.get("contraction_execution_target") == "upmem" for record in records):
        missing.append("UPMEM SDK simulator quantized records are absent.")
    return missing or ["No mandatory evidence class is obviously absent from loaded records."]


def _plot_cpu_gpu_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    usable = [row for row in rows if _plot_qubits(row) is not None]
    if not usable:
        return "no_performance_tier_cpu_gpu_rows"
    families = sorted({str(row["case_family"]) for row in usable})
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True)
    for axis, family in zip(axes.flat, families):
        group = sorted((row for row in usable if str(row["case_family"]) == family), key=lambda row: int(_plot_qubits(row) or 0))
        qubits = [int(_plot_qubits(row) or 0) for row in group]
        axis.errorbar(
            qubits,
            [float(row["cpu_simulation_compute_time_s_median"]) for row in group],
            yerr=_plot_iqr_error(group, "cpu_simulation_compute_time_s"),
            marker="o",
            capsize=3,
            label="QuEST CPU",
        )
        axis.errorbar(
            qubits,
            [float(row["gpu_simulation_compute_time_s_median"]) for row in group],
            yerr=_plot_iqr_error(group, "gpu_simulation_compute_time_s"),
            marker="s",
            capsize=3,
            label="QuEST GPU",
        )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Median compute time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle("CPU/GPU full-state compute runtime (performance tier)")
    _save_plot(fig, path)
    return None


def _plot_cpu_gpu_speedup(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        if _plot_qubits(row) is not None:
            grouped[str(row["case_family"])].append(row)
    if not grouped:
        return "no_actual_qubit_metadata"
    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    for family, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(_plot_qubits(row) or 0))
        ax.errorbar(
            [int(_plot_qubits(row) or 0) for row in ordered],
            [float(row["compute_speedup_cpu_over_gpu_median"]) for row in ordered],
            yerr=_plot_iqr_error(ordered, "compute_speedup_cpu_over_gpu"),
            marker="o",
            capsize=3,
            label=family,
        )
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Compute speedup (CPU time / GPU time)")
    ax.set_title("CPU/GPU compute speedup by circuit family")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize="small", ncol=2)
    _save_plot(fig, path)
    return None


def _plot_cpu_tn_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if row.get("route_id") in {"quimb_tn_exact", "quimb_tn_sliced_exact"} and _positive(row.get("simulation_compute_time_s_median")) is not None]
    if not selected:
        return "no_quimb_tn_rows"
    families = sorted({str(row["case_family"]) for row in selected})
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True)
    for axis, family in zip(axes.flat, families):
        for route, label, marker in (
            ("quimb_tn_exact", "Quimb unsliced", "o"),
            ("quimb_tn_sliced_exact", "Quimb sliced", "s"),
        ):
            group = sorted(
                (row for row in selected if row["route_id"] == route and str(row["case_family"]) == family),
                key=lambda row: int(_plot_qubits(row) or 0),
            )
            if group:
                axis.errorbar(
                    [int(_plot_qubits(row) or 0) for row in group],
                    [float(row["simulation_compute_time_s_median"]) for row in group],
                    yerr=_plot_iqr_error(group, "simulation_compute_time_s"),
                    marker=marker,
                    capsize=3,
                    label=label,
                )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Median contraction time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle("CPU tensor-network contraction runtime")
    _save_plot(fig, path)
    return None


def _plot_full_state_vs_tn_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    routes = {
        "quest_cpu_full_state_exact": ("QuEST CPU full state", "o"),
        "quimb_tn_exact": ("Quimb TN unsliced", "s"),
        "quimb_tn_sliced_exact": ("Quimb TN sliced", "^"),
    }
    selected = [
        row
        for row in rows
        if row.get("suite_id") in {"thesis_cpu_tn_quimb", "research_cpu_tn"}
        and row.get("route_id") in routes
        and _plot_qubits(row) is not None
        and _positive(row.get("simulation_compute_time_s_median")) is not None
    ]
    if not selected:
        return "no_matching_full_state_and_tn_rows"
    families = sorted({str(row["case_family"]) for row in selected})
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True)
    for axis, family in zip(axes.flat, families):
        for route, (label, marker) in routes.items():
            group = sorted(
                (row for row in selected if row["route_id"] == route and str(row["case_family"]) == family),
                key=lambda row: int(_plot_qubits(row) or 0),
            )
            if group:
                axis.errorbar(
                    [int(_plot_qubits(row) or 0) for row in group],
                    [float(row["simulation_compute_time_s_median"]) for row in group],
                    yerr=_plot_iqr_error(group, "simulation_compute_time_s"),
                    marker=marker,
                    capsize=3,
                    label=label,
                )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Median simulation/contraction time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="x-small")
    fig.suptitle("Full-state and tensor-network CPU implementations (different execution models)")
    _save_plot(fig, path)
    return None


def _plot_tn_planning_vs_contraction(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("route_id") in {"quimb_tn_exact", "quimb_tn_sliced_exact"}
        and _plot_qubits(row) is not None
        and _positive(row.get("simulation_compute_time_s_median")) is not None
    ]
    if not selected:
        return "no_quimb_timing_rows"
    families = sorted({str(row["case_family"]) for row in selected})
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True)
    styles = {
        "quimb_tn_exact": ("Quimb unsliced", "o"),
        "quimb_tn_sliced_exact": ("Quimb sliced", "s"),
    }
    for axis, family in zip(axes.flat, families):
        for route, (label, marker) in styles.items():
            group = sorted(
                (row for row in selected if row["route_id"] == route and str(row["case_family"]) == family),
                key=lambda row: int(_plot_qubits(row) or 0),
            )
            if not group:
                continue
            x = [int(_plot_qubits(row) or 0) for row in group]
            axis.plot(x, [max(float(row.get("planning_time_s_median") or 0.0), 1e-12) for row in group], marker=marker, linestyle="--", label=f"{label} planning")
            axis.plot(x, [float(row["simulation_compute_time_s_median"]) for row in group], marker=marker, linestyle="-", label=f"{label} contraction")
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Median time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="x-small")
    fig.suptitle("Tensor-network planning versus contraction time")
    _save_plot(fig, path)
    return None


def _plot_tn_path_metric(
    plt: Any,
    path: Path,
    rows: list[JsonDict],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("route_id") in {"quimb_tn_exact", "quimb_tn_sliced_exact"}
        and _plot_qubits(row) is not None
        and _positive(row.get(metric)) is not None
    ]
    if not selected:
        return f"no_{metric}_rows"
    families = sorted({str(row["case_family"]) for row in selected})
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True)
    for axis, family in zip(axes.flat, families):
        for route, label, marker in (
            ("quimb_tn_exact", "Quimb unsliced", "o"),
            ("quimb_tn_sliced_exact", "Quimb sliced", "s"),
        ):
            group = sorted(
                (row for row in selected if row["route_id"] == route and str(row["case_family"]) == family),
                key=lambda row: int(_plot_qubits(row) or 0),
            )
            if group:
                axis.plot(
                    [int(_plot_qubits(row) or 0) for row in group],
                    [float(row[metric]) for row in group],
                    marker=marker,
                    label=label,
                )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel(f"{ylabel} (log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle(title)
    _save_plot(fig, path)
    return None


def _plot_quantization_metric(
    plt: Any,
    path: Path,
    rows: list[JsonDict],
    *,
    metric: str,
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> str | None:
    selected = [row for row in rows if _plot_qubits(row) is not None and _positive(row.get(metric)) is not None]
    if not selected:
        return f"no_{metric}_rows"
    ordered = sorted(selected, key=lambda row: (str(row["case_family"]), int(_plot_qubits(row) or 0), str(row["case_id"])))
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.5), 5.2), constrained_layout=True)
    ax.bar(range(len(ordered)), [float(row[metric]) for row in ordered], color="#0f766e")
    if "ratio" in metric:
        ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save_plot(fig, path)
    return None


def _plot_same_plan_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_same_plan_cpu_upmem_rows"
    families = sorted({str(row["case_family"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True)
    for axis, family in zip(axes.flat, families):
        group = [row for row in rows if str(row["case_family"]) == family]
        by_qubits: dict[int, list[JsonDict]] = defaultdict(list)
        for row in group:
            by_qubits[int(row.get("benchmark_n_qubits") or 0)].append(row)
        qubits = sorted(by_qubits)
        axis.plot(
            qubits,
            [float(by_qubits[value][0]["cpu_time_s"]) for value in qubits],
            marker="o",
            label="CPU TaskGraph",
        )
        for mode, label, marker in (
            ("none", "UPMEM float32", "s"),
            ("per_task_input_quantize", "UPMEM int8", "^"),
        ):
            mode_rows = {
                int(row.get("benchmark_n_qubits") or 0): row
                for row in group
                if row.get("quantization_mode") == mode
            }
            mode_qubits = sorted(mode_rows)
            if mode_qubits:
                axis.plot(
                    mode_qubits,
                    [float(mode_rows[value]["upmem_simulator_time_s"]) for value in mode_qubits],
                    marker=marker,
                    label=label,
                )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Route compute time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle("Same-plan CPU and UPMEM SDK simulator execution")
    _save_plot(fig, path)
    return None


def _plot_planner_pressure(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if _positive(row.get("tn_estimated_flops")) is not None
        and _float_or_none(row.get("upmem_pressure_score")) is not None
        and row.get("contraction_plan_hash")
    ]
    if len({str(row.get("planner_id")) for row in selected}) < 2:
        return "multiple_planner_candidates_not_available"
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in selected:
        grouped[str(row.get("planner_id") or "unknown")].append(row)
    for planner_id, group in sorted(grouped.items()):
        ax.scatter(
            [float(row["tn_estimated_flops"]) for row in group],
            [float(row["upmem_pressure_score"]) for row in group],
            label=planner_id,
            alpha=0.75,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Estimated FLOPs")
    ax.set_ylabel("Modeled UPMEM pressure score")
    ax.set_title("Contraction-plan FLOPs versus UPMEM transfer pressure")
    ax.legend(fontsize="small")
    _save_plot(fig, path)
    return None


def _plot_slicing_ratio(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if row.get("route_id") == "quimb_tn_sliced_exact" and _positive(row.get("slicing_flop_ratio")) is not None]
    if not selected:
        return "no_slicing_flop_ratio_rows"
    ordered = sorted((row for row in selected if _plot_qubits(row) is not None), key=lambda row: (str(row["case_family"]), int(_plot_qubits(row) or 0), str(row["case_id"])))
    if not ordered:
        return "no_actual_qubit_metadata"
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.45), 5.2), constrained_layout=True)
    ax.bar(range(len(ordered)), [float(row["slicing_flop_ratio"]) for row in ordered], color="#7c3aed")
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Case")
    ax.set_ylabel("Sliced FLOPs / unsliced FLOPs")
    ax.set_title("Quimb/cotengra slicing FLOP ratio")
    _save_plot(fig, path)
    return None


def _plot_upmem_boundary(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if _is_strict_generic_upmem_record(row)]
    if not selected:
        return "no_upmem_rows"
    usable = [row for row in selected if _plot_qubits(row) is not None]
    if not usable:
        return "no_actual_qubit_metadata"
    families = sorted({str(row["case_family"]) for row in usable})
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.8), constrained_layout=True, sharey=True)
    for axis, family in zip(axes.flat, families):
        grouped: dict[tuple[str, int], list[JsonDict]] = defaultdict(list)
        for row in usable:
            if str(row["case_family"]) == family:
                grouped[(str(row["case_id"]), int(_plot_qubits(row) or 0))].append(row)
        points = sorted(grouped.items(), key=lambda item: item[0][1])
        qubits = [key[1] for key, _ in points]
        supported = [int(all(int(row.get("unsupported_count", 0) or 0) == 0 for row in group)) for _, group in points]
        axis.scatter(
            qubits,
            supported,
            c=["#16a34a" if value else "#dc2626" for value in supported],
            s=65,
            zorder=3,
        )
        axis.plot(qubits, supported, color="#94a3b8", linewidth=1.0, zorder=2)
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["unsupported", "supported"])
        axis.set_ylim(-0.25, 1.25)
        axis.grid(True, alpha=0.3)
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    fig.suptitle("Strict generic UPMEM SDK simulator support boundary")
    _save_plot(fig, path)
    return None


def _plot_upmem_accuracy(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if _is_strict_generic_upmem_record(row) and _positive(row.get("max_abs_error")) is not None]
    if not selected:
        return "no_upmem_error_rows"
    ordered = sorted((row for row in selected if _plot_qubits(row) is not None), key=lambda row: (str(row["case_family"]), int(_plot_qubits(row) or 0), str(row["case_id"])))
    if not ordered:
        return "no_actual_qubit_metadata"
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.45), 4.8), constrained_layout=True)
    ax.bar(range(len(ordered)), [float(row["max_abs_error"]) for row in ordered], color="#ea580c")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Max abs error (log scale)")
    ax.set_title("Strict generic UPMEM SDK simulator accuracy vs reference")
    _save_plot(fig, path)
    return None


def _plot_upmem_quantization_attribution(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    usable = [row for row in rows if _plot_qubits(row) is not None]
    if not usable:
        return "no_matched_generic_quantization_rows"
    ordered = sorted(usable, key=lambda row: (str(row["case_family"]), int(_plot_qubits(row) or 0), str(row["case_id"])))
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    runtime_values = [row.get("route_runtime_ratio_none_over_quantized") for row in ordered]
    transfer_values = [row.get("transfer_ratio_none_over_quantized") for row in ordered]
    if not any(_positive(value) is not None for value in [*runtime_values, *transfer_values]):
        return "no_generic_quantization_ratios"
    fig, (runtime_ax, transfer_ax) = plt.subplots(2, 1, figsize=(max(8.0, len(labels) * 0.5), 7.0), constrained_layout=True)
    x = list(range(len(ordered)))
    for axis, values, title, ylabel in (
        (runtime_ax, runtime_values, "Host-side residual time ratio", "float32 host residual time / int8 host residual time"),
        (transfer_ax, transfer_values, "Host/DPU transfer ratio", "float32 bytes / int8 bytes"),
    ):
        axis.bar(x, [float(value) if _positive(value) is not None else 0.0 for value in values], color="#0f766e")
        axis.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.3)
    transfer_ax.set_xticks(x)
    transfer_ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    transfer_ax.set_xlabel("Case")
    fig.suptitle("Strict generic UPMEM float32 vs int8 attribution (SDK simulator)")
    _save_plot(fig, path)
    return None


def _plot_internal_parallelism(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if row.get("route_id") in {"cpu_tn_frontier_exact", "cpu_tn_hybrid_sliced_frontier_exact"}]
    if not selected:
        return "no_internal_parallelism_rows"
    selected = [row for row in selected if _plot_qubits(row) is not None]
    if not selected:
        return "no_actual_qubit_metadata"
    labels = [f"{row['route_id']}_{row['case_family']}_{_plot_qubits(row)}q" for row in selected]
    widths = [float(row.get("max_frontier_width") or 0.0) for row in selected]
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.5), 4.8), constrained_layout=True)
    ax.bar(range(len(labels)), widths, color="#0891b2")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Max frontier width")
    ax.set_title("Internal diagnostic frontier width")
    _save_plot(fig, path)
    return None


def _save_plot(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # pragma: no cover
        fig.clf()


def _caption(filename: str) -> str:
    return {
        "cpu_gpu_runtime_by_qubits.png": "Performance-tier median QuEST CPU and verified QuEST GPU compute time by circuit size.",
        "cpu_gpu_speedup_by_qubits.png": "Performance-tier CPU/GPU compute speedup; values above 1 mean GPU faster.",
        "cpu_tn_runtime_by_qubits.png": "Quimb CPU tensor-network timing; sliced and unsliced routes are separate execution modes.",
        "full_state_vs_tn_runtime_by_qubits.png": "QuEST CPU full-state and Quimb CPU TN timing on the same shallow circuits; this is an algorithm/backend comparison, not same-plan speedup.",
        "tn_planning_vs_contraction.png": "Planning and contraction timing are reported separately for external CPU TN routes.",
        "tn_path_flops_by_family_size.png": "Reported contraction-plan FLOP estimates by circuit family and size.",
        "tn_path_peak_memory_by_family_size.png": "Reported peak intermediate tensor bytes by circuit family and size.",
        "cpu_tn_slicing_flop_ratio.png": "slicing_flop_ratio = sliced cotengra reported FLOPs / unsliced cotengra reported FLOPs.",
        "upmem_supported_boundary.png": "Supported versus unsupported strict generic-only UPMEM SDK simulator rows.",
        "upmem_accuracy_error.png": "Strict generic UPMEM SDK simulator max absolute error where validation data exists.",
        "upmem_quantization_attribution.png": "Same-route float32 versus int8 ratios for strict generic UPMEM SDK simulator execution; this is not hardware speedup.",
        "quantization_runtime_by_executor.png": "Same-plan SDK simulator float32/int8 host-side residual-time ratio; this is not hardware speedup.",
        "quantization_transfer_bytes.png": "Same-plan float32/int8 host-DPU transfer-volume ratio.",
        "quantization_error_by_family_size.png": "Int8 maximum absolute error against the full-precision TaskGraph reference.",
        "same_plan_cpu_upmem_runtime.png": "CPU replay and UPMEM SDK simulator rows share an identical contraction-plan hash; timing is not hardware speedup.",
        "planner_flops_vs_upmem_pressure.png": "Planner FLOP estimates versus modeled UPMEM pressure when multiple plan candidates are available.",
        "internal_parallelism_metadata_by_qubits.png": "Diagnostic internal TaskGraph frontier metadata, not serious baseline performance.",
    }.get(filename, "")


def _stats(prefix: str, values: list[float]) -> JsonDict:
    if not values:
        return {
            f"{prefix}_median": None,
            f"{prefix}_p25": None,
            f"{prefix}_p75": None,
            f"{prefix}_iqr": None,
            f"{prefix}_mean": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_std": None,
        }
    if len(values) == 1:
        p25 = p75 = values[0]
    else:
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        p25, p75 = quartiles[0], quartiles[2]
    return {
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_p25": p25,
        f"{prefix}_p75": p75,
        f"{prefix}_iqr": p75 - p25,
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _numbers(values: Iterable[Any]) -> list[float]:
    return [number for number in (_float_or_none(value) for value in values) if number is not None]


def _positive(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None or number <= 0:
        return None
    return number


def _reported_time(record: JsonDict) -> tuple[float | None, str]:
    """Select the recorded host residual without reconstructing it from totals."""
    residual = _float_or_none(record.get("total_host_residual_time_s"))
    if residual is not None and residual >= 0:
        return residual, "host_residual"
    total = _float_or_none(record.get("total_wall_time_s"))
    if total is not None and total >= 0:
        return total, "total_wall"
    return None, "unavailable"


def _reported_timing_basis(records: list[JsonDict]) -> str:
    bases = {_reported_time(record)[1] for record in records}
    if len(bases) == 1:
        return next(iter(bases))
    return "mixed"


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_value = _positive(numerator)
    denominator_value = _positive(denominator)
    if numerator_value is None or denominator_value is None:
        return None
    return numerator_value / denominator_value


def _difference(left: Any, right: Any) -> float | None:
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _max_number(values: Iterable[Any]) -> float | None:
    numbers = _numbers(values)
    return max(numbers) if numbers else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validation_errors(record: JsonDict) -> JsonDict:
    metrics = record.get("validation_error_metrics")
    if isinstance(metrics, dict):
        return metrics
    if isinstance(metrics, str):
        try:
            parsed = json.loads(metrics)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _full_precision_errors(record: JsonDict) -> JsonDict:
    """Read full-precision accuracy fields without replacing execution validation."""
    candidates: list[Any] = [
        record,
        record.get("final_full_precision_accuracy"),
        record.get("full_precision_accuracy"),
        _notes(record).get("final_full_precision_accuracy"),
        _notes(record).get("full_precision_accuracy"),
    ]
    output: JsonDict = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for output_key, keys in {
            "max_abs_error": ("full_precision_max_abs_error", "max_abs_error"),
            "l2_error": ("full_precision_l2_error", "l2_error"),
            "norm_drift": ("full_precision_norm_drift", "norm_drift"),
        }.items():
            if output_key not in output:
                for key in keys:
                    if candidate.get(key) not in {None, ""}:
                        output[output_key] = candidate[key]
                        break
    return output


def _accuracy_errors_for_reporting(record: JsonDict) -> JsonDict:
    execution = _validation_errors(record)
    full_precision = _full_precision_errors(record)
    if str(_record_value(record, "quantization_mode") or "") == "per_task_input_quantize":
        return {
            key: full_precision.get(key, execution.get(key))
            for key in ("max_abs_error", "l2_error", "norm_drift")
        }
    return execution


def _unsupported_reason(record: JsonDict) -> str:
    return str(
        _record_value(record, "resource_skip_reason")
        or _record_value(record, "reason")
        or record.get("validation_status")
        or record.get("status")
        or "unknown"
    )


def _tiling_status(record: JsonDict) -> bool | None:
    explicit = (
        record.get("tiling_implemented"),
        record.get("tiling_supported"),
        record.get("wram_output_tiled"),
        record.get("l2_tiled_execution"),
    )
    explicit_values = [_bool(value) for value in explicit if value is not None]
    if any(explicit_values):
        return True
    if explicit_values:
        return False
    for key in ("mram_tiled_task_count", "generic_output_tile_count"):
        value = _int_or_none(record.get(key))
        if value is not None:
            return value > 0 if key == "mram_tiled_task_count" else value > 1
    return None


def _notes(record: JsonDict) -> JsonDict:
    value = record.get("notes")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _record_value(record: JsonDict, key: str) -> Any:
    value = record.get(key)
    if value is not None and value != "":
        return value
    return _notes(record).get(key)


def _is_strict_generic_upmem_record(record: JsonDict) -> bool:
    return (
        record.get("contraction_execution_target") == "upmem"
        and str(_record_value(record, "policy") or "") == "generic-only"
        and str(_record_value(record, "quantization_mode") or "") in {"none", "per_task_input_quantize"}
    )


def _strict_generic_upmem_issues(record: JsonDict) -> list[str]:
    """Reject non-generic UPMEM data from the research boundary group.

    Unsupported generic rows remain valid boundary evidence. Completed rows must
    prove the SDK generic loop executed every TaskGraph task without CPU fallback.
    """
    case = str(record.get("case_id") or "unknown")
    route = str(record.get("route_id") or "unknown")
    policy = str(_record_value(record, "policy") or "")
    mode = str(_record_value(record, "quantization_mode") or "")
    issues: list[str] = []
    if policy != "generic-only":
        issues.append(f"UPMEM research row is not generic-only: {case}/{route} policy={policy or 'missing'}")
        return issues
    if mode not in {"none", "per_task_input_quantize"}:
        issues.append(f"UPMEM generic research row has unsupported quantization mode: {case}/{route} mode={mode or 'missing'}")
        return issues
    if str(record.get("upmem_execution_mode") or "") != "sdk_simulator":
        issues.append(f"UPMEM generic research row is not SDK simulator execution: {case}/{route}")
    if _bool(record.get("cpu_fallback_used")):
        issues.append(f"UPMEM generic research row used CPU fallback: {case}/{route}")
    if str(record.get("status") or "") != "completed":
        return issues
    if record.get("kernel_family") != "generic_loop_fallback":
        issues.append(f"completed UPMEM generic research row is not generic-loop evidence: {case}/{route}")
    if _record_value(record, "generic_only_all_tasks_used_generic_backend") is not True:
        issues.append(f"completed UPMEM generic research row lacks all-task generic proof: {case}/{route}")
    if _record_value(record, "valid_primary_upmem_codepath_result") is not True:
        issues.append(f"completed UPMEM generic research row lacks primary SDK-path proof: {case}/{route}")
    if _record_value(record, "upmem_program_executed") is not True:
        issues.append(f"completed UPMEM generic research row lacks DPU program execution proof: {case}/{route}")
    if (_int_or_none(_record_value(record, "dpu_program_invocations")) or 0) <= 0:
        issues.append(f"completed UPMEM generic research row lacks DPU invocations: {case}/{route}")
    return issues


def _family_and_qubits(record: JsonDict) -> tuple[str, JsonDict]:
    case_id = str(record.get("case_id") or "")
    family = case_id.split("_")[1] if case_id.startswith("quest_") and "_" in case_id else case_id.split("_")[0]
    for key in ("actual_n_qubits", "benchmark_n_qubits", "case_n_qubits", "workload_n_qubits", "circuit_n_qubits", "n_qubits", "allocated_qubits"):
        value = _int_or_none(record.get(key))
        if value is not None:
            return family, _qubit_metadata(value, key, None)
    workload_id = str(record.get("workload_id") or "")
    for source, text in (("case_id", case_id), ("workload_id", workload_id)):
        value = _qubits_from_identifier(text)
        if value is not None:
            return family, _qubit_metadata(value, source, None)
    warning = "actual_qubit_count_unresolved"
    return family, _qubit_metadata(None, None, warning)


def _qubit_metadata(value: int | None, source: str | None, warning: str | None) -> JsonDict:
    return {
        "actual_n_qubits": value,
        "benchmark_n_qubits": value,
        "actual_n_qubits_source": source,
        "actual_n_qubits_warning": warning,
    }


def _qubits_from_identifier(value: str) -> int | None:
    match = re.search(r"(?:^|_)(\d+)q(?:_|$)", value)
    if not match:
        return None
    return _int_or_none(match.group(1))


def _plot_qubits(row: JsonDict) -> int | None:
    for key in ("benchmark_n_qubits", "actual_n_qubits", "n_qubits"):
        value = _int_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _plot_iqr_error(rows: list[JsonDict], prefix: str) -> list[list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    for row in rows:
        median = _float_or_none(row.get(f"{prefix}_median"))
        p25 = _float_or_none(row.get(f"{prefix}_p25"))
        p75 = _float_or_none(row.get(f"{prefix}_p75"))
        if median is None or p25 is None or p75 is None:
            lower.append(0.0)
            upper.append(0.0)
        else:
            lower.append(max(0.0, median - p25))
            upper.append(max(0.0, p75 - median))
    return [lower, upper]


def _first_present(group: list[JsonDict], key: str) -> Any:
    for row in group:
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return None


def _first_record_value(group: list[JsonDict], key: str) -> Any:
    for row in group:
        value = _record_value(row, key)
        if value is not None and value != "":
            return value
    return None


def _is_unsupported(record: JsonDict) -> bool:
    return (
        str(record.get("validation_status") or "") == "skipped"
        or str(record.get("status") or "") in {"unsupported", "failed"}
        or int(record.get("unsupported_task_count", 0) or 0) > 0
        or bool(record.get("resource_skip_reason"))
    )


def _valid_for_pair(record: JsonDict) -> bool:
    return str(record.get("validation_status") or "") in {"passed", "passed_native_status", "passed_runtime_only"}


def _selected_suites(suite_filter: list[str] | None) -> list[str]:
    return [key for key in SUITE_COMMAND_ORDER if not suite_filter or key in suite_filter]


def _research_suite_argv(key: str, root: Path) -> list[str]:
    suite = RESEARCH_SUITES[key].relative_to(root).as_posix()
    if key in {"upmem_boundary", "upmem_quantization_stress"}:
        return [
            "upmem-mvp-benchmark",
            "--suite",
            suite,
            "--policies",
            "generic-only",
            "--quantization-modes",
            "none,per_task_input_quantize",
            "--execute-external",
            "--artifact-retention",
            "compact",
        ]
    if key == "planner_paths":
        return ["compare-planners", "--suite", suite]
    return ["simulation-backend-compare", "--suite", suite, "--artifact-retention", "compact"]


def _pack_dir(root: Path, out: Path | None) -> Path:
    if out is not None:
        return out if out.is_absolute() else root / out
    return DEFAULT_COMPARISON_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def _update_latest_link(parent: Path, target: Path) -> None:
    latest = parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(target.name)
    except OSError:
        return


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _latest_evidence_for_suite(root: Path, suite_id: str) -> Path | None:
    suite_root = root / "runs" / "evidence" / suite_id
    if not suite_root.exists():
        return None
    latest = suite_root / "latest"
    if latest.is_symlink() and latest.exists() and (latest.resolve() / "normalized_records.jsonl").is_file():
        return latest.resolve()
    candidates = [path.parent for path in suite_root.glob("*/*/normalized_records.jsonl")]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"read_error": "invalid_json", "path": path.as_posix()}


def _evidence_source_provenance(evidence_inputs: list[Path]) -> JsonDict:
    commits: set[str] = set()
    source_dirty = False
    repository_dirty = False
    for evidence in evidence_inputs:
        root = evidence if evidence.is_dir() else evidence.parent
        manifest = _read_optional_json(root / "run_manifest.json") or {}
        commit = str(manifest.get("benchmark_source_commit") or manifest.get("git_commit") or "")
        if commit:
            commits.add(commit)
        source_dirty = source_dirty or bool(
            manifest.get("benchmark_source_worktree_dirty")
            if manifest.get("benchmark_source_worktree_dirty") is not None
            else manifest.get("dirty_tree", manifest.get("dirty_worktree", False))
        )
        repository_dirty = repository_dirty or bool(manifest.get("repository_worktree_dirty", False))
    ordered_commits = sorted(commits)
    return {
        "commit": ordered_commits[0] if len(ordered_commits) == 1 else None,
        "commits": ordered_commits,
        "worktree_dirty": source_dirty,
        "repository_worktree_dirty": repository_dirty,
    }


def _gpu_verification_passed(root: Path) -> bool:
    payload = _read_optional_json(root / "build" / "gpu_verification" / "quest_gpu_full_state_exact.json")
    return bool(payload and payload.get("gpu_backend_verified") is True and payload.get("gpu_program_executed") is True)


def _gpu_blocker_reason(root: Path) -> str:
    payload = _read_optional_json(root / "build" / "gpu_verification" / "quest_gpu_full_state_exact.json") or {}
    return str(payload.get("blocker_reason") or payload.get("status") or "gpu_verification_failed")


def _skipped_group_result(group: str, reason: str) -> JsonDict:
    return {
        "command": f"skip research group {group}",
        "returncode": 0,
        "stdout": f"Skipped {group}: {reason}",
        "stderr": "",
        "skipped_group": group,
        "blocker_reason": reason,
        "benchmark_rows_emitted": False,
    }


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_metadata() -> JsonDict:
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {"model": model or platform.processor() or None, "logical_count": os.cpu_count()}


def _git(root: Path, args: list[str]) -> str | None:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _run_capture(root: Path, argv: list[str]) -> JsonDict:
    result = subprocess.run(argv, cwd=root, text=True, capture_output=True, check=False)
    return {"command": " ".join(argv), "returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}


def _bench_argv(args: list[str]) -> list[str]:
    return [sys.executable, "-m", "quantum_bench.bench", *args]


def _bench_command(args: list[str]) -> str:
    return f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench {' '.join(args)}"


def _command(args: list[str], *, python_script: bool = False) -> str:
    if python_script:
        return f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python {' '.join(args)}"
    return " ".join(args)


if __name__ == "__main__":
    raise SystemExit(main())
