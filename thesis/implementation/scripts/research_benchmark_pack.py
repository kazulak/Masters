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
    "cpu_gpu": ROOT / "configs" / "suites" / "manual" / "research_cpu_gpu.yml",
    "cpu_gpu_correctness": ROOT / "configs" / "suites" / "manual" / "research_cpu_gpu_correctness.yml",
    "cpu_tn": ROOT / "configs" / "suites" / "manual" / "research_cpu_tn.yml",
    "upmem_boundary": ROOT / "configs" / "suites" / "manual" / "research_upmem_boundary.yml",
    "internal_parallelism": ROOT / "configs" / "suites" / "manual" / "research_internal_parallelism.yml",
}

SUITE_COMMAND_ORDER = (
    "cpu_gpu_correctness",
    "cpu_gpu",
    "cpu_tn",
    "upmem_boundary",
    "internal_parallelism",
)

RELEVANT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
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
        suite = RESEARCH_SUITES[key].relative_to(root)
        print(f"- {key}:")
        print(f"  {_bench_command(['simulation-backend-compare', '--suite', suite.as_posix(), '--artifact-retention', 'compact'])}")
    print("")
    print("Pack commands:")
    print("  make research-plan")
    print("  make research-benchmarks")
    print("  RUN_RESEARCH=1 make research-benchmarks")
    print("  make research-report")
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
            suite = RESEARCH_SUITES[key].relative_to(root).as_posix()
            result = _run_capture(root, _bench_argv(["simulation-backend-compare", "--suite", suite, "--artifact-retention", "compact"]))
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
    return _write_pack(root, out_dir, evidence_inputs, command_results=command_results, selected_suite_keys=selected)


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
    return _write_pack(root, out_dir, evidence_inputs, command_results=command_results, selected_suite_keys=selected)


def _write_pack(
    root: Path,
    out_dir: Path,
    evidence_inputs: list[Path],
    *,
    command_results: list[JsonDict],
    selected_suite_keys: list[str],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_result_records(evidence_inputs) if evidence_inputs else []
    boundary = validate_artifact_boundaries(root)
    guard_issues = _claim_guard_issues(records)
    manifest = build_manifest(root, command_results=command_results, evidence_inputs=evidence_inputs, record_count=len(records), selected_suite_keys=selected_suite_keys)
    _write_json(out_dir / "benchmark_manifest.json", manifest)
    stats_rows = per_case_route_stats(records)
    speedup_rows = paired_speedups(records)
    unsupported_rows = unsupported_cases(records)
    validation_rows = validation_summary(records)
    capability_rows = route_capability_matrix(records)
    _write_csv(out_dir / "per_case_route_stats.csv", stats_rows, PER_CASE_ROUTE_STATS_FIELDS)
    _write_csv(out_dir / "paired_speedups.csv", speedup_rows, PAIRED_SPEEDUP_FIELDS)
    _write_csv(out_dir / "unsupported_cases.csv", unsupported_rows, UNSUPPORTED_FIELDS)
    _write_csv(out_dir / "validation_summary.csv", validation_rows, VALIDATION_SUMMARY_FIELDS)
    _write_csv(out_dir / "route_capability_matrix.csv", capability_rows, ROUTE_CAPABILITY_FIELDS)
    plot_manifest = write_plots(out_dir, stats_rows, speedup_rows, records)
    _write_json(out_dir / "plot_manifest.json", plot_manifest)
    (out_dir / "benchmark_summary.md").write_text(
        benchmark_summary(
            manifest,
            records,
            stats_rows,
            speedup_rows,
            unsupported_rows,
            validation_rows,
            capability_rows,
            plot_manifest,
            boundary,
            guard_issues,
        ),
        encoding="utf-8",
    )
    print(out_dir)
    return 1 if boundary["status"] == "failed" or guard_issues else 0


def build_manifest(
    root: Path,
    *,
    command_results: list[JsonDict],
    evidence_inputs: list[Path],
    record_count: int,
    selected_suite_keys: list[str],
) -> JsonDict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": RESEARCH_PACK_KIND,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": root.as_posix(),
        "git_commit": _git(root, ["rev-parse", "HEAD"]),
        "dirty_worktree": bool(_git(root, ["status", "--short"])),
        "command_line": " ".join(sys.argv),
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
    "contraction_execution_target",
    "upmem_execution_mode",
    "state_output_mode",
    "validation_method",
    "performance_tier",
    "timing_scope",
    "repeat_count",
    "total_wall_time_s_median",
    "total_wall_time_s_mean",
    "total_wall_time_s_min",
    "total_wall_time_s_max",
    "total_wall_time_s_std",
    "simulation_compute_time_s_median",
    "simulation_compute_time_s_mean",
    "simulation_compute_time_s_min",
    "simulation_compute_time_s_max",
    "simulation_compute_time_s_std",
    "validation_passed_count",
    "validation_failed_count",
    "unsupported_count",
    "slice_count",
    "slicing_flop_ratio",
    "slicing_flop_change_kind",
    "max_abs_error",
    "l2_error",
    "hardware_speedup_applicable",
    "gpu_backend_verified",
    "gpu_device_name",
    "cpu_fallback_used",
    "resource_skip_reason",
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
    "cpu_simulation_compute_time_s",
    "gpu_simulation_compute_time_s",
    "compute_speedup_cpu_over_gpu",
    "gpu_device_name",
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


def per_case_route_stats(records: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[Any, ...], list[JsonDict]] = defaultdict(list)
    for record in records:
        key = (
            record.get("suite_id"),
            record.get("case_id"),
            record.get("route_id"),
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
        compute_values = _numbers(row.get("simulation_compute_time_s") for row in group)
        family, qubits = _family_and_qubits(first)
        errors = [_validation_errors(row) for row in group]
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
                "contraction_execution_target": first.get("contraction_execution_target"),
                "upmem_execution_mode": first.get("upmem_execution_mode"),
                "state_output_mode": first.get("state_output_mode"),
                "validation_method": first.get("validation_method"),
                "performance_tier": bool(first.get("performance_tier", False)),
                "timing_scope": first.get("timing_scope"),
                "repeat_count": len(group),
                **_stats("total_wall_time_s", total_values),
                **_stats("simulation_compute_time_s", compute_values),
                "validation_passed_count": sum(1 for row in group if str(row.get("validation_status")) in {"passed", "passed_native_status", "passed_runtime_only"}),
                "validation_failed_count": sum(1 for row in group if str(row.get("validation_status")) not in {"passed", "passed_native_status", "passed_runtime_only", "skipped"}),
                "unsupported_count": sum(1 for row in group if _is_unsupported(row)),
                "slice_count": _first_present(group, "slice_count"),
                "slicing_flop_ratio": _first_present(group, "slicing_flop_ratio"),
                "slicing_flop_change_kind": _first_present(group, "slicing_flop_change_kind"),
                "max_abs_error": _max_number(error.get("max_abs_error") for error in errors),
                "l2_error": _max_number(error.get("l2_error") for error in errors),
                "hardware_speedup_applicable": any(bool(row.get("hardware_speedup_applicable", False)) for row in group),
                "gpu_backend_verified": any(bool(row.get("gpu_backend_verified", False)) for row in group),
                "gpu_device_name": _first_present(group, "gpu_device_name"),
                "cpu_fallback_used": any(bool(row.get("cpu_fallback_used", False)) for row in group),
                "resource_skip_reason": _first_present(group, "resource_skip_reason"),
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
                "cpu_simulation_compute_time_s": cpu_compute,
                "gpu_simulation_compute_time_s": gpu_compute,
                "compute_speedup_cpu_over_gpu": cpu_compute / gpu_compute,
                "gpu_device_name": gpu.get("gpu_device_name"),
            }
        )
    return rows


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
            "resource_skip_reason": record.get("resource_skip_reason"),
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
        first = group[0]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "route_id": route_id,
                "benchmark_role": first.get("benchmark_role"),
                "backend_family": first.get("backend_family"),
                "execution_model": first.get("execution_model"),
                "contraction_execution_target": first.get("contraction_execution_target"),
                "accelerator_kind": first.get("accelerator_kind"),
                "upmem_execution_mode": first.get("upmem_execution_mode"),
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


def write_plots(out_dir: Path, stats_rows: list[JsonDict], speedup_rows: list[JsonDict], records: list[JsonDict]) -> JsonDict:
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
        ("cpu_gpu_runtime_by_qubits.png", "CPU/GPU runtime by qubits", "paired_speedups.csv", lambda path: _plot_cpu_gpu_runtime(plt, path, speedup_rows)),
        ("cpu_gpu_speedup_by_qubits.png", "CPU/GPU speedup by qubits", "paired_speedups.csv", lambda path: _plot_cpu_gpu_speedup(plt, path, speedup_rows)),
        ("cpu_tn_runtime_by_qubits.png", "CPU tensor-network runtime by qubits", "per_case_route_stats.csv", lambda path: _plot_cpu_tn_runtime(plt, path, stats_rows)),
        ("cpu_tn_slicing_flop_ratio.png", "Quimb slicing FLOP ratio", "per_case_route_stats.csv", lambda path: _plot_slicing_ratio(plt, path, stats_rows)),
        ("upmem_supported_boundary.png", "UPMEM SDK simulator support boundary", "per_case_route_stats.csv", lambda path: _plot_upmem_boundary(plt, path, stats_rows)),
        ("upmem_accuracy_error.png", "UPMEM SDK simulator accuracy", "per_case_route_stats.csv", lambda path: _plot_upmem_accuracy(plt, path, stats_rows)),
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
            "Run `make research-plan` to print the command pack.",
            "Run `RUN_RESEARCH=1 make research-benchmarks` for full manual execution.",
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
            "CPU/GPU performance-tier speedup uses `simulation_compute_time_s`; wall-time ratios are reported separately.",
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
            f"- Unsupported/skipped rows preserved: {len(unsupported_rows)}.",
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
            "- UPMEM SDK simulator rows as strict SDK simulator code-path and boundary evidence.",
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
    return issues


def _upmem_readiness_lines(records: list[JsonDict], unsupported_rows: list[JsonDict]) -> list[str]:
    upmem_records = [row for row in records if row.get("contraction_execution_target") == "upmem" or row.get("upmem_execution_mode") == "sdk_simulator"]
    if not upmem_records:
        return [
            "- No UPMEM SDK simulator records were loaded in this pack.",
            "- Next target: run `research_upmem_boundary.yml` before making stronger UPMEM claims.",
        ]
    unsupported = [row for row in unsupported_rows if str(row.get("route_id") or "").startswith("upmem") or row.get("route_id") == "upmem_tn_sdk_simulator_quantized"]
    fallback_count = sum(1 for row in upmem_records if bool(row.get("cpu_fallback_used", False)))
    sdk_count = sum(1 for row in upmem_records if row.get("upmem_execution_mode") == "sdk_simulator")
    reasons = Counter(str(row.get("resource_skip_reason") or row.get("warnings") or row.get("validation_status") or "unknown") for row in unsupported)
    lines = [
        f"- UPMEM SDK simulator records loaded: {len(upmem_records)}; SDK simulator rows: {sdk_count}.",
        f"- CPU fallback flagged in UPMEM rows: {fallback_count}.",
        f"- Unsupported/boundary rows: {len(unsupported)}.",
    ]
    if reasons:
        lines.append(f"- Top blocker reasons: {', '.join(f'{reason}={count}' for reason, count in reasons.most_common(5))}.")
    lines.extend(
        [
            "- Evidence still blocks stronger UPMEM claims where tensor/task size caps, rank caps, lack of tiling, single-DPU execution, host-DPU transfer overhead, quantization/dequantization overhead, missing hardware timing, or missing multi-DPU scheduling appear in records.",
            "- Recommended next UPMEM implementation target: add conservative tiling/support expansion for the first unsupported generic TaskGraph boundary before pursuing multi-DPU or hardware-speedup claims.",
        ]
    )
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
    if "upmem_tn_sdk_simulator_quantized" not in routes:
        missing.append("UPMEM SDK simulator quantized records are absent.")
    return missing or ["No mandatory evidence class is obviously absent from loaded records."]


def _plot_cpu_gpu_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_paired_cpu_gpu_rows"
    usable = [row for row in rows if _plot_qubits(row) is not None]
    if not usable:
        return "no_actual_qubit_metadata"
    ordered = sorted(usable, key=lambda row: (str(row["case_family"]), int(_plot_qubits(row) or 0), str(row["case_id"])))
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(10.0, len(labels) * 0.4), 6.0), constrained_layout=True)
    ax.bar([i - width / 2 for i in x], [float(row["cpu_simulation_compute_time_s"]) for row in ordered], width=width, label="QuEST CPU")
    ax.bar([i + width / 2 for i in x], [float(row["gpu_simulation_compute_time_s"]) for row in ordered], width=width, label="QuEST GPU")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Circuit family and qubits")
    ax.set_ylabel("Compute time per repeat (s, log scale)")
    ax.set_title("CPU/GPU full-state compute runtime")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save_plot(fig, path)
    return None


def _plot_cpu_gpu_speedup(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_paired_cpu_gpu_rows"
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        if _plot_qubits(row) is not None:
            grouped[str(row["case_family"])].append(row)
    if not grouped:
        return "no_actual_qubit_metadata"
    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    for family, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(_plot_qubits(row) or 0))
        ax.plot([int(_plot_qubits(row) or 0) for row in ordered], [float(row["compute_speedup_cpu_over_gpu"]) for row in ordered], marker="o", label=family)
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
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for route in ("quimb_tn_exact", "quimb_tn_sliced_exact"):
        group = sorted(
            (row for row in selected if row["route_id"] == route and _plot_qubits(row) is not None),
            key=lambda row: (str(row["case_family"]), int(_plot_qubits(row) or 0), str(row["case_id"])),
        )
        if not group:
            continue
        labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in group]
        ax.plot(range(len(group)), [float(row["simulation_compute_time_s_median"]) for row in group], marker="o", label=route)
    ax.set_yscale("log")
    ax.set_xlabel("Case order")
    ax.set_ylabel("Median compute time (s, log scale)")
    ax.set_title("CPU TN runtime for Quimb routes")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
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
    selected = [row for row in rows if row.get("route_id") == "upmem_tn_sdk_simulator_quantized"]
    if not selected:
        return "no_upmem_rows"
    ordered = sorted((row for row in selected if _plot_qubits(row) is not None), key=lambda row: (str(row["case_family"]), int(_plot_qubits(row) or 0), str(row["case_id"])))
    if not ordered:
        return "no_actual_qubit_metadata"
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    values = [0 if int(row.get("unsupported_count", 0) or 0) else 1 for row in ordered]
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.45), 4.8), constrained_layout=True)
    ax.bar(range(len(ordered)), values, color=["#16a34a" if value else "#dc2626" for value in values])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["unsupported", "supported"])
    ax.set_title("UPMEM SDK simulator support boundary")
    _save_plot(fig, path)
    return None


def _plot_upmem_accuracy(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if row.get("route_id") == "upmem_tn_sdk_simulator_quantized" and _positive(row.get("max_abs_error")) is not None]
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
    ax.set_title("UPMEM SDK simulator accuracy vs reference")
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
        "cpu_gpu_runtime_by_qubits.png": "Median QuEST CPU and verified QuEST GPU compute time by circuit size.",
        "cpu_gpu_speedup_by_qubits.png": "CPU/GPU compute speedup; values above 1 mean GPU faster.",
        "cpu_tn_runtime_by_qubits.png": "Quimb CPU tensor-network timing; sliced and unsliced routes are separate execution modes.",
        "cpu_tn_slicing_flop_ratio.png": "slicing_flop_ratio = sliced cotengra reported FLOPs / unsliced cotengra reported FLOPs.",
        "upmem_supported_boundary.png": "Supported versus unsupported strict UPMEM SDK simulator rows.",
        "upmem_accuracy_error.png": "UPMEM quantized SDK simulator max absolute error where validation data exists.",
        "internal_parallelism_metadata_by_qubits.png": "Diagnostic internal TaskGraph frontier metadata, not serious baseline performance.",
    }.get(filename, "")


def _stats(prefix: str, values: list[float]) -> JsonDict:
    if not values:
        return {
            f"{prefix}_median": None,
            f"{prefix}_mean": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_std": None,
        }
    return {
        f"{prefix}_median": statistics.median(values),
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


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _first_present(group: list[JsonDict], key: str) -> Any:
    for row in group:
        value = row.get(key)
        if value not in {None, ""}:
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


def _pack_dir(root: Path, out: Path | None) -> Path:
    if out is not None:
        return out if out.is_absolute() else root / out
    return DEFAULT_COMPARISON_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _latest_evidence_for_suite(root: Path, suite_id: str) -> Path | None:
    suite_root = root / "runs" / "evidence" / suite_id
    if not suite_root.exists():
        return None
    candidates = [path.parent for path in suite_root.glob("*/*/normalized_records.jsonl")]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
