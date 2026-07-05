from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


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

TN_PATH_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "n_qubits",
    "repeat_id",
    "route_id",
    "benchmark_role",
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
    "unquantized_simulation_compute_time_s",
    "quantized_simulation_compute_time_s",
    "compute_ratio_unquantized_over_quantized",
    "unquantized_total_wall_time_s",
    "quantized_total_wall_time_s",
    "wall_ratio_unquantized_over_quantized",
    "quantization_max_abs_error",
    "quantization_l2_error",
    "max_abs_error_vs_reference",
    "l2_error_vs_reference",
    "validation_status",
    "quantized_replay_numeric_contract",
]

UPMEM_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "n_qubits",
    "repeat_id",
    "route_id",
    "quantization_mode",
    "status",
    "validation_status",
    "contraction_execution_target",
    "upmem_execution_mode",
    "execution_backend",
    "cpu_fallback_used",
    "upmem_program_executed",
    "dpu_program_invocations",
    "hardware_execution",
    "hardware_speedup_applicable",
    "simulation_compute_time_s",
    "total_wall_time_s",
    "max_abs_error",
    "l2_error",
    "resource_skip_reason",
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

    _write_csv(args.out / "full_state_cpu_gpu_by_circuit.csv", cpu_gpu_rows, FULL_STATE_FIELDS)
    _write_csv(args.out / "tn_path_comparison_by_circuit.csv", tn_path_rows, TN_PATH_FIELDS)
    _write_csv(args.out / "tn_quantization_comparison.csv", tn_quant_rows, TN_QUANT_FIELDS)
    _write_csv(args.out / "upmem_boundary_quantization.csv", upmem_rows, UPMEM_FIELDS)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "thesis_comparison_report",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_count": len(args.inputs),
        "record_count": len(records),
        "inputs": [path.as_posix() for path in args.inputs],
        "outputs": [
            "full_state_cpu_gpu_by_circuit.csv",
            "tn_path_comparison_by_circuit.csv",
            "tn_quantization_comparison.csv",
            "upmem_boundary_quantization.csv",
            "benchmark_summary.md",
            "plot_manifest.json",
        ],
        "claims": {
            "gpu_scope": "QuEST full-state GPU only; not GPU tensor-network evidence.",
            "tn_quantization_scope": "CPU TN path replay uses per-contraction operand quantization and complex128 accumulation.",
            "upmem_scope": "UPMEM SDK simulator evidence is not hardware timing or hardware speedup.",
        },
    }
    _write_json(args.out / "thesis_report_manifest.json", manifest)
    plot_manifest = write_plots(args.out, cpu_gpu_rows, tn_path_rows, tn_quant_rows, upmem_rows)
    _write_json(args.out / "plot_manifest.json", plot_manifest)
    (args.out / "benchmark_summary.md").write_text(
        benchmark_summary(args.title, records, cpu_gpu_rows, tn_path_rows, tn_quant_rows, upmem_rows, plot_manifest),
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
                "benchmark_role": record.get("benchmark_role"),
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
                "unquantized_simulation_compute_time_s": base_compute,
                "quantized_simulation_compute_time_s": quant_compute,
                "compute_ratio_unquantized_over_quantized": base_compute / quant_compute,
                "unquantized_total_wall_time_s": base_wall,
                "quantized_total_wall_time_s": quant_wall,
                "wall_ratio_unquantized_over_quantized": base_wall / quant_wall,
                "quantization_max_abs_error": quant.get("quantization_max_abs_error"),
                "quantization_l2_error": quant.get("quantization_l2_error"),
                "max_abs_error_vs_reference": errors.get("max_abs_error"),
                "l2_error_vs_reference": errors.get("l2_error"),
                "validation_status": quant.get("validation_status"),
                "quantized_replay_numeric_contract": quant.get("quantized_replay_numeric_contract"),
            }
        )
    return rows


def upmem_boundary_rows(records: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for record in records:
        if record.get("route_id") != "upmem_tn_sdk_simulator_quantized" and record.get("contraction_execution_target") != "upmem":
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
                "quantization_mode": record.get("quantization_mode"),
                "status": record.get("status"),
                "validation_status": record.get("validation_status"),
                "contraction_execution_target": record.get("contraction_execution_target"),
                "upmem_execution_mode": record.get("upmem_execution_mode"),
                "execution_backend": record.get("execution_backend"),
                "cpu_fallback_used": bool(record.get("cpu_fallback_used", False)),
                "upmem_program_executed": bool(record.get("upmem_program_executed", False)),
                "dpu_program_invocations": record.get("dpu_program_invocations"),
                "hardware_execution": bool(record.get("hardware_execution", False)),
                "hardware_speedup_applicable": bool(record.get("hardware_speedup_applicable", False)),
                "simulation_compute_time_s": record.get("simulation_compute_time_s"),
                "total_wall_time_s": record.get("total_wall_time_s"),
                "max_abs_error": errors.get("max_abs_error"),
                "l2_error": errors.get("l2_error"),
                "resource_skip_reason": record.get("resource_skip_reason"),
            }
        )
    return rows


def write_plots(
    out_dir: Path,
    cpu_gpu_rows: list[JsonDict],
    tn_path_rows: list[JsonDict],
    tn_quant_rows: list[JsonDict],
    upmem_rows: list[JsonDict],
) -> JsonDict:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"schema_version": SCHEMA_VERSION, "status": "skipped", "reason": "matplotlib_unavailable", "error": str(exc), "plots": []}
    entries = [
        _plot_entry(plt, plots_dir / "full_state_cpu_gpu_speedup_by_qubits.png", "Full-state CPU/GPU speedup", lambda path: _plot_speedup(plt, path, cpu_gpu_rows)),
        _plot_entry(plt, plots_dir / "tn_path_runtime_by_qubits.png", "TN route/path runtime", lambda path: _plot_tn_runtime(plt, path, tn_path_rows)),
        _plot_entry(plt, plots_dir / "tn_quantization_error_by_qubits.png", "TN quantization error", lambda path: _plot_tn_quant_error(plt, path, tn_quant_rows)),
        _plot_entry(plt, plots_dir / "upmem_boundary_status.png", "UPMEM SDK simulator boundary", lambda path: _plot_upmem_boundary(plt, path, upmem_rows)),
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
    tn_path_rows: list[JsonDict],
    tn_quant_rows: list[JsonDict],
    upmem_rows: list[JsonDict],
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
        f"- TN route/path rows: {len(tn_path_rows)}",
        f"- TN quantization matched rows: {len(tn_quant_rows)}",
        f"- UPMEM boundary rows: {len(upmem_rows)}",
        "",
        "## Claims Allowed",
        "",
        "- QuEST CPU vs QuEST GPU rows are direct full-state route comparisons when GPU rows are verified.",
        "- Quimb rows are serious CPU tensor-network evidence.",
        "- CPU path replay rows are diagnostic path and quantization attribution evidence.",
        "- UPMEM SDK simulator rows are strict code-path/boundary evidence only.",
        "",
        "## Claims Not Allowed",
        "",
        "- QuEST full-state GPU only: these rows are not GPU tensor-network evidence.",
        "- QuEST GPU full-state rows are not GPU tensor-network evidence.",
        "- CPU path replay rows are not serious external TN baselines.",
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
        return payload
    if isinstance(payload, str) and payload:
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


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


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(to_jsonable(rows))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plot_entry(plt: Any, path: Path, title: str, plotter: Any) -> JsonDict:
    reason = plotter(path)
    if reason:
        return {"plot": path.name, "title": title, "status": "skipped", "reason": reason}
    return {"plot": path.name, "title": title, "status": "generated", "reason": None, "size_bytes": path.stat().st_size}


def _plot_speedup(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_matched_cpu_gpu_rows"
    grouped = _median_by_family_qubits(rows, "compute_speedup_cpu_over_gpu")
    return _line_plot(plt, path, grouped, "Full-state CPU/GPU compute speedup", "CPU/GPU speedup")


def _plot_tn_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if _positive(row.get("simulation_compute_time_s")) is not None]
    if not selected:
        return "no_tn_runtime_rows"
    grouped: dict[str, list[tuple[int, float]]] = {}
    by_key: dict[tuple[str, int], list[float]] = {}
    for row in selected:
        qubits = _int_or_none(row.get("n_qubits"))
        if qubits is None:
            continue
        label = str(row.get("route_id"))
        by_key.setdefault((label, qubits), []).append(float(row["simulation_compute_time_s"]))
    for (label, qubits), values in by_key.items():
        grouped.setdefault(label, []).append((qubits, statistics.median(values)))
    return _line_plot(plt, path, grouped, "Tensor-network route runtime", "Median compute time (s)", log_y=True)


def _plot_tn_quant_error(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_tn_quantization_rows"
    grouped = _median_by_family_qubits(rows, "max_abs_error_vs_reference")
    return _line_plot(plt, path, grouped, "TN quantized replay error", "Max abs error", log_y=True)


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
