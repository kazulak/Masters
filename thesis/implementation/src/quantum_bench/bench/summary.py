from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from quantum_bench.core.jsonio import read_jsonl, write_json


def collect_raw_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((run_dir / "raw").glob("*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def write_summary(run_dir: Path) -> dict[str, Any]:
    records = collect_raw_records(run_dir)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["case_id"], record["route"])].append(record)

    rows = []
    for (case_id, route), group in sorted(grouped.items()):
        passed = [r for r in group if r["status"] == "passed"]
        failed = [r for r in group if r["status"] == "failed"]
        skipped = [r for r in group if r["status"] == "skipped"]
        times = [float(r["total_time_s"]) for r in passed if r.get("total_time_s") is not None]
        energies = [
            float(r["energy_joules"])
            for r in passed
            if r.get("energy_joules") is not None and float(r["energy_joules"]) > 0 and r.get("energy_source") not in {None, "unavailable"}
        ]
        rows.append(
            {
                "case_id": case_id,
                "route": route,
                "role": group[0].get("role"),
                "simulation_method": group[0].get("simulation_method"),
                "kernel_family": group[0].get("kernel_family"),
                "hardware_target": group[0].get("hardware_target"),
                "execution_mode": group[0].get("execution_mode"),
                "output_contract": group[0].get("output_contract"),
                "validation_mode": group[0].get("validation_mode"),
                "n_qubits": group[0].get("n_qubits"),
                "depth": group[0].get("depth"),
                "circuit_family": group[0].get("circuit_family"),
                "status": "passed" if passed else ("skipped" if skipped and not failed else "failed"),
                "passed_count": len(passed),
                "failed_count": len(failed),
                "skipped_count": len(skipped),
                "time_median_s": _median(times),
                "time_mean_s": _mean(times),
                "time_min_s": min(times) if times else None,
                "time_max_s": max(times) if times else None,
                "time_stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0 if times else None,
                "energy_median_j": _median(energies),
                "energy_source": _energy_source(group),
                "skip_reason": _first(group, "skip_reason"),
                "error": _first(group, "error"),
            }
        )

    summary = {
        "schema_version": "quantum_bench_summary_v2",
        "run_dir": str(run_dir),
        "record_count": len(records),
        "rows": rows,
        "validated_routes": [row for row in rows if row.get("validation_mode") == "compare_output"],
        "benchmark_only_baselines": [row for row in rows if row.get("validation_mode") == "benchmark_only"],
        "skipped_or_probe_routes": [row for row in rows if row.get("validation_mode") == "skip_with_reason" or row.get("status") == "skipped"],
        "energy_status": energy_status(records),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "metrics" / "metrics.json", rows)
    write_csv(run_dir / "metrics" / "metrics.csv", rows)
    write_markdown(run_dir / "summary.md", rows, summary["energy_status"])
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "route",
        "role",
        "simulation_method",
        "kernel_family",
        "hardware_target",
        "execution_mode",
        "output_contract",
        "validation_mode",
        "n_qubits",
        "depth",
        "circuit_family",
        "status",
        "passed_count",
        "failed_count",
        "skipped_count",
        "time_median_s",
        "time_mean_s",
        "time_min_s",
        "time_max_s",
        "time_stdev_s",
        "energy_median_j",
        "energy_source",
        "skip_reason",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown(path: Path, rows: list[dict[str, Any]], energy: dict[str, Any]) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        f"Records: {sum(int(row['passed_count']) + int(row['failed_count']) + int(row['skipped_count']) for row in rows)}",
        f"Energy status: {energy['status']}",
        "",
        "| case | route | status | passed | failed | skipped | median time s | median energy J |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['route']} | {row['status']} | {row['passed_count']} | {row['failed_count']} | "
            f"{row['skipped_count']} | {row['time_median_s']} | {row['energy_median_j']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def energy_status(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        record
        for record in records
        if record.get("energy_joules") is not None and float(record["energy_joules"]) > 0 and record.get("energy_source") not in {None, "unavailable"}
    ]
    if measured:
        return {"status": "measured", "measured_records": len(measured), "total_records": len(records)}
    return {"status": "unavailable_or_zero", "measured_records": 0, "total_records": len(records)}


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _first(records: list[dict[str, Any]], key: str) -> Any:
    for record in records:
        if record.get(key):
            return record[key]
    return None


def _energy_source(records: list[dict[str, Any]]) -> str:
    sources = sorted({str(record.get("energy_source")) for record in records if record.get("energy_source")})
    return ",".join(sources) if sources else "unavailable"
