"""Strict inspection report for the M4.2 SimplePIM qualification route."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from quantum_bench.bench.upmem_hardware_simplepim_rank1_m4_2 import (
    _require_response,
    BACKEND_ID,
    ROUTE_ID,
)


def inspect(run: Path) -> dict[str, object]:
    run = run.resolve()
    summary_path = run / "upmem_hardware_simplepim_rank1_m4_2_summary.json"
    response_path = run / "native_execute_response.json"
    records_path = run / "normalized_records.jsonl"
    if not summary_path.is_file() or not response_path.is_file() or not records_path.is_file():
        raise ValueError("M4.2 run must contain summary, native response, and normalized_records.jsonl")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    _require_response(response, parser=False)
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("M4.2 normalized records must contain only JSON objects")
    measured = [row for row in response["repetitions"] if row.get("warmup") is False]
    by_repeat = {row.get("repeat_id"): row for row in rows}
    native_by_repeat = {row.get("repeat_id"): row for row in measured}
    normalized_identity = (
        len(rows) == 5
        and set(by_repeat) == set(native_by_repeat) == set(range(5))
        and all(
            by_repeat[index].get("input_hash") == native_by_repeat[index].get("input_hash")
            and by_repeat[index].get("output_hash") == native_by_repeat[index].get("output_hash")
            and by_repeat[index].get("result_int64") == native_by_repeat[index].get("result_int64")
            and by_repeat[index].get("total_route_time_s") == native_by_repeat[index].get("total_time_s")
            for index in range(5)
        )
    )
    checks = {
        "completed": summary.get("status") == "completed",
        "identity": response.get("backend_id") == BACKEND_ID and response.get("route_id") == ROUTE_ID,
        "five_measured_rows": len(rows) == 5 and {row.get("repeat_id") for row in rows} == set(range(5)),
        "normalized_identity": normalized_identity,
        "simplepim_operator": all(row.get("simplepim_operator_api_used") is True for row in rows),
        "operator_metadata": all(row.get("operator_metadata_checks_passed") is True for row in rows),
        "host_mediated_communication": all(row.get("communication_provider") == "host_mediated" and row.get("pid_comm_invoked") is False for row in rows),
        "no_atim": all(row.get("atim_integrated") is False for row in rows),
        "no_speedup_claim": response.get("hardware_speedup_applicable") is False,
        "no_fake_energy": not any("energy" in row for row in rows),
        "fixed_qualification_fixture": all(
            row.get("qualification_task_count") == 1
            and row.get("task_graph_integrated") is False
            and row.get("execution_model") == "fixed_rank1_kernel_qualification_fixture"
            for row in rows
        ),
        "layout_bound_only": all(row.get("mram_capacity_verified") is False for row in rows),
    }
    return {"status": "valid_functionality_evidence" if all(checks.values()) else "invalid_or_incomplete", "run_dir": str(run), "row_count": len(rows), "checks": checks, "claim_boundary": summary.get("claim_boundary"), "next_blocker": "genuine operand transport from a real ContractionTask"}


def write_report(run: Path, root: Path) -> Path:
    result = inspect(run)
    out = root / "runs" / "comparisons" / "upmem_hardware_simplepim_rank1_m4_2"
    out.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target = out / stamp
    target.mkdir()
    (target / "inspection.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (target / "m4_2_records.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in (run / "normalized_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        fields = ["case_id", "repeat_id", "total_route_time_s", "application_visible_h2d_bytes", "application_visible_d2h_bytes", "exact_integer_match", "simplepim_operator_api_used"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    (target / "README.md").write_text(
        "# M4.2 SimplePIM rank-1 qualification\n\n"
        "This report is functionality evidence for one rank-1 operator on two DPUs. "
        "It contains no speedup, energy, scaling, or general TaskGraph claim.\n\n"
        f"Inspection: `{result['status']}`\n\n"
        "The fixed-vector route is not a complete TaskGraph adapter: genuine "
        "operand transport from a real ContractionTask remains missing. The "
        "native route records SimplePIM-managed allocation and metadata checks; "
        "void SimplePIM APIs do not independently validate intermediate contents.\n",
        encoding="utf-8",
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = write_report(args.input, root)
    print(target)
    inspection = json.loads((target / "inspection.json").read_text(encoding="utf-8"))
    return 0 if inspection.get("status") == "valid_functionality_evidence" else 2


if __name__ == "__main__":
    raise SystemExit(main())
