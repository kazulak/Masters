"""Validate and summarize one M4.3 physical TaskGraph run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from quantum_bench.bench.upmem_hardware_simplepim_taskgraph_m4_3 import (
    _load_workload,
    _graph,
    _is_integer,
    _operand_binding,
    _require_response,
    SUITE_ID,
)
from quantum_bench.tn import execution_identity_metadata
from quantum_bench.tn.execution_bundle import canonical_hash


def _canonical_json(value: object) -> str:
    """Normalize JSON round-trip differences such as tuple versus list."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_operand_binding(actual: object, expected: dict[str, object], label: str) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"M4.3 {label} operand binding must be an object")
    stored_hash = actual.get("binding_hash")
    if not isinstance(stored_hash, str):
        raise ValueError(f"M4.3 {label} operand binding is missing binding_hash")
    payload = {key: value for key, value in actual.items() if key != "binding_hash"}
    if canonical_hash(payload) != stored_hash:
        raise ValueError(f"M4.3 {label} operand binding hash mismatch")
    expected_payload = {key: value for key, value in expected.items() if key != "binding_hash"}
    if _canonical_json(payload) != _canonical_json(expected_payload):
        raise ValueError(f"M4.3 {label} operand binding mismatch")
    if stored_hash != expected["binding_hash"]:
        raise ValueError(f"M4.3 {label} operand binding hash does not match expected binding")


def inspect(run: Path) -> dict[str, object]:
    run = run.resolve()
    summary_path = run / f"{SUITE_ID}_summary.json"
    response_path = run / "native_execute_response.json"
    manifest_path = run / "input_manifest.json"
    records_path = run / "normalized_records.jsonl"
    bundle_path = run / "execution_bundle.json"
    if not all(path.is_file() for path in (summary_path, response_path, manifest_path, records_path, bundle_path)):
        raise ValueError("M4.3 run must contain summary, response, input manifest, execution bundle, and normalized records")
    operands_path = run / "operands.bin"
    if not operands_path.is_file():
        raise ValueError("M4.3 run must contain operands.bin before byte accounting")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workload = _load_workload()
    identities = execution_identity_metadata(_graph(workload), plan_reused=False)
    graph = _graph(workload)
    expected_binding = _operand_binding(graph, graph.tasks[0], input_sha256=manifest["input_file_sha256"])
    actual_input_sha256 = hashlib.sha256(operands_path.read_bytes()).hexdigest()
    if actual_input_sha256 != manifest["input_file_sha256"]:
        raise ValueError("M4.3 operands.bin SHA-256 does not match input manifest")
    _validate_operand_binding(manifest.get("operand_binding"), expected_binding, "input manifest")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    _validate_operand_binding(bundle.get("operand_binding"), expected_binding, "execution bundle")
    if not _is_integer(manifest.get("reference_int64")):
        raise ValueError("M4.3 manifest reference_int64 must be an integer")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    _require_response(payload=response, identities=identities, input_sha256=manifest["input_file_sha256"], input_hash=manifest["input_hash"], reference=manifest["reference_int64"])
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = {
        "completed": summary.get("status") == "completed",
        "five_measured_rows": len(rows) == 5 and {row.get("repeat_id") for row in rows} == set(range(5)),
        "taskgraph_derived_adapter": all(row.get("taskgraph_derived_operand_adapter") is True and row.get("host_taskgraph_operand_binding") is True and row.get("native_taskgraph_protocol") is False and row.get("native_plan_identity_binding") is False and row.get("contraction_plan_hash") == identities["contraction_plan_hash"] and row.get("operand_binding_hash") == expected_binding["binding_hash"] for row in rows),
        "physical_only": all(row.get("target_observed") == "physical_hardware" and row.get("cpu_fallback_used") is False and row.get("simulator_kernel_executed") is False for row in rows),
        "allocation_profile": response.get("allocation_profile") == "backend=hw",
        "input_byte_contract": operands_path.stat().st_size == 512 and all(row.get("scientific_input_file_bytes") == 512 and row.get("application_visible_h2d_bytes") == 2048 for row in rows),
        "no_extra_claims": all(row.get("hardware_speedup_applicable") is False and row.get("pid_comm_invoked") is False and row.get("atim_integrated") is False for row in rows),
        "no_energy": not any("energy" in row for row in rows),
    }
    return {
        "status": "taskgraph_derived_operand_adapter_functionality_evidence" if all(checks.values()) else "invalid_or_incomplete",
        "run_dir": str(run),
        "row_count": len(rows),
        "checks": checks,
        "claim_boundary": summary.get("claim_boundary"),
        "claim_verdict": summary.get("claim_verdict", "taskgraph_derived_operand_adapter_functionality_evidence"),
        "limitations": [
            "SimplePIM intermediate tables are not independently traced for contents.",
            "The result is a one-task synthetic rank-1 TaskGraph fixture, not general TN execution.",
            "Timing is bring-up timing and does not support speedup, energy, PID-Comm, ATiM, or scaling claims.",
        ],
    }


def write_report(run: Path, root: Path) -> Path:
    result = inspect(run)
    out = root / "runs" / "comparisons" / SUITE_ID
    out.mkdir(parents=True, exist_ok=True)
    target = out / __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target.mkdir()
    (target / "inspection.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [json.loads(line) for line in (run / "normalized_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = ["case_id", "repeat_id", "per_iteration_operator_time_s", "scientific_input_file_bytes", "application_visible_h2d_bytes", "application_visible_d2h_bytes", "exact_integer_match", "circuit_semantics_hash", "tensor_network_hash", "contraction_plan_hash", "map_binary_hash", "genred_binary_hash"]
    with (target / "m4_3_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    (target / "README.md").write_text(
        "# M4.3 SimplePIM TaskGraph adapter\n\n"
        "This report validates one bounded rank-1 `ContractionTask`-derived operand adapter transported to the existing SimplePIM operator. "
        "It is functionality evidence only: no general TN, speedup, energy, PID-Comm, ATiM, or scaling claim.\n\n"
        f"Inspection: `{result['status']}`\n\n"
        "The host binds the canonical task graph and exact operand-file SHA-256 in "
        "the input manifest and execution bundle. The native protocol remains "
        "operand-only: it does not parse the TaskGraph or bind the plan identity.\n",
        encoding="utf-8",
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    target = write_report(args.input, Path(__file__).resolve().parents[1])
    print(target)
    return 0 if json.loads((target / "inspection.json").read_text())["status"] == "taskgraph_derived_operand_adapter_functionality_evidence" else 2


if __name__ == "__main__":
    raise SystemExit(main())
