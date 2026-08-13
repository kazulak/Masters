from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.upmem_m5_report import M5_RECORD_FIELDS, ReportError, generate_report


HOST_BINARY_SHA256 = "a" * 64
DPU_BINARY_SHA256 = "b" * 64


def _row(
    *,
    case_id: str = "case-a",
    route_id: str = "route-a",
    workload_kind: str = "quantum_case",
    numeric_mode: str = "float32",
    partition_mode: str = "output",
    dpu_count: int = 1,
    requested_dpu_count: int | None = None,
    allocated_dpu_count: int | None = None,
    runtime_s: float | None = 1.0,
    status: str = "completed",
    repeat_id: int = 0,
    scaling_kind: str = "strong",
    tasklets: int = 1,
    timing_scope: str = "host_observed_total_time",
    target_observed: str = "physical_hardware",
    nested_timing: bool = False,
    host_binary_hash: str = HOST_BINARY_SHA256,
    dpu_binary_hash: str = DPU_BINARY_SHA256,
) -> dict[str, object]:
    requested = dpu_count if requested_dpu_count is None else requested_dpu_count
    allocated = dpu_count if allocated_dpu_count is None else allocated_dpu_count
    row: dict[str, object] = {
        "case_id": case_id,
        "schema_version": "upmem_m5_record_v3",
        "route_version": "upmem_route_v3",
        "route_id": route_id,
        "workload_kind": workload_kind,
        "quantum_case": workload_kind,
        "numeric_mode": numeric_mode,
        "partition_mode": partition_mode,
        "target_observed": target_observed,
        "requested_dpu_count": requested,
        "allocated_dpu_count": allocated,
        "observed_rank_count": 1,
        "one_rank": True,
        "rank_count": 1,
        "hardware_allocation_verified": True,
        "native_execution": True,
        "hardware_execution": True,
        "hardware_release_verified": True,
        "policy_reference_validation": {"passed": True},
        "simulator": False,
        "cpu_fallback": False,
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "fallback_used": False,
        "tasklets_per_dpu": tasklets,
        "timing_scope": timing_scope,
        "status": status,
        "repeat_id": repeat_id,
        "transfers": {"h2d_bytes": 100, "d2h_bytes": 20, "host_mediated_reduction_bytes": 8},
        "run_global_transfers": {"h2d_bytes": 100, "d2h_bytes": 20, "host_mediated_reduction_bytes": 8},
        "run_metadata": {
            "transfers": {"h2d_bytes": 100, "d2h_bytes": 20, "host_mediated_reduction_bytes": 8},
        },
        "load_balance": {"ratio": 1.0},
        # Policy-reference error is deliberately different from the report metric.
        "max_abs_error": 9.0 if numeric_mode == "int8" else 7.0,
        "full_precision_accuracy": {"max_abs_error": 0.0},
        "quantization_error_vs_float32": 0.125 if numeric_mode == "int8" else None,
        "quantization_mode": "per_task_resident_requantize" if numeric_mode == "int8" else "none",
        "numeric_arithmetic": "int8" if numeric_mode == "int8" else "float32",
        "numeric_transport": "float32_mram" if numeric_mode == "int8" else "float32",
        "requantization_scope": "per_task" if numeric_mode == "int8" else "none",
        "packed_int8_transfer": False,
        "collective_provider": "host_mediated_sum_v1" if partition_mode == "contracted" else "none",
        "reconstruction_provider": (
            "host_float64_reduction_v1" if partition_mode == "contracted" else "host_owned_range_assembly_v1"
        ),
        "task_hash": "task-a",
        "circuit_semantics_hash": "circuit-a",
        "tensor_network_hash": "network-a",
        "contraction_plan_hash": "plan-a",
        "contraction_path_structure_hash": "path-a",
        "operation_sha256": f"operation-{numeric_mode}",
        "host_binary_hash": host_binary_hash,
        "dpu_binary_hash": dpu_binary_hash,
    }
    row["scaling_kind"] = scaling_kind
    if nested_timing:
        row["timing"] = {"total_time_s": runtime_s}
    else:
        row["timing_s"] = runtime_s
    row["per_repeat_timing"] = {
        "total_time_s": runtime_s,
        "transfers": {"h2d_bytes": 10, "d2h_bytes": 2, "host_mediated_reduction_bytes": 1},
    }
    return row


def _write_run(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    run = tmp_path / "evidence"
    run.mkdir(parents=True)
    (run / "normalized_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_statistics_ratios_and_incompatible_pairing(tmp_path: Path) -> None:
    rows = [
        _row(runtime_s=10.0, repeat_id=0),
        _row(runtime_s=12.0, repeat_id=1),
        _row(dpu_count=2, runtime_s=5.0, repeat_id=0),
        _row(dpu_count=2, runtime_s=6.0, repeat_id=1),
        _row(partition_mode="contracted", runtime_s=5.0),
        _row(dpu_count=4, requested_dpu_count=4, allocated_dpu_count=2, runtime_s=None, status="unsupported", repeat_id=0),
        # These rows must not pair with route-a's one-DPU baseline.
        _row(route_id="route-b", dpu_count=2, runtime_s=2.0),
        _row(numeric_mode="int8", dpu_count=2, runtime_s=3.0),
        _row(case_id="case-weak", scaling_kind="same_route_dpu_weak_scaling", dpu_count=1, runtime_s=20.0, repeat_id=2),
        _row(dpu_count=2, tasklets=2, runtime_s=4.0),
        _row(dpu_count=2, timing_scope="kernel_only", runtime_s=4.0),
        _row(dpu_count=8, target_observed="simulator", runtime_s=99.0),
    ]
    run = _write_run(tmp_path, rows)
    output = generate_report(run, tmp_path / "report-root", timestamp="fixed")

    runtime = _csv_rows(output / "tables/m5_runtime_statistics.csv")
    baseline = next(
        row
        for row in runtime
        if row["route_id"] == "route-a"
        and row["numeric_mode"] == "float32"
        and row["partition_mode"] == "output"
        and row["dpu_count"] == "1"
    )
    assert float(baseline["median"]) == 11.0
    assert float(baseline["iqr"]) == 1.0
    assert float(baseline["min"]) == 10.0
    assert float(baseline["max"]) == 12.0
    assert baseline["repeat_count"] == "2"
    assert baseline["measured_repeat_count"] == "2"
    transfers = _csv_rows(output / "tables/m5_transfer_statistics.csv")
    assert next(row for row in transfers if row["dpu_count"] == "1")["median"] == "10.0"
    record = next(row for row in _csv_rows(output / "tables/m5_records.csv") if row["dpu_count"] == "1")
    assert record["h2d_bytes"] == "10.0"
    assert record["run_h2d_bytes_provenance"] == "100.0"

    ratios = _csv_rows(output / "tables/m5_strong_scaling.csv")
    route_a_dpu2 = next(
        row
        for row in ratios
        if row["route_id"] == "route-a" and row["dpu_count"] == "2" and row["tasklets_per_dpu"] == "1"
    )
    assert float(route_a_dpu2["speedup"]) == 2.0
    assert float(route_a_dpu2["efficiency"]) == 1.0
    assert not any(row["route_id"] == "route-b" for row in ratios)
    assert not any(row["numeric_mode"] == "int8" for row in ratios)
    assert not any(row["tasklets_per_dpu"] == "2" for row in ratios)
    assert not any(row["timing_scope"] == "kernel_only" for row in ratios)

    numeric_ratios = _csv_rows(output / "tables/m5_numeric_mode_ratios.csv")
    numeric = next(row for row in numeric_ratios if row["dpu_count"] == "2")
    assert float(numeric["runtime_ratio_float32_over_int8"]) > 1.0
    assert float(numeric["runtime_ratio_float32_over_int8"]) == 11.0 / 6.0
    partition_ratios = _csv_rows(output / "tables/m5_partition_ratios.csv")
    partition = next(row for row in partition_ratios if row["dpu_count"] == "1")
    assert float(partition["runtime_ratio_output_over_contracted"]) > 1.0
    assert float(partition["runtime_ratio_output_over_contracted"]) == 11.0 / 5.0
    assert (output / "plots/m5_numeric_mode_runtime_ratio.png").is_file()
    assert (output / "plots/m5_partition_runtime_ratio.png").is_file()

    records = _csv_rows(output / "tables/m5_records.csv")
    failed = next(row for row in records if row["status"] == "unsupported" and row["dpu_count"] == "4")
    assert failed["requested_dpu_count"] == "4"
    assert failed["allocated_dpu_count"] == "2"
    invalid = next(row for row in runtime if row["dpu_count"] == "8")
    assert invalid["median"] == ""
    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["supported_dpu_counts"] == [1, 2]
    assert manifest["failed_or_unsupported_dpu_counts"] == [4]
    assert len(manifest["plots"]) == 9
    assert manifest["source_sha256"]
    assert next(plot for plot in manifest["plots"] if "weak_scaling_runtime" in plot["path"])["status"] == "generated"
    quant_plot = next(plot for plot in manifest["plots"] if "quantization_accuracy" in plot["path"])
    assert "on-DPU int8 requantization" in quant_plot["caption"]
    assert "same-route measured one-rank physical diagnostics" in quant_plot["caption"].lower()
    assert "no CPU/GPU speedup or multi-rank claim" in quant_plot["caption"]


def test_float_only_caption_and_timestamp_traversal_are_fail_closed(tmp_path: Path) -> None:
    run = _write_run(tmp_path, [_row(nested_timing=True)])

    with pytest.raises(ReportError, match="timestamp"):
        generate_report(run, tmp_path / "report-root", timestamp="../outside")

    output = generate_report(run, tmp_path / "report-root", timestamp="float-only")
    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    quant_plot = next(plot for plot in manifest["plots"] if "quantization_accuracy" in plot["path"])
    assert "int8" not in quant_plot["caption"]


def test_claimed_but_unverified_physical_row_is_not_admitted(tmp_path: Path) -> None:
    row = _row()
    row["claims"] = {"physical_one_rank_measured": True, "speedup": True}
    row["hardware_allocation_verified"] = False
    run = _write_run(tmp_path, [row])

    output = generate_report(run, tmp_path / "report-root", timestamp="unverified")
    records = _csv_rows(output / "tables/m5_records.csv")
    assert records[0]["physical_one_rank_valid"] == "False"
    runtime = _csv_rows(output / "tables/m5_runtime_statistics.csv")
    assert runtime[0]["median"] == ""
    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["claims"]["physical_one_rank_measured"] is False
    assert all(entry["status"] == "todo_missing_data" for entry in manifest["plots"])


def test_incompatible_binary_hashes_do_not_pair(tmp_path: Path) -> None:
    rows = [
        _row(dpu_count=1, host_binary_hash="c" * 64, dpu_binary_hash="d" * 64),
        _row(dpu_count=2, host_binary_hash="e" * 64, dpu_binary_hash="f" * 64),
    ]
    output = generate_report(_write_run(tmp_path, rows), tmp_path / "report-root", timestamp="hashes")

    records = _csv_rows(output / "tables/m5_records.csv")
    assert all(row["physical_one_rank_valid"] == "True" for row in records)
    assert not _csv_rows(output / "tables/m5_strong_scaling.csv")


def test_incompatible_numeric_task_and_binary_identities_are_todo(tmp_path: Path) -> None:
    incompatible_task = _row(numeric_mode="int8", runtime_s=2.0)
    incompatible_task["task_hash"] = "task-b"
    incompatible_binary = _row(numeric_mode="int8", runtime_s=2.0, host_binary_hash="c" * 64)
    output = generate_report(
        _write_run(tmp_path, [_row(runtime_s=4.0), incompatible_task, incompatible_binary]),
        tmp_path / "report-root",
        timestamp="numeric-incompatible",
    )

    assert _csv_rows(output / "tables/m5_numeric_mode_ratios.csv") == []
    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    numeric_plot = next(plot for plot in manifest["plots"] if "numeric_mode_runtime_ratio" in plot["path"])
    assert numeric_plot["status"] == "todo_missing_data"


def test_missing_binary_hashes_are_not_physical_evidence(tmp_path: Path) -> None:
    row = _row()
    row.pop("host_binary_hash")
    row.pop("dpu_binary_hash")

    output = generate_report(_write_run(tmp_path, [row]), tmp_path / "report-root", timestamp="missing-hashes")

    record = _csv_rows(output / "tables/m5_records.csv")[0]
    assert record["physical_one_rank_valid"] == "False"
    assert record["runtime_s"] == ""
    assert _csv_rows(output / "tables/m5_strong_scaling.csv") == []


def test_run_transfer_provenance_ignores_conflicting_repeat_transfers(tmp_path: Path) -> None:
    row = _row()
    row["run_global_transfers"] = {"h2d_bytes": 300, "d2h_bytes": 30, "host_mediated_reduction_bytes": 3}
    row["run_metadata"] = {"transfers": {"h2d_bytes": 200, "d2h_bytes": 20, "host_mediated_reduction_bytes": 2}}
    row["transfers"] = {"h2d_bytes": 900, "d2h_bytes": 90, "host_mediated_reduction_bytes": 9}
    row["per_repeat_timing"] = {
        "total_time_s": 1.0,
        "transfers": {"h2d_bytes": 10, "d2h_bytes": 2, "host_mediated_reduction_bytes": 1},
    }

    output = generate_report(_write_run(tmp_path, [row]), tmp_path / "report-root", timestamp="transfer-provenance")
    record = _csv_rows(output / "tables/m5_records.csv")[0]
    assert record["h2d_bytes"] == "10.0"
    assert record["run_h2d_bytes_provenance"] == "300.0"


def test_quantization_accuracy_ignores_policy_reference_error(tmp_path: Path) -> None:
    output = generate_report(_write_run(tmp_path, [_row(numeric_mode="int8")]), tmp_path / "report-root", timestamp="quantization")

    accuracy = _csv_rows(output / "tables/m5_accuracy_statistics.csv")
    assert accuracy[0]["median"] == "0.125"
    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    plot = next(entry for entry in manifest["plots"] if "quantization_accuracy" in entry["path"])
    assert "policy-reference validation" in plot["caption"]


def test_quantization_accuracy_is_todo_without_canonical_error(tmp_path: Path) -> None:
    row = _row(numeric_mode="int8")
    row.pop("quantization_error_vs_float32")
    row.pop("full_precision_accuracy")
    output = generate_report(_write_run(tmp_path, [row]), tmp_path / "report-root", timestamp="quantization-missing")

    accuracy = _csv_rows(output / "tables/m5_accuracy_statistics.csv")
    assert accuracy[0]["median"] == ""
    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    plot = next(entry for entry in manifest["plots"] if "quantization_accuracy" in entry["path"])
    assert plot["status"] == "todo_missing_data"


def test_missing_data_has_todo_plots_and_respects_output_boundary(tmp_path: Path) -> None:
    run = _write_run(tmp_path, [])
    output_root = tmp_path / "comparison-root"
    output = generate_report(run, output_root, timestamp="missing")

    assert output == output_root / "runs/comparisons/upmem_m5/missing"
    assert output.is_relative_to(output_root)
    assert not output.is_relative_to(run)
    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "todo_missing_data"
    assert len(manifest["plots"]) == 9
    assert all(entry["status"] == "todo_missing_data" for entry in manifest["plots"])
    assert "TODO: no measured data" in (output / "m5_summary.md").read_text(encoding="utf-8")
    assert all((output / entry["path"]).is_file() for entry in manifest["plots"])
    assert not list(run.glob("*.png"))

    with pytest.raises(ReportError, match="inside the source evidence run"):
        generate_report(run, run, timestamp="rejected")


@pytest.mark.parametrize("missing_field", ["cpu_fallback_used", "simulator_kernel_executed", "fallback_used"])
def test_physical_admission_requires_each_canonical_fallback_field(tmp_path: Path, missing_field: str) -> None:
    row = _row()
    row.pop(missing_field)
    row["cpu_fallback"] = False
    row["simulator"] = False

    output = generate_report(_write_run(tmp_path, [row]), tmp_path / "report-root", timestamp=f"missing-{missing_field}")

    record = _csv_rows(output / "tables/m5_records.csv")[0]
    assert record["physical_one_rank_valid"] == "False"
    assert record["runtime_s"] == ""


def test_numeric_ratio_requires_canonical_quantization_evidence(tmp_path: Path) -> None:
    int8 = _row(numeric_mode="int8", runtime_s=2.0)
    int8.pop("numeric_transport")
    output = generate_report(
        _write_run(tmp_path, [_row(runtime_s=4.0), int8]),
        tmp_path / "report-root",
        timestamp="numeric-evidence-required",
    )

    assert _csv_rows(output / "tables/m5_numeric_mode_ratios.csv") == []
    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    plot = next(entry for entry in manifest["plots"] if "numeric_mode_runtime_ratio" in entry["path"])
    assert plot["status"] == "todo_missing_data"
    assert plot["caption"].startswith("TODO:")


def test_partition_ratio_requires_canonical_reduction_evidence(tmp_path: Path) -> None:
    contracted = _row(partition_mode="contracted", runtime_s=2.0)
    contracted.pop("collective_provider")
    contracted.pop("reconstruction_provider")
    output = generate_report(
        _write_run(tmp_path, [_row(runtime_s=4.0), contracted]),
        tmp_path / "report-root",
        timestamp="partition-evidence-required",
    )

    assert _csv_rows(output / "tables/m5_partition_ratios.csv") == []
    manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
    plot = next(entry for entry in manifest["plots"] if "partition_runtime_ratio" in entry["path"])
    assert plot["status"] == "todo_missing_data"
    assert plot["caption"].startswith("TODO:")


def test_one_rank_accepts_explicit_flag_or_observed_count_but_rejects_contradiction(tmp_path: Path) -> None:
    explicit = _row()
    explicit.pop("observed_rank_count")
    explicit.pop("rank_count")
    observed = _row(case_id="observed")
    observed.pop("one_rank")
    contradictory = _row(case_id="contradictory")
    contradictory["observed_rank_count"] = 2
    contradictory["rank_count"] = 2

    output = generate_report(
        _write_run(tmp_path, [explicit, observed, contradictory]),
        tmp_path / "report-root",
        timestamp="rank-contract",
    )

    records = {row["case_id"]: row for row in _csv_rows(output / "tables/m5_records.csv")}
    assert records["case-a"]["physical_one_rank_valid"] == "True"
    assert records["observed"]["physical_one_rank_valid"] == "True"
    assert records["contradictory"]["physical_one_rank_valid"] == "False"


def test_strong_scaling_omits_nonpositive_runtime(tmp_path: Path) -> None:
    output = generate_report(
        _write_run(tmp_path, [_row(runtime_s=0.0), _row(dpu_count=2, runtime_s=2.0)]),
        tmp_path / "report-root",
        timestamp="nonpositive-runtime",
    )

    runtime = _csv_rows(output / "tables/m5_runtime_statistics.csv")
    baseline = next(row for row in runtime if row["dpu_count"] == "1")
    assert baseline["median"] == ""
    assert _csv_rows(output / "tables/m5_strong_scaling.csv") == []


def test_m5_record_schema_is_stable_for_populated_and_empty_runs(tmp_path: Path) -> None:
    populated = generate_report(
        _write_run(tmp_path / "populated", [_row()]),
        tmp_path / "report-root",
        timestamp="schema-populated",
    )
    empty = generate_report(
        _write_run(tmp_path / "empty", []),
        tmp_path / "report-root",
        timestamp="schema-empty",
    )

    headers = []
    for output in (populated, empty):
        with (output / "tables/m5_records.csv").open(newline="", encoding="utf-8") as handle:
            headers.append(next(csv.reader(handle)))
    assert headers[0] == list(M5_RECORD_FIELDS)
    assert headers[1] == list(M5_RECORD_FIELDS)
    assert len(headers[1]) == len(set(headers[1]))
