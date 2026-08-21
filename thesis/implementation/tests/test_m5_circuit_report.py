from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from quantum_bench.bench.m5_circuit_report import (
    _faceted_plot,
    _plot_exact_identity,
    _record_row,
    _short_series_label,
    _timing_bar_label,
    _timing_breakdown_plot,
    _timing_plot_summary,
    generate_report,
    load_records,
)


def test_timing_bar_label_compacts_recorded_topology_syntax() -> None:
    assert _timing_bar_label(14, "2/8DPU + 1/1R") == "14q\n2/8 DPU"
    assert _timing_bar_label(20, "8DPU/1R") == "20q\n8/8 DPU"


def _row(
    *,
    engine: str,
    runtime: float,
    path: str = "opt_einsum_greedy",
    numeric: str = "float32",
    repeat: int = 0,
    scope: str = "whole_circuit_steady_state_v1",
    hashes: bool = True,
    dpu_count: int | None = None,
    timing_breakdown: dict[str, float] | None = None,
    status: str = "completed",
) -> dict[str, object]:
    is_upmem = "upmem" in engine or "dpu" in engine
    topology_id = f"upmem:{dpu_count or 'unspecified'}:1" if is_upmem else "cpu"
    row: dict[str, object] = {
        "case_id": "bv-8",
        "circuit_family": "BV",
        "qubits": 8,
        "engine_id": engine,
        "path_variant_id": path,
        "numeric_policy": numeric,
        "repeat_id": repeat,
        "timing_scope": scope,
        "status": status,
        "validation_status": "passed",
        "scientific_validation_status": "passed",
        "exact_once": True,
        "no_fallback_used": True,
        "executor_config_hash": "executor-config-a",
        "route_modules": {
            "tensor_network": {"implementation": "quantum_gate_tn_v1"},
            "planner": {"implementation": path},
            "numeric": {"implementation": numeric},
            "executor": {"implementation": engine},
            "topology": {"implementation": topology_id},
        },
        "timing_s": runtime,
        "actual_h2d_bytes": 80,
        "actual_d2h_bytes": 20,
        "actual_transfer_bytes": 100,
        "max_abs_error": 1e-6,
        "timing_breakdown": timing_breakdown
        or {"h2d_s": 0.1, "kernel_s": runtime / 2, "d2h_s": 0.1},
    }
    if "int8" in numeric.lower() or "quant" in numeric.lower():
        row.update(
            numeric_policy_id=numeric,
            packed_int8_transfer=True,
            numeric_transport="host_packed_int8_mram",
        )
    if is_upmem:
        row.update(
            target_observed="physical_hardware",
            hardware_allocation_verified=True,
            native_kernel_executed=True,
            hardware_kernel_executed=True,
            simulator=False,
            simulator_kernel_executed=False,
            cpu_fallback=False,
            cpu_fallback_used=False,
            release_succeeded=True,
            hardware_speedup_applicable=True,
            timing_is_bringup_only=False,
        )
        if dpu_count is not None:
            row["engine_metadata"] = {
                "active_dpu_ids": list(range(dpu_count)),
                "active_rank_indices": [0],
                "active_rank_count": 1,
            }
    if hashes:
        row.update(
            circuit_semantics_hash="circuit-bv-8",
            tensor_network_hash="network-bv-8",
            contraction_plan_hash=f"plan-{path}",
        )
    if dpu_count is not None:
        row["requested_dpu_count"] = dpu_count
        row["allocated_dpu_count"] = dpu_count
        row["rank_count"] = 1
    return row


def _csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _valid_dag_v2(row: dict[str, object], dag_hash: str) -> None:
    row.update(
        contraction_dag_schema_version="contraction_dag_v2",
        contraction_dag_hash=dag_hash,
        host_dag_node_completion_coverage=True,
        exact_once_scope="host_dag_node_completion_per_route",
    )
    if row.get("target_observed") == "physical_hardware":
        row["native_identity_verified"] = True
        row["physical_plan_consumed"] = True


def _assert_no_duplicate_plot_identities(report_dir: Path) -> None:
    manifest = json.loads((report_dir / "plot_manifest.json").read_text())
    x_axis = {
        "runtime_by_qubits": "qubits",
        "same_plan_cpu_upmem_speedup": "qubits",
        "upmem_strong_scaling": "active_dpu_count",
        "upmem_rank_scaling": "rank_count",
        "path_runtime_ratio": "qubits",
        "float32_int8_ratio": "qubits",
        "validation_accuracy": "qubits",
        "timing_breakdown": "qubits",
        "transfer_bytes": "qubits",
        "supported_boundary": "qubits",
    }
    derived_series = {
        "upmem_strong_scaling": lambda row: f"{row['path']} | {row['numeric_policy']}",
        "upmem_rank_scaling": lambda row: f"{row['path']} | {row['numeric_policy']}",
        "path_runtime_ratio": lambda row: (
            f"{row['engine']} | {row['numeric_policy']} | "
            f"{row['activity_label'] or row['topology']}"
        ),
        "float32_int8_ratio": lambda row: (
            f"{row['engine']} | {row['path']} | "
            f"{row['activity_label'] or row['topology']}"
        ),
    }
    for entry in manifest["plots"]:
        if entry["status"] != "generated_valid":
            continue
        csv_path = report_dir / entry["source_csv"]
        if not csv_path.exists():
            continue
        rows = _csv(csv_path)
        if not rows:
            continue
        if "series" not in rows[0]:
            continue
        x = x_axis.get(entry["name"])
        if x not in rows[0]:
            continue
        if entry["name"] == "timing_breakdown":
            summary = _timing_plot_summary(rows)
            identities = {
                (
                    row["engine"],
                    row["route_id"],
                    row["route_config_hash"],
                    row["executor_config_hash"],
                    row["timing_scope"],
                    row["provisioned_dpu_count"],
                    row["provisioned_rank_count"],
                    row["active_dpu_count"],
                    row["active_rank_count"],
                    row["path"],
                    row["numeric_policy"],
                    row["qubits"],
                    row["stage"],
                )
                for row in summary
            }
            assert len(identities) == len(summary)
            continue
        seen: set[tuple[str, str, tuple[object, ...], str]] = set()
        for row in rows:
            group = "_plot_series" if entry["name"] in derived_series else "series"
            effective_row = dict(row)
            if group == "_plot_series":
                effective_row[group] = derived_series[entry["name"]](row)
            key = (
                row.get("family", ""),
                row.get(x, ""),
                _plot_exact_identity(effective_row, group),
                row.get("stage", ""),
            )
            assert key not in seen
            seen.add(key)


def test_report_writes_expected_tables_plots_and_energy_todo(tmp_path: Path) -> None:
    rows = [
        _row(engine="cpu_numpy", runtime=10.0),
        _row(engine="upmem_m5", runtime=5.0),
        _row(engine="upmem_m5", runtime=2.5, dpu_count=1),
        _row(engine="upmem_m5", runtime=1.25, dpu_count=2),
        _row(engine="cpu_numpy", runtime=12.0, numeric="host_packed_int8"),
        _row(engine="quimb_tn", runtime=3.0),
    ]
    result = generate_report(rows, tmp_path / "report")
    assert result.output_dir == tmp_path / "report"
    for name in (
        "runtime_by_qubits.csv",
        "runtime_by_case_median.csv",
        "same_plan_cpu_upmem_speedup.csv",
        "upmem_strong_scaling.csv",
        "path_runtime_ratio.csv",
        "float32_int8_ratios.csv",
        "validation_accuracy.csv",
        "timing_breakdown.csv",
        "transfer_bytes.csv",
        "supported_boundary.csv",
        "energy.csv",
    ):
        assert (tmp_path / "report" / "tables" / name).is_file()
    assert len(_csv(tmp_path / "report" / "tables" / "runtime_by_qubits.csv")) == len(
        rows
    )
    assert (tmp_path / "report" / "plots" / "energy_todo.png").stat().st_size > 1000
    manifest = json.loads((tmp_path / "report" / "plot_manifest.json").read_text())
    energy = next(item for item in manifest["plots"] if item["name"] == "energy")
    assert energy["status"] == "generated_todo_missing_data"
    assert "unavailable" in energy["reason"]


def test_same_plan_pairing_requires_all_hashes_and_timing_scope(tmp_path: Path) -> None:
    cpu = _row(engine="cpu_numpy", runtime=10.0)
    upmem = _row(engine="upmem_m5", runtime=5.0)
    missing_hash = _row(engine="upmem_m5", runtime=5.0, hashes=False)
    mismatched_scope = _row(engine="upmem_m5", runtime=5.0, scope="kernel_only")
    generate_report([cpu, upmem, missing_hash, mismatched_scope], tmp_path / "report")
    pairs = _csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv")
    assert len(pairs) == 1
    assert float(pairs[0]["speedup_cpu_over_upmem"]) == pytest.approx(2.0)


def test_same_plan_pairing_requires_matching_dag_identity(tmp_path: Path) -> None:
    cpu = _row(engine="cpu_numpy", runtime=10.0)
    upmem = _row(engine="upmem_m5", runtime=5.0)
    _valid_dag_v2(cpu, "dag-a")
    _valid_dag_v2(upmem, "dag-a")
    generate_report([cpu, upmem], tmp_path / "matched")
    assert len(
        _csv(tmp_path / "matched" / "tables" / "same_plan_cpu_upmem_speedup.csv")
    ) == 1

    mismatched = dict(upmem, contraction_dag_hash="dag-b")
    generate_report([cpu, mismatched], tmp_path / "mismatched")
    assert _csv(
        tmp_path / "mismatched" / "tables" / "same_plan_cpu_upmem_speedup.csv"
    ) == []

    legacy = _row(engine="cpu_numpy", runtime=10.0)
    generate_report([legacy, upmem], tmp_path / "mixed")
    assert _csv(tmp_path / "mixed" / "tables" / "same_plan_cpu_upmem_speedup.csv") == []

    broken_v2 = dict(cpu, contraction_dag_hash="")
    generate_report([broken_v2, upmem], tmp_path / "broken-v2")
    assert _csv(
        tmp_path / "broken-v2" / "tables" / "same_plan_cpu_upmem_speedup.csv"
    ) == []


def test_same_plan_pairing_uses_claim_policy_for_speedup_admission(
    tmp_path: Path,
) -> None:
    cpu = _row(engine="cpu_numpy", runtime=10.0)
    simulator = _row(engine="upmem_m5", runtime=5.0)
    simulator["simulator_kernel_executed"] = True

    generate_report([cpu, simulator], tmp_path / "report")

    assert _csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv") == []


def test_numeric_pairing_requires_one_complete_matching_dag_v2_identity(
    tmp_path: Path,
) -> None:
    float32 = _row(engine="upmem_m5", runtime=10.0, numeric="float32")
    int8 = _row(engine="upmem_m5", runtime=5.0, numeric="host_packed_int8")
    for row in (float32, int8):
        _valid_dag_v2(row, "dag-a")
    generate_report([float32, int8], tmp_path / "matched")
    assert len(_csv(tmp_path / "matched" / "tables" / "float32_int8_ratios.csv")) == 1

    broken = dict(int8, contraction_dag_hash="")
    generate_report([float32, broken], tmp_path / "broken")
    assert _csv(tmp_path / "broken" / "tables" / "float32_int8_ratios.csv") == []
    assert len(_csv(tmp_path / "broken" / "tables" / "runtime_by_case_median.csv")) == 1


def test_path_pairing_requires_distinct_dag_v2_hashes(tmp_path: Path) -> None:
    greedy = _row(engine="upmem_m5", runtime=10.0, path="opt_einsum_greedy")
    cotengra = _row(engine="upmem_m5", runtime=5.0, path="cotengra_flops_seed0")
    for row, dag_hash in ((greedy, "dag-greedy"), (cotengra, "dag-cotengra")):
        _valid_dag_v2(row, dag_hash)
    generate_report([greedy, cotengra], tmp_path / "distinct")
    assert len(_csv(tmp_path / "distinct" / "tables" / "path_runtime_ratio.csv")) == 1

    same_dag = dict(cotengra, contraction_dag_hash="dag-greedy")
    generate_report([greedy, same_dag], tmp_path / "same")
    assert _csv(tmp_path / "same" / "tables" / "path_runtime_ratio.csv") == []


def test_same_plan_pairing_keeps_each_upmem_topology(tmp_path: Path) -> None:
    cpu = _row(engine="cpu_numpy", runtime=10.0)
    upmem_rows = [
        _row(engine="upmem_m5", runtime=8.0, dpu_count=1),
        _row(engine="upmem_m5", runtime=4.0, dpu_count=2),
        _row(engine="upmem_m5", runtime=1.0, dpu_count=8),
    ]
    generate_report([cpu, *upmem_rows], tmp_path / "report")
    pairs = _csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv")
    assert len(pairs) == 3
    assert [
        (row["local_dpu_count"], row["rank_count"], row["total_dpu_count"])
        for row in pairs
    ] == [
        ("1", "1", "1"),
        ("2", "1", "2"),
        ("8", "1", "8"),
    ]
    assert [float(row["speedup_cpu_over_upmem"]) for row in pairs] == pytest.approx(
        [1.25, 2.5, 10.0]
    )
    manifest = json.loads((tmp_path / "report" / "plot_manifest.json").read_text())
    speedup_plot = next(
        item
        for item in manifest["plots"]
        if item["name"] == "same_plan_cpu_upmem_speedup"
    )
    assert "Measured same-plan" in speedup_plot["title"]
    assert ">1 favors UPMEM" in speedup_plot["title"]


def test_same_plan_pairing_rejects_ambiguous_cpu_executor_baselines(
    tmp_path: Path,
) -> None:
    cpu_a = _row(engine="cpu_numpy", runtime=10.0)
    cpu_b = _row(engine="cpu_numpy", runtime=8.0)
    cpu_b["route_id"] = "alternate-cpu"
    cpu_b["route_config_hash"] = "alternate-route-config"
    cpu_b["executor_config_hash"] = "alternate-executor-config"
    upmem = _row(engine="upmem_m5", runtime=5.0)

    generate_report([cpu_a, cpu_b, upmem], tmp_path / "report")

    assert (
        _csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv") == []
    )
    rejections = _csv(tmp_path / "report" / "tables" / "claim_rejections.csv")
    assert any("ambiguous CPU baselines" in row["reasons"] for row in rejections)


def test_same_plan_pairing_deduplicates_identical_rows(tmp_path: Path) -> None:
    cpu = _row(engine="cpu_numpy", runtime=10.0)
    upmem = _row(engine="upmem_m5", runtime=5.0)

    generate_report([cpu, dict(cpu), upmem, dict(upmem)], tmp_path / "report")

    pairs = _csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv")
    assert len(pairs) == 1
    assert float(pairs[0]["speedup_cpu_over_upmem"]) == pytest.approx(2.0)
    assert _csv(tmp_path / "report" / "tables" / "claim_rejections.csv") == []


@pytest.mark.parametrize("conflicting_engine", ["cpu_numpy", "upmem_m5"])
def test_same_plan_pairing_rejects_conflicting_duplicate_rows(
    tmp_path: Path, conflicting_engine: str
) -> None:
    cpu = _row(engine="cpu_numpy", runtime=10.0)
    upmem = _row(engine="upmem_m5", runtime=5.0)
    conflicting = dict(cpu if conflicting_engine == "cpu_numpy" else upmem)
    conflicting["timing_s"] = 7.0

    generate_report([cpu, upmem, conflicting], tmp_path / "report")

    assert _csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv") == []
    rejections = _csv(tmp_path / "report" / "tables" / "claim_rejections.csv")
    expected_class = "CPU" if conflicting_engine == "cpu_numpy" else "UPMEM"
    assert any(
        f"conflicting duplicate {expected_class} rows" in row["reasons"]
        for row in rejections
    )


def test_bringup_and_non_applicable_upmem_rows_stay_out_of_performance_ratios(
    tmp_path: Path,
) -> None:
    cpu = _row(engine="cpu_numpy", runtime=10.0)
    bringup = _row(engine="upmem_m5", runtime=2.0, dpu_count=1)
    bringup["timing_is_bringup_only"] = True
    non_applicable = _row(engine="upmem_m5", runtime=3.0, dpu_count=2)
    non_applicable["hardware_speedup_applicable"] = False
    valid = _row(engine="upmem_m5", runtime=5.0, dpu_count=4)
    generate_report([cpu, bringup, non_applicable, valid], tmp_path / "report")
    pairs = _csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv")
    assert len(pairs) == 1
    assert pairs[0]["local_dpu_count"] == "4"
    scaling = _csv(tmp_path / "report" / "tables" / "upmem_strong_scaling.csv")
    assert len(scaling) == 1
    assert scaling[0]["fully_active_topology"] == "True"
    assert scaling[0]["speedup"] == ""
    raw = _csv(tmp_path / "report" / "tables" / "runtime_by_qubits.csv")
    assert len(raw) == 4
    assert sum(row["scientific_admitted"] == "True" for row in raw) == 4


def test_speedup_todo_reason_reports_matched_but_ineligible_rows(
    tmp_path: Path,
) -> None:
    cpu = _row(engine="cpu_numpy", runtime=10.0)
    bringup = _row(engine="upmem_m5", runtime=2.0, dpu_count=1)
    bringup["timing_is_bringup_only"] = True
    non_applicable = _row(engine="upmem_m5", runtime=3.0, dpu_count=2)
    non_applicable["hardware_speedup_applicable"] = False
    generate_report([cpu, bringup, non_applicable], tmp_path / "report")
    manifest = json.loads((tmp_path / "report" / "plot_manifest.json").read_text())
    speedup_plot = next(
        item
        for item in manifest["plots"]
        if item["name"] == "same_plan_cpu_upmem_speedup"
    )
    assert speedup_plot["status"] == "generated_todo_missing_data"
    assert (
        speedup_plot["reason"]
        == "matching CPU/UPMEM rows exist, but no CPU/UPMEM pairs are "
        "performance-eligible/repeated"
    )


def test_speedup_todo_reason_preserves_identity_mismatch_classification(
    tmp_path: Path,
) -> None:
    cpu = _row(engine="cpu_numpy", runtime=10.0)
    upmem = _row(engine="upmem_m5", runtime=5.0, hashes=False)
    generate_report([cpu, upmem], tmp_path / "report")
    manifest = json.loads((tmp_path / "report" / "plot_manifest.json").read_text())
    speedup_plot = next(
        item
        for item in manifest["plots"]
        if item["name"] == "same_plan_cpu_upmem_speedup"
    )
    assert (
        speedup_plot["reason"]
        == "no matching CPU/UPMEM rows with all hashes and timing_scope"
    )


def test_ratios_and_scaling_use_the_declared_baselines(tmp_path: Path) -> None:
    rows = [
        _row(
            engine="upmem_m5",
            runtime=8.0,
            path="opt_einsum_greedy",
            numeric="float32",
            dpu_count=1,
        ),
        _row(
            engine="upmem_m5",
            runtime=4.0,
            path="opt_einsum_greedy",
            numeric="float32",
            dpu_count=2,
        ),
        _row(
            engine="upmem_m5",
            runtime=2.0,
            path="cotengra_flops_seed0",
            numeric="float32",
            dpu_count=2,
        ),
        _row(
            engine="upmem_m5",
            runtime=8.0,
            path="opt_einsum_greedy",
            numeric="host_packed_int8",
            dpu_count=2,
        ),
    ]
    generate_report(rows, tmp_path / "report")
    scaling = _csv(tmp_path / "report" / "tables" / "upmem_strong_scaling.csv")
    assert len(scaling) == 4
    baseline = next(row for row in scaling if row["active_dpu_count"] == "1")
    assert float(baseline["speedup"]) == pytest.approx(1.0)
    assert float(baseline["efficiency"]) == pytest.approx(1.0)
    scaled = next(
        row
        for row in scaling
        if row["path"] == "opt_einsum_greedy"
        and row["numeric_policy"] == "float32"
        and row["active_dpu_count"] == "2"
    )
    assert float(scaled["speedup"]) == pytest.approx(2.0)
    assert float(scaled["efficiency"]) == pytest.approx(1.0)
    assert scaled["series"] == (
        "opt_einsum_greedy | float32 | 2DPU x 1 rank(s) = 2 total"
    )
    numeric = _csv(tmp_path / "report" / "tables" / "float32_int8_ratios.csv")
    assert len(numeric) == 1
    assert float(numeric[0]["runtime_ratio_float32_over_int8"]) == pytest.approx(0.5)


def test_runtime_curve_uses_per_case_medians_without_duplicating_raw_rows(
    tmp_path: Path,
) -> None:
    rows = [
        _row(engine="cpu_numpy", runtime=10.0, repeat=0),
        _row(engine="cpu_numpy", runtime=14.0, repeat=1),
    ]
    generate_report(rows, tmp_path / "report")
    raw = _csv(tmp_path / "report" / "tables" / "runtime_by_qubits.csv")
    summary = _csv(tmp_path / "report" / "tables" / "runtime_by_case_median.csv")
    assert len(raw) == 2
    assert len(summary) == 1
    assert float(summary[0]["median_runtime_s"]) == pytest.approx(12.0)
    assert summary[0]["repeat_count"] == "2"
    manifest = json.loads((tmp_path / "report" / "plot_manifest.json").read_text())
    runtime_plot = next(
        item for item in manifest["plots"] if item["name"] == "runtime_by_qubits"
    )
    assert runtime_plot["source_csv"] == "tables/runtime_by_case_median.csv"


def test_faceted_plot_is_family_local_and_layout_is_warning_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    families = ["BV", "GHZ", "QFT", "Random", "XOR", "WState"]
    rows = [
        {
            "family": family,
            "qubits": qubits,
            "series": series,
            "runtime_s": runtime,
        }
        for family in families
        for series, runtime in (
            ("cpu_numpy | float32", 1.0),
            ("upmem_m5 | float32", 2.0),
        )
        for qubits in (4, 8)
    ]
    saved: list[object] = []
    monkeypatch.setattr(
        plt.Figure,
        "savefig",
        lambda figure, *args, **kwargs: saved.append(figure),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detailed_title = "Detailed scientific runtime title retained in the figure"
        assert _faceted_plot(
            tmp_path / "runtime.png",
            detailed_title,
            rows,
            "qubits",
            "runtime_s",
            "series",
            log_y=True,
            panel_title="median runtime (s)",
        )
    assert not [warning for warning in caught if "tight_layout" in str(warning.message)]
    figure = saved[0]
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == detailed_title
    assert any(legend is not None for legend in figure.legends)
    visible_axes = [axis for axis in figure.axes if axis.get_visible()]
    assert len(visible_axes) == 6
    assert all(axis.get_yscale() == "log" for axis in visible_axes)
    assert all(len(axis.lines) == 2 for axis in visible_axes)
    assert {axis.get_title().split(" | ", 1)[0] for axis in visible_axes} == set(
        families
    )
    assert all(
        "|" not in line.get_label() for axis in visible_axes for line in axis.lines
    )
    plt.close(figure)


def test_short_series_labels_are_stable_and_compact() -> None:
    assert (
        _short_series_label(
            "BV | opt_einsum_greedy | float32 | 2DPU x 1 rank(s) = 2 total",
            "BV",
        )
        == "greedy / Float32 / 2DPU/1R"
    )
    assert (
        _short_series_label(
            "BV | numpy_cpu | opt_einsum_greedy | f32_real | "
            "1 local / None ranks / 1 total",
            "BV",
        )
        == "CPU / greedy / Float32"
    )
    assert (
        _short_series_label(
            "BV | upmem_physical_1rank_8dpu | opt_einsum_greedy | f32_real | "
            "8DPU x 1 rank(s) = 8 total",
            "BV",
        )
        == "UPMEM physical / greedy / Float32 / 8DPU/1R"
    )
    assert (
        _short_series_label(
            "BV | numpy_cpu | f32_real | 1DPU x ? rank(s) = 1 total",
            "BV",
        )
        == "CPU / Float32"
    )
    assert (
        _short_series_label(
            "BV | upmem_physical_1rank_8dpu | host_packed_int8 | "
            "8DPU x 1 rank(s) = 8 total",
            "BV",
        )
        == "UPMEM physical / host-packed Int8 / 8DPU/1R"
    )
    assert (
        _short_series_label(
            "BV | numpy_cpu | opt_einsum_greedy",
            "BV",
        )
        == "CPU / greedy"
    )
    assert (
        _short_series_label(
            "BV | upmem_physical_1rank_8dpu | cotengra_flops_seed0 | "
            "8DPU x 1 rank(s) = 8 total",
            "BV",
        )
        == "UPMEM physical / cotengra FLOPs / 8DPU/1R"
    )


def test_human_figure_labels_hide_raw_route_and_engine_ids() -> None:
    label = _short_series_label(
        "case-8 | upmem_physical_1rank_8dpu | opt_einsum_greedy | "
        "host_packed_int8 | whole_route_including_session_lifecycle | 8DPU/1R"
    )
    assert "UPMEM physical" in label
    assert "greedy" in label
    assert "host-packed Int8" in label
    for raw_id in (
        "upmem_physical_1rank_8dpu",
        "opt_einsum_greedy",
        "host_packed_int8",
        "whole_route_including_session_lifecycle",
    ):
        assert raw_id not in label


def test_partial_activity_is_explicit_in_record_and_legend_label() -> None:
    row = _row(engine="upmem_m5", runtime=1.0, dpu_count=32)
    row["engine_metadata"] = {
        "active_dpu_ids": list(range(16)),
        "active_rank_indices": [0],
        "active_rank_count": 1,
    }
    normalized = _record_row(row)
    assert normalized["provisioned_dpu_count"] == 32
    assert normalized["provisioned_rank_count"] == 1
    assert normalized["active_dpu_count"] == 16
    assert normalized["active_rank_count"] == 1
    assert normalized["activity_label"] == "16/32DPU + 1/1R"
    assert "16/32DPU + 1/1R" in normalized["series"]
    assert _short_series_label(normalized["series"], "BV").endswith("16/32DPU + 1/1R")


def test_plot_uses_compact_semantic_legend_and_separate_topology_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    saved: list[object] = []
    monkeypatch.setattr(
        plt.Figure,
        "savefig",
        lambda figure, *args, **kwargs: saved.append(figure),
    )
    rows = [
        {
            "family": "BV",
            "qubits": qubits,
            "runtime_s": float(qubits),
            "engine": "upmem_m5",
            "semantic": "UPMEM physical | opt_einsum_greedy | float32_real",
            "series": (
                f"BV | upmem_m5 | opt_einsum_greedy | f32_real | {active}/8DPU + 1/1R"
            ),
            "provisioned_dpu_count": 8,
            "provisioned_rank_count": 1,
            "active_dpu_count": active,
            "active_rank_count": 1,
        }
        for qubits, active in zip((8, 12, 16, 20), (1, 2, 4, 8))
    ]
    assert _faceted_plot(
        tmp_path / "activity-range.png",
        "Activity range regression",
        rows,
        "qubits",
        "runtime_s",
        "series",
        semantic_group="semantic",
    )
    figure = saved[0]
    axis = next(axis for axis in figure.axes if axis.get_visible())
    data_lines = [line for line in axis.lines if not line.get_label().startswith("_")]
    assert data_lines == []
    assert len(axis.collections) == 4
    assert [text.get_text() for text in figure.legends[0].texts] == [
        "UPMEM / greedy / Float32"
    ]
    assert {text.get_text() for text in figure.legends[1].texts} == {
        "1/8 DPU, 1/1 R",
        "2/8 DPU, 1/1 R",
        "4/8 DPU, 1/1 R",
        "8/8 DPU, 1/1 R",
    }
    plt.close(figure)


def test_plot_dodges_distinct_exact_upmem_configurations_at_one_x(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    saved: list[object] = []
    monkeypatch.setattr(
        plt.Figure,
        "savefig",
        lambda figure, *args, **kwargs: saved.append(figure),
    )
    rows = [
        {
            "family": "BV",
            "qubits": 8,
            "runtime_s": float(index + 1),
            "series": "BV | upmem_m5 | opt_einsum_greedy | float32_real",
            "_visual_series": "UPMEM physical | opt_einsum_greedy | float32_real",
            "engine": "upmem_m5",
            "route_id": f"route-{index}",
            "route_config_hash": f"config-{index}",
            "executor_config_hash": f"executor-{index}",
            "timing_scope": "whole_route",
            "admission_identity": f"admission-{index}",
            "provisioned_dpu_count": 8,
            "provisioned_rank_count": 1,
            "active_dpu_count": 4,
            "active_rank_count": 1,
        }
        for index in range(2)
    ]
    assert _faceted_plot(
        tmp_path / "dodge.png",
        "Dodge regression",
        rows,
        "qubits",
        "runtime_s",
        "series",
        semantic_group="_visual_series",
    )
    figure = saved[0]
    axis = next(axis for axis in figure.axes if axis.get_visible())
    offsets = [collection.get_offsets()[0, 0] for collection in axis.collections]
    assert len(offsets) == 2
    assert len(set(offsets)) == 2
    assert [text.get_text() for text in figure.legends[0].texts] == [
        "UPMEM / greedy / Float32"
    ]
    plt.close(figure)


def test_single_qubit_topology_matrix_uses_todo_general_plots(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for dpu_count in (1, 2, 4, 8, 16, 32, 64, 128):
        for numeric in ("float32", "host_packed_int8"):
            rows.append(
                _row(
                    engine="upmem_m5",
                    runtime=float(dpu_count),
                    numeric=numeric,
                    dpu_count=dpu_count,
                )
            )

    output = tmp_path / "scaling-report"
    generate_report(rows, output)
    manifest = json.loads((output / "plot_manifest.json").read_text())
    entries = {item["name"]: item for item in manifest["plots"]}
    reason = (
        "selected evidence has one qubit size and multiple topologies; use the "
        "dedicated strong-scaling figure and source CSV"
    )
    for name in (
        "runtime_by_qubits",
        "transfer_bytes",
        "float32_int8_ratio",
    ):
        assert entries[name]["status"] == "generated_todo_missing_data"
        assert entries[name]["reason"] == reason
    assert entries["timing_breakdown"]["status"] == "generated_valid"
    assert entries["upmem_strong_scaling"]["status"] == "generated_valid"
    expected_pngs = {
        "runtime_by_qubits.png",
        "same_plan_cpu_upmem_speedup.png",
        "upmem_strong_scaling.png",
        "upmem_rank_scaling.png",
        "path_runtime_ratio.png",
        "float32_int8_ratio.png",
        "validation_accuracy.png",
        "timing_breakdown.png",
        "transfer_bytes.png",
        "supported_boundary.png",
        "energy_todo.png",
    }
    assert expected_pngs <= {path.name for path in (output / "plots").glob("*.png")}


def test_supported_boundary_preserves_cpu_and_upmem_engine_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    saved: list[object] = []
    monkeypatch.setattr(
        plt.Figure,
        "savefig",
        lambda figure, *args, **kwargs: saved.append(figure),
    )
    generate_report(
        [
            _row(engine="cpu_numpy", runtime=1.0),
            _row(
                engine="upmem_physical_1rank_64dpu",
                runtime=2.0,
                dpu_count=64,
            ),
        ],
        tmp_path / "report",
    )
    boundary = next(
        figure
        for figure in saved
        if figure._suptitle is not None
        and figure._suptitle.get_text() == "M5.5 supported boundary"
    )
    labels = {text.get_text() for legend in boundary.legends for text in legend.texts}
    assert "CPU" in labels
    assert "UPMEM physical / 64DPU/1R" in labels
    assert "unknown" not in labels
    plt.close(boundary)


def test_path_and_numeric_pairing_separate_active_topologies(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for active_count, suffix in ((16, "partial"), (32, "full")):
        metadata = {
            "active_dpu_ids": list(range(active_count)),
            "active_rank_indices": [0],
            "active_rank_count": 1,
        }
        path_a = _row(
            engine="upmem_m5", runtime=8.0, path="opt_einsum_greedy", dpu_count=32
        )
        path_b = _row(
            engine="upmem_m5", runtime=4.0, path="cotengra_flops_seed0", dpu_count=32
        )
        float32 = _row(engine="upmem_m5", runtime=8.0, numeric="float32", dpu_count=32)
        int8 = _row(
            engine="upmem_m5",
            runtime=4.0,
            numeric="host_packed_int8",
            dpu_count=32,
        )
        for row in (path_a, path_b, float32, int8):
            row["engine_metadata"] = metadata
            row["case_id"] = f"bv-8-{suffix}"
            row["circuit_semantics_hash"] = "circuit-shared"
            row["tensor_network_hash"] = "network-shared"
            rows.append(row)

    generate_report(rows, tmp_path / "report")
    path_rows = _csv(tmp_path / "report" / "tables" / "path_runtime_ratio.csv")
    numeric_rows = _csv(tmp_path / "report" / "tables" / "float32_int8_ratios.csv")
    assert len(path_rows) == 2
    assert len(numeric_rows) == 2
    assert {row["active_dpu_count"] for row in path_rows} == {"16", "32"}
    assert {row["activity_label"] for row in path_rows} == {
        "16/32DPU + 1/1R",
        "32DPU/1R",
    }
    assert {row["active_dpu_count"] for row in numeric_rows} == {"16", "32"}


def test_ratio_facets_use_independent_engine_numeric_path_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    saved: dict[str, object] = {}
    monkeypatch.setattr(
        plt.Figure,
        "savefig",
        lambda figure, filename, *args, **kwargs: saved.__setitem__(
            Path(filename).name, figure
        ),
    )
    rows: list[dict[str, object]] = []
    for qubits in (8, 12):
        for engine, dpu_count in (("cpu_numpy", None), ("upmem_m5", 1)):
            for path, numeric in (
                ("opt_einsum_greedy", "float32"),
                ("cotengra_flops_seed0", "host_packed_int8"),
            ):
                row = _row(
                    engine=engine,
                    runtime=float(qubits),
                    path=path,
                    numeric=numeric,
                    dpu_count=dpu_count,
                )
                row.update(
                    family="BV",
                    case_id=f"bv-{qubits}",
                    qubits=qubits,
                    engine_class="cpu" if engine == "cpu_numpy" else "upmem",
                    _plot_series=(
                        f"{'CPU' if engine == 'cpu_numpy' else 'UPMEM physical'} | "
                        f"{path} | {numeric}"
                    ),
                    route_id=f"{engine}-{path}-{numeric}",
                    route_config_hash=f"route-{engine}-{path}-{numeric}",
                    circuit_semantics_hash=f"circuit-bv-{qubits}",
                    tensor_network_hash=f"network-bv-{qubits}",
                    contraction_plan_hash=f"plan-{path}",
                    active_dpu_count=dpu_count,
                    active_rank_count=1 if dpu_count is not None else None,
                    provisioned_dpu_count=dpu_count,
                    provisioned_rank_count=1 if dpu_count is not None else None,
                )
                rows.append(row)

    assert _faceted_plot(
        tmp_path / "ratio-facets.png",
        "Ratio facets",
        rows,
        "qubits",
        "timing_s",
        "_plot_series",
        semantic_group="_plot_series",
    )

    figure = next(iter(saved.values()))
    axes = [axis for axis in figure.axes if axis.get_visible()]
    assert len(axes) == 1
    data_lines = axes[0].lines
    assert len(data_lines) == 2
    assert all(list(line.get_xdata()) == [8, 12] for line in data_lines)
    assert len(axes[0].collections) == 2
    plt.close(figure)


def test_ratio_tables_aggregate_only_matched_repetitions_with_quartiles(
    tmp_path: Path,
) -> None:
    rows = [
        _row(engine="cpu_numpy", runtime=10.0, repeat=0),
        _row(engine="upmem_m5", runtime=5.0, repeat=0, dpu_count=1),
        _row(engine="cpu_numpy", runtime=20.0, repeat=1),
        _row(engine="upmem_m5", runtime=8.0, repeat=1, dpu_count=1),
        _row(
            engine="upmem_m5",
            runtime=10.0,
            path="cotengra_flops_seed0",
            repeat=0,
            dpu_count=1,
        ),
        _row(
            engine="upmem_m5",
            runtime=4.0,
            path="cotengra_flops_seed0",
            repeat=1,
            dpu_count=1,
        ),
        _row(
            engine="upmem_m5",
            runtime=10.0,
            numeric="host_packed_int8",
            repeat=0,
            dpu_count=1,
        ),
        _row(
            engine="upmem_m5",
            runtime=4.0,
            numeric="host_packed_int8",
            repeat=1,
            dpu_count=1,
        ),
        _row(engine="upmem_m5", runtime=10.0, repeat=0, dpu_count=2),
        _row(engine="upmem_m5", runtime=5.0, repeat=1, dpu_count=2),
    ]
    generate_report(rows, tmp_path / "report")

    speedup = _csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv")
    assert len(speedup) == 2
    one_dpu = next(row for row in speedup if row["total_dpu_count"] == "1")
    assert one_dpu["repeat_count"] == "2"
    assert "repeat_id" not in one_dpu
    assert float(one_dpu["speedup_cpu_over_upmem"]) == pytest.approx(2.25)
    assert float(one_dpu["speedup_cpu_over_upmem_q1"]) < 2.25
    assert float(one_dpu["speedup_cpu_over_upmem_q3"]) > 2.25

    path = _csv(tmp_path / "report" / "tables" / "path_runtime_ratio.csv")
    assert len(path) == 1
    assert path[0]["path_a"] == "opt_einsum_greedy"
    assert path[0]["path_b"] == "cotengra_flops_seed0"
    assert path[0]["repeat_count"] == "2"
    assert float(path[0]["runtime_ratio_a_over_b"]) == pytest.approx(1.25)

    numeric = _csv(tmp_path / "report" / "tables" / "float32_int8_ratios.csv")
    assert len(numeric) == 1
    assert numeric[0]["repeat_count"] == "2"
    assert float(numeric[0]["runtime_ratio_float32_over_int8"]) == pytest.approx(1.25)

    scaling = _csv(tmp_path / "report" / "tables" / "upmem_strong_scaling.csv")
    assert len(scaling) == 4
    scaled = next(
        row
        for row in scaling
        if row["active_dpu_count"] == "2"
        and row["path"] == "opt_einsum_greedy"
        and row["numeric_policy"] == "float32"
    )
    assert scaled["repeat_count"] == "2"
    assert float(scaled["speedup"]) == pytest.approx(1.05)


def test_report_preserves_study_timing_field_names(tmp_path: Path) -> None:
    row = _row(engine="upmem_m5", runtime=2.0, dpu_count=1)
    row.update(
        h2d_time_s=0.1,
        kernel_time_s=0.2,
        d2h_time_s=0.3,
        host_quantization_time_s=0.4,
        host_dequantization_time_s=0.5,
        graph_execution_s=0.6,
        session_open_s=0.7,
        session_close_s=0.8,
        timing_breakdown={
            "h2d_time_s": 0.1,
            "kernel_time_s": 0.2,
            "d2h_time_s": 0.3,
            "host_quantization_time_s": 0.4,
            "host_dequantization_time_s": 0.5,
            "graph_execution_s": 0.6,
            "session_open_s": 0.7,
            "session_close_s": 0.8,
        },
    )
    generate_report([row], tmp_path / "report")
    timing = _csv(tmp_path / "report" / "tables" / "timing_breakdown.csv")
    by_stage = {item["stage"]: float(item["time_s"]) for item in timing}
    assert by_stage == {
        "h2d": pytest.approx(0.1),
        "kernel": pytest.approx(0.2),
        "d2h": pytest.approx(0.3),
        "host_quantization": pytest.approx(0.4),
        "host_dequantization": pytest.approx(0.5),
        "session_open": pytest.approx(0.7),
        "session_close": pytest.approx(0.8),
    }


def test_transfer_table_reads_physical_study_transfer_contract(tmp_path: Path) -> None:
    row = _row(engine="upmem_m5", runtime=2.0, dpu_count=1)
    for key in ("actual_h2d_bytes", "actual_d2h_bytes", "actual_transfer_bytes"):
        row.pop(key)
    row["transfer"] = {
        "application_visible_h2d_bytes": 80,
        "application_visible_d2h_bytes": 20,
        "application_visible_transfer_bytes": 100,
    }
    generate_report([row], tmp_path / "report")
    transfers = _csv(tmp_path / "report" / "tables" / "transfer_bytes.csv")
    assert len(transfers) == 1
    assert float(transfers[0]["h2d_bytes"]) == 80
    assert float(transfers[0]["d2h_bytes"]) == 20
    assert float(transfers[0]["transfer_bytes"]) == 100
    assert transfers[0]["invariant_passed"] == "True"

    row["transfer_accounting_verified"] = False
    generate_report([row], tmp_path / "unverified-report")
    unverified = _csv(tmp_path / "unverified-report" / "tables" / "transfer_bytes.csv")
    assert unverified[0]["invariant_passed"] == "False"


def test_timing_transfer_validation_aggregate_three_repeats_with_medians(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            engine="upmem_m5",
            runtime=1.0,
            repeat=0,
            dpu_count=1,
            timing_breakdown={"h2d_s": 1.0, "d2h_s": 2.0},
        ),
        _row(
            engine="upmem_m5",
            runtime=1.0,
            repeat=1,
            dpu_count=1,
            timing_breakdown={"h2d_s": 2.0, "d2h_s": 4.0},
        ),
        _row(
            engine="upmem_m5",
            runtime=1.0,
            repeat=2,
            dpu_count=1,
            timing_breakdown={"h2d_s": 3.0, "d2h_s": 6.0},
        ),
    ]
    for index, row in enumerate(rows):
        row.pop("actual_h2d_bytes", None)
        row.pop("actual_d2h_bytes", None)
        row.pop("actual_transfer_bytes", None)
        row["max_abs_error"] = [0.9, 1.1, 1.0][index]
        row["actual_h2d_bytes"] = [80, 100, 90][index]
        row["actual_d2h_bytes"] = [20, 20, 10][index]
        row["actual_transfer_bytes"] = [100, 120, 100][index]
    generate_report(rows, tmp_path / "report")
    validation = _csv(tmp_path / "report" / "tables" / "validation_accuracy.csv")
    assert len(validation) == 1
    assert validation[0]["repeat_count"] == "3"
    assert float(validation[0]["max_abs_error"]) == pytest.approx(1.0)
    timing = _csv(tmp_path / "report" / "tables" / "timing_breakdown.csv")
    assert len(timing) == 2
    assert sorted({row["repeat_count"] for row in timing}) == ["3"]
    assert sorted({row["stage"] for row in timing}) == ["d2h", "h2d"]
    transfer = _csv(tmp_path / "report" / "tables" / "transfer_bytes.csv")
    assert len(transfer) == 1
    assert transfer[0]["repeat_count"] == "3"
    assert float(transfer[0]["transfer_bytes"]) == pytest.approx(100.0)
    assert transfer[0]["raw_invariants_all_passed"] == "True"
    assert transfer[0]["invariant_passed"] == "True"
    assert transfer[0]["aggregate_component_medians_additive"] == "False"


def test_planner_and_numeric_variants_stay_distinct_in_validation_and_transfer_series(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            engine="upmem_m5",
            runtime=1.0,
            path="opt_einsum_greedy",
            numeric="float32",
            dpu_count=1,
        ),
        _row(
            engine="upmem_m5",
            runtime=1.0,
            path="cotengra_flops_seed0",
            numeric="float32",
            dpu_count=1,
        ),
        _row(
            engine="upmem_m5",
            runtime=1.0,
            path="opt_einsum_greedy",
            numeric="host_packed_int8",
            dpu_count=1,
        ),
    ]
    for row in rows:
        row["timing_breakdown"] = {"h2d_s": 1.0, "d2h_s": 2.0}
    generate_report(rows, tmp_path / "report")
    validation = _csv(tmp_path / "report" / "tables" / "validation_accuracy.csv")
    assert len({row["series"] for row in validation}) == 3
    transfer = _csv(tmp_path / "report" / "tables" / "transfer_bytes.csv")
    assert len({row["series"] for row in transfer}) == 3
    timing = _csv(tmp_path / "report" / "tables" / "timing_breakdown.csv")
    assert len({row["series"] for row in timing}) == 6


def test_timing_breakdown_uses_only_leaf_stages_without_totals(
    tmp_path: Path,
) -> None:
    row = _row(engine="upmem_m5", runtime=2.0, dpu_count=1)
    row.update(
        planning_time_s=0.01,
        session_open_s=0.02,
        host_quantization_time_s=0.03,
        h2d_s=0.04,
        dpu_kernel_time_s=0.05,
        d2h_s=0.06,
        assembly_s=0.07,
        host_dequantization_s=0.08,
        validation_s=0.09,
        session_close_s=0.1,
        graph_execution_s=1.0,
        total_time_s=2.0,
    )
    generate_report([row], tmp_path / "report")
    timing = _csv(tmp_path / "report" / "tables" / "timing_breakdown.csv")
    assert len(timing) == 10
    assert {row["stage"] for row in timing} == {
        "planning",
        "session_open",
        "host_quantization",
        "h2d",
        "kernel",
        "d2h",
        "assembly",
        "host_dequantization",
        "validation",
        "session_close",
    }


def test_mixed_repeat_statuses_do_not_collapse_into_one_aggregate(
    tmp_path: Path,
) -> None:
    completed = _row(engine="upmem_m5", runtime=1.0, dpu_count=1, repeat=0)
    passed = _row(
        engine="upmem_m5", runtime=1.0, dpu_count=1, repeat=1, status="passed"
    )
    generate_report([completed, passed], tmp_path / "report")

    validation = _csv(tmp_path / "report" / "tables" / "validation_accuracy.csv")
    assert {row["status"] for row in validation} == {"completed", "passed"}
    assert {row["repeat_count"] for row in validation} == {"1"}
    transfer = _csv(tmp_path / "report" / "tables" / "transfer_bytes.csv")
    assert {row["status"] for row in transfer} == {"completed", "passed"}
    assert {row["repeat_count"] for row in transfer} == {"1"}


def test_transfer_aggregate_requires_all_repeat_invariants_for_plotting(
    tmp_path: Path,
) -> None:
    rows = [
        _row(engine="upmem_m5", runtime=1.0, dpu_count=1, repeat=repeat)
        for repeat in range(3)
    ]
    rows[1]["transfer_accounting_verified"] = False
    generate_report(rows, tmp_path / "report")

    transfer = _csv(tmp_path / "report" / "tables" / "transfer_bytes.csv")
    assert len(transfer) == 1
    assert transfer[0]["raw_invariants_all_passed"] == "False"
    assert transfer[0]["invariant_passed"] == "False"
    assert transfer[0]["aggregate_component_medians_additive"] == "True"
    manifest = json.loads((tmp_path / "report" / "plot_manifest.json").read_text())
    entry = next(item for item in manifest["plots"] if item["name"] == "transfer_bytes")
    assert entry["status"] == "generated_todo_missing_data"


def test_lifecycle_only_timing_stays_in_csv_but_not_timing_plot(tmp_path: Path) -> None:
    row = _row(
        engine="cpu_numpy",
        runtime=1.0,
        timing_breakdown={
            "session_open_s": 0.1,
            "graph_execution_s": 0.2,
            "session_close_s": 0.3,
        },
    )
    generate_report([row], tmp_path / "report")

    timing = _csv(tmp_path / "report" / "tables" / "timing_breakdown.csv")
    assert {row["stage"] for row in timing} == {"session_open", "session_close"}
    assert {row["timing_coverage"] for row in timing} == {"lifecycle_only"}
    manifest = json.loads((tmp_path / "report" / "plot_manifest.json").read_text())
    entry = next(
        item for item in manifest["plots"] if item["name"] == "timing_breakdown"
    )
    assert entry["status"] == "generated_todo_missing_data"
    assert "physical execution stages" in entry["title"]


def test_timing_plot_summary_uses_equal_weight_family_medians() -> None:
    rows = [
        {
            "case_id": "family-a-1",
            "family": "A",
            "engine": "upmem_physical_1rank_8dpu",
            "path": "opt_einsum_greedy",
            "numeric_policy": "float32_real",
            "qubits": 8,
            "stage": "h2d",
            "time_s": value,
            "timing_coverage": "execution_stage_leaves",
        }
        for value in (1.0, 3.0)
    ]
    rows.extend(
        [
            {
                "case_id": "family-b",
                "family": "B",
                "engine": "upmem_physical_1rank_8dpu",
                "path": "opt_einsum_greedy",
                "numeric_policy": "float32_real",
                "qubits": 8,
                "stage": "h2d",
                "time_s": 10.0,
                "timing_coverage": "execution_stage_leaves",
            },
            {
                "case_id": "cpu-is-excluded",
                "family": "C",
                "engine": "numpy_cpu",
                "path": "opt_einsum_greedy",
                "numeric_policy": "float32_real",
                "qubits": 8,
                "stage": "h2d",
                "time_s": 100.0,
                "timing_coverage": "execution_stage_leaves",
            },
            {
                "case_id": "lifecycle-is-excluded",
                "family": "D",
                "engine": "upmem_physical_1rank_8dpu",
                "path": "opt_einsum_greedy",
                "numeric_policy": "float32_real",
                "qubits": 8,
                "stage": "session_open",
                "time_s": 100.0,
                "timing_coverage": "lifecycle_only",
            },
        ]
    )

    summary = _timing_plot_summary(rows)
    assert len(summary) == 1
    assert summary[0]["median_time_s"] == pytest.approx(6.0)
    assert summary[0]["family_count"] == 2


def test_timing_summary_never_aggregates_distinct_execution_identities() -> None:
    common = {
        "case_id": "bv-8",
        "family": "BV",
        "engine": "upmem_physical_1rank_8dpu",
        "path": "opt_einsum_greedy",
        "numeric_policy": "float32_real",
        "qubits": 8,
        "stage": "kernel",
        "timing_coverage": "execution_stage_leaves",
    }
    rows = [
        {
            **common,
            "time_s": 1.0,
            "route_id": "route-a",
            "route_config_hash": "route-config-a",
            "executor_config_hash": "executor-a",
            "timing_scope": "whole_route",
            "provisioned_dpu_count": 1,
            "provisioned_rank_count": 1,
            "active_dpu_count": 1,
            "active_rank_count": 1,
        },
        {
            **common,
            "time_s": 2.0,
            "route_id": "route-b",
            "route_config_hash": "route-config-b",
            "executor_config_hash": "executor-b",
            "timing_scope": "kernel_only",
            "provisioned_dpu_count": 8,
            "provisioned_rank_count": 1,
            "active_dpu_count": 4,
            "active_rank_count": 1,
        },
    ]

    summary = _timing_plot_summary(rows)

    assert len(summary) == 2
    assert {row["median_time_s"] for row in summary} == {1.0, 2.0}
    assert {
        (
            row["route_id"],
            row["executor_config_hash"],
            row["timing_scope"],
            row["provisioned_dpu_count"],
            row["active_dpu_count"],
        )
        for row in summary
    } == {
        ("route-a", "executor-a", "whole_route", 1, 1),
        ("route-b", "executor-b", "kernel_only", 8, 4),
    }


def test_timing_plot_has_four_route_panels_and_one_bounded_stage_legend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    paths = ("opt_einsum_greedy", "cotengra_flops_seed0")
    numerics = ("float32_real", "host_packed_int8")
    rows = [
        {
            "case_id": f"private-case-{family}-{qubits}",
            "family": family,
            "engine": "upmem_physical_1rank_8dpu",
            "path": path_id,
            "numeric_policy": numeric,
            "qubits": qubits,
            "stage": stage,
            "time_s": float(qubits) / (10 if stage == "h2d" else 5),
            "timing_coverage": "execution_stage_leaves",
        }
        for path_id in paths
        for numeric in numerics
        for family in ("BV", "GHZ")
        for qubits in (8, 10)
        for stage in ("h2d", "kernel")
    ]
    saved: list[object] = []
    monkeypatch.setattr(
        plt.Figure,
        "savefig",
        lambda figure, *args, **kwargs: saved.append(figure),
    )
    title = (
        "M5.5 measured non-overlapping physical execution stages\n"
        "Medians across circuit families"
    )
    valid, reason = _timing_breakdown_plot(tmp_path / "timing.png", title, rows)
    assert valid
    assert reason == ""

    figure = saved[0]
    visible_axes = [axis for axis in figure.axes if axis.get_visible()]
    assert {axis.get_title() for axis in visible_axes} == {
        "greedy / Float32",
        "greedy / host-packed Int8",
        "cotengra FLOPs / Float32",
        "cotengra FLOPs / host-packed Int8",
    }
    assert len(visible_axes) == 4
    assert all(axis.get_yscale() == "linear" for axis in visible_axes)
    legend_labels = [text.get_text() for text in figure.legends[0].texts]
    assert legend_labels == ["H2D", "Kernel"]
    assert len(legend_labels) <= 10
    human_text = " ".join(
        [figure._suptitle.get_text(), *legend_labels]
        + [axis.get_title() for axis in visible_axes]
    )
    assert "private-case" not in human_text
    for raw_id in (*paths, *numerics, "upmem_physical_1rank_8dpu"):
        assert raw_id not in human_text
    plt.close(figure)


def test_timing_plot_rejects_more_than_eight_route_panels(tmp_path: Path) -> None:
    rows = [
        {
            "family": "BV",
            "engine": "upmem_physical_1rank_8dpu",
            "path": f"planner_{index:02d}",
            "numeric_policy": "float32_real",
            "qubits": 8,
            "stage": "kernel",
            "time_s": 1.0,
            "timing_coverage": "execution_stage_leaves",
        }
        for index in range(9)
    ]
    valid, reason = _timing_breakdown_plot(
        tmp_path / "timing.png",
        "Measured non-overlapping physical execution stages",
        rows,
    )
    assert not valid
    assert (
        reason == "9 planner/numeric timing panels exceed the 8-panel readability limit"
    )
    assert (tmp_path / "timing.png").is_file()


def test_timing_plot_rejects_incompatible_identities_in_one_intended_bar(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "family": "BV",
            "engine": "upmem_physical_1rank_8dpu",
            "path": "opt_einsum_greedy",
            "numeric_policy": "float32_real",
            "qubits": 8,
            "stage": "kernel",
            "time_s": float(index + 1),
            "timing_coverage": "execution_stage_leaves",
            "route_id": f"route-{index}",
            "route_config_hash": f"config-{index}",
            "executor_config_hash": f"executor-{index}",
            "timing_scope": "whole_route",
            "provisioned_dpu_count": 8,
            "provisioned_rank_count": 1,
            "active_dpu_count": 4,
            "active_rank_count": 1,
            "admission_identity": f"admission-{index}",
        }
        for index in range(2)
    ]
    valid, reason = _timing_breakdown_plot(
        tmp_path / "timing.png", "Physical timings", rows
    )
    assert not valid
    assert "incompatible physical timing identities" in reason
    assert (tmp_path / "timing.png").is_file()


def test_dpu_kernel_prefers_dpu_kernel_without_double_counting(
    tmp_path: Path,
) -> None:
    row = _row(engine="upmem_m5", runtime=1.0, dpu_count=1)
    row["timing_breakdown"] = {"kernel_s": 2.0, "dpu_kernel_s": 1.0}
    generate_report([row], tmp_path / "report")
    timing = _csv(tmp_path / "report" / "tables" / "timing_breakdown.csv")
    assert len(timing) == 1
    assert timing[0]["stage"] == "kernel"
    assert float(timing[0]["time_s"]) == pytest.approx(1.0)


def test_cross_algorithm_rows_are_retained_but_not_same_plan_claims(
    tmp_path: Path,
) -> None:
    rows = [
        _row(engine="cpu_numpy", runtime=10.0),
        _row(engine="quest_cpu_full_state", runtime=4.0),
        _row(engine="upmem_m5", runtime=5.0),
    ]
    generate_report(rows, tmp_path / "report")
    runtime = _csv(tmp_path / "report" / "tables" / "runtime_by_qubits.csv")
    assert any(row["cross_algorithm"] == "True" for row in runtime)
    assert (
        len(_csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv"))
        == 1
    )


def test_rank_scaling_is_reported_separately(tmp_path: Path) -> None:
    rows = [
        _row(engine="upmem_m5", runtime=8.0, dpu_count=64),
        _row(engine="upmem_m5", runtime=4.0, dpu_count=128),
    ]
    rows[0].update(local_dpu_count=64, rank_count=1, total_dpu_count=64)
    rows[1].update(local_dpu_count=64, rank_count=2, total_dpu_count=128)
    rows[1]["engine_metadata"] = {
        "active_dpu_ids": list(range(128)),
        "active_rank_indices": [0, 1],
        "active_rank_count": 2,
    }
    generate_report(rows, tmp_path / "report")
    scaling = _csv(tmp_path / "report" / "tables" / "upmem_strong_scaling.csv")
    rank = [row for row in scaling if row["scale_dimension"] == "rank"]
    assert len(rank) == 1
    assert float(rank[0]["speedup"]) == pytest.approx(2.0)
    assert float(rank[0]["efficiency"]) == pytest.approx(1.0)
    assert rank[0]["baseline_local_dpu_count"] == "64"
    assert rank[0]["baseline_total_dpu_count"] == "64"
    assert rank[0]["local_dpu_count"] == "64"
    assert rank[0]["total_dpu_count"] == "128"
    dpu = [row for row in scaling if row["scale_dimension"] == "dpu"]
    assert len(dpu) == 1
    assert dpu[0]["speedup"] == ""
    assert (tmp_path / "report" / "plots" / "upmem_rank_scaling.png").is_file()


def test_scaling_keeps_dpu_and_rank_topologies_distinct(tmp_path: Path) -> None:
    rows = [
        _row(engine="upmem_m5", runtime=8.0, dpu_count=1),
        _row(engine="upmem_m5", runtime=4.0, dpu_count=2),
        _row(engine="upmem_m5", runtime=8.0, dpu_count=2),
    ]
    rows[2].update(local_dpu_count=1, rank_count=2, total_dpu_count=2)
    rows[2]["engine_metadata"] = {
        "active_dpu_ids": [0, 1],
        "active_rank_indices": [0, 1],
        "active_rank_count": 2,
    }
    generate_report(rows, tmp_path / "report")
    scaling = _csv(tmp_path / "report" / "tables" / "upmem_strong_scaling.csv")
    assert {
        (
            row["scale_dimension"],
            row["local_dpu_count"],
            row["rank_count"],
            row["total_dpu_count"],
        )
        for row in scaling
    } == {
        ("dpu", "1", "1", "1"),
        ("dpu", "2", "1", "2"),
        ("rank", "1", "2", "2"),
    }


def test_scaling_admission_uses_active_metadata_and_provisioned_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    saved: dict[str, object] = {}
    monkeypatch.setattr(
        plt.Figure,
        "savefig",
        lambda figure, filename, *args, **kwargs: saved.__setitem__(
            Path(filename).name, figure
        ),
    )
    baseline = _row(engine="upmem_m5", runtime=10.0, dpu_count=1)
    active_two = _row(engine="upmem_m5", runtime=5.0, dpu_count=2)
    overprovisioned = _row(engine="upmem_m5", runtime=1.0, dpu_count=32)
    overprovisioned["engine_metadata"] = {
        "active_dpu_ids": list(range(16)),
        "active_rank_indices": [0],
        "active_rank_count": 1,
    }
    inactive_rank = _row(engine="upmem_m5", runtime=4.0, dpu_count=128)
    inactive_rank.update(rank_count=2)
    inactive_rank["engine_metadata"] = {
        "active_dpu_ids": list(range(64)),
        "active_rank_indices": [0],
        "active_rank_count": 1,
    }
    generate_report(
        [baseline, active_two, overprovisioned, inactive_rank],
        tmp_path / "report",
    )

    scaling = _csv(tmp_path / "report" / "tables" / "upmem_strong_scaling.csv")
    assert len(scaling) == 4
    by_dimension = {
        (row["scale_dimension"], row["total_dpu_count"]): row for row in scaling
    }
    assert float(by_dimension[("dpu", "1")]["speedup"]) == pytest.approx(1.0)
    assert float(by_dimension[("dpu", "2")]["speedup"]) == pytest.approx(2.0)
    over = by_dimension[("dpu", "32")]
    assert over["active_dpu_count"] == "16"
    assert over["fully_active_topology"] == "False"
    assert over["speedup"] == ""
    rank = by_dimension[("rank", "128")]
    assert rank["local_dpu_count"] == "64"
    assert rank["active_dpu_count"] == "64"
    assert rank["active_rank_count"] == "1"
    assert rank["fully_active_topology"] == "False"
    assert rank["speedup"] == ""

    scaling_figure = saved["upmem_strong_scaling.png"]
    axis = next(axis for axis in scaling_figure.axes if axis.get_visible())
    data_lines = [line for line in axis.lines if not line.get_label().startswith("_")]
    assert data_lines == []
    assert len(axis.collections) == 2
    manifest = json.loads((tmp_path / "report" / "plot_manifest.json").read_text())
    rank_plot = next(
        item for item in manifest["plots"] if item["name"] == "upmem_rank_scaling"
    )
    assert rank_plot["status"] == "generated_todo_missing_data"
    assert "fully active multi-rank" in rank_plot["reason"]
    plt.close("all")


def test_ratio_pairing_requires_matching_evidence_hashes(tmp_path: Path) -> None:
    path_a = _row(engine="upmem_m5", runtime=8.0, path="opt_einsum_greedy")
    path_b = _row(engine="upmem_m5", runtime=4.0, path="cotengra_flops_seed0")
    wrong_network = dict(path_b, tensor_network_hash="different-network", timing_s=1.0)
    float32 = _row(engine="upmem_m5", runtime=8.0, numeric="float32")
    int8 = _row(engine="upmem_m5", runtime=4.0, numeric="host_packed_int8")
    wrong_plan = dict(int8, contraction_plan_hash="different-plan", timing_s=1.0)

    generate_report(
        [path_a, path_b, wrong_network, float32, int8, wrong_plan],
        tmp_path / "report",
    )
    paths = _csv(tmp_path / "report" / "tables" / "path_runtime_ratio.csv")
    numeric = _csv(tmp_path / "report" / "tables" / "float32_int8_ratios.csv")
    assert len(paths) == 1
    assert float(paths[0]["runtime_ratio_a_over_b"]) == pytest.approx(2.0)
    assert len(numeric) == 1
    assert float(numeric[0]["runtime_ratio_float32_over_int8"]) == pytest.approx(2.0)


def test_path_ratio_is_same_circuit_tn_but_not_same_plan(tmp_path: Path) -> None:
    path_a = _row(engine="upmem_m5", runtime=8.0, path="opt_einsum_greedy")
    path_b = _row(engine="upmem_m5", runtime=4.0, path="cotengra_flops_seed0")
    generate_report([path_a, path_b], tmp_path / "report")
    manifest = json.loads((tmp_path / "report" / "plot_manifest.json").read_text())
    entry = next(
        item for item in manifest["plots"] if item["name"] == "path_runtime_ratio"
    )
    assert "same-circuit/TN" in entry["title"]
    assert "same-plan" not in entry["title"]


def test_ambiguous_path_variants_are_rejected_and_recorded(tmp_path: Path) -> None:
    path_a = _row(engine="upmem_m5", runtime=8.0, path="opt_einsum_greedy")
    conflicting_a = dict(path_a, timing_s=9.0)
    path_b = _row(engine="upmem_m5", runtime=4.0, path="cotengra_flops_seed0")

    generate_report([path_a, conflicting_a, path_b], tmp_path / "report")

    assert _csv(tmp_path / "report" / "tables" / "path_runtime_ratio.csv") == []
    rejections = _csv(tmp_path / "report" / "tables" / "claim_rejections.csv")
    assert len(rejections) == 1
    assert rejections[0]["claim"] == "path_ablation"
    assert "ambiguous duplicate path variants" in rejections[0]["reasons"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scientific_validation_status", "failed"),
        ("scientific_validation_status", None),
        ("exact_once", False),
        ("no_fallback_used", False),
    ],
)
def test_cpu_scientific_admission_rejects_incomplete_contracts(
    tmp_path: Path, field: str, value: object
) -> None:
    row = _row(engine="cpu_numpy", runtime=2.0)
    row[field] = value
    generate_report([row], tmp_path / "report")
    assert _csv(tmp_path / "report" / "tables" / "runtime_by_case_median.csv") == []
    raw = _csv(tmp_path / "report" / "tables" / "runtime_by_qubits.csv")
    assert raw[0]["scientific_admitted"] == "False"
    validation = _csv(tmp_path / "report" / "tables" / "validation_accuracy.csv")
    assert validation[0]["scientific_admitted"] == "False"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_observed", "hardware"),
        ("target_observed", None),
        ("hardware_allocation_verified", False),
        ("native_kernel_executed", False),
        ("hardware_kernel_executed", False),
        ("simulator", True),
        ("cpu_fallback", True),
        ("release_succeeded", False),
    ],
)
def test_upmem_scientific_admission_rejects_unverified_hardware(
    tmp_path: Path, field: str, value: object
) -> None:
    row = _row(engine="upmem_m5", runtime=2.0, dpu_count=1)
    row[field] = value
    generate_report([row], tmp_path / "report")
    assert _csv(tmp_path / "report" / "tables" / "runtime_by_case_median.csv") == []
    assert _csv(tmp_path / "report" / "tables" / "timing_breakdown.csv") == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("simulator_kernel_executed", True), ("cpu_fallback_used", True)],
)
def test_upmem_scientific_admission_rejects_contradictory_aliases(
    tmp_path: Path, field: str, value: object
) -> None:
    row = _row(engine="upmem_m5", runtime=2.0, dpu_count=1)
    row[field] = value
    generate_report([row], tmp_path / "report")
    assert _csv(tmp_path / "report" / "tables" / "runtime_by_case_median.csv") == []


def test_cross_algorithm_uses_established_validation_without_same_plan_admission(
    tmp_path: Path,
) -> None:
    valid = _row(engine="quest_cpu_full_state", runtime=3.0)
    invalid = _row(engine="quimb_tn", runtime=4.0)
    for row in (valid, invalid):
        row.pop("scientific_validation_status")
        row.pop("exact_once")
        row.pop("no_fallback_used")
        row.pop("executor_config_hash")
    invalid["validation_status"] = "not_run"
    generate_report([valid, invalid], tmp_path / "report")
    summary = _csv(tmp_path / "report" / "tables" / "runtime_by_case_median.csv")
    assert len(summary) == 1
    assert summary[0]["engine"] == "quest_cpu_full_state"
    assert (
        _csv(tmp_path / "report" / "tables" / "same_plan_cpu_upmem_speedup.csv") == []
    )


def test_pairing_rejects_missing_or_mismatched_executor_configuration(
    tmp_path: Path,
) -> None:
    path_a = _row(engine="upmem_m5", runtime=8.0, path="opt_einsum_greedy")
    path_b = _row(engine="upmem_m5", runtime=4.0, path="cotengra_flops_seed0")
    path_b["executor_config_hash"] = "executor-config-b"
    float32 = _row(engine="upmem_m5", runtime=8.0, numeric="float32")
    int8 = _row(engine="upmem_m5", runtime=4.0, numeric="host_packed_int8")
    int8.pop("executor_config_hash")
    baseline = _row(engine="upmem_m5", runtime=8.0, dpu_count=1)
    target = _row(engine="upmem_m5", runtime=4.0, dpu_count=2)
    target["executor_config_hash"] = "executor-config-b"

    generate_report(
        [path_a, path_b, float32, int8, baseline, target],
        tmp_path / "report",
    )
    assert _csv(tmp_path / "report" / "tables" / "path_runtime_ratio.csv") == []
    assert _csv(tmp_path / "report" / "tables" / "float32_int8_ratios.csv") == []
    scaling = _csv(tmp_path / "report" / "tables" / "upmem_strong_scaling.csv")
    assert len(scaling) == 2
    assert sorted(row["speedup"] for row in scaling) == ["", "1.0"]

    baseline.pop("executor_config_hash")
    target.pop("executor_config_hash")
    generate_report([baseline, target], tmp_path / "missing-executor-report")
    assert (
        _csv(
            tmp_path / "missing-executor-report" / "tables" / "upmem_strong_scaling.csv"
        )
        == []
    )


def test_jsonl_input_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "normalized_records.jsonl"
    path.write_text(
        json.dumps(_row(engine="cpu_numpy", runtime=1.0)) + "\n", encoding="utf-8"
    )
    assert len(load_records(path)) == 1


def test_generated_plot_sources_have_no_duplicate_series_x_panel_identities(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            engine="cpu_numpy",
            runtime=12.0,
            dpu_count=None,
            path="opt_einsum_greedy",
            numeric="float32",
            repeat=0,
        ),
        _row(
            engine="upmem_m5",
            runtime=6.0,
            dpu_count=1,
            path="opt_einsum_greedy",
            numeric="float32",
            repeat=0,
        ),
        _row(
            engine="upmem_m5",
            runtime=5.0,
            dpu_count=1,
            path="opt_einsum_greedy",
            numeric="host_packed_int8",
            repeat=0,
        ),
    ]
    for row in rows:
        row["timing_breakdown"] = {"h2d_s": 1.0, "d2h_s": 1.0}
        row["actual_h2d_bytes"] = 80
        row["actual_d2h_bytes"] = 20
        row["actual_transfer_bytes"] = 100
        row["max_abs_error"] = 1.0
    report = tmp_path / "report"
    generate_report(rows, report)
    _assert_no_duplicate_plot_identities(report)


def test_validation_timing_and_transfer_preserve_active_topology_series(
    tmp_path: Path,
) -> None:
    partially_active = _row(engine="upmem_m5", runtime=1.0, dpu_count=2)
    partially_active["engine_metadata"] = {
        "active_dpu_ids": [0],
        "active_rank_indices": [0],
        "active_rank_count": 1,
    }
    fully_active = _row(engine="upmem_m5", runtime=1.0, dpu_count=2)
    report = tmp_path / "report"
    generate_report([partially_active, fully_active], report)

    for table in (
        "validation_accuracy.csv",
        "timing_breakdown.csv",
        "transfer_bytes.csv",
    ):
        rows = _csv(report / "tables" / table)
        assert len({row["series"] for row in rows}) > 1
    _assert_no_duplicate_plot_identities(report)


def test_generate_report_from_canonical_runs_if_available_no_duplicate_plot_rows(
    tmp_path: Path,
) -> None:
    sources = sorted(
        Path("runs/inbox/eth/m5_5_accepted_2026-08-14/m5_circuit_canonical").glob(
            "**/normalized_records.jsonl"
        )
    )
    if not sources:
        pytest.skip("canonical run not available in this environment")
    assert len(sources) == 1, "expected exactly one canonical normalized record file"
    source = sources[0]
    rows = load_records(source)
    if not rows:
        pytest.skip("canonical run file is empty")
    report = tmp_path / "canonical-report"
    generate_report(rows, report)
    _assert_no_duplicate_plot_identities(report)
