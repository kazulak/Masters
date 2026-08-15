from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from quantum_bench.bench.m5_circuit_report import (
    _faceted_plot,
    _record_row,
    _short_series_label,
    generate_report,
    load_records,
)


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
    status: str = "completed",
) -> dict[str, object]:
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
        "timing_s": runtime,
        "actual_h2d_bytes": 80,
        "actual_d2h_bytes": 20,
        "actual_transfer_bytes": 100,
        "max_abs_error": 1e-6,
        "timing_breakdown": {"h2d_s": 0.1, "kernel_s": runtime / 2, "d2h_s": 0.1},
    }
    if "upmem" in engine or "dpu" in engine:
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
        == "greedy / f32 / 2DPU/1R"
    )
    assert (
        _short_series_label(
            "BV | numpy_cpu | opt_einsum_greedy | f32_real | "
            "1 local / None ranks / 1 total",
            "BV",
        )
        == "CPU / greedy / f32_real"
    )
    assert (
        _short_series_label(
            "BV | upmem_physical_1rank_8dpu | opt_einsum_greedy | f32_real | "
            "8DPU x 1 rank(s) = 8 total",
            "BV",
        )
        == "UPMEM / greedy / f32_real / 8DPU/1R"
    )
    assert (
        _short_series_label(
            "BV | numpy_cpu | f32_real | 1DPU x ? rank(s) = 1 total",
            "BV",
        )
        == "CPU / f32_real"
    )
    assert (
        _short_series_label(
            "BV | upmem_physical_1rank_8dpu | host_packed_int8 | "
            "8DPU x 1 rank(s) = 8 total",
            "BV",
        )
        == "UPMEM / int8 / 8DPU/1R"
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
        == "UPMEM / cotengra / 8DPU/1R"
    )


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
    assert _short_series_label(normalized["series"], "BV").endswith(
        "16/32DPU + 1/1R"
    )


def test_plot_collapses_active_counts_into_one_provisioned_topology_line(
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
            "series": (
                "BV | upmem_m5 | opt_einsum_greedy | f32_real | "
                f"{active}/8DPU + 1/1R"
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
    )
    figure = saved[0]
    axis = next(axis for axis in figure.axes if axis.get_visible())
    data_lines = [line for line in axis.lines if not line.get_label().startswith("_")]
    assert len(data_lines) == 1
    assert list(data_lines[0].get_xdata()) == [8, 12, 16, 20]
    assert [text.get_text() for text in figure.legends[0].texts] == [
        "UPMEM / greedy / f32_real / 1-8 active of 8 provisioned DPU / 1 of 1 rank"
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
        "timing_breakdown",
        "transfer_bytes",
        "float32_int8_ratio",
    ):
        assert entries[name]["status"] == "generated_todo_missing_data"
        assert entries[name]["reason"] == reason
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
        float32 = _row(
            engine="upmem_m5", runtime=8.0, numeric="float32", dpu_count=32
        )
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
            for numeric in ("float32", "host_packed_int8"):
                for path in ("opt_einsum_greedy", "cotengra_flops_seed0"):
                    row = _row(
                        engine=engine,
                        runtime=float(qubits),
                        path=path,
                        numeric=numeric,
                        dpu_count=dpu_count,
                    )
                    row.update(
                        case_id=f"bv-{qubits}",
                        qubits=qubits,
                        circuit_semantics_hash=f"circuit-bv-{qubits}",
                        tensor_network_hash=f"network-bv-{qubits}",
                    )
                    rows.append(row)

    generate_report(rows, tmp_path / "report")

    for filename in ("path_runtime_ratio.png", "float32_int8_ratio.png"):
        figure = saved[filename]
        axes = [axis for axis in figure.axes if axis.get_visible()]
        assert len(axes) == 1
        data_lines = [
            line for line in axes[0].lines if not line.get_label().startswith("_")
        ]
        assert len(data_lines) == 4
        assert all(
            len(set(line.get_xdata())) == len(line.get_xdata()) for line in data_lines
        )
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
        "graph_execution": pytest.approx(0.6),
        "session_open": pytest.approx(0.7),
        "session_close": pytest.approx(0.8),
        "total": pytest.approx(2.0),
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
    assert len(data_lines) == 1
    assert list(data_lines[0].get_xdata()) == [1, 2]
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
