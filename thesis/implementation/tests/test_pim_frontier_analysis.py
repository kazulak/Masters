from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from quantum_bench.bench.pim_frontier_analysis import run_pim_frontier_analysis, validate_cli_options
import quantum_bench.bench.pim_frontier_analysis as frontier_cmd
from quantum_bench.core.records import CircuitSpec, ContractionTask, PathSummary, TaskGraph, TensorNetworkSpec
from quantum_bench.targets.upmem import (
    MEMORY_LEVEL_L1_WRAM,
    MEMORY_LEVEL_L2_SINGLE_DPU_MRAM,
    MEMORY_LEVEL_L3_MULTI_DPU,
    MEMORY_LEVEL_L4_OUT_OF_SCOPE,
    MEMORY_LEVEL_NOT_DENSE_GEMM,
    UpmemResourceModel,
    analyze_task,
    analyze_task_graph,
)


def _task(
    task_id: str,
    gemm_m: int,
    gemm_k: int,
    gemm_n: int,
    *,
    input_ids: tuple[str, str] | None = None,
    output_id: str | None = None,
    dependencies: tuple[str, ...] = (),
    structure: str = "dense",
) -> ContractionTask:
    return ContractionTask(
        id=task_id,
        input_tensor_ids=input_ids or (f"{task_id}_left", f"{task_id}_right"),
        output_tensor_id=output_id or f"{task_id}_out",
        dependencies=dependencies,
        index_expression="ab,bc->ac",
        input_shapes=((gemm_m, gemm_k), (gemm_k, gemm_n)),
        output_shape=(gemm_m, gemm_n),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=gemm_m,
        gemm_k=gemm_k,
        gemm_n=gemm_n,
        structure=structure,
        estimated_flops=8 * gemm_m * gemm_k * gemm_n,
        estimated_bytes=gemm_m * gemm_k + gemm_k * gemm_n + gemm_m * gemm_n,
    )


def _graph(tasks: tuple[ContractionTask, ...]) -> TaskGraph:
    circuit = CircuitSpec(name="synthetic", n_qubits=2, operations=(), source={"kind": "synthetic"})
    network = TensorNetworkSpec(circuit=circuit, tensors=(), output_labels=(), einsum_expression="")
    summary = PathSummary(
        planner="synthetic",
        optimize="synthetic",
        path_length=len(tasks),
        largest_intermediate=None,
        naive_flops=None,
        optimized_flops=None,
        text="synthetic",
        planner_engine="synthetic",
        planner_id="synthetic",
        planner_kind="synthetic",
        optimize_mode="synthetic",
    )
    return TaskGraph(network=network, tasks=tasks, path=(), path_summary=summary, planning_time_s=0.0)


def test_resource_model_validation() -> None:
    model = UpmemResourceModel()

    assert model.aggregate_mram_bytes == model.available_dpus * model.per_dpu_mram_bytes

    with pytest.raises(ValueError, match="effective_wram_bytes"):
        UpmemResourceModel(per_dpu_wram_bytes=1024, effective_wram_bytes=2048)
    with pytest.raises(ValueError, match="positive"):
        UpmemResourceModel(available_dpus=0)
    with pytest.raises(ValueError, match="max_task_group_dpus"):
        UpmemResourceModel(available_dpus=4, max_task_group_dpus=8)


def test_memory_level_classification_for_synthetic_gemms() -> None:
    default = UpmemResourceModel()

    assert analyze_task(_task("l1", 8, 8, 8), 0, default).memory_level == MEMORY_LEVEL_L1_WRAM
    assert analyze_task(_task("l2", 256, 256, 256), 0, default).memory_level == MEMORY_LEVEL_L2_SINGLE_DPU_MRAM
    l3 = analyze_task(_task("l3", 8192, 8192, 16), 0, default)
    assert l3.memory_level == MEMORY_LEVEL_L3_MULTI_DPU
    assert l3.memory_capacity_min_dpus >= 2

    tiny_cluster = UpmemResourceModel(available_dpus=1, max_task_group_dpus=1)
    l4 = analyze_task(_task("l4", 8192, 8192, 16), 0, tiny_cluster)
    assert l4.memory_level == MEMORY_LEVEL_L4_OUT_OF_SCOPE
    assert l4.memory_reason == "aggregate_mram_capacity_exceeded"


def test_non_dense_and_missing_gemm_reasons_are_not_memory_overflow() -> None:
    sparse = analyze_task(_task("sparse", 8, 8, 8, structure="sparse"), 0)
    missing = analyze_task(_task("missing", 0, 8, 8), 1)

    assert sparse.memory_level == MEMORY_LEVEL_NOT_DENSE_GEMM
    assert sparse.memory_reason == "not_lowerable_to_dense_gemm"
    assert missing.memory_level == MEMORY_LEVEL_NOT_DENSE_GEMM
    assert missing.memory_reason == "missing_gemm_dimensions"


def test_serial_graph_reports_width_one_and_serialized_source() -> None:
    graph = _graph(
        (
            _task("task_0", 8, 8, 8, input_ids=("a", "b"), output_id="r0"),
            _task("task_1", 8, 8, 8, input_ids=("r0", "c"), output_id="r1", dependencies=("task_0",)),
        )
    )
    analysis = analyze_task_graph(graph)
    summary = analysis.summary()

    assert [wave.ready_task_count for wave in analysis.waves] == [1, 1]
    assert summary["max_frontier_width"] == 1
    assert summary["potential_parallelism_source"] == "task_graph_serialized_by_planner"


def test_independent_tasks_produce_parallel_frontier() -> None:
    graph = _graph(
        (
            _task("task_0", 8, 8, 8, input_ids=("a", "b"), output_id="r0"),
            _task("task_1", 8, 8, 8, input_ids=("c", "d"), output_id="r1"),
            _task("task_2", 8, 8, 8, input_ids=("r0", "r1"), output_id="r2", dependencies=("task_0", "task_1")),
        )
    )
    analysis = analyze_task_graph(graph)
    summary = analysis.summary()

    assert [wave.ready_task_count for wave in analysis.waves] == [2, 1]
    assert analysis.waves[0].dominant_source == "inter_task"
    assert summary["max_frontier_width"] == 2
    assert summary["critical_path_length_tasks"] == 2


def test_pim_frontier_analysis_case_writes_json_csv_and_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(frontier_cmd, "capture_environment", lambda root_dir: {})

    run_dir = run_pim_frontier_analysis(tmp_path, case="bell_2q", n_qubits=2, output_plots=False)
    payload = json.loads((run_dir / "pim_frontier_analysis.json").read_text(encoding="utf-8"))
    encoded = json.dumps(payload)

    assert payload["schema_version"] == "pim_frontier_analysis_v1"
    assert payload["metadata"]["analysis_only"] is True
    assert payload["metadata"]["suite_routes_ignored"] is True
    assert payload["task_rows"]
    assert payload["case_summaries"][0]["memory_level_counts"]
    assert payload["case_summaries"][0]["dominant_source_counts"]
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded
    assert str(tmp_path) not in encoded
    assert (run_dir / "pim_frontier_analysis_tasks.csv").exists()
    assert (run_dir / "pim_frontier_analysis_cases.csv").exists()
    assert (run_dir / "pim_frontier_analysis_waves.csv").exists()
    assert (run_dir / "pim_frontier_analysis_summary.md").exists()


def test_pim_frontier_analysis_suite_ignores_routes_and_csv_nested_values_are_json(tmp_path: Path, monkeypatch) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: frontier_route_ignore_test
defaults:
  planner: {engine: opt_einsum, optimize: greedy}
workloads:
  - id: bell_2q
    circuit: {kind: builtin, name: bell_2q}
routes:
  - id: cpu_tn_einsum_exact
  - id: made_up_route_that_must_not_run
validation: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(frontier_cmd, "capture_environment", lambda root_dir: {})

    run_dir = run_pim_frontier_analysis(tmp_path, suite_path=suite_path, output_plots=False)
    payload = json.loads((run_dir / "pim_frontier_analysis.json").read_text(encoding="utf-8"))
    with (run_dir / "pim_frontier_analysis_cases.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert payload["metadata"]["normal_suite_routes_executed"] is False
    assert payload["metadata"]["suite_routes_ignored"] is True
    assert len(payload["case_summaries"]) == 1
    assert len(payload["task_rows"]) == payload["case_summaries"][0]["task_count"]
    assert json.loads(row["memory_level_counts"])
    assert json.loads(row["dominant_source_counts"])


def test_pim_frontier_analysis_cli_validation_rules() -> None:
    model = UpmemResourceModel()

    with pytest.raises(ValueError, match="exactly one"):
        validate_cli_options(suite_path=None, case=None, n_qubits=None, resource_model=model)
    with pytest.raises(ValueError, match="bell_2q"):
        validate_cli_options(suite_path=None, case="bell_2q", n_qubits=3, resource_model=model)
    with pytest.raises(ValueError, match="requires --n-qubits"):
        validate_cli_options(suite_path=None, case="QRNG", n_qubits=None, resource_model=model)
