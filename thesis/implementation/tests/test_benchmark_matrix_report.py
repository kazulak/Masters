from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from quantum_bench.bench.benchmark_matrix_report import (
    load_benchmark_matrix,
    run_benchmark_matrix_report,
    validate_benchmark_matrix,
)
import quantum_bench.bench.benchmark_matrix_report as matrix_module
from quantum_bench.bench.pim_frontier_analysis import run_pim_frontier_analysis
import quantum_bench.bench.pim_frontier_analysis as frontier_module
from quantum_bench.bench.runner import run_suite
from quantum_bench.circuits import load_circuit
from quantum_bench.targets.upmem import (
    MEMORY_LEVEL_L1_WRAM,
    MEMORY_LEVEL_L2_SINGLE_DPU_MRAM,
    MEMORY_LEVEL_L3_MULTI_DPU,
    MEMORY_LEVEL_L4_OUT_OF_SCOPE,
    SYNTHETIC_PRESSURE_ERROR,
    UPMEM_DENSE_ESTIMATE_KEY,
    build_synthetic_pressure_task_graph,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_case() -> dict[str, object]:
    return {
        "case_id": "synthetic_l1_test",
        "workload_id": "synthetic_l1_test",
        "metadata": {
            "workload_type": "synthetic_pressure",
            "execution_scope": "model_only",
            "not_real_quantum_circuit": True,
        },
        "circuit": {
            "kind": "synthetic_pressure",
            "name": "synthetic_l1_test",
            "profile": "independent_gemm",
            "task_count": 2,
            "gemm_m": 16,
            "gemm_k": 16,
            "gemm_n": 16,
        },
    }


def test_benchmark_matrix_config_loads_and_keeps_upmem_unified() -> None:
    matrix = load_benchmark_matrix(ROOT / "configs" / "benchmark_matrix.yml")
    categories = [route["route_category"] for route in matrix["route_categories"]]

    assert categories.count("upmem_tn_runtime") == 1
    assert "upmem_l1" not in categories
    assert "upmem_l2" not in categories
    assert "upmem_l3" not in categories
    assert any(route["route_id"] == "cpu_tn_einsum_exact" for route in matrix["route_categories"])
    quest = next(route for route in matrix["route_categories"] if route["route_id"] == "quest_cpu_full_state_exact")
    assert quest["output_authority"] == "authoritative"
    assert quest["validation_policy"] == "full_exact"
    gpu_tn = next(route for route in matrix["route_categories"] if route["route_category"] == "gpu_tn_exact")
    assert gpu_tn["route_status"] == "planned"
    assert gpu_tn["evidence_type"] == "planned"
    assert gpu_tn["output_authority"] == "future_only"
    assert gpu_tn["validation_policy"] == "not_applicable_until_verified"
    gpu_full_state = next(route for route in matrix["route_categories"] if route["route_category"] == "gpu_full_state")
    assert gpu_full_state["route_id"] == "quest_gpu_full_state_exact"
    assert gpu_full_state["route_status"] == "implemented_optional"
    assert gpu_full_state["evidence_type"] == "measured_when_verified"
    assert gpu_full_state["output_authority"] == "authoritative_when_verified"


def test_benchmark_matrix_rejects_upmem_internal_classes_as_top_level_routes() -> None:
    matrix = load_benchmark_matrix(ROOT / "configs" / "benchmark_matrix.yml")
    matrix["route_categories"] = [dict(route) for route in matrix["route_categories"]]
    matrix["route_categories"].append(
        {
            "route_category": "upmem_l1",
            "route_id": "bad",
            "route_status": "planned",
            "execution_scope": "task_level",
            "target": "upmem_simulator",
            "validation_scope": "task_output",
            "evidence_type": "planned",
            "output_authority": "task_level",
            "validation_policy": "not_applicable",
        }
    )

    with pytest.raises(ValueError, match="internal UPMEM classes"):
        validate_benchmark_matrix(matrix)


def test_benchmark_matrix_rejects_missing_quest_exact_semantics() -> None:
    matrix = load_benchmark_matrix(ROOT / "configs" / "benchmark_matrix.yml")
    matrix["route_categories"] = [dict(route) for route in matrix["route_categories"]]
    quest = next(route for route in matrix["route_categories"] if route["route_id"] == "quest_cpu_full_state_exact")
    quest["output_authority"] = "benchmark_only"

    with pytest.raises(ValueError, match="authoritative"):
        validate_benchmark_matrix(matrix)


def test_benchmark_matrix_rejects_gpu_tn_authoritative_claim_before_verified_route() -> None:
    matrix = load_benchmark_matrix(ROOT / "configs" / "benchmark_matrix.yml")
    matrix["route_categories"] = [dict(route) for route in matrix["route_categories"]]
    gpu_tn = next(route for route in matrix["route_categories"] if route["route_category"] == "gpu_tn_exact")
    gpu_tn["output_authority"] = "authoritative"
    gpu_tn["validation_policy"] = "full_exact"

    with pytest.raises(ValueError, match="planned GPU TN"):
        validate_benchmark_matrix(matrix)


def test_benchmark_matrix_rejects_gpu_full_state_without_verified_route_semantics() -> None:
    matrix = load_benchmark_matrix(ROOT / "configs" / "benchmark_matrix.yml")
    matrix["route_categories"] = [dict(route) for route in matrix["route_categories"]]
    gpu_full_state = next(route for route in matrix["route_categories"] if route["route_category"] == "gpu_full_state")
    gpu_full_state["route_id"] = "planned"
    gpu_full_state["route_status"] = "planned"

    with pytest.raises(ValueError, match="optional verified QuEST GPU"):
        validate_benchmark_matrix(matrix)


def test_synthetic_pressure_task_graph_has_analyzer_fields_and_no_arrays() -> None:
    graph = build_synthetic_pressure_task_graph(_synthetic_case())
    encoded = json.dumps(graph, default=str)

    assert len(graph.tasks) == 2
    assert graph.network.tensors == ()
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded
    for index, task in enumerate(graph.tasks):
        assert task.id == f"task_{index}"
        assert task.input_tensor_ids
        assert task.output_tensor_id
        assert task.gemm_m == 16
        assert task.gemm_k == 16
        assert task.gemm_n == 16
        assert UPMEM_DENSE_ESTIMATE_KEY in task.target_estimates


def test_synthetic_pressure_is_rejected_by_normal_circuit_loader_and_runner(tmp_path: Path) -> None:
    case = _synthetic_case()

    with pytest.raises(ValueError, match="analysis-only"):
        load_circuit(case, ROOT)

    suite_path = tmp_path / "synthetic_normal_run.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: synthetic_normal_run
workloads:
  - id: synthetic_l1_test
    metadata:
      workload_type: synthetic_pressure
      execution_scope: model_only
      not_real_quantum_circuit: true
    circuit:
      kind: synthetic_pressure
      name: synthetic_l1_test
      profile: independent_gemm
      task_count: 2
      gemm_m: 16
      gemm_k: 16
      gemm_n: 16
routes:
  - id: cpu_tn_einsum_exact
validation: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=SYNTHETIC_PRESSURE_ERROR[:30]):
        run_suite(suite_path, tmp_path)


def test_pim_frontier_pressure_suite_handles_synthetic_cases(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(frontier_module, "capture_environment", lambda root_dir: {})

    run_dir = run_pim_frontier_analysis(
        tmp_path,
        suite_path=ROOT / "configs" / "suites" / "diagnostics" / "pim_frontier_pressure_quick.yml",
        output_plots=False,
    )
    payload = json.loads((run_dir / "pim_frontier_analysis.json").read_text(encoding="utf-8"))
    levels = payload["summary"]["memory_level_counts"]
    synthetic = [
        row
        for row in payload["case_summaries"]
        if row["circuit"].get("source", {}).get("metadata", {}).get("workload_type") == "synthetic_pressure"
    ]

    assert synthetic
    assert levels[MEMORY_LEVEL_L1_WRAM] > 0
    assert levels[MEMORY_LEVEL_L2_SINGLE_DPU_MRAM] > 0
    assert levels[MEMORY_LEVEL_L3_MULTI_DPU] > 0
    assert levels[MEMORY_LEVEL_L4_OUT_OF_SCOPE] > 0
    assert payload["metadata"]["suite_routes_ignored"] is True


def test_benchmark_matrix_report_writes_artifacts_and_preserves_semantics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(matrix_module, "capture_environment", lambda root_dir: {})

    run_dir = run_benchmark_matrix_report(
        tmp_path,
        ROOT / "configs" / "benchmark_matrix.yml",
        output_plots=False,
    )
    payload = json.loads((run_dir / "benchmark_matrix.json").read_text(encoding="utf-8"))
    encoded = json.dumps(payload)
    upmem_categories = {row["route_category"] for row in payload["benchmark_matrix_rows"] if row["route_category"].startswith("upmem")}
    upmem_rows = [row for row in payload["benchmark_matrix_rows"] if row["route_category"] == "upmem_tn_runtime"]
    with (run_dir / "benchmark_matrix.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert payload["schema_version"] == "benchmark_matrix_v1"
    assert payload["summary"]["upmem_is_unified_runtime"] is True
    assert upmem_categories == {"upmem_tn_runtime"}
    assert upmem_rows
    assert all(row["resource_model_id"] != "not_applicable" for row in upmem_rows)
    assert all(row["execution_scope"] != "full_circuit" for row in upmem_rows)
    assert any(int(row["upmem_executable_task_count_current"]) > 0 for row in upmem_rows)
    assert payload["pim_pressure_resource_model_rows"][0]["first_l2_l3_synthetic_pressure"]
    assert payload["pim_pressure_resource_model_rows"][0]["first_l2_l3_real_circuits"] is not None
    assert csv_rows
    assert str(tmp_path) not in encoded
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded


def test_benchmark_matrix_report_rejects_upmem_full_circuit_speedup_claim() -> None:
    matrix = load_benchmark_matrix(ROOT / "configs" / "benchmark_matrix.yml")
    matrix["route_categories"] = [dict(route) for route in matrix["route_categories"]]
    upmem = next(route for route in matrix["route_categories"] if route["route_category"] == "upmem_tn_runtime")
    upmem["execution_scope"] = "full_circuit"

    with pytest.raises(ValueError, match="full-circuit authority"):
        validate_benchmark_matrix(matrix)
