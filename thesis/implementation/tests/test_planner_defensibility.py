"""Planner Defensibility v3 comprehensive test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from quantum_bench.bench.planner_compare import COMPARISON_FIELDS, compare_planners
from quantum_bench.core.records import (
    CircuitSpec,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
)
from quantum_bench.core.target_estimates import TARGET_ESTIMATE_SIDECAR_SCHEMA_VERSION
from quantum_bench.targets.upmem.schedule import UPMEM_DENSE_MODEL
from quantum_bench.tn.exact_modeled_planner import (
    DEFAULT_MAX_INPUT_TENSORS,
    HARD_MAX_INPUT_TENSORS,
    ExactModeledPlanner,
)
from quantum_bench.tn.network import TensorNetworkValue, build_full_einsum_expression
from quantum_bench.tn.planner_motifs import build_planner_motif_workload
from quantum_bench.tn.planners import planner_from_config
from quantum_bench.tn.upmem_path_cost_v2 import (
    UPMEM_PATH_OBJECTIVE_V2,
    metric_contract_v2,
    model_upmem_task_cost_v2,
    upmem_path_cost_profile_v2,
)
from quantum_bench.tn.upmem_planner import (
    PlannerInfeasibleError,
    UpmemAwareProjectedPrefixPlanner,
)
from .support import dense_task


def _motif_network(name: str = "chain"):
    return build_planner_motif_workload(
        {
            "case_id": f"planner_motif_{name}",
            "circuit": {"kind": "planner_motif", "name": name},
            "metadata": {
                "workload_type": "synthetic_planner_motif",
                "execution_scope": "model_only",
                "not_real_quantum_circuit": True,
            },
        }
    ).network


def _exact_config(
    profile: str = "balanced_literature_informed", max_tensors: int = 6
) -> dict[str, Any]:
    return {
        "engine": "exact_modeled",
        "algorithm": "exact_modeled",
        "objective_version": UPMEM_PATH_OBJECTIVE_V2,
        "selection_scope": "exact_finite_search",
        "max_input_tensors": max_tensors,
        "weight_profile": profile,
        "normalization": "fixed_log1p_generic_budgets_v2",
        "execution_policy": "generic_single_dpu_split_complex_v2",
    }


def _greedy_config(profile: str = "balanced_literature_informed") -> dict[str, Any]:
    return {
        "engine": "custom_upmem",
        "algorithm": "greedy",
        "objective_version": UPMEM_PATH_OBJECTIVE_V2,
        "selection_scope": "projected_prefix",
        "weight_profile": profile,
        "normalization": "fixed_log1p_generic_budgets_v2",
        "execution_policy": "generic_single_dpu_split_complex_v2",
    }


def test_exact_modeled_planner_exact_complete_deterministic_path() -> None:
    network = _motif_network("grid")
    planner = ExactModeledPlanner()
    first = planner.plan(network)
    second = planner.plan(network)

    assert isinstance(first.path, tuple)
    assert len(first.path) == len(network.tensors) - 1
    assert all(isinstance(step, tuple) and len(step) == 2 for step in first.path)
    assert first.path == second.path
    assert first.metadata["modeled_score"] == second.metadata["modeled_score"]


def test_exact_modeled_planner_explored_complete_paths_greater_than_one() -> None:
    network = _motif_network("grid")  # 6 tensors -> 2700 complete paths
    planner = ExactModeledPlanner()
    result = planner.plan(network)

    assert len(network.tensors) == 6
    assert result.metadata["explored_complete_paths"] == 2700
    assert result.metadata["feasible_complete_paths"] >= 1
    assert (
        result.metadata["explored_complete_paths"]
        >= result.metadata["feasible_complete_paths"]
    )


def test_exact_modeled_score_less_than_or_equal_custom_greedy() -> None:
    network = _motif_network("grid")
    profile = upmem_path_cost_profile_v2("balanced_literature_informed")

    exact_planner = ExactModeledPlanner(profile=profile)
    greedy_planner = UpmemAwareProjectedPrefixPlanner(profile=profile)

    exact_res = exact_planner.plan(network)
    greedy_res = greedy_planner.plan(network)

    exact_score = float(exact_res.metadata["modeled_score"])
    greedy_score = float(greedy_res.metadata["modeled_score"])

    assert exact_score <= greedy_score + 1e-12


def test_exact_modeled_cap_rejection() -> None:
    assert HARD_MAX_INPUT_TENSORS == 6
    assert DEFAULT_MAX_INPUT_TENSORS == 6
    network = _motif_network("grid")  # 6 tensors
    capped_planner = ExactModeledPlanner(max_input_tensors=3)

    with pytest.raises(
        PlannerInfeasibleError, match="input tensor cap exceeded"
    ) as exc_info:
        capped_planner.plan(network)

    assert exc_info.value.rejection_reasons == ("input_tensor_cap_exceeded",)

    with pytest.raises(ValueError, match="max_input_tensors cannot exceed 6"):
        ExactModeledPlanner(max_input_tensors=7)


def test_v2_metric_contract_fields() -> None:
    contract = metric_contract_v2()
    assert len(contract) > 0

    required_fields = {"unit", "origin", "scope", "model_id"}
    for key, entry in contract.items():
        assert required_fields <= set(entry.keys()), (
            f"Metric '{key}' missing contract fields"
        )
        assert entry["origin"] == "analytic_model"
        assert entry["model_id"] == UPMEM_PATH_OBJECTIVE_V2

    numeric_fields = {
        "estimated_flops",
        "largest_tensor_bytes",
        "host_to_dpu_payload_bytes",
        "dpu_to_host_payload_bytes",
        "mram_dma_window_bytes_model",
        "tile_iterations",
        "host_completion_events",
        "numeric_component_invocations",
        "numeric_recombination_flops",
        "numeric_representation_penalty",
        "task_mram_payload_bytes",
        "native_static_mram_reservation_bytes",
        "mram_capacity_bytes",
        "mram_static_reservation_pressure_ratio",
        "mram_max_region_payload_ratio",
        "mram_payload_pressure_ratio",
        "known_wram_static_bytes",
        "wram_budget_bytes",
        "wram_known_pressure_ratio",
    }
    assert set(contract.keys()) == numeric_fields
    assert "feasibility" not in contract
    assert "rejection_reasons" not in contract

    components = model_upmem_task_cost_v2(dense_task("task", 2, 3, 4))
    serialized = components.to_json_dict()
    assert "metric_contract" in serialized
    assert serialized["metric_contract"] == contract


def test_largest_tensor_differs_from_task_payload() -> None:
    task = dense_task("task", 4, 4, 4)
    components = model_upmem_task_cost_v2(task)

    assert components.largest_tensor_bytes > 0
    assert components.task_mram_payload_bytes > 0
    assert components.largest_tensor_bytes != components.task_mram_payload_bytes


def test_baseline_configs_and_seeds_reproducible(tmp_path: Path) -> None:
    suite_path = Path("configs/suites/diagnostics/planner_defensibility_v3.yml")

    run_dir1 = compare_planners(suite_path, tmp_path / "run1")
    run_dir2 = compare_planners(suite_path, tmp_path / "run2")

    json1 = json.loads(
        (run_dir1 / "planner_comparison.json").read_text(encoding="utf-8")
    )
    json2 = json.loads(
        (run_dir2 / "planner_comparison.json").read_text(encoding="utf-8")
    )

    assert len(json1["rows"]) == len(json2["rows"])
    for row1, row2 in zip(json1["rows"], json2["rows"]):
        assert row1["case_id"] == row2["case_id"]
        assert row1["planner_id"] == row2["planner_id"]
        assert row1["pim_objective_score"] == row2["pim_objective_score"]
        assert (
            row1["contraction_path_structure_hash"]
            == row2["contraction_path_structure_hash"]
        )
        assert (
            row1.get("planner_config_hash") is not None
            and len(str(row1["planner_config_hash"])) > 0
        )
        assert isinstance(row1.get("planner_config"), dict)

        if row1["planner_engine"] == "cotengra":
            cfg = row1["planner_config"]
            assert "seed" in cfg and cfg["seed"] is not None
            assert "max_repeats" in cfg and cfg["max_repeats"] is not None


def test_modeled_only_wording_and_no_hardware_performance_claim(tmp_path: Path) -> None:
    network = _motif_network("chain")
    planner = ExactModeledPlanner()
    result = planner.plan(network)

    assert "modeled-only" in result.path_info_text
    assert "not hardware-optimal" in result.path_info_text
    assert (
        result.metadata["selection_claim"]
        == "exact_optimum_under_modeled_objective_only_not_hardware_optimal"
    )
    assert result.metadata["execution_plan_executed"] is False

    suite_path = Path("configs/suites/diagnostics/planner_defensibility_v3.yml")
    run_dir = compare_planners(suite_path, tmp_path)
    records = [
        json.loads(line)
        for line in (run_dir / "normalized_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for rec in records:
        assert rec["hardware_execution"] is False
        assert rec["hardware_speedup_applicable"] is False
        assert rec["execution_plan_executed"] is False
        assert rec["parallelism_evidence_type"] == "modeled"


def test_pim_largest_tensor_bytes_and_alias_definition(tmp_path: Path) -> None:
    assert "pim_largest_tensor_bytes" in COMPARISON_FIELDS
    assert "pim_peak_intermediate_bytes" in COMPARISON_FIELDS
    assert "pim_peak_intermediate_bytes_alias_definition" in COMPARISON_FIELDS

    suite_path = Path("configs/suites/diagnostics/planner_defensibility_v3.yml")
    run_dir = compare_planners(suite_path, tmp_path)
    data = json.loads((run_dir / "planner_comparison.json").read_text(encoding="utf-8"))

    for row in data["rows"]:
        assert "pim_largest_tensor_bytes" in row
        assert "pim_peak_intermediate_bytes" in row
        assert "pim_peak_intermediate_bytes_alias_definition" in row
        alias_def = row["pim_peak_intermediate_bytes_alias_definition"]
        assert (
            alias_def
            == "compatibility_alias_for_pim_largest_tensor_bytes_not_path_level_peak_memory"
        )
        assert "not_path_level_peak_memory" in alias_def
        if row.get("candidate_status") == "completed":
            assert row["pim_peak_intermediate_bytes"] == row["pim_largest_tensor_bytes"]


def test_exact_planner_numeric_semantics_matches_projected_prefix_v2() -> None:
    network = _motif_network("chain")
    exact_planner = ExactModeledPlanner()
    greedy_planner = UpmemAwareProjectedPrefixPlanner(
        profile=upmem_path_cost_profile_v2("balanced_literature_informed")
    )

    exact_res = exact_planner.plan(network)
    greedy_res = greedy_planner.plan(network)

    assert (
        exact_res.metadata["numeric_contract"]
        == greedy_res.identity.options["numeric_contract"]
    )
    assert "task_numeric_executions" in exact_res.metadata
    assert "task_numeric_executions" in greedy_res.metadata


def test_defensibility_v3_suite_workload_mix(tmp_path: Path) -> None:
    suite_path = Path("configs/suites/diagnostics/planner_defensibility_v3.yml")
    run_dir = compare_planners(suite_path, tmp_path)
    data = json.loads((run_dir / "planner_comparison.json").read_text(encoding="utf-8"))

    assert len(data["rows"]) == 24
    assert all(row["candidate_status"] == "completed" for row in data["rows"])

    cotengra_rows = [row for row in data["rows"] if row["planner_engine"] == "cotengra"]
    assert len(cotengra_rows) == 12
    assert all(row["candidate_status"] == "completed" for row in cotengra_rows)

    case_ids = {row["case_id"] for row in data["rows"]}
    assert case_ids == {
        "planner_motif_chain",
        "planner_motif_flop_memory_tradeoff",
        "qrng_3q",
    }


def test_exact_modeled_planner_genuinely_complex_valued_tensor_network() -> None:
    s0 = TensorSpec("t0", (0, 1), (2, 4), "dense", dtype="complex128")
    s1 = TensorSpec("t1", (1, 2), (4, 2), "dense", dtype="complex128")
    s2 = TensorSpec("t2", (2, 3), (2, 2), "dense", dtype="complex128")

    v0 = np.ones((2, 4), dtype=np.complex128) + 1j * np.ones(
        (2, 4), dtype=np.complex128
    )
    v1 = np.ones((4, 2), dtype=np.complex128) + 1j * np.ones(
        (4, 2), dtype=np.complex128
    )
    v2 = np.ones((2, 2), dtype=np.complex128) + 1j * np.ones(
        (2, 2), dtype=np.complex128
    )

    tensors = [
        TensorValue(s0, v0),
        TensorValue(s1, v1),
        TensorValue(s2, v2),
    ]
    circuit = CircuitSpec("test_complex", 0, (), {"kind": "custom"})
    specs = tuple(t.spec for t in tensors)
    net_spec = TensorNetworkSpec(
        circuit, specs, (), build_full_einsum_expression(list(specs), ())
    )
    complex_network = TensorNetworkValue(net_spec, tensors)

    real_tensors = [
        TensorValue(
            TensorSpec(s.id, s.labels, s.shape, s.structure, dtype="float64"),
            np.ones(s.shape, dtype=np.float64),
        )
        for s in specs
    ]
    real_specs = tuple(t.spec for t in real_tensors)
    real_network = TensorNetworkValue(
        TensorNetworkSpec(
            circuit, real_specs, (), build_full_einsum_expression(list(real_specs), ())
        ),
        real_tensors,
    )

    planner = ExactModeledPlanner()
    complex_res = planner.plan(complex_network)
    real_res = planner.plan(real_network)

    complex_execs = complex_res.metadata["task_numeric_executions"]
    assert len(complex_execs) == len(complex_network.tensors) - 1
    for task_id, task_exec in complex_execs.items():
        assert task_exec["representation"] == "split_real_imag"
        assert task_exec["component_invocations"] == 4
        assert task_exec["component_invocations"] > 1
        assert task_exec["recombination_flops"] > 0

    complex_components = complex_res.metadata["components"]
    real_components = real_res.metadata["components"]

    assert complex_components["numeric_component_invocations"] == 8
    assert real_components["numeric_component_invocations"] == 2
    assert (
        complex_components["numeric_component_invocations"]
        > real_components["numeric_component_invocations"]
    )

    assert complex_components["numeric_recombination_flops"] == 6
    assert real_components["numeric_recombination_flops"] == 0
    assert complex_components["numeric_recombination_flops"] > 0


def test_sidecar_artifact_persistence_and_consistency(tmp_path: Path) -> None:
    suite_path = Path("configs/suites/diagnostics/planner_defensibility_v3.yml")
    run_dir = compare_planners(suite_path, tmp_path)
    data = json.loads((run_dir / "planner_comparison.json").read_text(encoding="utf-8"))

    for row in data["rows"]:
        if row.get("candidate_status") != "completed":
            continue

        task_graph_path = run_dir / row["task_graph_artifact"]
        assert task_graph_path.exists()
        task_graph_json = json.loads(task_graph_path.read_text(encoding="utf-8"))
        for task in task_graph_json["tasks"]:
            assert task["target_estimates"] == {}

        target_estimates_path = run_dir / row["target_estimates_artifact"]
        assert target_estimates_path.exists()
        jsonl_lines = [
            json.loads(line)
            for line in target_estimates_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(jsonl_lines) == len(task_graph_json["tasks"])
        for est_row in jsonl_lines:
            assert est_row["schema_version"] == TARGET_ESTIMATE_SIDECAR_SCHEMA_VERSION
            assert (
                est_row["scientific_plan_hash"]
                == task_graph_json["contraction_plan_hash"]
            )
            assert est_row["scientific_plan_hash"] == row["contraction_plan_hash"]
            assert est_row["target_id"] == "upmem_dense_gemm"
            assert est_row["model_id"] == UPMEM_DENSE_MODEL
            assert isinstance(est_row["metric_provenance"], list)
            assert len(est_row["metric_provenance"]) > 0

        planner_dir = task_graph_path.parent
        target_summary_path = (
            planner_dir / "target_estimates" / "upmem_path_summary.json"
        )
        assert target_summary_path.exists()
        target_summary = json.loads(target_summary_path.read_text(encoding="utf-8"))

        assert target_summary["scientific_plan_hash"] == row["contraction_plan_hash"]
        assert target_summary["target_id"] == "upmem_dense_gemm"
        assert target_summary["model_id"] == UPMEM_DENSE_MODEL
        assert (
            row["total_host_to_dpu_bytes"] == target_summary["total_host_to_dpu_bytes"]
        )
        assert (
            row["total_dpu_to_host_bytes"] == target_summary["total_dpu_to_host_bytes"]
        )
        assert (
            row["total_mram_to_wram_bytes"]
            == target_summary["total_mram_to_wram_bytes"]
        )
        assert row["unsupported_task_count"] == target_summary["unsupported_tasks"]
        assert (
            row["tiling_required_task_count"]
            == target_summary["tasks_requiring_tiling"]
        )
        assert (
            row["estimated_total_tile_count"]
            == target_summary["total_estimated_tile_count"]
        )
        assert (
            row["estimated_max_parallel_tiles"]
            == target_summary["max_estimated_parallel_tiles"]
        )

        assert (
            sum(r["host_to_dpu_bytes"] for r in jsonl_lines)
            == target_summary["total_host_to_dpu_bytes"]
        )
        assert (
            sum(r["dpu_to_host_bytes"] for r in jsonl_lines)
            == target_summary["total_dpu_to_host_bytes"]
        )
        assert (
            sum(r["mram_to_wram_bytes"] for r in jsonl_lines)
            == target_summary["total_mram_to_wram_bytes"]
        )
        assert (
            sum(r["estimated_tile_count"] for r in jsonl_lines)
            == target_summary["total_estimated_tile_count"]
        )


def test_parse_defensibility_v3_suite_config_and_planners() -> None:
    suite_path = Path("configs/suites/diagnostics/planner_defensibility_v3.yml")
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))

    assert suite["suite_id"] == "planner_defensibility_v3"
    assert suite["metadata"]["execution_tier"] == "modeled_planning_only"

    workloads = suite["workloads"]
    assert len(workloads) == 3
    assert [w["id"] for w in workloads] == [
        "planner_motif_chain",
        "planner_motif_flop_memory_tradeoff",
        "qrng_3q",
    ]

    for motif_id in ("planner_motif_chain", "planner_motif_flop_memory_tradeoff"):
        w = next(item for item in workloads if item["id"] == motif_id)
        assert w["circuit"] == {
            "kind": "planner_motif",
            "name": motif_id.replace("planner_motif_", ""),
        }
        assert w["metadata"] == {
            "workload_type": "synthetic_planner_motif",
            "execution_scope": "model_only",
            "not_real_quantum_circuit": True,
        }

    builtin_w = next(item for item in workloads if item["id"] == "qrng_3q")
    assert builtin_w["circuit"]["kind"] == "builtin"
    assert builtin_w["circuit"]["name"] == "qrng"
    assert builtin_w["circuit"]["n_qubits"] in (3, 4)

    routes = suite["routes"]
    assert len(routes) == 1
    assert routes[0]["role"] == "planning_reference_only"

    planners = suite["planner_comparison"]["planners"]
    assert len(planners) == 8

    cotengra_planners = [p for p in planners if p["engine"] == "cotengra"]
    assert len(cotengra_planners) == 4
    for c_cfg in cotengra_planners:
        assert "seed" in c_cfg and c_cfg["seed"] == 0
        assert "max_repeats" in c_cfg and c_cfg["max_repeats"] == 1
        assert c_cfg["objective"] in {"flops", "size", "write", "combo"}

    exact_planners = [p for p in planners if p["engine"] == "exact_modeled"]
    assert len(exact_planners) == 1
    assert exact_planners[0]["max_input_tensors"] == 6
    assert exact_planners[0]["max_input_tensors"] <= HARD_MAX_INPUT_TENSORS

    custom_planners = [p for p in planners if p["engine"] == "custom_upmem"]
    assert len(custom_planners) == 1
    assert custom_planners[0]["algorithm"] == "greedy"
    assert custom_planners[0]["objective_version"] == "upmem_path_cost_v2"

    opt_planners = [p for p in planners if p["engine"] == "opt_einsum"]
    assert len(opt_planners) == 2
    assert {p["optimize"] for p in opt_planners} == {"greedy", "optimal"}

    for p_cfg in planners:
        planner_inst = planner_from_config(p_cfg)
        identity = planner_inst.identity
        assert identity.planner_engine == p_cfg["engine"]
        assert isinstance(identity.planner_id, str) and len(identity.planner_id) > 0
        assert (
            isinstance(identity.planner_config_hash, str)
            and len(identity.planner_config_hash) > 0
        )
        if p_cfg["engine"] == "cotengra":
            assert identity.options["seed"] == p_cfg["seed"]
            assert identity.options["max_repeats"] == p_cfg["max_repeats"]


def test_defensibility_suite_summary_and_json_modeled_only_no_forbidden_claims(
    tmp_path: Path,
) -> None:
    suite_path = Path("configs/suites/diagnostics/planner_defensibility_v3.yml")
    run_dir = compare_planners(suite_path, tmp_path)

    summary_md = (run_dir / "planner_comparison_summary.md").read_text(encoding="utf-8")
    comparison_json_text = (run_dir / "planner_comparison.json").read_text(
        encoding="utf-8"
    )

    assert "modeled" in summary_md.lower()
    assert "modeled" in comparison_json_text.lower()

    forbidden_words = (
        "hardware performance",
        "speedup",
        "superior",
        "hardware-optimal",
    )
    for word in forbidden_words:
        assert word not in summary_md.lower()
        assert word not in comparison_json_text.lower()
