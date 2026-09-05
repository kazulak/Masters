from __future__ import annotations

from dataclasses import replace
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from quantum_bench.model import (
    ContractNode,
    ContractionDAG,
    ReduceNode,
    TensorSpec,
    TensorView,
)
from quantum_bench.upmem.plan import UpmemWorkUnit
from quantum_bench.upmem.protocol import MRAM_POOL_BYTES


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "characterize_upmem_frontiers.py"
SPEC = importlib.util.spec_from_file_location("upmem_frontiers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
frontiers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontiers)


def _spec(tensor_id: str, labels: tuple[int, ...], shape: tuple[int, ...]) -> TensorSpec:
    return TensorSpec(tensor_id, labels, shape, "dense", dtype="complex128")


def _contract(
    node_id: str,
    left_id: str,
    right_id: str,
    output_id: str,
    left_labels: tuple[int, ...],
    left_shape: tuple[int, ...],
    right_labels: tuple[int, ...],
    right_shape: tuple[int, ...],
    output_labels: tuple[int, ...],
    output_shape: tuple[int, ...],
    dependencies: tuple[str, ...] = (),
) -> ContractNode:
    contracted = tuple(
        label
        for label in left_labels
        if label in right_labels and label not in output_labels
    )
    return ContractNode(
        node_id=node_id,
        left=TensorView(tensor_id=left_id, labels=left_labels, shape=left_shape),
        right=TensorView(tensor_id=right_id, labels=right_labels, shape=right_shape),
        output=_spec(output_id, output_labels, output_shape),
        contracted_labels=contracted,
        output_labels=output_labels,
        dependencies=dependencies,
    )


def _chain_dag() -> ContractionDAG:
    first = _contract(
        "first",
        "a",
        "b",
        "p",
        (0, 1),
        (2, 3),
        (1, 2),
        (3, 4),
        (0, 2),
        (2, 4),
    )
    second = _contract(
        "second",
        "p",
        "c",
        "out",
        (0, 2),
        (2, 4),
        (2, 3),
        (4, 5),
        (0, 3),
        (2, 5),
        dependencies=("first",),
    )
    return ContractionDAG(
        tensors=(
            _spec("a", (0, 1), (2, 3)),
            _spec("b", (1, 2), (3, 4)),
            _spec("c", (2, 3), (4, 5)),
        ),
        nodes=(first, second),
        output=TensorView(tensor_id="out", labels=(0, 3), shape=(2, 5)),
    )


def _fork_join_dag() -> ContractionDAG:
    left = _contract(
        "left",
        "a0",
        "a1",
        "p",
        (0, 1),
        (2, 3),
        (1, 2),
        (3, 2),
        (0, 2),
        (2, 2),
    )
    right = _contract(
        "right",
        "b0",
        "b1",
        "q",
        (0, 3),
        (2, 3),
        (3, 2),
        (3, 2),
        (0, 2),
        (2, 2),
    )
    join = ReduceNode(
        node_id="join",
        inputs=(
            TensorView(tensor_id="p", labels=(0, 2), shape=(2, 2)),
            TensorView(tensor_id="q", labels=(0, 2), shape=(2, 2)),
        ),
        output=_spec("result", (0, 2), (2, 2)),
        dependencies=("left", "right"),
    )
    return ContractionDAG(
        tensors=(
            _spec("a0", (0, 1), (2, 3)),
            _spec("a1", (1, 2), (3, 2)),
            _spec("b0", (0, 3), (2, 3)),
            _spec("b1", (3, 2), (3, 2)),
        ),
        nodes=(left, right, join),
        output=TensorView(tensor_id="result", labels=(0, 2), shape=(2, 2)),
    )


def _uneven_dag() -> ContractionDAG:
    root_a = _contract(
        "root_a",
        "a0",
        "a1",
        "p",
        (0, 1),
        (2, 2),
        (1, 2),
        (2, 2),
        (0, 2),
        (2, 2),
    )
    root_b = _contract(
        "root_b",
        "b0",
        "b1",
        "q",
        (0, 1),
        (2, 2),
        (1, 2),
        (2, 2),
        (0, 2),
        (2, 2),
    )
    branch_a = _contract(
        "branch_a",
        "p",
        "a2",
        "r",
        (0, 2),
        (2, 2),
        (2, 3),
        (2, 2),
        (0, 3),
        (2, 2),
        dependencies=("root_a",),
    )
    branch_b = _contract(
        "branch_b",
        "q",
        "b2",
        "s",
        (0, 2),
        (2, 2),
        (2, 5),
        (2, 2),
        (0, 5),
        (2, 2),
        dependencies=("root_b",),
    )
    long_branch = _contract(
        "long_branch",
        "r",
        "a3",
        "t",
        (0, 3),
        (2, 2),
        (3, 5),
        (2, 2),
        (0, 5),
        (2, 2),
        dependencies=("branch_a",),
    )
    join = ReduceNode(
        node_id="join",
        inputs=(
            TensorView(tensor_id="t", labels=(0, 5), shape=(2, 2)),
            TensorView(tensor_id="s", labels=(0, 5), shape=(2, 2)),
        ),
        output=_spec("result", (0, 5), (2, 2)),
        dependencies=("long_branch", "branch_b"),
    )
    return ContractionDAG(
        tensors=tuple(
            _spec(tensor_id, labels, shape)
            for tensor_id, labels, shape in (
                ("a0", (0, 1), (2, 2)),
                ("a1", (1, 2), (2, 2)),
                ("a2", (2, 3), (2, 2)),
                ("a3", (3, 5), (2, 2)),
                ("b0", (0, 1), (2, 2)),
                ("b1", (1, 2), (2, 2)),
                ("b2", (2, 5), (2, 2)),
            )
        ),
        nodes=(root_a, root_b, branch_a, branch_b, long_branch, join),
        output=TensorView(tensor_id="result", labels=(0, 5), shape=(2, 2)),
    )


def test_frozen_cells_and_pool_hashes_are_unchanged() -> None:
    cells, pools = frontiers.frozen_cells()

    assert len(cells) == 40
    assert sum(cell["logical_plan_id"] is None for cell in cells) == 4
    assert {cell["cell_id"] for cell in cells} == {
        cell["cell_id"] for cell in frontiers.frozen_cells()[0]
    }
    assert pools == {
        "generalization": {
            "path": "thesis_results/upmem_path_heuristic_generalization_v1/software/candidate_paths.json",
            "sha256": "d95150ddf89f6aafa861000b0db2d8447d64456a035c5404463a878c3a319049",
        },
        "pilot": {
            "path": "thesis_results/upmem_path_heuristic_v1/software/candidate_paths.json",
            "sha256": "5269bfea2a1777b041e10edf9618ba613d44021ce89b3ff0d95093a81c095b65",
        },
    }


@pytest.mark.parametrize(
    ("dag", "expected_cohorts"),
    (
        (_chain_dag(), (("first",), ("second",))),
        (_fork_join_dag(), (("left", "right"), ("join",))),
        (
            _uneven_dag(),
            (("root_a", "root_b"), ("branch_a", "branch_b"), ("long_branch",), ("join",)),
        ),
    ),
)
def test_dependency_ready_cohorts_and_operation_edges(
    dag: ContractionDAG, expected_cohorts: tuple[tuple[str, ...], ...]
) -> None:
    facts = frontiers.characterize_dag(dag)

    assert tuple(tuple(row) for row in facts["dependency_ready_cohorts"]) == expected_cohorts
    assert tuple(facts["frontier_widths"]) == tuple(map(len, expected_cohorts))
    operations = {row["node_id"]: row for row in facts["operations"]}
    for node_id, operation in operations.items():
        assert operation["predecessors"] == facts["predecessors"][node_id]
        assert operation["successors"] == facts["successors"][node_id]
        assert operation["measured_timing"] is None
        assert operation["measured_timing_reason"] == frontiers.TIMING_REASON


def test_critical_path_uses_four_product_real_mac_work() -> None:
    facts = frontiers.characterize_dag(_chain_dag())

    assert facts["critical_path"]["node_ids"] == ["first", "second"]
    assert facts["critical_path"]["real_mac_work"] == 4 * (2 * 4 * 3 + 2 * 5 * 4)
    assert facts["total_real_mac_work"] == facts["critical_path"]["real_mac_work"]


def test_dependency_validation_rejects_duplicate_missing_and_cyclic_edges() -> None:
    dag = _chain_dag()
    first, second = dag.nodes
    assert isinstance(first, ContractNode)
    assert isinstance(second, ContractNode)

    with pytest.raises(ValueError, match="duplicate DAG node ID"):
        frontiers.validate_dependency_graph(replace(dag, nodes=(first, first)))
    with pytest.raises(ValueError, match="missing dependency"):
        frontiers.validate_dependency_graph(
            replace(dag, nodes=(first, replace(second, dependencies=("missing",))))
        )
    with pytest.raises(ValueError, match="duplicate dependency"):
        frontiers.validate_dependency_graph(
            replace(dag, nodes=(first, replace(second, dependencies=("first", "first"))))
        )
    with pytest.raises(ValueError, match="cyclic"):
        frontiers.validate_dependency_graph(
            replace(
                dag,
                nodes=(
                    replace(first, dependencies=("second",)),
                    replace(second, dependencies=("first",)),
                ),
            )
        )


def test_fused_four_product_mram_uses_existing_512_kib_limit() -> None:
    unit = UpmemWorkUnit(
        node_id="small",
        stable_tile_id="small:0",
        wave=0,
        logical_rank=0,
        logical_dpu=0,
        batch_start=0,
        batch_size=1,
        m_start=0,
        m_size=128,
        n_start=0,
        n_size=128,
        k_start=0,
        k_size=128,
        estimated_input_bytes=1,
        estimated_output_bytes=1,
        aligned_mram_bytes=1,
        estimated_arithmetic_work=1,
    )
    boundary = frontiers.fused_four_product_mram(unit, frontiers.POLICIES[0])
    assert boundary["fused_four_product_live_mram_bytes"] == MRAM_POOL_BYTES
    assert boundary["fused_four_product_admitted"] is True
    assert boundary["mram_reservation_components"] == ["2A", "2B", "4C"]
    assert boundary["control_completion_mram_reserved_bytes"] == 0
    assert boundary["control_completion_memory_scope"] == "WRAM_outside_MRAM_reservation_v1"
    assert boundary["retile_requested"] is False
    assert boundary["fusion_route"] == "fused_four_product_generic_upmem"

    oversized = replace(unit, m_size=129)
    rejected = frontiers.fused_four_product_mram(oversized, frontiers.POLICIES[0])
    assert rejected["fused_four_product_live_mram_bytes"] > MRAM_POOL_BYTES
    assert rejected["fused_four_product_admitted"] is False
    assert rejected["fusion_route"] == "generic_upmem_no_retile"
    assert rejected["nonfit_fallback"] == "generic_upmem_no_retile"
    assert rejected["retile_requested"] is False


def test_resident_pairs_are_unqualified_memory_candidates() -> None:
    cell = frontiers.frozen_cells()[0][0]
    result = frontiers.characterize_frontier_cell(cell)
    pairs = result["resident_pairs"]

    assert pairs
    assert all(pair["admitted"] is False for pair in pairs)
    candidate = next(pair for pair in pairs if pair["memory_candidate"])
    assert candidate["admission_status"] == "unqualified_memory_candidate"
    assert candidate["resident_pair_scope"] == "bounded_memory_candidate_only_v1"
    assert candidate["memory_candidate_reasons"] == []
    assert {
        "same_dpu_locality_unverified",
        "full_intermediate_layout_unverified",
        "intermediate_reconstruction_unverified",
        "no_split_k_unverified",
        "scale_handling_unverified",
    } <= set(candidate["admission_reasons"])
    assert candidate["same_dpu_locality_verified"] is False
    assert candidate["full_intermediate_layout_verified"] is False
    assert candidate["no_split_k_verified"] is False
    assert candidate["scale_handling_verified"] is False
    assert candidate["numeric_policy"] == frontiers.RESIDENT_PAIR_NUMERIC_POLICY

    int8_facts = frontiers.characterize_dag(
        _chain_dag(), numeric_policy="complex_int8_shared_scale_v1"
    )
    assert int8_facts["resident_pairs"]
    assert all(
        "numeric_policy_not_float32_initial_probe" in pair["memory_candidate_reasons"]
        for pair in int8_facts["resident_pairs"]
    )


def test_liveness_is_partial_logical_payload_accounting_not_whole_host_admission() -> None:
    facts = frontiers.characterize_dag(_chain_dag())
    liveness = facts["liveness"]
    actual = liveness["actual_retained_runtime"]

    assert liveness["memory_scope"] == (
        "partial logical tensor payload accounting, not whole-host admission/RSS"
    )
    assert liveness["whole_host_admission"] is False
    assert actual["estimate_kind"] == "partial_logical_tensor_payload_accounting"
    assert actual["process_memory_lower_bound_claimed"] is False
    assert actual["numpy_storage_aliases_deduplicated"] is False
    assert actual["logical_nbytes_alias_double_count_possible"] is True
    assert actual["whole_host_admission"] is False
    assert set(actual["excluded_retained_categories"]) == {
        "encoded_operands",
        "raw_lane_values",
        "transport_copies",
        "runtime_object_overheads",
    }
    assert "process RSS" in actual["accounting_caveats"][1]


def test_excluded_cell_short_circuits_before_lowering(monkeypatch) -> None:
    excluded = next(
        cell for cell in frontiers.frozen_cells()[0] if cell["logical_plan_id"] is None
    )

    def fail_lowering(*args, **kwargs):
        raise AssertionError("excluded cell must not lower its path")

    monkeypatch.setattr(frontiers, "lower_tensor_network", fail_lowering)
    result = frontiers.characterize_frontier_cell(excluded)

    assert result["status"] == "rejected"
    assert result["reconstruction_performed"] is False
    assert result["frontier_census"] is None
    assert result["frontier_census_reason"] == "no_logical_plan_identity"


def test_eligible_reconstruction_uses_bounded_worker_timeout(monkeypatch) -> None:
    cell = frontiers.frozen_cells()[0][0]

    def timeout(*args, **kwargs):
        raise frontiers.subprocess.TimeoutExpired(kwargs.get("args", args[0]), 60.0)

    monkeypatch.setattr(frontiers.subprocess, "run", timeout)
    result = frontiers.isolated_frontier_cell(cell, timeout=60.0)

    assert result["status"] == "rejected"
    assert result["rejection_reasons"] == ["lowering_timeout"]
    assert result["frontier_census"] is None
    assert result["frontier_census_reason"] == "lowering_timeout"


def test_existing_cell_reconstructs_exact_plan_and_keeps_source_only_boundary() -> None:
    cell = frontiers.frozen_cells()[0][0]
    result = frontiers.characterize_frontier_cell(cell)

    assert result["status"] == "eligible"
    assert result["physical_plan_id"]
    assert result["frontier_census"]["fused_four_product_mram"]["plan_backed"] is True
    assert result["measured_timing"] is None
    assert result["measured_timing_reason"] == frontiers.TIMING_REASON
    assert result["cpu_fallback_used"] is False
    assert result["hardware_execution"] is False
    assert result["liveness"]["actual_retained_runtime"]["peak_live_tensor_bytes"] >= (
        result["liveness"]["minimal_theoretical"]["peak_live_tensor_bytes"]
    )
    assert result["liveness"]["actual_retained_runtime"]["estimate_kind"] == (
        "partial_logical_tensor_payload_accounting"
    )
    assert result["liveness"]["whole_host_admission"] is False
    assert result["frontier_census"]["k_accumulation"]["implicit_resident_k"] is False
    assert result["fused_four_product_mram"]["reservation_components"] == [
        "2A",
        "2B",
        "4C",
    ]
    assert result["fused_four_product_mram"]["control_completion_mram_reserved_bytes"] == 0
    assert result["fused_four_product_mram"]["retile_requested"] is False
    assert result["per_operation"]
    assert all(operation["measured_timing"] is None for operation in result["per_operation"])

    excluded = frontiers.frozen_cells()[0][28]
    excluded_result = frontiers.characterize_frontier_cell(excluded)
    assert excluded_result["status"] == "rejected"
    assert excluded_result["rejection_reasons"] == [
        "retained_candidate_has_no_logical_identity"
    ]
    assert excluded_result["frontier_census"] is None


def test_cli_writer_emits_ignored_run_json_and_csv_without_execution(monkeypatch, tmp_path: Path) -> None:
    cells, pools = frontiers.frozen_cells()

    def fake_characterize(cell: dict[str, object]) -> dict[str, object]:
        excluded = cell["logical_plan_id"] is None
        graph = {
            "frontier_widths": [],
            "maximum_frontier_width": 0,
            "critical_path": {"real_mac_work": 0},
            "liveness": {
                "minimal_theoretical": {"peak_live_tensor_bytes": 0},
                "actual_retained_runtime": {"peak_live_tensor_bytes": 0},
            },
            "fused_four_product_mram": {
                "peak_live_mram_bytes": None,
                "all_contract_operations_admitted": None,
            },
            "operations": [],
            "geometry_category_counts": {},
        }
        result = {
            **cell,
            "status": "rejected" if excluded else "eligible",
            "rejection_reasons": ["retained_candidate_has_no_logical_identity"]
            if excluded
            else [],
            "frontier_census": graph,
            "critical_path_real_mac_work": 0,
            "geometry_category_counts": {},
        }
        if excluded:
            result["frontier_census"] = None
            result["frontier_census_reason"] = "no_logical_plan_identity"
        return result

    monkeypatch.setattr(frontiers, "frozen_cells", lambda: (cells, pools))
    monkeypatch.setattr(frontiers, "characterize_frontier_cell", fake_characterize)
    monkeypatch.setattr(frontiers, "isolated_frontier_cell", fake_characterize)
    report = frontiers.write_census(tmp_path / "run")

    output = tmp_path / "run"
    assert report["cell_count"] == 40
    assert report["exclusion_count"] == 4
    assert json.loads((output / "upmem_frontier_census.json").read_text())["timing"] == {
        "measured": None,
        "reason": frontiers.TIMING_REASON,
    }
    assert (output / "upmem_frontier_census.csv").exists()
    assert (output / "selected_paths.json").exists()
    assert (output / "SHA256SUMS").exists()
    with (output / "upmem_frontier_census.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 40
    excluded_rows = [
        row
        for row in rows
        if row.get("frontier_census_reason") == "no_logical_plan_identity"
    ]
    assert len(excluded_rows) == 4
    assert all(row["row_kind"] == "cell" for row in excluded_rows)
