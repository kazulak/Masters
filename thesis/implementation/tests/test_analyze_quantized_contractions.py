from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from quantum_bench.model import ContractNode, ContractionDAG, TensorSpec, TensorView


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_quantized_contractions.py"
SPEC = importlib.util.spec_from_file_location("analyze_quantized_contractions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def _matrix_dag() -> tuple[ContractionDAG, dict[str, np.ndarray]]:
    left = TensorSpec("left", (0, 1), (2, 3), "dense", dtype="complex128")
    right = TensorSpec("right", (1, 2), (3, 2), "dense", dtype="complex128")
    output = TensorSpec(
        "out", (0, 2), (2, 2), "dense", dtype="complex128", produced_by="contract"
    )
    node = ContractNode(
        node_id="contract",
        left=TensorView(tensor_id="left", labels=left.labels, shape=left.shape),
        right=TensorView(tensor_id="right", labels=right.labels, shape=right.shape),
        output=output,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    return (
        ContractionDAG(
            tensors=(left, right),
            nodes=(node,),
            output=TensorView(tensor_id="out", labels=output.labels, shape=output.shape),
        ),
        {
            "left": np.array(
                [[1 + 2j, -2 + 0.5j, 3 - 1j], [0.25 - 1j, 4, -1 + 2j]],
                dtype=np.complex128,
            ),
            "right": np.array(
                [[2 - 1j, 1 + 3j], [-2 + 2j, 3], [1 - 1j, -0.5 + 2j]],
                dtype=np.complex128,
            ),
        },
    )


def test_accumulator_bound_uses_full_complex_component_and_has_no_magic_limit() -> None:
    safe = (np.iinfo(np.int32).max // (2 * 127**2))
    assert analyzer.int32_accumulator_bound(safe) <= np.iinfo(np.int32).max
    assert analyzer.int32_accumulator_safe(safe)
    assert not analyzer.int32_accumulator_safe(safe + 1)
    with pytest.raises(ValueError, match="positive"):
        analyzer.int32_accumulator_bound(0)


def test_injected_analysis_has_separate_local_and_cumulative_trace() -> None:
    dag, inputs = _matrix_dag()
    report = analyzer.analyze_suite(
        cases=()
    )
    assert report["summary"]["suite"]["case_count"] == 0

    analysis = analyzer.analyze_dag("injected", dag, inputs, logical_plan_id="plan")
    assert analysis["contraction_count"] == 1
    assert len(analysis["nodes"]) == 1
    node = analysis["nodes"][0]
    assert node["node_id"] == "contract"
    assert node["M"] == 2
    assert node["N"] == 2
    assert node["K"] == 3
    assert node["int32_accumulator_safe"] is True
    assert node["scale_computation_count"] == 2
    assert node["quantization_event_count"] == 2
    assert node["requantization_event_count"] == 0
    assert node["logical_encoded_bytes"] == 2 * (6 + 6) + 16
    assert "local_output_max_abs_error_vs_same_node_float32" in node
    assert "cumulative_output_max_abs_error_vs_same_node_float32" in node


def test_serialization_is_exactly_three_files_and_byte_identical(tmp_path: Path) -> None:
    dag, inputs = _matrix_dag()
    analysis = analyzer.analyze_dag("injected", dag, inputs, logical_plan_id="plan")
    summary = analyzer._circuit_summary(analysis)
    report = {
        "summary": {
            "schema_version": analyzer.SCHEMA_VERSION,
            "policy_id": analyzer.POLICY_ID,
            "accuracy_status": "accuracy_unqualified",
            "execution": {
                "scope": "cpu_only_software_analysis",
                "planner_engine": "opt_einsum",
                "planner_mode": "greedy",
                "host_timing_recorded": False,
                "physical_execution": False,
                "simulator_execution": False,
            },
            "claim_boundary": analyzer.CLAIM_BOUNDARY,
            "suite": {"fixed_deterministic": False, "case_count": 1, "circuit_ids": ["injected"]},
            "artifacts": {
                "summary": "quantization_summary.json",
                "nodes": "quantization_nodes.csv",
                "circuits": "quantization_circuits.csv",
                "node_row_count": 1,
                "node_columns": list(analyzer.NODE_COLUMNS),
                "circuit_columns": list(analyzer.CIRCUIT_COLUMNS),
            },
            "circuits": [summary],
        },
        "circuit_rows": [analyzer._circuit_row(summary)],
        "node_rows": [{"circuit_id": "injected", "logical_plan_id": "plan", **analysis["nodes"][0]}],
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    analyzer.write_outputs(report, first)
    analyzer.write_outputs(report, second)
    assert sorted(path.name for path in first.iterdir()) == [
        "quantization_circuits.csv",
        "quantization_nodes.csv",
        "quantization_summary.json",
    ]
    for name in ("quantization_summary.json", "quantization_nodes.csv", "quantization_circuits.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    summary_data = json.loads((first / "quantization_summary.json").read_text())
    assert summary_data["accuracy_status"] == "accuracy_unqualified"
    assert summary_data["claim_boundary"]["forbidden_claims"]
    with (first / "quantization_nodes.csv").open(newline="") as handle:
        assert tuple(next(csv.reader(handle))) == analyzer.NODE_COLUMNS
    with (first / "quantization_circuits.csv").open(newline="") as handle:
        assert tuple(next(csv.reader(handle))) == analyzer.CIRCUIT_COLUMNS


def test_fixed_suite_names_are_stable() -> None:
    assert tuple(case.circuit_id for case in analyzer.FIXED_SUITE) == (
        "bell2",
        "stress4_l2",
        "ghz18",
        "hs18_d1",
        "stress18_l2",
    )
    assert analyzer.POLICY_ID == "complex_int8_shared_scale_v1"
