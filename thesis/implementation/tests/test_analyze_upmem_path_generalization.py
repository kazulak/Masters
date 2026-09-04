from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_upmem_path_generalization.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_upmem_path_generalization", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


@pytest.fixture(autouse=True)
def _clean_reporting_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis, "_source_state", lambda: ("d" * 40, False))


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(analysis._canonical_bytes(value))


def _rewrite_runtime(path: Path, mutate) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fields = tuple(rows[0])
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _candidate(path_id: str, host_bytes: float, *, greedy: bool) -> dict:
    return {
        "candidate_path_id": path_id,
        "source_kind": "fixture",
        "is_greedy": greedy,
        "logical_plan_id": path_id,
        "conventional_features": {
            "flops": host_bytes,
            "macs": host_bytes / 2,
            "peak_intermediate_elements": host_bytes,
            "peak_intermediate_bytes": host_bytes * 16,
            "total_intermediate_writes": host_bytes,
            "maximum_intermediate_rank": 2,
            "contraction_count": 1,
        },
        "topologies": [
            {
                "topology_id": "1dpu_t8",
                "feasible": True,
                "physical_plan_id": path_id,
                "resource_admission": {
                    "collection_resource_admission_passed": True,
                },
                "topology": {
                    "dpu_count": 1,
                    "rank_count": 1,
                    "tasklets_per_dpu": 8,
                },
                "features": {
                    "B_host_dpu": host_bytes,
                    "B_mram_wram": host_bytes,
                    "I_dpu": host_bytes,
                    "N_sync": host_bytes,
                    "E_num": 0,
                    "P_wram": 0,
                },
            }
        ],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    workload_path = tmp_path / "workload.json"
    workload = {
        "workload": [
            {
                "circuit_id": "train_a",
                "family": "family_a",
                "candidate_source": "generalization_v1",
                "split": "training",
                "circuit_definition": {
                    "kind": "builtin",
                    "name": "train_a",
                    "parameters": {"n_qubits": 2},
                },
            },
            {
                "circuit_id": "validation_b",
                "family": "family_b",
                "candidate_source": "generalization_v1",
                "split": "validation",
                "circuit_definition": {
                    "kind": "builtin",
                    "name": "validation_b",
                    "parameters": {"n_qubits": 2},
                },
            },
        ]
    }
    _write_json(workload_path, workload)
    source = "a" * 40
    dataset = {
        "schema_version": "upmem_path_candidate_dataset_v1",
        "source_sha": source,
        "workload_manifest_sha256": hashlib.sha256(
            workload_path.read_bytes()
        ).hexdigest(),
        "circuits": [],
    }
    cell_specs = (
        ("train_a", "training", "family_a", 10.0, 5.0),
        ("validation_b", "validation", "family_b", 8.0, 4.0),
    )
    calibration_cells = []
    runtime_values = {}
    for index, (circuit_id, split, _family, greedy_time, candidate_time) in enumerate(
        cell_specs
    ):
        greedy = f"{index + 1:x}" * 64
        selected = f"{index + 3:x}" * 64
        dataset["circuits"].append(
            {
                "circuit_id": circuit_id,
                "split": split,
                "circuit": {
                    "kind": "builtin",
                    "name": circuit_id,
                    "parameters": {"n_qubits": 2},
                },
                "problem_id": f"{index + 5:x}" * 64,
                "tensor_network_structure_id": f"{index + 7:x}" * 64,
                "candidates": [
                    _candidate(greedy, 100.0, greedy=True),
                    _candidate(selected, 50.0, greedy=False),
                ],
            }
        )
        cell_id = f"{circuit_id}:1dpu_t8"
        calibration_cells.append(
            {
                "cell_id": cell_id,
                "circuit_id": circuit_id,
                "topology_id": "1dpu_t8",
                "greedy_path_id": greedy,
                "candidate_path_ids": [greedy, selected],
                "candidate_roles": [
                    {"role": role, "candidate_path_id": greedy if role == "greedy" else selected}
                    for role in sorted(analysis.CALIBRATION_ROLES)
                ],
            }
        )
        runtime_values[(cell_id, greedy)] = greedy_time
        runtime_values[(cell_id, selected)] = candidate_time
    candidate_path = tmp_path / "candidates.json"
    _write_json(candidate_path, dataset)
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    calibration = {
        "schema_version": "upmem_path_calibration_candidate_set_v1",
        "source_sha": source,
        "candidate_set_sha256": candidate_sha,
        "timing_used_for_selection": False,
        "selection_profile_sha256": "9" * 64,
        "selection_profile_model": {"mode": "grouped"},
        "cells": calibration_cells,
    }
    calibration_path = tmp_path / "calibration.json"
    _write_json(calibration_path, calibration)
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    physical_source = "b" * 40
    experiment_id = "c" * 64
    run_id = "run-fixture"
    runtime_path = tmp_path / "runtime.csv"
    fields = (
        "split",
        "attempt_type",
        "cell_id",
        "circuit_id",
        "candidate_path_id",
        "block",
        "total_wall_s",
        "candidate_set_sha256",
        "calibration_set_sha256",
        "candidate_generation_source_sha",
        "physical_execution_source_sha",
        "experiment_id",
        "run_id",
        "timing_scope",
        "status",
        "validation",
        "fallback",
        "topology_id",
        "route_id",
        "plan_id",
        "source_sha",
        "problem_id",
        "tensor_network_structure_id",
        "logical_plan_id",
        "physical_plan_id",
        "output_sha256",
        "request_transport",
        "requested_dpus",
        "allocated_dpus",
        "tasklets_per_dpu",
        "rank_count",
        "collection_resource_admission_passed",
        "execution_resource_admission_passed",
        "startup_resource_admission_passed",
        "physical_target_verified",
        "hardware_kernel_executed",
        "simulator_kernel_executed",
        "cpu_fallback_used",
        "binary_identity_verified",
        "native_identity_verified",
        "hardware_release_verified",
        "full_precision_passed",
        "policy_reference_passed",
    )
    with runtime_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in calibration_cells:
            circuit_id = cell["circuit_id"]
            split = next(
                item["split"]
                for item in dataset["circuits"]
                if item["circuit_id"] == circuit_id
            )
            for candidate_id in cell["candidate_path_ids"]:
                base = runtime_values[(cell["cell_id"], candidate_id)]
                for attempt, block in (("warmup", 0), ("measurement", 1), ("measurement", 2), ("measurement", 3)):
                    writer.writerow(
                        {
                            "split": split,
                            "attempt_type": attempt,
                            "cell_id": cell["cell_id"],
                            "circuit_id": circuit_id,
                            "candidate_path_id": candidate_id,
                            "block": block,
                            "total_wall_s": base + (0.01 * block),
                            "candidate_set_sha256": candidate_sha,
                            "calibration_set_sha256": calibration_sha,
                            "candidate_generation_source_sha": source,
                            "physical_execution_source_sha": physical_source,
                            "experiment_id": experiment_id,
                            "run_id": run_id,
                            "timing_scope": "steady_execution_v1",
                            "status": "success",
                            "validation": "passed",
                            "fallback": "false",
                            "topology_id": "1dpu_t8",
                            "route_id": "1dpu_t8",
                            "plan_id": f"path_{candidate_id}",
                            "source_sha": source,
                            "problem_id": next(
                                item["problem_id"]
                                for item in dataset["circuits"]
                                if item["circuit_id"] == circuit_id
                            ),
                            "tensor_network_structure_id": next(
                                item["tensor_network_structure_id"]
                                for item in dataset["circuits"]
                                if item["circuit_id"] == circuit_id
                            ),
                            "logical_plan_id": candidate_id,
                            "physical_plan_id": candidate_id,
                            "output_sha256": hashlib.sha256(circuit_id.encode()).hexdigest(),
                            "request_transport": "packed_operation_v1",
                            "requested_dpus": 1,
                            "allocated_dpus": 1,
                            "tasklets_per_dpu": 8,
                            "rank_count": 1,
                            "collection_resource_admission_passed": "true",
                            "execution_resource_admission_passed": "true",
                            "startup_resource_admission_passed": "true",
                            "physical_target_verified": "true",
                            "hardware_kernel_executed": "true",
                            "simulator_kernel_executed": "false",
                            "cpu_fallback_used": "false",
                            "binary_identity_verified": "true",
                            "native_identity_verified": "true",
                            "hardware_release_verified": "true",
                            "full_precision_passed": "true",
                            "policy_reference_passed": "true",
                        }
                    )
    summary_path = tmp_path / "runtime.json"
    _write_json(
        summary_path,
        {
            "schema_version": "upmem_path_runtime_calibration_v1",
            "candidate_set_sha256": candidate_sha,
            "calibration_set_sha256": calibration_sha,
            "candidate_generation_source_sha": source,
            "physical_execution_source_sha": physical_source,
            "numeric_policy": "split_complex_float32_v1",
            "request_transport": "packed_operation_v1",
            "timing_scope": "steady_execution_v1",
            "claim_policy": "diagnostic_v1",
            "sample_count": 16,
            "session_count": 16,
            "expected_candidate_cell_count": 4,
            "expected_cell_count": 2,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "fallback_used": False,
            "all_successful_physical_sessions": True,
            "all_resource_admission_passed": True,
            "all_accuracy_qualified": True,
        },
    )
    return (
        candidate_path,
        calibration_path,
        runtime_path,
        summary_path,
        workload_path,
    )


def test_analysis_is_fold_isolated_and_freezes_grouped_profile(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "output"
    result = analysis.analyze(
        *paths,
        output,
        weight_samples=16,
        weight_seed=7,
        bootstrap_count=20,
        bootstrap_seed=9,
    )

    assert result["pilot_rows_used"] == 0
    assert result["test_rows_used"] == 0
    assert result["selected_model_form"] == "grouped"
    assert len(result["family_fold_fits"]) == 4
    assert all(
        not set(row["training_cell_ids"]) & set(row["held_out_cell_ids"])
        for row in result["family_fold_fits"]
    )
    profile = json.loads(
        (output / "upmem_thesis_workload_float32_pretest_v1.json").read_text()
    )
    assert profile["profile_id"] == "upmem_thesis_workload_float32_pretest_v1"
    assert profile["final_test_timing_used"] is False
    with (output / "oracle_headroom.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert all(float(row["oracle_regret"]) == pytest.approx(1.0) for row in rows)
    assert all(float(row["captured_headroom"]) == pytest.approx(1.0) for row in rows)
    with (output / "candidate_uncertainty.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        uncertainty = list(csv.DictReader(stream))
    assert len(uncertainty) == 4
    assert all(row["measurement_count"] == "3" for row in uncertainty)
    with (output / "weight_stability.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        stability = list(csv.DictReader(stream))
    assert len(stability) == 40


def test_analysis_rejects_pilot_runtime_rows(tmp_path: Path) -> None:
    paths = list(_fixture(tmp_path))
    runtime_path = paths[2]
    rows = runtime_path.read_text(encoding="utf-8").splitlines()
    values = rows[1].split(",")
    values[0] = "pilot_development"
    rows[1] = ",".join(values)
    runtime_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pilot, test, or unknown"):
        analysis.analyze(
            *paths,
            tmp_path / "output",
            weight_samples=8,
            weight_seed=1,
            bootstrap_count=10,
            bootstrap_seed=2,
        )


def test_analysis_is_independent_of_runtime_row_order(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = tmp_path / "first"
    analysis.analyze(
        *paths,
        first,
        weight_samples=16,
        weight_seed=7,
        bootstrap_count=20,
        bootstrap_seed=9,
    )
    _rewrite_runtime(paths[2], lambda rows: rows.reverse())
    second = tmp_path / "second"
    analysis.analyze(
        *paths,
        second,
        weight_samples=16,
        weight_seed=7,
        bootstrap_count=20,
        bootstrap_seed=9,
    )
    for filename in (
        "leave_one_family_out.csv",
        "leave_one_instance_out.csv",
        "model_comparison.csv",
        "candidate_uncertainty.csv",
        "weight_stability.csv",
        "path_selection_stability.csv",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_analysis_rejects_forged_physical_plan_identity(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _rewrite_runtime(
        paths[2], lambda rows: rows[0].__setitem__("physical_plan_id", "0" * 64)
    )
    with pytest.raises(ValueError, match="physical_plan_id identity mismatch"):
        analysis.analyze(
            *paths,
            tmp_path / "output",
            weight_samples=8,
            weight_seed=1,
            bootstrap_count=10,
            bootstrap_seed=2,
        )


def test_analysis_rejects_manifest_split_reclassification(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    workload = json.loads(paths[4].read_text(encoding="utf-8"))
    workload["workload"][0]["split"] = "test"
    _write_json(paths[4], workload)
    with pytest.raises(ValueError, match="workload manifest"):
        analysis.analyze(
            *paths,
            tmp_path / "output",
            weight_samples=8,
            weight_seed=1,
            bootstrap_count=10,
            bootstrap_seed=2,
        )


def test_analysis_accepts_noncanonical_json_serialization(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    candidate = json.loads(paths[0].read_text(encoding="utf-8"))
    paths[0].write_text(json.dumps(candidate), encoding="ascii")
    result = analysis.analyze(
        *paths,
        tmp_path / "output",
        weight_samples=8,
        weight_seed=1,
        bootstrap_count=10,
        bootstrap_seed=2,
    )
    assert result["development_cell_count"] == 2


def test_analysis_rejects_dirty_reporting_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(analysis, "_source_state", lambda: ("d" * 40, True))
    with pytest.raises(ValueError, match="clean committed worktree"):
        analysis.analyze(
            *paths,
            tmp_path / "output",
            weight_samples=8,
            weight_seed=1,
            bootstrap_count=10,
            bootstrap_seed=2,
        )


def test_identifiability_screen_excludes_redundant_and_constant_terms() -> None:
    greedy_id = "a" * 64
    other_id = "b" * 64
    cell = analysis.TrainingCell(
        cell_id="fixture:1dpu_t8",
        topology="1dpu_t8",
        candidates=(
            analysis._candidate(_candidate(greedy_id, 100.0, greedy=True)),
            analysis._candidate(_candidate(other_id, 50.0, greedy=False)),
        ),
        greedy_path_id=greedy_id,
    )
    model = analysis._identifiable_model(
        {cell.cell_id: cell}, {cell.cell_id}, "six_term"
    )
    assert model.active_features == ("B_host_dpu",)
    assert set(model.zero_range_features) == {"E_num", "P_wram"}


def test_no_measurable_headroom_is_reported_as_null() -> None:
    greedy_id = "a" * 64
    other_id = "b" * 64
    cell = analysis.TrainingCell(
        cell_id="fixture:1dpu_t8",
        topology="1dpu_t8",
        candidates=(
            analysis._candidate(_candidate(greedy_id, 100.0, greedy=True)),
            analysis._candidate(_candidate(other_id, 50.0, greedy=False)),
        ),
        greedy_path_id=greedy_id,
    )
    fit = analysis.fit_weights(
        (cell,),
        tuple(
            analysis.RuntimeMeasurement(
                cell_id=cell.cell_id,
                candidate_id=candidate_id,
                runtime_s=value,
                observation_id=str(block),
            )
            for candidate_id, values in (
                (greedy_id, (10.0, 10.2, 9.8)),
                (other_id, (10.0, 10.2, 9.8)),
            )
            for block, value in enumerate(values, start=1)
        ),
        model=analysis._identifiable_model(
            {cell.cell_id: cell}, {cell.cell_id}, "grouped"
        ),
        random_sample_count=4,
        seed=3,
    )
    rows = [
        {
            "attempt_type": "measurement",
            "candidate_path_id": candidate_id,
            "total_wall_s": str(value),
        }
        for candidate_id, values in (
            (greedy_id, (10.0, 10.2, 9.8)),
            (other_id, (10.0, 10.2, 9.8)),
        )
        for block, value in enumerate(values, start=1)
    ]
    for block, row in enumerate(rows):
        row["block"] = str((block % 3) + 1)
    result = analysis._cell_metrics(
        cell,
        rows,
        fit,
        fold_kind="fixture",
        fold_id="fixture",
        circuit_id="fixture",
        family="fixture",
    )
    assert result["captured_headroom"] is None
    assert result["classification"] == "neutral"
