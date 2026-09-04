#!/usr/bin/env python3
"""Analyze the frozen UPMEM path-heuristic generalization calibration."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
from typing import Any

import numpy as np

from quantum_bench.upmem.path_heuristic import (
    COST_MODEL_ID,
    FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    ConventionalPathFeatures,
    FeatureModelDecision,
    PathCandidate,
    RawFeatureVector,
    RuntimeMeasurement,
    TrainingCell,
    WeightFitResult,
    fit_weights,
    geometric_mean,
    normalize_features,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_FORMS = ("six_term", "grouped")
NORMALIZATION = "log((candidate+1)/(greedy+1))"
DEVELOPMENT_SPLITS = frozenset({"training", "validation"})
OUTPUT_COLUMNS = (
    "fold_kind",
    "fold_id",
    "model_form",
    "circuit_id",
    "family",
    "topology_id",
    "greedy_path_id",
    "selected_path_id",
    "oracle_path_id",
    "greedy_median_s",
    "selected_median_s",
    "oracle_median_s",
    "greedy_mad_s",
    "selected_mad_s",
    "oracle_mad_s",
    "greedy_min_s",
    "selected_min_s",
    "oracle_min_s",
    "greedy_max_s",
    "selected_max_s",
    "oracle_max_s",
    "measurement_count",
    "speedup",
    "minimum_candidate_rank",
    "top_1",
    "top_3",
    "oracle_regret",
    "greedy_regret",
    "oracle_speedup",
    "captured_headroom",
    "score_runtime_spearman",
    "score_runtime_kendall_tau_b",
    "classification",
)
CANDIDATE_STAT_COLUMNS = (
    "cell_id",
    "circuit_id",
    "family",
    "split",
    "topology_id",
    "candidate_path_id",
    "measurement_count",
    "median_s",
    "mad_s",
    "minimum_s",
    "maximum_s",
    "median_bootstrap_low_s",
    "median_bootstrap_high_s",
)
CALIBRATION_ROLES = frozenset(
    {
        "greedy",
        "minimum_flops",
        "minimum_peak_intermediate",
        "minimum_writes",
        "frozen_v1_selected",
        "feature_diverse",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_state() -> tuple[str, bool]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or dirty.returncode or len(value) != 40:
        raise ValueError("cannot determine reporting source SHA")
    return value, bool(dirty.stdout.strip())


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _candidate(record: dict[str, Any]) -> PathCandidate:
    features = []
    feasible = []
    for topology in record["topologies"]:
        if topology["feasible"]:
            topology_id = str(topology["topology_id"])
            features.append(
                (topology_id, RawFeatureVector.from_mapping(topology["features"]))
            )
            feasible.append(topology_id)
    return PathCandidate(
        path_id=str(record["candidate_path_id"]),
        conventional=ConventionalPathFeatures(**record["conventional_features"]),
        features_by_topology=tuple(features),
        feasible_topologies=tuple(feasible),
        is_greedy=bool(record["is_greedy"]),
        source=str(record["source_kind"]),
    )


def _mad(values: list[float]) -> float:
    center = median(values)
    return float(median(abs(value - center) for value in values))


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if np.ptp(left_array) == 0.0 or np.ptp(right_array) == 0.0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _kendall_tau_b(left: list[float], right: list[float]) -> float | None:
    concordant = discordant = left_ties = right_ties = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if left_delta == 0.0 and right_delta == 0.0:
                continue
            if left_delta == 0.0:
                left_ties += 1
            elif right_delta == 0.0:
                right_ties += 1
            elif left_delta * right_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    return (concordant - discordant) / denominator if denominator else None


def _load_inputs(
    candidate_path: Path,
    calibration_path: Path,
    runtime_path: Path,
    runtime_summary_path: Path,
    workload_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
]:
    dataset = _load(candidate_path)
    calibration = _load(calibration_path)
    summary = _load(runtime_summary_path)
    workload = _load(workload_path)
    candidate_sha = _object_sha256(dataset)
    calibration_sha = _object_sha256(calibration)
    workload_sha = _object_sha256(workload)
    if dataset.get("schema_version") != "upmem_path_candidate_dataset_v1":
        raise ValueError("candidate dataset schema is invalid")
    if calibration.get("candidate_set_sha256") != candidate_sha:
        raise ValueError("calibration set does not match candidate dataset")
    if calibration.get("source_sha") != dataset.get("source_sha"):
        raise ValueError("calibration and candidate sources differ")
    if calibration.get("timing_used_for_selection") is not False:
        raise ValueError("calibration selection must not use physical timing")
    selection_profile_sha = calibration.get("selection_profile_sha256")
    if not isinstance(selection_profile_sha, str) or len(selection_profile_sha) != 64:
        raise ValueError("calibration lacks the frozen selection profile identity")
    selection_model = calibration.get("selection_profile_model")
    if not isinstance(selection_model, dict) or selection_model.get("mode") not in {
        "six_term",
        "grouped",
    }:
        raise ValueError("calibration selection profile model is invalid")
    if dataset.get("workload_manifest_sha256") != workload_sha:
        raise ValueError("candidate dataset does not match workload manifest")
    if summary.get("schema_version") != "upmem_path_runtime_calibration_v1":
        raise ValueError("runtime summary schema is invalid")
    calibration_cells = calibration.get("cells")
    if not isinstance(calibration_cells, list) or not calibration_cells:
        raise ValueError("calibration set has no cells")
    expected_cell_count = len(calibration_cells)
    expected_candidate_count = sum(
        len(cell.get("candidate_path_ids", ())) for cell in calibration_cells
    )
    expected_sample_count = expected_candidate_count * 4
    expected_summary = {
        "candidate_set_sha256": candidate_sha,
        "calibration_set_sha256": calibration_sha,
        "candidate_generation_source_sha": dataset.get("source_sha"),
        "numeric_policy": "split_complex_float32_v1",
        "request_transport": "packed_operation_v1",
        "timing_scope": "steady_execution_v1",
        "claim_policy": "diagnostic_v1",
        "sample_count": expected_sample_count,
        "session_count": expected_sample_count,
        "expected_candidate_cell_count": expected_candidate_count,
        "expected_cell_count": expected_cell_count,
        "fallback_used": False,
        "all_successful_physical_sessions": True,
        "all_resource_admission_passed": True,
        "all_accuracy_qualified": True,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"runtime summary {field} is not the frozen contract")

    workload_entries = {
        str(item["circuit_id"]): item for item in workload["workload"]
    }
    circuits = {str(item["circuit_id"]): item for item in dataset["circuits"]}
    if set(circuits) != {
        circuit_id
        for circuit_id, item in workload_entries.items()
        if item.get("candidate_source") == "generalization_v1"
    }:
        raise ValueError("candidate circuits do not match the new workload subset")
    if any(item.get("split") not in DEVELOPMENT_SPLITS | {"test"} for item in circuits.values()):
        raise ValueError("candidate dataset contains a pilot or unknown split")

    candidate_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for circuit_id, circuit in circuits.items():
        manifest = workload_entries[circuit_id]
        circuit_contract = {
            "split": manifest.get("split"),
            "circuit": manifest.get("circuit_definition"),
        }
        for field, expected_value in circuit_contract.items():
            if circuit.get(field) != expected_value:
                raise ValueError(
                    f"candidate circuit {circuit_id} {field} differs from workload manifest"
                )
        for candidate in circuit["candidates"]:
            candidate_id = str(candidate["candidate_path_id"])
            logical_plan_id = candidate.get("logical_plan_id")
            if not isinstance(logical_plan_id, str) or len(logical_plan_id) != 64:
                raise ValueError("candidate logical plan identity is invalid")
            for topology in candidate["topologies"]:
                topology_id = str(topology["topology_id"])
                if topology.get("feasible") is not True:
                    continue
                admission = topology.get("resource_admission")
                if not isinstance(admission, dict) or admission.get(
                    "collection_resource_admission_passed"
                ) is not True:
                    raise ValueError("feasible candidate lacks resource admission")
                physical_plan_id = topology.get("physical_plan_id")
                if not isinstance(physical_plan_id, str) or len(physical_plan_id) != 64:
                    raise ValueError("feasible candidate physical plan identity is invalid")
                key = (circuit_id, topology_id, candidate_id)
                if key in candidate_records:
                    raise ValueError("candidate topology identity is duplicated")
                candidate_records[key] = {
                    "circuit": circuit,
                    "candidate": candidate,
                    "topology": topology,
                    "manifest": manifest,
                }

    expected: dict[tuple[str, str, str], str] = {}
    cell_records: dict[str, dict[str, Any]] = {}
    for cell in calibration["cells"]:
        circuit_id = str(cell["circuit_id"])
        split = str(circuits[circuit_id]["split"])
        if split not in DEVELOPMENT_SPLITS:
            raise ValueError("calibration set contains a test or pilot circuit")
        cell_id = str(cell["cell_id"])
        topology_id = str(cell["topology_id"])
        if cell_id != f"{circuit_id}:{topology_id}":
            raise ValueError("calibration cell identity is not canonical")
        roles = cell.get("candidate_roles")
        if not isinstance(roles, list) or {
            str(item.get("role")) for item in roles if isinstance(item, dict)
        } != CALIBRATION_ROLES:
            raise ValueError("calibration cell roles are not the frozen contract")
        role_ids = {
            str(item["candidate_path_id"])
            for item in roles
            if isinstance(item, dict)
        }
        if not role_ids <= {str(value) for value in cell["candidate_path_ids"]}:
            raise ValueError("calibration role references an unselected candidate")
        cell_records[cell_id] = cell
        for candidate_id in cell["candidate_path_ids"]:
            key = (circuit_id, topology_id, str(candidate_id))
            if key not in candidate_records:
                raise ValueError("calibration candidate is not physically feasible")
            expected[(cell_id, str(candidate_id), "warmup")] = "0"
            for block in (1, 2, 3):
                expected[(cell_id, str(candidate_id), f"measurement:{block}")] = str(block)

    rows_by_cell: dict[str, list[dict[str, Any]]] = {key: [] for key in cell_records}
    seen: set[tuple[str, str, str]] = set()
    with runtime_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        split = str(row.get("split"))
        if split not in DEVELOPMENT_SPLITS:
            raise ValueError("runtime table contains pilot, test, or unknown rows")
        if row.get("candidate_set_sha256") != candidate_sha:
            raise ValueError("runtime row candidate-set identity mismatch")
        if row.get("calibration_set_sha256") != calibration_sha:
            raise ValueError("runtime row calibration-set identity mismatch")
        if row.get("candidate_generation_source_sha") != dataset.get("source_sha"):
            raise ValueError("runtime row candidate source mismatch")
        if row.get("physical_execution_source_sha") != summary.get(
            "physical_execution_source_sha"
        ):
            raise ValueError("runtime row physical source mismatch")
        if row.get("experiment_id") != summary.get("experiment_id") or row.get(
            "run_id"
        ) != summary.get("run_id"):
            raise ValueError("runtime row experiment identity mismatch")
        if row.get("timing_scope") != "steady_execution_v1":
            raise ValueError("runtime row timing scope mismatch")
        if row.get("status") != "success" or row.get("validation") not in {
            "true",
            "passed",
            "1",
        }:
            raise ValueError("runtime row is not successful and valid")
        if row.get("fallback") not in {"false", "0"}:
            raise ValueError("runtime row used fallback")
        cell_id = str(row.get("cell_id"))
        candidate_id = str(row.get("candidate_path_id"))
        attempt = str(row.get("attempt_type"))
        block = str(row.get("block"))
        if (attempt, block) not in {
            ("warmup", "0"),
            ("measurement", "1"),
            ("measurement", "2"),
            ("measurement", "3"),
        }:
            raise ValueError("runtime row attempt/block contract is invalid")
        key = (
            cell_id,
            candidate_id,
            "warmup" if attempt == "warmup" else f"measurement:{block}",
        )
        if key not in expected or key in seen:
            raise ValueError("runtime table has an unexpected or duplicate observation")
        circuit_id = str(row.get("circuit_id"))
        if circuits[circuit_id]["split"] != split:
            raise ValueError("runtime row split does not match candidate dataset")
        cell = cell_records.get(cell_id)
        if cell is None or circuit_id != str(cell["circuit_id"]):
            raise ValueError("runtime row cell/circuit identity mismatch")
        topology_id = str(cell["topology_id"])
        candidate_record = candidate_records.get(
            (circuit_id, topology_id, candidate_id)
        )
        if candidate_record is None:
            raise ValueError("runtime row candidate/topology identity mismatch")
        topology = candidate_record["topology"]
        candidate = candidate_record["candidate"]
        circuit = candidate_record["circuit"]
        topology_facts = topology["topology"]
        identity_contract = {
            "topology_id": topology_id,
            "route_id": topology_id,
            "plan_id": f"path_{candidate_id}",
            "source_sha": dataset.get("source_sha"),
            "problem_id": circuit["problem_id"],
            "tensor_network_structure_id": circuit["tensor_network_structure_id"],
            "logical_plan_id": candidate["logical_plan_id"],
            "physical_plan_id": topology["physical_plan_id"],
            "request_transport": "packed_operation_v1",
            "requested_dpus": str(topology_facts["dpu_count"]),
            "allocated_dpus": str(topology_facts["dpu_count"]),
            "tasklets_per_dpu": str(topology_facts["tasklets_per_dpu"]),
            "rank_count": str(topology_facts["rank_count"]),
        }
        for field, expected_value in identity_contract.items():
            if row.get(field) != str(expected_value):
                raise ValueError(f"runtime row {field} identity mismatch")
        truth_contract = {
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
        for field, expected_value in truth_contract.items():
            if str(row.get(field)).lower() != expected_value:
                raise ValueError(f"runtime row {field} contract failed")
        if not math.isfinite(float(row["total_wall_s"])) or float(
            row["total_wall_s"]
        ) <= 0.0:
            raise ValueError("runtime row total wall must be positive")
        seen.add(key)
        rows_by_cell[cell_id].append(row)
    if seen != set(expected):
        raise ValueError("runtime table is incomplete")
    if len(rows) != int(summary["sample_count"]):
        raise ValueError("runtime table count does not match summary")
    # Different contraction orders can produce different valid float32 bytes.
    output_hashes: dict[tuple[str, str, str, str], str] = {}
    for rows_for_cell in rows_by_cell.values():
        for row in rows_for_cell:
            configuration = tuple(
                str(row[field])
                for field in (
                    "circuit_id", "candidate_path_id", "topology_id", "physical_plan_id"
                )
            )
            digest = str(row["output_sha256"])
            if output_hashes.setdefault(configuration, digest) != digest:
                raise ValueError("inconsistent outputs for the same physical configuration")
    return dataset, calibration, summary, workload, rows_by_cell


def _cells(
    dataset: dict[str, Any], calibration: dict[str, Any]
) -> tuple[dict[str, TrainingCell], dict[str, str]]:
    circuit_map = {str(item["circuit_id"]): item for item in dataset["circuits"]}
    candidate_map = {
        (str(circuit["circuit_id"]), str(candidate["candidate_path_id"])): candidate
        for circuit in dataset["circuits"]
        for candidate in circuit["candidates"]
    }
    result = {}
    splits = {}
    for record in calibration["cells"]:
        circuit_id = str(record["circuit_id"])
        cell_id = str(record["cell_id"])
        result[cell_id] = TrainingCell(
            cell_id=cell_id,
            topology=str(record["topology_id"]),
            candidates=tuple(
                _candidate(candidate_map[(circuit_id, str(candidate_id))])
                for candidate_id in record["candidate_path_ids"]
            ),
            greedy_path_id=str(record["greedy_path_id"]),
        )
        splits[cell_id] = str(circuit_map[circuit_id]["split"])
    return result, splits


def _identifiable_model(
    cells: dict[str, TrainingCell], cell_ids: set[str], model_form: str
) -> FeatureModelDecision:
    vectors = []
    for cell_id in sorted(cell_ids):
        cell = cells[cell_id]
        greedy = next(
            candidate
            for candidate in cell.candidates
            if candidate.path_id == cell.greedy_path_id
        )
        vectors.extend(
            normalize_features(
                candidate.raw_for(cell.topology), greedy.raw_for(cell.topology)
            ).values
            for candidate in cell.candidates
        )
    raw_matrix = np.asarray(vectors, dtype=np.float64)
    if model_form == "six_term":
        names = FEATURE_NAMES
        matrix = raw_matrix
    elif model_form == "grouped":
        names = GROUP_FEATURE_NAMES
        matrix = np.column_stack(
            ((raw_matrix[:, 0] + raw_matrix[:, 1]) / 2.0, raw_matrix[:, 2:4])
        )
    else:
        raise ValueError(f"unknown model form: {model_form}")

    ranges = np.ptp(matrix, axis=0)
    zero_range = tuple(name for name, value in zip(names, ranges, strict=True) if value == 0.0)
    candidate_indices = [index for index, value in enumerate(ranges) if value > 0.0]
    correlations = []
    for position, left in enumerate(candidate_indices):
        for right in candidate_indices[position + 1 :]:
            coefficient = float(np.corrcoef(matrix[:, left], matrix[:, right])[0, 1])
            if math.isfinite(coefficient) and abs(coefficient) >= 0.98:
                correlations.append((names[left], names[right]))

    retained: list[int] = []
    current_rank = 0
    for index in candidate_indices:
        proposed = matrix[:, (*retained, index)]
        proposed_rank = int(np.linalg.matrix_rank(proposed))
        correlated = any(
            abs(float(np.corrcoef(matrix[:, index], matrix[:, other])[0, 1])) >= 0.98
            for other in retained
        )
        if proposed_rank > current_rank and not correlated:
            retained.append(index)
            current_rank = proposed_rank
    if not retained:
        raise ValueError(f"{model_form} has no identifiable feature in this fit scope")
    return FeatureModelDecision(
        mode=model_form,
        active_features=tuple(names[index] for index in retained),
        zero_range_features=zero_range,
        correlated_pairs=tuple(correlations),
        matrix_rank=current_rank,
        rank_tolerance=0.0,
        reason="deterministic zero-range, rank, and correlation screening",
    )


def _measurements(
    cell_ids: set[str], rows_by_cell: dict[str, list[dict[str, Any]]]
) -> tuple[RuntimeMeasurement, ...]:
    return tuple(
        RuntimeMeasurement(
            cell_id=cell_id,
            candidate_id=str(row["candidate_path_id"]),
            runtime_s=float(row["total_wall_s"]),
            # fit_weights intentionally accepts only the explicit fit scope as train.
            split="train",
            source_sha=str(row["physical_execution_source_sha"]),
            timing_scope=str(row["timing_scope"]),
            status="success",
            observation_id=str(row["block"]),
        )
        for cell_id in sorted(cell_ids)
        for row in rows_by_cell[cell_id]
        if row["attempt_type"] == "measurement"
    )


def _fit(
    cells: dict[str, TrainingCell],
    rows_by_cell: dict[str, list[dict[str, Any]]],
    cell_ids: set[str],
    model_form: str,
    *,
    samples: int,
    seed: int,
) -> WeightFitResult:
    return fit_weights(
        tuple(cells[cell_id] for cell_id in sorted(cell_ids)),
        _measurements(cell_ids, rows_by_cell),
        model=_identifiable_model(cells, cell_ids, model_form),
        random_sample_count=samples,
        seed=seed,
    )


def _values_by_candidate_and_block(
    rows: list[dict[str, Any]],
) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {}
    for row in rows:
        if row["attempt_type"] != "measurement":
            continue
        candidate_id = str(row["candidate_path_id"])
        block = int(row["block"])
        if block in values.setdefault(candidate_id, {}):
            raise ValueError("duplicate candidate measurement block")
        values[candidate_id][block] = float(row["total_wall_s"])
    if any(set(blocks) != {1, 2, 3} for blocks in values.values()):
        raise ValueError("candidate measurements do not contain exact blocks 1..3")
    return values


def _median_interval(
    values: list[float], *, count: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = rng.choice(array, size=(count, len(array)), replace=True)
    return tuple(float(value) for value in np.quantile(np.median(samples, axis=1), [0.025, 0.975]))


def _candidate_statistics(
    cells: dict[str, TrainingCell],
    rows_by_cell: dict[str, list[dict[str, Any]]],
    metadata: dict[str, tuple[str, str, str]],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    result = []
    for cell_index, cell_id in enumerate(sorted(cells)):
        circuit_id, family, split = metadata[cell_id]
        values_by_candidate = _values_by_candidate_and_block(rows_by_cell[cell_id])
        for candidate_index, candidate_id in enumerate(sorted(values_by_candidate)):
            values = [
                values_by_candidate[candidate_id][block]
                for block in sorted(values_by_candidate[candidate_id])
            ]
            interval = _median_interval(
                values,
                count=bootstrap_count,
                seed=bootstrap_seed + 1000 * cell_index + candidate_index,
            )
            result.append(
                {
                    "cell_id": cell_id,
                    "circuit_id": circuit_id,
                    "family": family,
                    "split": split,
                    "topology_id": cells[cell_id].topology,
                    "candidate_path_id": candidate_id,
                    "measurement_count": len(values),
                    "median_s": float(median(values)),
                    "mad_s": _mad(values),
                    "minimum_s": min(values),
                    "maximum_s": max(values),
                    "median_bootstrap_low_s": interval[0],
                    "median_bootstrap_high_s": interval[1],
                }
            )
    return result


def _cell_metrics(
    cell: TrainingCell,
    rows: list[dict[str, Any]],
    result: WeightFitResult,
    *,
    fold_kind: str,
    fold_id: str,
    circuit_id: str,
    family: str,
) -> dict[str, Any]:
    values_by_block = _values_by_candidate_and_block(rows)
    values = {
        key: [item[block] for block in sorted(item)]
        for key, item in values_by_block.items()
    }
    medians = {key: float(median(item)) for key, item in values.items()}
    mads = {key: _mad(item) for key, item in values.items()}
    greedy_id = str(cell.greedy_path_id)
    greedy = next(item for item in cell.candidates if item.path_id == greedy_id)
    normalized = {
        item.path_id: normalize_features(
            item.raw_for(cell.topology), greedy.raw_for(cell.topology)
        )
        for item in cell.candidates
    }
    selected = min(
        cell.candidates,
        key=lambda item: (
            result.model.score(normalized[item.path_id], result.weights),
            item.path_id,
        ),
    ).path_id
    measured_ids = sorted(medians)
    scores = [
        result.model.score(normalized[candidate_id], result.weights)
        for candidate_id in measured_ids
    ]
    measured_runtimes = [medians[candidate_id] for candidate_id in measured_ids]
    ordered = sorted(medians, key=lambda key: (medians[key], key))
    oracle = ordered[0]
    rank = ordered.index(selected) + 1
    greedy_time = medians[greedy_id]
    selected_time = medians[selected]
    oracle_time = medians[oracle]
    measurable = (greedy_time - oracle_time) > max(
        mads[greedy_id], mads[oracle]
    )
    captured = (
        (greedy_time - selected_time) / (greedy_time - oracle_time)
        if measurable
        else None
    )
    tolerance = max(mads[greedy_id], mads[selected])
    if selected_time < greedy_time - tolerance:
        classification = "improved"
    elif selected_time > greedy_time + tolerance:
        classification = "regressed"
    else:
        classification = "neutral"
    return {
        "fold_kind": fold_kind,
        "fold_id": fold_id,
        "model_form": result.model.mode,
        "circuit_id": circuit_id,
        "family": family,
        "topology_id": cell.topology,
        "greedy_path_id": greedy_id,
        "selected_path_id": selected,
        "oracle_path_id": oracle,
        "greedy_median_s": greedy_time,
        "selected_median_s": selected_time,
        "oracle_median_s": oracle_time,
        "greedy_mad_s": mads[greedy_id],
        "selected_mad_s": mads[selected],
        "oracle_mad_s": mads[oracle],
        "greedy_min_s": min(values[greedy_id]),
        "selected_min_s": min(values[selected]),
        "oracle_min_s": min(values[oracle]),
        "greedy_max_s": max(values[greedy_id]),
        "selected_max_s": max(values[selected]),
        "oracle_max_s": max(values[oracle]),
        "measurement_count": len(values[selected]),
        "speedup": greedy_time / selected_time,
        "minimum_candidate_rank": rank,
        "top_1": rank == 1,
        "top_3": rank <= 3,
        "oracle_regret": selected_time / oracle_time,
        "greedy_regret": greedy_time / oracle_time,
        "oracle_speedup": greedy_time / oracle_time,
        "captured_headroom": captured,
        "score_runtime_spearman": _pearson(
            _average_ranks(scores), _average_ranks(measured_runtimes)
        ),
        "score_runtime_kendall_tau_b": _kendall_tau_b(
            scores, measured_runtimes
        ),
        "classification": classification,
    }


def _cross_validate(
    *,
    fold_kind: str,
    groups: dict[str, set[str]],
    cells: dict[str, TrainingCell],
    rows_by_cell: dict[str, list[dict[str, Any]]],
    metadata: dict[str, tuple[str, str, str]],
    model_form: str,
    samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    fits = []
    all_ids = set(cells)
    for index, (fold_id, held_out) in enumerate(sorted(groups.items())):
        training = all_ids - held_out
        if not training:
            raise ValueError("cross-validation fold has no training cells")
        result = _fit(
            cells,
            rows_by_cell,
            training,
            model_form,
            samples=samples,
            seed=seed + index,
        )
        fits.append(
            {
                "fold_kind": fold_kind,
                "fold_id": fold_id,
                "model_form": model_form,
                "training_cell_ids": sorted(training),
                "held_out_cell_ids": sorted(held_out),
                "weights": result.weights.as_mapping(),
                "feature_model": asdict(result.model),
            }
        )
        for cell_id in sorted(held_out):
            circuit_id, family, _split = metadata[cell_id]
            rows.append(
                _cell_metrics(
                    cells[cell_id],
                    rows_by_cell[cell_id],
                    result,
                    fold_kind=fold_kind,
                    fold_id=fold_id,
                    circuit_id=circuit_id,
                    family=family,
                )
            )
    return rows, fits


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speeds = [float(row["speedup"]) for row in rows]
    return {
        "geometric_mean_speedup": geometric_mean(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "improved_cells": sum(row["classification"] == "improved" for row in rows),
        "neutral_cells": sum(row["classification"] == "neutral" for row in rows),
        "regressed_cells": sum(row["classification"] == "regressed" for row in rows),
        "top_1_cells": sum(bool(row["top_1"]) for row in rows),
        "top_3_cells": sum(bool(row["top_3"]) for row in rows),
    }


def _resample_rows_by_block(
    rows_by_cell: dict[str, list[dict[str, Any]]],
    *,
    rng: np.random.Generator,
) -> dict[str, list[dict[str, Any]]]:
    sampled: dict[str, list[dict[str, Any]]] = {}
    for cell_id, rows in rows_by_cell.items():
        by_candidate: dict[str, dict[int, dict[str, Any]]] = {}
        warmups = []
        for row in rows:
            if row["attempt_type"] == "warmup":
                warmups.append(dict(row))
                continue
            by_candidate.setdefault(str(row["candidate_path_id"]), {})[
                int(row["block"])
            ] = row
        selected_blocks = [int(value) for value in rng.choice((1, 2, 3), 3)]
        result = warmups
        for candidate_id in sorted(by_candidate):
            for output_block, source_block in enumerate(selected_blocks, start=1):
                record = dict(by_candidate[candidate_id][source_block])
                record["block"] = str(output_block)
                result.append(record)
        sampled[cell_id] = result
    return sampled


def _bootstrap_robustness(
    *,
    cells: dict[str, TrainingCell],
    groups: dict[str, set[str]],
    metadata: dict[str, tuple[str, str, str]],
    rows_by_cell: dict[str, list[dict[str, Any]]],
    count: int,
    seed: int,
    weight_samples: int,
    weight_seed: int,
) -> tuple[tuple[float, float], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    differences = []
    weight_rows = []
    path_counts: dict[tuple[str, str, str], int] = {}
    all_ids = set(cells)
    for replicate in range(count):
        sampled_rows = _resample_rows_by_block(rows_by_cell, rng=rng)
        by_model = {}
        for model_form in MODEL_FORMS:
            cv_rows, _ = _cross_validate(
                fold_kind="family_bootstrap",
                groups=groups,
                cells=cells,
                rows_by_cell=sampled_rows,
                metadata=metadata,
                model_form=model_form,
                samples=weight_samples,
                seed=weight_seed,
            )
            by_model[model_form] = cv_rows
            fit = _fit(
                cells,
                sampled_rows,
                all_ids,
                model_form,
                samples=weight_samples,
                seed=weight_seed,
            )
            weight_rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "model_form": model_form,
                    **fit.weights.as_mapping(),
                    "geometric_mean_speedup": fit.geometric_mean_speedup,
                    "minimum_speedup": fit.minimum_cell_speedup,
                }
            )
            for cell_id, path_id in fit.selected_path_ids:
                path_counts[(model_form, cell_id, path_id)] = (
                    path_counts.get((model_form, cell_id, path_id), 0) + 1
                )
        differences.append(
            sum(math.log(row["speedup"]) for row in by_model["grouped"])
            / len(by_model["grouped"])
            - sum(math.log(row["speedup"]) for row in by_model["six_term"])
            / len(by_model["six_term"])
        )
    interval = tuple(
        float(value) for value in np.quantile(differences, [0.025, 0.975])
    )
    path_rows = [
        {
            "model_form": model_form,
            "cell_id": cell_id,
            "candidate_path_id": path_id,
            "selection_count": selected_count,
            "selection_frequency": selected_count / count,
        }
        for (model_form, cell_id, path_id), selected_count in sorted(path_counts.items())
    ]
    return interval, weight_rows, path_rows


def _choose_model(
    six_rows: list[dict[str, Any]],
    grouped_rows: list[dict[str, Any]],
    bootstrap_interval: tuple[float, float],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
) -> tuple[str, dict[str, Any]]:
    six_summary = _summary(six_rows)
    grouped_summary = _summary(grouped_rows)
    agreement = all(
        left["selected_path_id"] == right["selected_path_id"]
        for left, right in zip(
            sorted(six_rows, key=lambda row: (row["circuit_id"], row["topology_id"])),
            sorted(
                grouped_rows,
                key=lambda row: (row["circuit_id"], row["topology_id"]),
            ),
            strict=True,
        )
    )
    interval = bootstrap_interval
    if agreement:
        selected = "grouped"
        reason = "all cross-validated development-cell selections agree"
    elif interval[0] <= 0.0 <= interval[1] and grouped_summary[
        "minimum_speedup"
    ] >= six_summary["minimum_speedup"]:
        selected = "grouped"
        reason = "bootstrap-indistinguishable with no worse worst cell"
    else:
        six_order = (
            six_summary["geometric_mean_speedup"],
            six_summary["minimum_speedup"],
            0,
        )
        grouped_order = (
            grouped_summary["geometric_mean_speedup"],
            grouped_summary["minimum_speedup"],
            1,
        )
        selected = "grouped" if grouped_order > six_order else "six_term"
        reason = "lexicographic cross-validated speedup and worst-cell decision"
    return selected, {
        "rule": (
            "grouped_on_complete_selection_agreement_else_grouped_when_paired_block_"
            "bootstrap_95pct_interval_includes_zero_and_worst_cell_is_no_worse_else_"
            "lexicographic_geomean_minimum_speedup_with_grouped_tie_break"
        ),
        "selection_agreement": agreement,
        "grouped_minus_six_mean_log_speedup_bootstrap_95pct": list(interval),
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "reason": reason,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    candidate_path: Path,
    calibration_path: Path,
    runtime_path: Path,
    runtime_summary_path: Path,
    workload_path: Path,
    output_dir: Path,
    *,
    weight_samples: int,
    weight_seed: int,
    bootstrap_count: int,
    bootstrap_seed: int,
    bootstrap_weight_samples: int | None = None,
) -> dict[str, Any]:
    if bootstrap_count < 1:
        raise ValueError("bootstrap_count must be positive")
    if bootstrap_weight_samples is not None and bootstrap_weight_samples < 1:
        raise ValueError("bootstrap_weight_samples must be positive")
    reporting_source_sha, dirty = _source_state()
    if dirty:
        raise ValueError("generalization analysis requires a clean committed worktree")
    dataset, calibration, runtime_summary, workload, rows_by_cell = _load_inputs(
        candidate_path,
        calibration_path,
        runtime_path,
        runtime_summary_path,
        workload_path,
    )
    workload_entries = {
        str(item["circuit_id"]): item for item in workload["workload"]
    }
    cells, splits = _cells(dataset, calibration)
    metadata = {}
    family_groups: dict[str, set[str]] = {}
    instance_groups: dict[str, set[str]] = {}
    calibration_by_id = {
        str(item["cell_id"]): item for item in calibration["cells"]
    }
    for cell_id in cells:
        circuit_id = str(calibration_by_id[cell_id]["circuit_id"])
        family = str(workload_entries[circuit_id]["family"])
        split = str(workload_entries[circuit_id]["split"])
        metadata[cell_id] = (circuit_id, family, split)
        family_groups.setdefault(family, set()).add(cell_id)
        instance_groups.setdefault(circuit_id, set()).add(cell_id)

    family_rows = []
    family_fits = []
    instance_rows = []
    instance_fits = []
    for model_form in MODEL_FORMS:
        rows, fits = _cross_validate(
            fold_kind="family",
            groups=family_groups,
            cells=cells,
            rows_by_cell=rows_by_cell,
            metadata=metadata,
            model_form=model_form,
            samples=weight_samples,
            seed=weight_seed,
        )
        family_rows.extend(rows)
        family_fits.extend(fits)
        rows, fits = _cross_validate(
            fold_kind="instance",
            groups=instance_groups,
            cells=cells,
            rows_by_cell=rows_by_cell,
            metadata=metadata,
            model_form=model_form,
            samples=weight_samples,
            seed=weight_seed + 100,
        )
        instance_rows.extend(rows)
        instance_fits.extend(fits)

    by_model = {
        model: [row for row in family_rows if row["model_form"] == model]
        for model in MODEL_FORMS
    }
    interval, weight_stability, path_stability = _bootstrap_robustness(
        cells=cells,
        groups=family_groups,
        metadata=metadata,
        rows_by_cell=rows_by_cell,
        count=bootstrap_count,
        seed=bootstrap_seed,
        weight_samples=(
            bootstrap_weight_samples
            if bootstrap_weight_samples is not None
            else min(weight_samples, 512)
        ),
        weight_seed=weight_seed,
    )
    selected_model, decision = _choose_model(
        by_model["six_term"],
        by_model["grouped"],
        interval,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
    )
    model_rows = []
    for model_form in MODEL_FORMS:
        values = _summary(by_model[model_form])
        model_rows.append({"model_form": model_form, **values})

    development_ids = set(cells)
    final_fit = _fit(
        cells,
        rows_by_cell,
        development_ids,
        selected_model,
        samples=weight_samples,
        seed=weight_seed,
    )
    candidate_statistics = _candidate_statistics(
        cells,
        rows_by_cell,
        metadata,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
    )
    candidate_sha = _object_sha256(dataset)
    calibration_sha = _object_sha256(calibration)
    runtime_sha = _file_sha256(runtime_path)
    workload_sha = _object_sha256(workload)
    profile = {
        "schema_version": "physical_speedup_fit_v1",
        "profile_id": "upmem_thesis_workload_float32_pretest_v1",
        "score_id": COST_MODEL_ID,
        "source_sha": dataset["source_sha"],
        "candidate_generation_source_sha": dataset["source_sha"],
        "physical_execution_source_sha": runtime_summary[
            "physical_execution_source_sha"
        ],
        "reporting_tool_source_sha": reporting_source_sha,
        "candidate_set_sha256": candidate_sha,
        "calibration_set_sha256": calibration_sha,
        "runtime_table_sha256": runtime_sha,
        "workload_manifest_sha256": workload_sha,
        "weights": final_fit.weights.as_mapping(),
        "feature_model": asdict(final_fit.model),
        "selected_model_form": selected_model,
        "fit_splits": sorted(set(splits.values())),
        "fit_scope_semantics": (
            "training_and_validation_cells_fit_only_after_cross_validated_model_"
            "selection; final_test_timing_excluded"
        ),
        "training_cell_ids": sorted(development_ids),
        "selected_path_ids": dict(final_fit.selected_path_ids),
        "cell_speedups": dict(final_fit.cell_speedups),
        "geometric_mean_speedup": final_fit.geometric_mean_speedup,
        "minimum_cell_speedup": final_fit.minimum_cell_speedup,
        "improved_cell_count": final_fit.improved_cell_count,
        "weight_search_seed": weight_seed,
        "random_weight_samples": weight_samples,
        "normalization": NORMALIZATION,
        "primary_objective": "geometric_mean_greedy_relative_speedup",
        "final_test_timing_used": False,
        "model_choice": decision,
    }
    summary = {
        "schema_version": "upmem_path_generalization_analysis_v1",
        "source_sha": reporting_source_sha,
        "candidate_generation_source_sha": dataset["source_sha"],
        "physical_execution_source_sha": runtime_summary[
            "physical_execution_source_sha"
        ],
        "candidate_set_sha256": candidate_sha,
        "calibration_set_sha256": calibration_sha,
        "runtime_table_sha256": runtime_sha,
        "runtime_summary_sha256": _object_sha256(runtime_summary),
        "workload_manifest_sha256": workload_sha,
        "pilot_rows_used": 0,
        "test_rows_used": 0,
        "development_cell_count": len(cells),
        "model_comparison": {row["model_form"]: row for row in model_rows},
        "selected_model_form": selected_model,
        "model_choice": decision,
        "family_fold_fits": family_fits,
        "instance_fold_fits": instance_fits,
        "bootstrap_weight_refits": bootstrap_count * len(MODEL_FORMS),
        "bootstrap_weight_samples_per_refit": (
            bootstrap_weight_samples
            if bootstrap_weight_samples is not None
            else min(weight_samples, 512)
        ),
        "pretest_profile_sha256": hashlib.sha256(
            _canonical_bytes(profile)
        ).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "leave_one_family_out.csv", family_rows, OUTPUT_COLUMNS)
    _write_csv(
        output_dir / "leave_one_instance_out.csv", instance_rows, OUTPUT_COLUMNS
    )
    _write_csv(
        output_dir / "oracle_headroom.csv",
        [
            row
            for row in family_rows
            if row["model_form"] == selected_model
        ],
        OUTPUT_COLUMNS,
    )
    model_columns = tuple(model_rows[0])
    _write_csv(output_dir / "model_comparison.csv", model_rows, model_columns)
    _write_csv(
        output_dir / "candidate_uncertainty.csv",
        candidate_statistics,
        CANDIDATE_STAT_COLUMNS,
    )
    _write_csv(
        output_dir / "weight_stability.csv",
        weight_stability,
        tuple(weight_stability[0]),
    )
    _write_csv(
        output_dir / "path_selection_stability.csv",
        path_stability,
        tuple(path_stability[0]),
    )
    (output_dir / "generalization_summary.json").write_bytes(
        _canonical_bytes(summary)
    )
    (output_dir / "upmem_thesis_workload_float32_pretest_v1.json").write_bytes(
        _canonical_bytes(profile)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-paths", type=Path, required=True)
    parser.add_argument("--calibration-set", type=Path, required=True)
    parser.add_argument("--runtime-table", type=Path, required=True)
    parser.add_argument("--runtime-summary", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-samples", type=int, default=100_000)
    parser.add_argument("--weight-seed", type=int, default=20260903)
    parser.add_argument("--bootstrap-count", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    parser.add_argument("--bootstrap-weight-samples", type=int, default=512)
    args = parser.parse_args()
    result = analyze(
        args.candidate_paths,
        args.calibration_set,
        args.runtime_table,
        args.runtime_summary,
        args.workload_manifest,
        args.output_dir,
        weight_samples=args.weight_samples,
        weight_seed=args.weight_seed,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_weight_samples=args.bootstrap_weight_samples,
    )
    print(
        json.dumps(
            {
                "selected_model_form": result["selected_model_form"],
                "development_cell_count": result["development_cell_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
