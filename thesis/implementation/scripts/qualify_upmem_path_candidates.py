#!/usr/bin/env python3
"""Qualify frozen path candidates and prepare canonical experiment configs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import yaml

from quantum_bench.cpu import run_complex128_reference, run_cpu_once
from quantum_bench.evidence import canonical_json
from quantum_bench.experiment import load_experiment_config
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network
from quantum_bench.model import make_simulation_job
from quantum_bench.planning import plan_cotengra, plan_opt_einsum
from quantum_bench.upmem.path_heuristic import (
    FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    FeatureModelDecision,
    RawFeatureVector,
    WeightVector,
    path_id,
    score_features,
)
from quantum_bench.upmem.plan import (
    UpmemTopology,
    collection_resource_admission,
    physical_plan_id,
    plan_upmem,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from upmem_path_heuristic import (  # noqa: E402
    EXECUTION_PROFILE,
    WAVE_ADAPTIVE_STAGE,
    _circuit_from_definition,
    _resolve_qasm_path,
    _validate_calibration_execution_contract,
    _validate_dataset_execution_contract,
    _require_wave_execution_coverage,
    _wave_scaling_admission,
    _wave_score,
    _wave_stage_metadata,
    execution_contract,
    validate_wave_stage_provenance,
)

FLOAT32 = "split_complex_float32_v1"
_PROFILE_SCHEMA = "physical_speedup_fit_v1"
_SCORE_ID = "upmem_slr_cost_v1"
_WAVE_SCORE_ID = "upmem_slr_wave_cost_v1"
_WAVE_PRIMARY_QUANTITY = "session_inclusive_s"
_WAVE_TIMING_SCOPE = "steady_execution_v1"
_WAVE_ACTIVE_SIX_TERM = frozenset(FEATURE_NAMES[:4])
_WAVE_ACTIVE_GROUPED = frozenset(GROUP_FEATURE_NAMES)
_WAVE_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024
_WAVE_MEMORY_RESERVE_BYTES = 0
_WAVE_MEMORY_SCOPE = "declared_prepared_wave_executor_bytes_v1"
_EXPLICIT_PILOT_FIELDS = (
    "pilot_weights",
    "migrated_pilot_weights",
    "prior_pilot_weights",
    "legacy_weights",
    "pilot_profile",
    "migrated_profile",
    "reuse_prior_pilot_weights",
    "reuse_prior_calibration_profile",
    "old_pilot_weight_or_profile_imported",
    "mix_lost_raw_calibration",
)
_NORMALIZATION = "log((candidate+1)/(greedy+1))"


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _candidate_set_sha256(dataset: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dataset)).hexdigest()


def _candidate_map(dataset: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (circuit["circuit_id"], candidate["candidate_path_id"]): candidate
        for circuit in dataset["circuits"]
        for candidate in circuit["candidates"]
    }


def _circuit_map(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {circuit["circuit_id"]: circuit for circuit in dataset["circuits"]}


def _prepared_circuit_definition(
    definition: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Return public circuit fields and private staged-source identity."""

    if definition.get("kind") != "qasm_file":
        return {**dict(definition), "path": None}, None
    _circuit_from_definition(definition)
    source_path = _resolve_qasm_path(definition.get("path"))
    parameters = dict(definition.get("parameters", {}))
    digest = parameters.pop("qasm_sha256", None)
    if not isinstance(digest, str):
        raise ValueError("qasm_file source lacks its private qasm_sha256")
    prepared_path = Path("qasm") / digest / source_path.name
    return (
        {
            "kind": "qasm_file",
            "name": None,
            "path": prepared_path.as_posix(),
            "parameters": parameters,
        },
        {
            "source_path": str(source_path),
            "prepared_path": prepared_path.as_posix(),
            "qasm_sha256": digest,
        },
    )


def _profile_model(profile: dict[str, Any]) -> FeatureModelDecision:
    raw_model = profile.get("feature_model")
    if not isinstance(raw_model, dict):
        raise ValueError("frozen profile has no feature model")
    mode = raw_model.get("mode")
    active = raw_model.get("active_features")
    if not isinstance(active, list) or not all(
        isinstance(feature, str) for feature in active
    ):
        raise ValueError("frozen profile feature model has invalid active features")
    allowed = FEATURE_NAMES if mode == "six_term" else GROUP_FEATURE_NAMES
    if mode not in {"six_term", "grouped"} or any(
        feature not in allowed for feature in active
    ):
        raise ValueError("frozen profile feature model is invalid")
    if len(set(active)) != len(active):
        raise ValueError("frozen profile feature model repeats an active feature")
    zero_range = raw_model.get("zero_range_features", [])
    correlated_pairs = raw_model.get("correlated_pairs", [])
    if not isinstance(zero_range, list) or not all(
        isinstance(feature, str) for feature in zero_range
    ):
        raise ValueError("frozen profile feature model has invalid zero-range features")
    if not isinstance(correlated_pairs, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(feature, str) for feature in pair)
        for pair in correlated_pairs
    ):
        raise ValueError("frozen profile feature model has invalid correlations")
    matrix_rank = raw_model.get("matrix_rank", len(active))
    rank_tolerance = raw_model.get("rank_tolerance", 0.0)
    reason = raw_model.get("reason", "explicit frozen profile")
    if isinstance(matrix_rank, bool) or not isinstance(matrix_rank, int):
        raise ValueError("frozen profile feature model has invalid matrix rank")
    if isinstance(rank_tolerance, bool) or not isinstance(
        rank_tolerance, (int, float)
    ):
        raise ValueError("frozen profile feature model has invalid rank tolerance")
    if not isinstance(reason, str):
        raise ValueError("frozen profile feature model has invalid reason")
    return FeatureModelDecision(
        mode=mode,
        active_features=tuple(active),
        zero_range_features=tuple(zero_range),
        correlated_pairs=tuple(tuple(pair) for pair in correlated_pairs),
        matrix_rank=matrix_rank,
        rank_tolerance=float(rank_tolerance),
        reason=reason,
    )


def _require_hex_digest(value: object, field: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase hexadecimal digest")
    return value


def _require_wave_metadata(
    record: Mapping[str, Any], contract: Mapping[str, Any], owner: str
) -> None:
    expected = {
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": _WAVE_SCORE_ID,
        "primary_quantity": _WAVE_PRIMARY_QUANTITY,
        "timing_scope": _WAVE_TIMING_SCOPE,
        "numeric_policy": FLOAT32,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"{owner} has a mismatched wave {field}")


def _require_wave_contract_metadata(
    record: Mapping[str, Any], contract: Mapping[str, Any], owner: str
) -> None:
    expected = {
        "execution_profile": EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": _WAVE_SCORE_ID,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"{owner} has a mismatched wave {field}")


def _validate_wave_profile(
    profile: dict[str, Any],
    dataset: dict[str, Any],
    contract: Mapping[str, Any],
    model: FeatureModelDecision,
    raw_weights: dict[str, Any],
    weights: WeightVector,
) -> None:
    _require_wave_metadata(profile, contract, "frozen profile")
    if profile.get("fit_splits") != ["training"]:
        raise ValueError("frozen wave profile must be fitted on training only")
    validate_wave_stage_provenance(profile)
    source_sha = _require_hex_digest(dataset.get("source_sha"), "source_sha", 40)
    if profile.get("source_sha") != source_sha:
        raise ValueError("frozen profile source identity does not match dataset")
    if profile.get("candidate_generation_source_sha") != source_sha:
        raise ValueError(
            "frozen profile candidate-generation source does not match dataset"
        )
    _require_hex_digest(
        profile.get("candidate_generation_source_sha"),
        "candidate_generation_source_sha",
        40,
    )
    _require_hex_digest(profile.get("candidate_set_sha256"), "candidate_set_sha256", 64)
    preregistration_sha = _require_hex_digest(
        dataset.get("preregistration_sha256"), "preregistration_sha256", 64
    )
    if profile.get("preregistration_sha256") != preregistration_sha:
        raise ValueError("frozen profile preregistration identity does not match dataset")
    if profile.get("normalization") != _NORMALIZATION:
        raise ValueError("frozen profile normalization is not supported")

    for field in ("calibration_set_sha256", "runtime_table_sha256"):
        _require_hex_digest(profile.get(field), field, 64)
        if field in dataset and profile[field] != dataset[field]:
            raise ValueError(f"frozen profile {field} does not match dataset")
    _require_hex_digest(
        profile.get("physical_execution_source_sha"),
        "physical_execution_source_sha",
        40,
    )
    if profile["physical_execution_source_sha"] != contract.get("execution_source"):
        raise ValueError("frozen profile physical execution source does not match wave contract")
    if "reporting_tool_source_sha" in profile:
        _require_hex_digest(profile["reporting_tool_source_sha"], "reporting_tool_source_sha", 40)

    for field in _EXPLICIT_PILOT_FIELDS:
        if field in profile and profile[field] not in (False, None, "", [], {}):
            raise ValueError("frozen profile contains migrated pilot weights or profile")

    if model.mode == "six_term":
        if not set(model.active_features) <= _WAVE_ACTIVE_SIX_TERM:
            raise ValueError("wave six-term profile may activate only the first four features")
    elif not set(model.active_features) <= _WAVE_ACTIVE_GROUPED:
        raise ValueError("wave grouped profile has unsupported active features")

    declared_weights = (
        weights.as_mapping()
        if model.mode == "six_term"
        else {
            "movement": weights.host_dpu + weights.mram_wram,
            "compute": weights.dpu_work,
            "coordination": weights.sync,
        }
    )
    if any(
        value != 0.0 and feature not in model.active_features
        for feature, value in declared_weights.items()
    ):
        raise ValueError("wave profile assigns weight to an inactive feature")

    for feature in ("E_num", "P_wram"):
        value = raw_weights[feature]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"wave profile {feature} weight must be zero")
        if float(value) != 0.0:
            raise ValueError(f"wave profile {feature} weight must be zero")
    if weights.numeric != 0.0 or weights.wram != 0.0:
        raise ValueError("wave profile E_num and P_wram weights must be zero")
    if model.mode == "grouped" and not math.isclose(
        weights.host_dpu,
        weights.mram_wram,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("wave grouped profile requires equal-half movement weights")


def _load_frozen_profile(
    profile_path: Path,
    *,
    dataset: dict[str, Any],
    dataset_hash: str,
) -> tuple[dict[str, Any], str, WeightVector, FeatureModelDecision]:
    profile = _load(profile_path)
    profile_hash = hashlib.sha256(_canonical_bytes(profile)).hexdigest()
    if profile.get("schema_version") != _PROFILE_SCHEMA:
        raise ValueError("frozen profile has an invalid schema")
    if profile.get("candidate_set_sha256") != dataset_hash:
        raise ValueError("frozen profile candidate-set identity does not match dataset")
    source_sha = dataset.get("source_sha")
    if profile.get("source_sha") != source_sha:
        raise ValueError("frozen profile source identity does not match dataset")
    candidate_source = profile.get("candidate_generation_source_sha")
    if candidate_source is not None and candidate_source != source_sha:
        raise ValueError(
            "frozen profile candidate-generation source does not match dataset"
        )
    contract = execution_contract(dataset)
    _validate_dataset_execution_contract(dataset, contract)
    if contract is not None and profile.get("score_id") != _WAVE_SCORE_ID:
        raise ValueError("frozen profile score identity is not supported for wave evaluation")
    if contract is None and profile.get("score_id") != _SCORE_ID:
        raise ValueError("frozen profile score identity is not supported")
    if profile.get("normalization") != _NORMALIZATION:
        raise ValueError("frozen profile normalization is not supported")
    for field in (
        "preregistration_sha256",
        "workload_id",
        "workload_sha256",
        "workload_manifest_sha256",
    ):
        if field in profile and field in dataset and profile[field] != dataset[field]:
            raise ValueError(f"frozen profile {field} does not match dataset")
    raw_weights = profile.get("weights")
    if not isinstance(raw_weights, dict) or set(raw_weights) != set(FEATURE_NAMES):
        raise ValueError("frozen profile weights must contain the six canonical features")
    model = _profile_model(profile)
    try:
        weights = WeightVector.from_values(
            raw_weights,
            inactive=("E_num", "P_wram") if contract is not None else (),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen profile weights are invalid") from exc
    if contract is not None:
        _validate_wave_profile(
            profile, dataset, contract, model, raw_weights, weights
        )
    return profile, profile_hash, weights, model


def _profile_selected_candidate(
    candidates: list[dict[str, Any]],
    topology_id: str,
    greedy: dict[str, Any],
    *,
    weights: WeightVector,
    model: FeatureModelDecision,
    contract: Mapping[str, Any] | None = None,
) -> tuple[str, float]:
    scored = []
    for candidate in candidates:
        if contract is not None:
            score = _wave_score(candidate, greedy, topology_id, weights)
        else:
            greedy_features = RawFeatureVector.from_mapping(
                _topology_record(greedy, topology_id).get("features", {})
            )
            raw = RawFeatureVector.from_mapping(
                _topology_record(candidate, topology_id).get("features", {})
            )
            score = score_features(raw, greedy_features, weights, model=model)
        scored.append((score, candidate["candidate_path_id"]))
    score, candidate_id = min(scored, key=lambda item: (item[0], item[1]))
    return candidate_id, score


def _planner_config(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate["source_kind"] == "opt_einsum_greedy":
        return {"engine": "opt_einsum", "mode": "greedy"}
    if candidate["source_kind"] == "cotengra_one_trial":
        return {
            "engine": "cotengra",
            "mode": "greedy",
            "max_repeats": 1,
            "seed": int(candidate["source_seed"]),
        }
    raise ValueError(f"unsupported candidate source: {candidate['source_kind']}")


def _regenerate(circuit: dict[str, Any], candidate: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    definition = circuit["circuit"]
    spec = _circuit_from_definition(definition)
    network, inputs = lower_tensor_network(make_simulation_job(spec))
    planner = _planner_config(candidate)
    if planner["engine"] == "opt_einsum":
        path, provenance = plan_opt_einsum(network, optimize="greedy")
    else:
        path, provenance = plan_cotengra(
            network,
            methods="greedy",
            max_repeats=1,
            seed=planner["seed"],
        )
    identifier = path_id(path, circuit_id=circuit["circuit_id"])
    if identifier != candidate["candidate_path_id"]:
        raise ValueError(
            f"candidate regeneration mismatch for {circuit['circuit_id']}/{candidate['candidate_path_id']}"
        )
    if provenance["planner_config_hash"] != candidate["planner_config_hash"]:
        raise ValueError(
            f"planner provenance mismatch for {circuit['circuit_id']}/{identifier}"
        )
    dag = build_contraction_dag(network, path)
    if contraction_dag_hash(dag) != candidate["logical_plan_id"]:
        raise ValueError(
            f"logical-plan mismatch for {circuit['circuit_id']}/{identifier}"
        )
    return dag, inputs


def _ranking_best(rankings_path: Path) -> dict[tuple[str, str], str]:
    result = {}
    with rankings_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["equal_weight_rank"]) == 1:
                result[(row["circuit_id"], row["topology_id"])] = row["candidate_path_id"]
    return result


def _representative_ids_for_topology(
    circuit: dict[str, Any], rankings: dict[tuple[str, str], str], topology_id: str
) -> tuple[str, ...]:
    candidates = tuple(
        candidate
        for candidate in circuit["candidates"]
        if candidate.get("conventional_features") is not None
        and next(
            item for item in candidate["topologies"]
            if item["topology_id"] == topology_id
        ).get("feasible") is True
    )
    greedy = next(candidate for candidate in candidates if candidate["is_greedy"])
    selected = {
        greedy["candidate_path_id"],
        min(
            candidates,
            key=lambda item: (
                item["conventional_features"]["flops"], item["candidate_path_id"]
            ),
        )["candidate_path_id"],
    }
    selected.add(rankings[(circuit["circuit_id"], topology_id)])
    selected.add(
        min(
            candidates,
            key=lambda candidate: (
                next(
                    item for item in candidate["topologies"]
                    if item["topology_id"] == topology_id
                )["features"]["B_host_dpu"],
                candidate["candidate_path_id"],
            ),
        )["candidate_path_id"]
    )
    return tuple(sorted(selected))


def _representative_ids(
    circuit: dict[str, Any], rankings: dict[tuple[str, str], str]
) -> tuple[str, ...]:
    return tuple(sorted({
        candidate_id
        for topology_id in ("1dpu_t8", "4dpu_t8")
        for candidate_id in _representative_ids_for_topology(
            circuit, rankings, topology_id
        )
    }))


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _cpu_reference_errors(
    actual: np.ndarray, reference: np.ndarray
) -> tuple[float | None, float | None, float | None, bool]:
    if actual.shape != reference.shape:
        return None, None, None, False
    difference = np.asarray(actual, dtype=np.complex128) - reference
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    denominator = float(np.linalg.norm(reference.reshape(-1)))
    relative_l2 = (
        float(np.linalg.norm(difference.reshape(-1))) / denominator
        if denominator
        else float(np.linalg.norm(difference.reshape(-1)))
    )
    norm_drift = abs(
        float(np.linalg.norm(np.asarray(actual, dtype=np.complex128).reshape(-1)))
        - denominator
    )
    passed = bool(
        np.allclose(actual, reference, rtol=1.0e-5, atol=1.0e-5)
    )
    return maximum, relative_l2, norm_drift, passed


def qualify_frozen_selection(
    dataset_path: Path,
    selection_path: Path,
    output_path: Path,
    *,
    split: str,
    profile_path: Path,
) -> dict[str, Any]:
    """CPU-qualify every unique candidate named by a frozen selection artifact."""

    dataset = _load(dataset_path)
    selection = _load(selection_path)
    dataset_hash = _candidate_set_sha256(dataset)
    candidates = _candidate_map(dataset)
    circuits = _circuit_map(dataset)
    contract = execution_contract(dataset)
    _validate_dataset_execution_contract(dataset, contract)
    selected, selection_roles, _ = _evaluation_selection(
        dataset=dataset,
        dataset_hash=dataset_hash,
        candidate_map=candidates,
        circuit_map=circuits,
        selection_path=selection_path,
        profile_path=profile_path,
        split=split,
    )

    unique: dict[tuple[str, str], dict[str, set[str]]] = {}
    for circuit_id, topology_id, candidate_id in selected:
        entry = unique.setdefault(
            (circuit_id, candidate_id),
            {"topology_ids": set(), "roles": set()},
        )
        entry["topology_ids"].add(topology_id)
        entry["roles"].update(
            selection_roles[(circuit_id, topology_id, candidate_id)]
        )

    rows: list[dict[str, Any]] = []
    references: dict[str, tuple[np.ndarray, str]] = {}
    for circuit_id, candidate_id in sorted(unique):
        circuit = circuits[circuit_id]
        candidate = candidates[(circuit_id, candidate_id)]
        if circuit_id not in references:
            definition = circuit["circuit"]
            spec = _circuit_from_definition(definition)
            network, reference_inputs = lower_tensor_network(make_simulation_job(spec))
            greedy_path, _ = plan_opt_einsum(network, optimize="greedy")
            reference = np.asarray(
                run_complex128_reference(
                    build_contraction_dag(network, greedy_path), reference_inputs
                ),
                dtype=np.complex128,
            )
            references[circuit_id] = (reference, _array_sha256(reference))
        reference, reference_hash = references[circuit_id]
        row: dict[str, Any] = {
            "circuit_id": circuit_id,
            "split": split,
            "candidate_path_id": candidate_id,
            "roles": sorted(unique[(circuit_id, candidate_id)]["roles"]),
            "topology_ids": sorted(
                unique[(circuit_id, candidate_id)]["topology_ids"]
            ),
            "logical_plan_id": candidate["logical_plan_id"],
            "reference_output_sha256": reference_hash,
            "output_sha256": None,
            "max_absolute_error": None,
            "relative_l2_error": None,
            "norm_drift": None,
            "errors": {
                "max_absolute_error": None,
                "relative_l2_error": None,
                "norm_drift": None,
            },
            "error": None,
            "passed": False,
        }
        try:
            dag, inputs = _regenerate(circuit, candidate)
            if contract is not None:
                for topology_id in sorted(
                    unique[(circuit_id, candidate_id)]["topology_ids"]
                ):
                    _verify_wave_plan(
                        dag,
                        _topology_record(candidate, topology_id),
                        topology_id,
                    )
            actual = np.asarray(run_cpu_once(dag, inputs, FLOAT32).output)
            row["output_sha256"] = _array_sha256(actual)
            maximum, relative_l2, norm_drift, passed = _cpu_reference_errors(
                actual, reference
            )
            row["max_absolute_error"] = maximum
            row["relative_l2_error"] = relative_l2
            row["norm_drift"] = norm_drift
            row["errors"] = {
                "max_absolute_error": maximum,
                "relative_l2_error": relative_l2,
                "norm_drift": norm_drift,
            }
            row["passed"] = passed
            if actual.shape != reference.shape:
                row["error"] = (
                    f"output shape mismatch: {actual.shape!r} != {reference.shape!r}"
                )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    result = {
        "schema_version": "upmem_path_frozen_selection_cpu_qualification_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": dataset_hash,
        "selection_sha256": hashlib.sha256(_canonical_bytes(selection)).hexdigest(),
        "selection_schema_version": selection["schema_version"],
        "split": split,
        "numeric_policy": FLOAT32,
        "reference_numeric_policy": "complex128",
        "rtol": 1.0e-5,
        "atol": 1.0e-5,
        "selection_contract_passed": True,
        "selected_cell_count": len(
            {(circuit, topology) for circuit, topology, _ in selected}
        ),
        "qualified_candidate_count": len(rows),
        "all_passed": bool(rows) and all(row["passed"] for row in rows),
        "candidates": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(result))
    return result


def qualify_cpu(
    dataset_path: Path,
    rankings_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    dataset = _load(dataset_path)
    candidates = _candidate_map(dataset)
    rankings = _ranking_best(rankings_path)
    rows = []
    for circuit in dataset["circuits"]:
        definition = circuit["circuit"]
        spec = _circuit_from_definition(definition)
        network, reference_inputs = lower_tensor_network(make_simulation_job(spec))
        greedy_path, _ = plan_opt_einsum(network, optimize="greedy")
        reference = run_complex128_reference(
            build_contraction_dag(network, greedy_path), reference_inputs
        )
        for candidate_id in _representative_ids(circuit, rankings):
            candidate = candidates[(circuit["circuit_id"], candidate_id)]
            dag, inputs = _regenerate(circuit, candidate)
            actual = run_cpu_once(dag, inputs, FLOAT32)
            actual_output = np.asarray(actual.output)
            maximum, relative_l2, _, passed = _cpu_reference_errors(
                actual_output, np.asarray(reference, dtype=np.complex128)
            )
            if not passed:
                raise ValueError(
                    f"CPU candidate validation failed for {circuit['circuit_id']}/{candidate_id}"
                )
            output_bytes = np.ascontiguousarray(actual.output).tobytes()
            rows.append(
                {
                    "circuit_id": circuit["circuit_id"],
                    "split": circuit["split"],
                    "candidate_path_id": candidate_id,
                    "logical_plan_id": contraction_dag_hash(dag),
                    "max_absolute_error": maximum,
                    "relative_l2_error": relative_l2,
                    "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                    "passed": True,
                }
            )
    record = {
        "schema_version": "upmem_path_cpu_candidate_qualification_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": _candidate_set_sha256(dataset),
        "numeric_policy": FLOAT32,
        "qualified_candidate_count": len(rows),
        "all_passed": True,
        "candidates": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(record))
    return record


def _collection(*, warmups: int, measurements: int, seed: int) -> dict[str, Any]:
    return {
        "claim_policy": "diagnostic_v1",
        "base_seed": seed,
        "warmup_blocks": warmups,
        "measurement_blocks": measurements,
        "session_policy": "fresh_session_per_attempt_v1",
        "block_cooldown_s": 0.0,
        "machine_policy": {
            "machine_exclusivity": {"mode": "observed_v1"},
            "cpu_governor": {"mode": "observed_v1"},
            "affinity": {"mode": "exact_required_v1", "expected_cpus": [0]},
            "numa_policy": {"mode": "observed_v1"},
            "background_load": {
                "mode": "observed_v1",
                "max_load1_per_online_cpu": None,
            },
        },
    }


def _route(
    topology_id: str, *, simulator: bool, prepared_waves: bool = False,
    execution_root: Path | None = None,
) -> dict[str, Any]:
    if topology_id not in {"1dpu_t8", "4dpu_t8"}:
        raise ValueError("unsupported path-study topology")
    if type(prepared_waves) is not bool:
        raise TypeError("prepared_waves must be a bool")
    dpus = 1 if topology_id == "1dpu_t8" else 4
    options = {
        "dpu_count": dpus,
        "rank_count": 1,
        "tasklets_per_dpu": 8,
        "session_root": f"../runs/upmem_sessions/path_heuristic_{'sdk' if simulator else 'physical'}_{topology_id}",
        "host_binary": "../native/upmem/runtime/bin/host_upmem_execution_plan_v4_t8",
        "dpu_binary": "../native/upmem/runtime/bin/dpu_gemm_tile_v4_t8",
        "initialization_binary": "../native/upmem/runtime/bin/dpu_simplepim_management_init_t8",
    }
    if prepared_waves:
        options.update({
            "request_transport": "packed_wave_v1",
            "schedule_policy": "static_dag_waves_v1",
            "fuse_complex": True,
            "geometry_policy": "panel_only_v1",
            "dpu_binary": "../native/upmem/runtime/bin/dpu_wave_v5_t8",
        })
        root = ROOT if execution_root is None else Path(execution_root)
        if not root.is_absolute():
            raise ValueError("wave execution root must be an absolute implementation path")
        for field in ("session_root", "host_binary", "dpu_binary", "initialization_binary"):
            options[field] = str(root / Path(options[field]).relative_to(".."))
    if not simulator:
        options["rank_paths"] = ["/dev/dpu_rank1"]
    return {
        "executor": "upmem_sdk_simulator" if simulator else "upmem_physical",
        "numeric_policy": FLOAT32,
        "options": options,
    }


_EVALUATION_TOPOLOGIES = ("1dpu_t8", "4dpu_t8")


def _topology_record(candidate: dict[str, Any], topology_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in candidate.get("topologies", [])
        if item.get("topology_id") == topology_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"candidate {candidate.get('candidate_path_id')} must have exactly "
            f"one {topology_id} topology record"
        )
    return matches[0]


def _require_collection_admission_fact(
    topology: Mapping[str, Any],
    circuit_id: str,
    topology_id: str,
    candidate_id: object,
    *,
    wave: bool,
) -> bool:
    admission = topology.get("resource_admission")
    if not isinstance(admission, dict):
        raise ValueError(
            f"evaluation candidate lacks resource admission for "
            f"{circuit_id}/{topology_id}/{candidate_id}"
        )
    if wave:
        resources = topology.get("topology")
        if not isinstance(resources, Mapping):
            raise ValueError(
                f"candidate topology resources are invalid for "
                f"{circuit_id}/{topology_id}/{candidate_id}"
            )
        facts = _wave_scaling_admission(
            admission,
            resources,
            field=(
                "candidate wave scaling admission "
                f"{circuit_id}/{topology_id}/{candidate_id}"
            ),
        )
        return facts["collection_resource_admission_passed"]
    value = admission.get("collection_resource_admission_passed")
    if value is not True:
        raise ValueError(
            f"evaluation candidate lacks passed resource admission for "
            f"{circuit_id}/{topology_id}/{candidate_id}"
        )
    return True


def _verify_wave_plan(dag: Any, topology: dict[str, Any], topology_id: str) -> None:
    if topology_id not in _EVALUATION_TOPOLOGIES:
        raise ValueError("unsupported path-study topology")
    resources = {
        "dpu_count": 1 if topology_id == "1dpu_t8" else 4,
        "rank_count": 1,
        "tasklets_per_dpu": 8,
    }
    if topology.get("topology") != resources:
        raise ValueError("candidate topology differs from execution route")
    plan = plan_upmem(
        dag, numeric_policy=FLOAT32, topology=UpmemTopology(**resources),
        schedule_policy="static_dag_waves_v1",
    )
    _require_wave_execution_coverage(plan)
    declared_logical = topology.get("logical_plan_id")
    if declared_logical is not None and declared_logical != plan.logical_plan_id:
        raise ValueError("candidate logical-plan identity differs from wave execution")
    if physical_plan_id(plan) != topology.get("physical_plan_id"):
        raise ValueError("candidate physical-plan identity differs from wave execution")
    admission = topology.get("resource_admission")
    if not isinstance(admission, Mapping):
        raise ValueError("candidate lacks wave resource admission facts")
    _wave_scaling_admission(
        admission,
        resources,
        expected=collection_resource_admission(plan),
        field=f"candidate wave scaling admission {topology_id}",
    )


def _require_evaluation_candidate(
    candidate: dict[str, Any],
    circuit_id: str,
    topology_id: str,
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    topology = _topology_record(candidate, topology_id)
    if topology.get("feasible") is not True:
        raise ValueError(
            f"evaluation candidate is infeasible for {circuit_id}/{topology_id}/"
            f"{candidate.get('candidate_path_id')}"
        )
    _require_collection_admission_fact(
        topology,
        circuit_id,
        topology_id,
        candidate.get("candidate_path_id"),
        wave=contract is not None,
    )
    if contract is not None:
        for field, value in (
            ("execution_profile", EXECUTION_PROFILE),
            ("execution_contract", dict(contract)),
            ("score_id", _WAVE_SCORE_ID),
        ):
            if topology.get(field) != value:
                raise ValueError(f"candidate topology has a mismatched wave {field}")
        memory = topology.get("memory_admission")
        if not isinstance(memory, dict) or memory.get("passed") is not True:
            raise ValueError(
                f"evaluation candidate lacks passed memory admission for "
                f"{circuit_id}/{topology_id}/{candidate.get('candidate_path_id')}"
            )
        if memory.get("scope") != _WAVE_MEMORY_SCOPE:
            raise ValueError("candidate memory admission has an unsupported scope")
        if memory.get("configured_budget_bytes") != _WAVE_MEMORY_BUDGET_BYTES:
            raise ValueError("candidate memory admission budget differs from frozen executor")
        if memory.get("configured_reserve_bytes") != _WAVE_MEMORY_RESERVE_BYTES:
            raise ValueError("candidate memory admission reserve differs from frozen executor")
        estimate = memory.get("declared_executor_memory_estimate_bytes")
        if isinstance(estimate, bool) or not isinstance(estimate, int) or estimate < 0:
            raise ValueError("candidate memory admission estimate is invalid")
        if topology.get("host_memory_estimate_bytes") != estimate:
            raise ValueError("candidate memory admission estimate does not match topology")
        required = memory.get("required_bytes")
        if isinstance(required, bool) or not isinstance(required, int) or required < 0:
            raise ValueError("candidate memory admission requirement is invalid")
        if required != estimate + _WAVE_MEMORY_RESERVE_BYTES:
            raise ValueError("candidate memory admission requirement is inconsistent")
        if required > _WAVE_MEMORY_BUDGET_BYTES:
            raise ValueError("candidate memory admission exceeds frozen executor budget")
    if not isinstance(topology.get("physical_plan_id"), str) or not topology[
        "physical_plan_id"
    ]:
        raise ValueError(
            f"evaluation candidate lacks physical-plan identity for "
            f"{circuit_id}/{topology_id}/{candidate.get('candidate_path_id')}"
        )
    return topology


def _evaluation_selection(
    *,
    dataset: dict[str, Any],
    dataset_hash: str,
    candidate_map: dict[tuple[str, str], dict[str, Any]],
    circuit_map: dict[str, dict[str, Any]],
    selection_path: Path,
    profile_path: Path,
    split: str,
) -> tuple[
    list[tuple[str, str, str]],
    dict[tuple[str, str, str], tuple[str, ...]],
    str,
]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    selection = _load(selection_path)
    contract = execution_contract(dataset)
    _validate_dataset_execution_contract(dataset, contract)
    profile, profile_hash, weights, model = _load_frozen_profile(
        profile_path,
        dataset=dataset,
        dataset_hash=dataset_hash,
    )
    if selection.get("schema_version") != "upmem_path_frozen_selection_v1":
        raise ValueError("evaluation selection has an invalid schema")
    if selection.get("candidate_set_sha256") != dataset_hash:
        raise ValueError("evaluation candidate-set identity does not match dataset")
    if selection.get("source_sha") != dataset.get("source_sha"):
        raise ValueError("evaluation source identity does not match dataset")
    if selection.get("fitted_profile_sha256") != profile_hash:
        raise ValueError("evaluation frozen-profile identity does not match profile")
    if contract is not None:
        _require_wave_contract_metadata(selection, contract, "evaluation selection")
        _require_hex_digest(
            selection.get("fitted_profile_sha256"),
            "fitted_profile_sha256",
            64,
        )
        for field in ("primary_quantity", "timing_scope", "numeric_policy"):
            if field in selection and selection[field] != profile[field]:
                raise ValueError(f"evaluation selection {field} does not match profile")
        if (
            "preregistration_sha256" in selection
            and selection["preregistration_sha256"]
            != dataset.get("preregistration_sha256")
        ):
            raise ValueError(
                "evaluation selection preregistration identity does not match dataset"
            )
    if "score_id" in selection and selection["score_id"] != profile["score_id"]:
        raise ValueError("evaluation selection score identity does not match profile")
    if "weights" in selection and selection["weights"] != profile["weights"]:
        raise ValueError("evaluation selection weights do not match profile")
    if (
        "feature_model" in selection
        and selection["feature_model"] != profile["feature_model"]
    ):
        raise ValueError("evaluation selection feature model does not match profile")
    if contract is not None:
        for field in ("weights", "feature_model"):
            if field in selection and selection[field] != profile[field]:
                raise ValueError(f"evaluation selection {field} does not match profile")
    for field in (
        "preregistration_sha256",
        "workload_id",
        "workload_sha256",
        "workload_manifest_sha256",
    ):
        if field in selection and field in dataset and selection[field] != dataset[field]:
            raise ValueError(f"evaluation selection {field} does not match dataset")
    if selection.get("split") != split:
        raise ValueError("evaluation selection split does not match requested split")
    if selection.get("timing_used_for_selection") is not False:
        raise ValueError("evaluation selection must be timing-independent")
    rows = selection.get("selections")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation selection must contain selections")

    expected_circuits = {
        circuit_id
        for circuit_id, circuit in circuit_map.items()
        if circuit.get("split") == split
    }
    if not expected_circuits:
        raise ValueError(f"candidate dataset contains no {split} circuits")
    expected_keys = {
        (circuit_id, topology_id)
        for circuit_id in expected_circuits
        for topology_id in _EVALUATION_TOPOLOGIES
    }
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("evaluation selection rows must be objects")
        if row.get("split") != split:
            raise ValueError("evaluation selection contains another split")
        circuit_id = row.get("circuit_id")
        topology_id = row.get("topology_id")
        if not isinstance(circuit_id, str) or not isinstance(topology_id, str):
            raise ValueError("evaluation selection cell identities must be strings")
        key = (circuit_id, topology_id)
        if key in rows_by_key:
            raise ValueError(
                f"evaluation selection contains duplicate cell {circuit_id}/{topology_id}"
            )
        if key not in expected_keys:
            raise ValueError(
                f"evaluation selection contains an unexpected cell {circuit_id}/{topology_id}"
            )
        rows_by_key[key] = row
    if set(rows_by_key) != expected_keys:
        missing = sorted(expected_keys - set(rows_by_key))
        raise ValueError(f"evaluation selection is missing cells: {missing}")

    selected: set[tuple[str, str, str]] = set()
    roles: dict[tuple[str, str, str], set[str]] = {}
    for (circuit_id, topology_id), row in sorted(rows_by_key.items()):
        circuit_candidates = [
            candidate
            for (candidate_circuit_id, _), candidate in candidate_map.items()
            if candidate_circuit_id == circuit_id
        ]
        feasible = []
        for candidate in circuit_candidates:
            topology = _topology_record(candidate, topology_id)
            admission = topology.get("resource_admission")
            collection_admission_valid = (
                isinstance(admission, dict)
                and admission.get("collection_resource_admission_passed") is True
            )
            if contract is not None and topology.get("feasible") is True:
                _require_collection_admission_fact(
                    topology,
                    circuit_id,
                    topology_id,
                    candidate.get("candidate_path_id"),
                    wave=True,
                )
                collection_admission_valid = True
            if (
                topology.get("feasible") is True
                and collection_admission_valid
                and (
                    contract is None
                    or isinstance(topology.get("memory_admission"), dict)
                    and topology["memory_admission"].get("passed") is True
                )
            ):
                feasible.append(candidate)
        if not feasible:
            raise ValueError(f"no admitted candidates for {circuit_id}/{topology_id}")
        greedy_candidates = [
            candidate for candidate in feasible if candidate.get("is_greedy") is True
        ]
        if len(greedy_candidates) != 1:
            raise ValueError(
                f"evaluation requires exactly one feasible greedy candidate for "
                f"{circuit_id}/{topology_id}"
            )
        expected_greedy = greedy_candidates[0]["candidate_path_id"]
        expected_flops = min(
            feasible,
            key=lambda candidate: (
                candidate["conventional_features"]["flops"],
                candidate["candidate_path_id"],
            ),
        )["candidate_path_id"]
        expected_roles = {
            "greedy_path_id": expected_greedy,
            "minimum_flops_path_id": expected_flops,
        }
        for field, expected in expected_roles.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"evaluation {field} is not deterministic for "
                    f"{circuit_id}/{topology_id}"
                )
        expected_upmem, _ = _profile_selected_candidate(
            feasible,
            topology_id,
            greedy_candidates[0],
            weights=weights,
            model=model,
            contract=contract,
        )
        upmem_selected = row.get("upmem_selected_path_id")
        if not isinstance(upmem_selected, str) or not upmem_selected:
            raise ValueError(
                f"evaluation selection has no UPMEM-selected path for "
                f"{circuit_id}/{topology_id}"
            )
        claimed_candidate = candidate_map.get((circuit_id, upmem_selected))
        if claimed_candidate is None:
            raise ValueError(
                f"evaluation selection references unknown candidate "
                f"{circuit_id}/{upmem_selected}"
            )
        _require_evaluation_candidate(
            claimed_candidate, circuit_id, topology_id, contract=contract
        )
        if upmem_selected != expected_upmem:
            raise ValueError(
                f"evaluation UPMEM-selected path is not selected by frozen profile "
                f"for {circuit_id}/{topology_id}"
            )
        if "upmem_score" in row:
            claimed_score = row["upmem_score"]
            if isinstance(claimed_score, bool) or not isinstance(
                claimed_score, (int, float)
            ) or not math.isfinite(float(claimed_score)):
                raise ValueError("evaluation selection score is invalid")
            expected_score = _profile_selected_candidate(
                feasible,
                topology_id,
                greedy_candidates[0],
                weights=weights,
                model=model,
                contract=contract,
            )[1]
            if not math.isclose(
                float(claimed_score), expected_score, rel_tol=1.0e-12, abs_tol=1.0e-12
            ):
                raise ValueError("evaluation selection score does not match frozen profile")
        role_values = (
            *expected_roles.items(),
            ("upmem_selected_path_id", expected_upmem),
        )
        for field, candidate_id in role_values:
            candidate = candidate_map.get((circuit_id, candidate_id))
            if candidate is None:
                raise ValueError(
                    f"evaluation selection references unknown candidate "
                    f"{circuit_id}/{candidate_id}"
                )
            _require_evaluation_candidate(
                candidate, circuit_id, topology_id, contract=contract
            )
            selected_key = (circuit_id, topology_id, candidate_id)
            selected.add(selected_key)
            roles.setdefault(selected_key, set()).add(field.removesuffix("_path_id"))
    return (
        sorted(selected),
        {key: tuple(sorted(value)) for key, value in roles.items()},
        profile_hash,
    )


def _confirmation_selection(
    dataset: dict[str, Any], profile_path: Path,
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str, str], tuple[str, ...]], str]:
    contract = execution_contract(dataset)
    if contract is None:
        raise ValueError("confirmation requires the frozen wave execution profile")
    profile, profile_hash, weights, model = _load_frozen_profile(
        profile_path, dataset=dataset, dataset_hash=_candidate_set_sha256(dataset)
    )
    circuits = [item for item in dataset["circuits"] if item.get("split") == "training"]
    expected_cells = {
        f"{item['circuit_id']}:{topology}" for item in circuits
        for topology in _EVALUATION_TOPOLOGIES
    }
    selected_ids = profile.get("selected_path_ids")
    measured = profile.get("candidate_speedups")
    if not expected_cells or any(
        not isinstance(value, dict) or set(value) != expected_cells
        for value in (selected_ids, measured)
    ):
        raise ValueError("confirmation profile must cover exactly the training cells")
    roles: dict[tuple[str, str, str], set[str]] = {}
    for circuit in circuits:
        circuit_id = circuit["circuit_id"]
        by_id = {item["candidate_path_id"]: item for item in circuit["candidates"]}
        for topology in _EVALUATION_TOPOLOGIES:
            cell_id = f"{circuit_id}:{topology}"
            pool = measured[cell_id]
            if not isinstance(pool, dict) or not pool or not set(pool) <= set(by_id):
                raise ValueError("confirmation measured candidate pool is invalid")
            candidates = [by_id[path] for path in sorted(pool)]
            for candidate in candidates:
                _require_evaluation_candidate(candidate, circuit_id, topology, contract=contract)
            greedy = [candidate for candidate in candidates if candidate.get("is_greedy") is True]
            if len(greedy) != 1:
                raise ValueError("confirmation requires exactly one measured greedy candidate")
            selected, _ = _profile_selected_candidate(
                candidates, topology, greedy[0], weights=weights, model=model, contract=contract
            )
            if selected_ids[cell_id] != selected:
                raise ValueError("confirmation selection does not match frozen measured-pool score")
            for role, candidate_id in (("greedy", greedy[0]["candidate_path_id"]), ("upmem_selected", selected)):
                roles.setdefault((circuit_id, topology, candidate_id), set()).add(role)
    return sorted(roles), {key: tuple(sorted(value)) for key, value in roles.items()}, profile_hash


def prepare_config(
    *,
    dataset_path: Path,
    output_path: Path,
    mode: str,
    calibration_path: Path | None = None,
    rankings_path: Path | None = None,
    selection_path: Path | None = None,
    profile_path: Path | None = None,
    split: str | None = None,
    execution_target: str = "physical",
    experiment_id: str | None = None,
    execution_root: Path | None = None,
) -> dict[str, Any]:
    if execution_target not in {"physical", "sdk"}:
        raise ValueError("execution target must be physical or sdk")
    dataset = _load(dataset_path)
    contract = execution_contract(dataset)
    _validate_dataset_execution_contract(dataset, contract)
    if contract is not None:
        if dataset.get("execution_contract") != contract or dataset.get("score_id") != contract[
            "cost_model_id"
        ]:
            raise ValueError("wave candidate dataset lacks its complete execution contract")
    dataset_hash = _candidate_set_sha256(dataset)
    candidate_map = _candidate_map(dataset)
    circuit_map = _circuit_map(dataset)
    selected: list[tuple[str, str, str]] = []
    selection_roles: dict[tuple[str, str, str], tuple[str, ...]] = {}
    selection_provenance: list[tuple[str, str, str]] = []
    profile_hash: str | None = None
    calibration_hash: str | None = None
    stage_metadata: dict[str, Any] | None = None
    if mode == "calibration" or (mode == "sdk" and calibration_path is not None):
        if calibration_path is None:
            raise ValueError("calibration mode requires a calibration set")
        calibration = _load(calibration_path)
        _validate_calibration_execution_contract(calibration, contract)
        if contract is not None:
            stage_metadata = _wave_stage_metadata(calibration)
            calibration_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        if calibration.get("candidate_set_sha256") != dataset_hash:
            raise ValueError("calibration candidate-set identity does not match dataset")
        if calibration.get("source_sha") != dataset.get("source_sha"):
            raise ValueError("calibration source identity does not match dataset")
        for cell in calibration["cells"]:
            if contract is not None and (
                ("split" in cell and cell["split"] != "training")
                or circuit_map.get(cell["circuit_id"], {}).get("split") != "training"
            ):
                raise ValueError("wave calibration cells must be training only")
            for candidate_id in cell["candidate_path_ids"]:
                selected.append((cell["circuit_id"], cell["topology_id"], candidate_id))
        if mode == "sdk":
            warmups, measurements, seed, simulator = 0, 1, 20260909, True
        else:
            warmups, measurements, seed, simulator = 1, 3, 20260910, False
        topology_ids = ("1dpu_t8", "4dpu_t8")
    elif mode == "sdk":
        if rankings_path is None:
            raise ValueError("sdk mode requires rankings")
        rankings = _ranking_best(rankings_path)
        topology_ids = ("1dpu_t8", "4dpu_t8") if contract else ("1dpu_t8",)
        for circuit in dataset["circuits"]:
            for topology_id in topology_ids:
                ids = _representative_ids_for_topology(
                    circuit, rankings, topology_id
                )
                selected.extend((circuit["circuit_id"], topology_id, path) for path in ids)
        warmups, measurements, seed, simulator = 0, 1, 20260909, True
    elif mode == "confirmation":
        if split != "training" or profile_path is None or selection_path is not None:
            raise ValueError("confirmation requires training split and a profile, not a selection file")
        selected, selection_roles, profile_hash = _confirmation_selection(dataset, profile_path)
        selection_provenance = list(selected)
        topology_ids = _EVALUATION_TOPOLOGIES
        simulator = execution_target == "sdk"
        warmups, measurements, seed = (0, 1, 20260913) if simulator else (1, 5, 20260914)
    elif mode == "evaluation":
        if selection_path is None:
            raise ValueError("evaluation mode requires a frozen selection")
        if profile_path is None:
            raise ValueError("evaluation mode requires a frozen profile")
        if split is None:
            raise ValueError("evaluation mode requires validation or test split")
        topology_ids = _EVALUATION_TOPOLOGIES
        selected, selection_roles, profile_hash = _evaluation_selection(
            dataset=dataset,
            dataset_hash=dataset_hash,
            candidate_map=candidate_map,
            circuit_map=circuit_map,
            selection_path=selection_path,
            profile_path=profile_path,
            split=split,
        )
        selection_provenance = list(selected)
        if execution_target == "sdk":
            if contract is None:
                selected = sorted({
                    (circuit_id, "1dpu_t8", candidate_id)
                    for circuit_id, _topology_id, candidate_id in selected
                })
                topology_ids = ("1dpu_t8",)
            warmups, measurements, seed, simulator = 0, 1, 20260912, True
        else:
            warmups, measurements, seed, simulator = 1, 5, 20260911, False
    else:
        raise ValueError("mode must be sdk, calibration, confirmation, or evaluation")
    if contract is not None and mode in {"confirmation", "evaluation"}:
        stage_metadata = {
            "stage_id": "development_confirmation" if mode == "confirmation" else split,
            "round_ordinal": 0,
            "prior_stage_hashes": [],
            "selection_profile_sha256": profile_hash,
            "timing_used_for_selection": False,
        }
    if mode not in {"evaluation", "confirmation"} and execution_target != "physical":
        raise ValueError(
            "execution target is only configurable for evaluation mode"
        )
    selected = sorted(set(selected))
    cases: dict[str, dict[str, Any]] = {}
    qasm_sources: list[dict[str, str]] = []
    for circuit_id in sorted({item[0] for item in selected}):
        public_definition, source_binding = _prepared_circuit_definition(
            circuit_map[circuit_id]["circuit"]
        )
        cases[circuit_id] = {"circuit": public_definition}
        if source_binding is not None:
            qasm_sources.append({"circuit_id": circuit_id, **source_binding})
    plans = {}
    matrix = []
    for circuit_id, topology_id, candidate_id in selected:
        candidate = candidate_map[(circuit_id, candidate_id)]
        topology = _topology_record(candidate, topology_id)
        if contract is not None:
            if any(
                candidate.get(field) != dataset[field]
                for field in ("execution_profile", "execution_contract", "score_id")
            ):
                raise ValueError("selected candidate execution contract differs from dataset")
            _require_evaluation_candidate(
                candidate, circuit_id, topology_id, contract=contract
            )
        if topology.get("feasible") is not True:
            raise ValueError(
                f"selected candidate is infeasible for {circuit_id}/{topology_id}/{candidate_id}"
            )
        dag, _inputs = _regenerate(circuit_map[circuit_id], candidate)
        if contract is not None:
            _verify_wave_plan(dag, topology, topology_id)
        plan_id = f"path_{candidate_id}"
        plans[plan_id] = {"planner": _planner_config(candidate), "slicing": None}
        matrix.append(
            {
                "case_id": circuit_id,
                "plan_id": plan_id,
                "route_ids": [f"upmem_{topology_id}"],
            }
        )
    if mode in {"evaluation", "confirmation"}:
        grouped: dict[tuple[str, str], set[str]] = {}
        for circuit_id, topology_id, candidate_id in selected:
            grouped.setdefault((circuit_id, candidate_id), set()).add(topology_id)
        topology_order = {
            topology_id: index for index, topology_id in enumerate(topology_ids)
        }
        matrix = [
            {
                "case_id": circuit_id,
                "plan_id": f"path_{candidate_id}",
                "route_ids": sorted(
                    route_ids, key=lambda route_id: topology_order[route_id]
                ),
            }
            for (circuit_id, candidate_id), route_ids in sorted(grouped.items())
        ]
    default_experiment_id = (
        f"upmem-path-heuristic-evaluation-{split}"
        f"{'-sdk' if execution_target == 'sdk' else ''}-v1"
        if mode == "evaluation"
        else f"upmem-path-heuristic-{mode}-v1"
    )
    if contract is not None:
        if stage_metadata is not None and stage_metadata["stage_id"] == WAVE_ADAPTIVE_STAGE:
            default_experiment_id = (
                f"upmem-final-system-path-adaptive-{stage_metadata['round_ordinal']}"
                f"{'-sdk' if simulator else ''}-v2"
            )
        elif mode == "confirmation":
            default_experiment_id = (
                f"upmem-final-system-path-confirmation{'-sdk' if simulator else ''}-v2"
            )
        elif mode == "evaluation":
            default_experiment_id = (
                f"upmem-final-system-path-evaluation-{split}"
                f"{'-sdk' if execution_target == 'sdk' else ''}-v2"
            )
        else:
            default_experiment_id = f"upmem-final-system-path-{mode}-v2"
    if experiment_id is not None and not experiment_id.strip():
        raise ValueError("experiment_id must be nonempty when provided")
    config = {
        "schema_version": "tn_benchmark_v3",
        "experiment_id": experiment_id or default_experiment_id,
        "defaults": {"timeout_s": 120.0},
        "collection": _collection(warmups=warmups, measurements=measurements, seed=seed),
        "cases": cases,
        "plans": plans,
        "routes": {
            topology_id: _route(
                topology_id, simulator=simulator, prepared_waves=contract is not None,
                execution_root=execution_root,
            )
            for topology_id in topology_ids
        },
        "matrix": matrix,
    }
    # Matrix route names use the canonical mapping keys above.
    for item in config["matrix"]:
        item["route_ids"] = [
            route_id.removeprefix("upmem_") for route_id in item["route_ids"]
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for binding in qasm_sources:
        source_path = Path(binding["source_path"])
        target_path = output_path.resolve().parent / binding["prepared_path"]
        payload = source_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != binding["qasm_sha256"]:
            raise ValueError(
                f"qasm_file source changed while preparing {binding['circuit_id']}"
            )
        if target_path.resolve() == source_path.resolve():
            continue
        if target_path.exists():
            if target_path.read_bytes() != payload:
                raise ValueError(
                    f"qasm_file basename collision in output bundle: {target_path.name}"
                )
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(payload)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    provenance = {
        "schema_version": "upmem_path_experiment_provenance_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": dataset_hash,
        "preregistration_sha256": dataset["preregistration_sha256"],
        "mode": mode,
        **({"stage": stage_metadata} if stage_metadata is not None else {}),
        **(
            {
                "execution_profile": dataset["execution_profile"],
                "execution_contract": contract,
                "score_id": dataset["score_id"],
                "numeric_policy": FLOAT32,
                "primary_quantity": _WAVE_PRIMARY_QUANTITY,
                "timing_scope": _WAVE_TIMING_SCOPE,
                "configuration_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "normalized_configuration_sha256": hashlib.sha256(
                    _canonical_bytes(json.loads(canonical_json(load_experiment_config(output_path))))
                ).hexdigest(),
                "calibration_set_sha256": calibration_hash,
            }
            if contract is not None
            else {}
        ),
        **(
            {
                "selection_split": split,
                "execution_target": execution_target,
                "selection_path": str(selection_path) if selection_path is not None else None,
                "profile_path": str(profile_path),
                "fitted_profile_sha256": profile_hash,
                "selection_roles": [
                    {
                        "circuit_id": circuit_id,
                        "topology_id": topology_id,
                        "candidate_path_id": candidate_id,
                        "roles": list(
                            selection_roles[(circuit_id, topology_id, candidate_id)]
                        ),
                    }
                    for circuit_id, topology_id, candidate_id in selection_provenance
                ],
            }
            if mode in {"evaluation", "confirmation"}
            else {}
        ),
        "selected": [
            {
                "circuit_id": circuit_id,
                "topology_id": topology_id,
                "candidate_path_id": candidate_id,
                "logical_plan_id": candidate_map[(circuit_id, candidate_id)]["logical_plan_id"],
                "physical_plan_id": next(
                    item["physical_plan_id"]
                    for item in candidate_map[(circuit_id, candidate_id)]["topologies"]
                    if item["topology_id"] == topology_id
                ),
            }
            for circuit_id, topology_id, candidate_id in selected
        ],
        **({"qasm_source_bindings": qasm_sources} if qasm_sources else {}),
    }
    output_path.with_suffix(output_path.suffix + ".provenance.json").write_bytes(
        _canonical_bytes(provenance)
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cpu = subparsers.add_parser("cpu")
    cpu.add_argument("--candidate-paths", type=Path, required=True)
    cpu.add_argument("--rankings", type=Path, required=True)
    cpu.add_argument("--output", type=Path, required=True)
    frozen_cpu = subparsers.add_parser(
        "qualify-frozen-selection",
        aliases=(
            "cpu-selection",
            "evaluation-cpu",
            "cpu-evaluate",
            "qualify-evaluation",
        ),
    )
    frozen_cpu.add_argument("--candidate-paths", type=Path, required=True)
    frozen_cpu.add_argument(
        "--selection", "--selection-artifact", dest="selection", type=Path, required=True
    )
    frozen_cpu.add_argument(
        "--profile", "--frozen-profile", dest="profile", type=Path, required=True
    )
    frozen_cpu.add_argument("--split", choices=("validation", "test"), required=True)
    frozen_cpu.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--candidate-paths", type=Path, required=True)
    prepare.add_argument("--calibration-set", type=Path)
    prepare.add_argument("--selection", "--selection-artifact", dest="selection", type=Path)
    prepare.add_argument("--profile", "--frozen-profile", dest="profile", type=Path)
    prepare.add_argument("--rankings", type=Path)
    prepare.add_argument(
        "--mode", choices=("sdk", "calibration", "confirmation", "evaluation"), required=True
    )
    prepare.add_argument("--split", choices=("training", "validation", "test"))
    prepare.add_argument(
        "--execution-target",
        "--evaluation-target",
        "--target",
        dest="execution_target",
        choices=("physical", "sdk"),
        default="physical",
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--experiment-id")
    prepare.add_argument(
        "--execution-root", type=Path,
        help="Absolute implementation directory of the frozen wave executor; defaults to this checkout",
    )
    args = parser.parse_args()
    if args.command == "cpu":
        record = qualify_cpu(args.candidate_paths, args.rankings, args.output)
        print(json.dumps({"qualified_candidate_count": record["qualified_candidate_count"]}))
    elif args.command in {
        "qualify-frozen-selection",
        "cpu-selection",
        "evaluation-cpu",
        "cpu-evaluate",
        "qualify-evaluation",
    }:
        record = qualify_frozen_selection(
            args.candidate_paths,
            args.selection,
            args.output,
            split=args.split,
            profile_path=args.profile,
        )
        print(
            json.dumps(
                {
                    "all_passed": record["all_passed"],
                    "qualified_candidate_count": record[
                        "qualified_candidate_count"
                    ],
                },
                sort_keys=True,
            )
        )
    else:
        config = prepare_config(
            dataset_path=args.candidate_paths,
            output_path=args.output,
            mode=args.mode,
            calibration_path=args.calibration_set,
            rankings_path=args.rankings,
            selection_path=args.selection,
            profile_path=args.profile,
            split=args.split,
            execution_target=args.execution_target,
            experiment_id=args.experiment_id,
            execution_root=args.execution_root,
        )
        print(json.dumps({"matrix_count": len(config["matrix"])}))


if __name__ == "__main__":
    main()
