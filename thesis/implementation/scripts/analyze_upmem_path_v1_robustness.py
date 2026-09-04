#!/usr/bin/env python3
"""Audit the surviving UPMEM path-heuristic v1 evidence.

This is deliberately a reporting-only analyzer.  It reads the compact
calibration observations and historical aggregate validation/test reports;
it never reconstructs missing validation/test observations or invokes a
backend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "thesis_results" / "upmem_path_heuristic_v1"
FEATURES = (
    "B_host_dpu",
    "B_mram_wram",
    "I_dpu",
    "N_sync",
    "E_num",
    "P_wram",
)
GROUPS = ("movement", "compute", "coordination")
METRICS = (
    "total_wall_s",
    "session_inclusive_s",
    "kernel_s",
    "h2d_s",
    "d2h_s",
    "request_build_s",
    "request_wave_s",
)
BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 20260903
EPSILON = 1.0
SCHEMA_VERSION = "upmem_path_v1_robustness_v1"
RANKING_COLUMNS = (
    "cell_id",
    "circuit_id",
    "topology_id",
    "candidate_path_id",
    "score",
    "score_rank",
    "runtime_median_s",
    "runtime_raw_mad_s",
    "runtime_min_s",
    "runtime_max_s",
    "is_greedy",
    "is_profile_selected",
    "is_fastest_measured",
    "profile_selected_rank",
    "spearman_score_runtime",
    "flops",
    "peak_intermediate_elements",
    "total_intermediate_writes",
)
CORRELATION_COLUMNS = (
    "row_kind",
    "scope",
    "representation",
    "feature",
    "left_feature",
    "right_feature",
    "variance_population",
    "distinct_count",
    "pair_tie_fraction",
    "pair_count",
    "pearson",
    "spearman",
)
STABILITY_COLUMNS = (
    "resample_index",
    "profile_index",
    "geometric_mean_speedup",
    "minimum_cell_speedup",
    "improved_cell_count",
    "movement_weight",
    "compute_weight",
    "coordination_weight",
    "B_host_dpu_weight",
    "B_mram_wram_weight",
    "I_dpu_weight",
    "N_sync_weight",
    "E_num_weight",
    "P_wram_weight",
    "selected_path_ids_json",
)
LOO_COLUMNS = (
    "omitted_training_circuit_id",
    "evaluation_scope",
    "cell_id",
    "circuit_id",
    "topology_id",
    "profile_index",
    "selected_path_id",
    "greedy_path_id",
    "selected_rank",
    "greedy_median_s",
    "selected_median_s",
    "oracle_median_s",
    "speedup_vs_greedy",
    "speedup_vs_oracle",
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    checked = [_finite(value, "timing") for value in values]
    return float(statistics.median(checked))


def _mad(values: Sequence[float]) -> float:
    center = _median(values)
    return float(statistics.median(abs(value - center) for value in values))


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile requires nonempty values and [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bootstrap_median_interval(
    values: Sequence[float], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if resamples < 1:
        raise ValueError("bootstrap requires a positive resample count")
    checked = tuple(_finite(value, "timing") for value in values)
    rng = random.Random(seed)
    medians = [
        _median([checked[rng.randrange(len(checked))] for _ in checked])
        for _ in range(resamples)
    ]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original, _ in indexed[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("correlation vectors must have equal nonzero length")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _grouped_weights(weights: Mapping[str, object]) -> dict[str, float]:
    return {
        "movement": _finite(weights["B_host_dpu"], "B_host_dpu")
        + _finite(weights["B_mram_wram"], "B_mram_wram"),
        "compute": _finite(weights["I_dpu"], "I_dpu"),
        "coordination": _finite(weights["N_sync"], "N_sync"),
    }


def _grouped_score(normalized: Mapping[str, float], weights: Mapping[str, float]) -> float:
    grouped = _grouped_weights(weights)
    return (
        grouped["movement"] * (normalized["B_host_dpu"] + normalized["B_mram_wram"])
        / 2.0
        + grouped["compute"] * normalized["I_dpu"]
        + grouped["coordination"] * normalized["N_sync"]
    )


def _score(raw: Mapping[str, object], greedy: Mapping[str, object], weights: Mapping[str, object]) -> float:
    normalized = {
        feature: math.log(
            (_finite(raw[feature], feature) + EPSILON)
            / (_finite(greedy[feature], feature) + EPSILON)
        )
        for feature in FEATURES
    }
    return _grouped_score(normalized, weights)


def _profile_weights(row: Mapping[str, object]) -> dict[str, float]:
    raw = json.loads(str(row["weights_json"]))
    if not isinstance(raw, dict) or set(raw) != set(FEATURES):
        raise ValueError("weight-search row has an invalid six-term weight vector")
    weights = {feature: _finite(raw[feature], feature) for feature in FEATURES}
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("weight-search weights do not sum to one")
    return weights


def _read_profiles(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("weight-search candidate table is empty")
    profiles = []
    for index, row in enumerate(rows):
        weights = _profile_weights(row)
        selected = json.loads(row["selected_path_ids_json"])
        if not isinstance(selected, dict):
            raise ValueError("weight-search selected paths must be an object")
        profiles.append(
            {
                "profile_index": index,
                "weights": weights,
                "selected_path_ids": {str(key): str(value) for key, value in selected.items()},
                "equivalent_weight_vector_count": int(row.get("equivalent_weight_vector_count", 1)),
            }
        )
    return profiles


def _candidate_index(dataset: Mapping[str, Any]) -> tuple[
    dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]
]:
    circuits: dict[str, dict[str, Any]] = {}
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    if dataset.get("schema_version") != "upmem_path_candidate_dataset_v1":
        raise ValueError("candidate dataset schema is invalid")
    for circuit in dataset.get("circuits", []):
        circuit_id = str(circuit["circuit_id"])
        circuits[circuit_id] = circuit
        candidates = circuit.get("candidates", [])
        greedy = [item for item in candidates if item.get("is_greedy") is True]
        if len(greedy) != 1:
            raise ValueError(f"{circuit_id} does not have exactly one greedy candidate")
        for candidate in candidates:
            for topology in candidate.get("topologies", []):
                topology_id = str(topology["topology_id"])
                key = (circuit_id, topology_id)
                cell = cells.setdefault(
                    key,
                    {
                        "circuit_id": circuit_id,
                        "topology_id": topology_id,
                        "circuit_split": circuit.get("split"),
                        "greedy_path_id": greedy[0]["candidate_path_id"],
                        "candidates": {},
                    },
                )
                candidate_id = str(candidate["candidate_path_id"])
                cell["candidates"][candidate_id] = {
                    "candidate": candidate,
                    "topology": topology,
                    "features": topology.get("features", {}),
                }
    if not circuits or not cells:
        raise ValueError("candidate dataset contains no circuits or cells")
    return circuits, cells


def _calibration_groups(
    calibration: Mapping[str, Any], cells: Mapping[tuple[str, str], Mapping[str, Any]]
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, Any]]],
    dict[tuple[str, str], list[str]],
]:
    if calibration.get("schema_version") != "upmem_path_runtime_calibration_v1":
        raise ValueError("calibration schema is invalid")
    if calibration.get("all_accuracy_qualified") is not True:
        raise ValueError("calibration accuracy qualification is false")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    observations = calibration.get("observations")
    if not isinstance(observations, list):
        raise ValueError("calibration observations must be a list")
    if calibration.get("sample_count") != len(observations):
        raise ValueError("calibration sample count does not match observations")
    for row in observations:
        cell_id = str(row["cell_id"])
        circuit_id, topology_id = cell_id.split(":", 1)
        key = (circuit_id, topology_id, str(row["candidate_path_id"]))
        if (circuit_id, topology_id) not in cells:
            raise ValueError(f"calibration row references unknown cell {cell_id!r}")
        if key[2] not in cells[(circuit_id, topology_id)]["candidates"]:
            raise ValueError(f"calibration row references unknown path {key[2]!r}")
        if row.get("status") != "success" or row.get("request_transport") != "packed_operation_v1":
            raise ValueError("calibration contains a non-qualified observation")
        if row.get("timing_scope") != "steady_execution_v1":
            raise ValueError("calibration timing scope is not steady_execution_v1")
        if row.get("attempt_type") not in {"warmup", "measurement"}:
            raise ValueError("calibration attempt type is invalid")
        groups.setdefault(key, []).append(dict(row))

    expected_cells: dict[tuple[str, str], list[str]] = {}
    for cell_key, cell in cells.items():
        cell_id = f"{cell_key[0]}:{cell_key[1]}"
        observed = sorted(
            path_id for (circuit, topology, path_id) in groups if (circuit, topology) == cell_key
        )
        if not observed:
            continue
        expected_cells[cell_key] = observed
        for path_id in observed:
            rows = groups[(cell_key[0], cell_key[1], path_id)]
            warmups = [row for row in rows if row["attempt_type"] == "warmup"]
            measurements = [row for row in rows if row["attempt_type"] == "measurement"]
            if len(warmups) != 1 or len(measurements) != 3:
                raise ValueError(f"{cell_id}/{path_id} does not have one warmup and three measurements")
            if sorted(int(row["block"]) for row in warmups) != [0] or sorted(
                int(row["block"]) for row in measurements
            ) != [1, 2, 3]:
                raise ValueError(f"{cell_id}/{path_id} has an invalid block schedule")
            if len({str(row["sample_id"]) for row in rows}) != 4:
                raise ValueError(f"{cell_id}/{path_id} has duplicate sample IDs")
    if len(expected_cells) != int(calibration.get("expected_cell_count", len(expected_cells))):
        raise ValueError("calibration cell count does not match its manifest")
    expected_observations = int(calibration.get("expected_candidate_cell_count", 0)) * 4
    if expected_observations and len(observations) != expected_observations:
        raise ValueError("calibration observation count does not match candidate cells")
    return groups, expected_cells


def _measurement_values(rows: Iterable[Mapping[str, Any]], metric: str) -> list[float]:
    return [
        _finite(row[metric], metric, positive=metric in {"total_wall_s", "session_inclusive_s"})
        for row in rows
        if row["attempt_type"] == "measurement"
    ]


def _stat_record(
    key: tuple[str, str, str], rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    measurements = [row for row in rows if row["attempt_type"] == "measurement"]
    result: dict[str, Any] = {
        "cell_id": f"{key[0]}:{key[1]}",
        "circuit_id": key[0],
        "topology_id": key[1],
        "candidate_path_id": key[2],
        "warmup_count": sum(row["attempt_type"] == "warmup" for row in rows),
        "measurement_count": len(measurements),
        "measurement_blocks": [int(row["block"]) for row in measurements],
        "measurement_observations": [
            {
                "block": int(row["block"]),
                "sample_index": int(row["sample_index"]),
                "sample_id": str(row["sample_id"]),
                "metrics": {metric: _finite(row[metric], metric) for metric in METRICS},
            }
            for row in sorted(measurements, key=lambda item: int(item["block"]))
        ],
        "metrics": {},
    }
    for metric in METRICS:
        values = _measurement_values(measurements, metric)
        low, high = _bootstrap_median_interval(
            values, seed=_stable_seed(seed, key[0], key[1], key[2], metric)
        )
        result["metrics"][metric] = {
            "median": _median(values),
            "raw_mad": _mad(values),
            "minimum": min(values),
            "maximum": max(values),
            "bootstrap_low": low,
            "bootstrap_high": high,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_method": "percentile_bootstrap_median_v1",
        }
    return result


def _stats_map(
    groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]], *, seed: int
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        key: _stat_record(key, rows, seed=seed)
        for key, rows in sorted(groups.items())
    }


def _median_runtime(stats: Mapping[tuple[str, str, str], Mapping[str, Any]], key: tuple[str, str, str]) -> float:
    return float(stats[key]["metrics"]["total_wall_s"]["median"])


def _ranking(
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
    stats: Mapping[tuple[str, str, str], Mapping[str, Any]],
    profile: Mapping[str, Any],
    weights: Mapping[str, object],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cell_key in sorted(cells):
        cell = cells[cell_key]
        observed = [
            path_id
            for (circuit, topology, path_id) in stats
            if (circuit, topology) == cell_key
        ]
        if not observed:
            continue
        greedy_id = str(cell["greedy_path_id"])
        if (cell_key[0], cell_key[1], greedy_id) not in stats:
            raise ValueError(f"greedy path is not measured for {cell_key}")
        greedy_features = cell["candidates"][greedy_id]["features"]
        scored = []
        for path_id in observed:
            candidate = cell["candidates"][path_id]["candidate"]
            scored.append(
                {
                    "path_id": path_id,
                    "score": _score(cell["candidates"][path_id]["features"], greedy_features, weights),
                    "runtime": _median_runtime(stats, (*cell_key, path_id)),
                }
            )
        by_score = sorted(scored, key=lambda item: (item["score"], item["path_id"]))
        by_runtime = sorted(scored, key=lambda item: (item["runtime"], item["path_id"]))
        score_rank = {item["path_id"]: index for index, item in enumerate(by_score, 1)}
        runtime_rank = {
            item["path_id"]: index for index, item in enumerate(by_runtime, 1)
        }
        fastest_id = by_runtime[0]["path_id"]
        score_runtime = _spearman(
            [item["score"] for item in scored], [item["runtime"] for item in scored]
        )
        selected_id = str(profile["selected_path_ids"].get(f"{cell_key[0]}:{cell_key[1]}", ""))
        if selected_id not in score_rank:
            raise ValueError(f"fitted selected path is not measured for {cell_key}")
        runtimes = [item["runtime"] for item in scored]
        greedy_runtime = _median_runtime(stats, (*cell_key, greedy_id))
        selected_runtime = _median_runtime(stats, (*cell_key, selected_id))
        oracle_runtime = min(runtimes)
        denominator = greedy_runtime - oracle_runtime
        greedy_mad = float(
            stats[(*cell_key, greedy_id)]["metrics"]["total_wall_s"]["raw_mad"]
        )
        oracle_mad = float(
            stats[(*cell_key, fastest_id)]["metrics"]["total_wall_s"]["raw_mad"]
        )
        measurable_headroom = denominator > max(greedy_mad, oracle_mad)
        captured = (
            (greedy_runtime - selected_runtime) / denominator
            if measurable_headroom
            else None
        )
        summary = {
            "cell_id": f"{cell_key[0]}:{cell_key[1]}",
            "circuit_id": cell_key[0],
            "topology_id": cell_key[1],
            "measured_candidate_count": len(observed),
            "greedy_path_id": greedy_id,
            "profile_selected_path_id": selected_id,
            "profile_selected_score_rank": score_rank[selected_id],
            "profile_selected_rank": runtime_rank[selected_id],
            "fastest_measured_path_id": fastest_id,
            "score_runtime_spearman": score_runtime,
            "top1_hit": runtime_rank[selected_id] == 1,
            "top3_hit": runtime_rank[selected_id] <= 3,
            "greedy_median_s": greedy_runtime,
            "selected_median_s": selected_runtime,
            "oracle_median_s": oracle_runtime,
            "oracle_speedup_vs_greedy": greedy_runtime / oracle_runtime,
            "selected_speedup_vs_greedy": greedy_runtime / selected_runtime,
            "oracle_regret": selected_runtime / oracle_runtime,
            "greedy_regret": greedy_runtime / oracle_runtime,
            "captured_headroom": captured,
            "captured_headroom_reason": (
                "no_measurable_candidate_pool_headroom" if captured is None else None
            ),
        }
        summaries.append(summary)
        for item in by_score:
            path_id = item["path_id"]
            candidate = cell["candidates"][path_id]["candidate"]
            timing = stats[(*cell_key, path_id)]["metrics"]["total_wall_s"]
            rows.append(
                {
                    "cell_id": summary["cell_id"],
                    "circuit_id": cell_key[0],
                    "topology_id": cell_key[1],
                    "candidate_path_id": path_id,
                    "score": item["score"],
                    "score_rank": score_rank[path_id],
                    "runtime_median_s": timing["median"],
                    "runtime_raw_mad_s": timing["raw_mad"],
                    "runtime_min_s": timing["minimum"],
                    "runtime_max_s": timing["maximum"],
                    "is_greedy": path_id == greedy_id,
                    "is_profile_selected": path_id == selected_id,
                    "is_fastest_measured": path_id == fastest_id,
                    "profile_selected_rank": runtime_rank[selected_id],
                    "spearman_score_runtime": score_runtime,
                    "flops": candidate["conventional_features"]["flops"],
                    "peak_intermediate_elements": candidate["conventional_features"]["peak_intermediate_elements"],
                    "total_intermediate_writes": candidate["conventional_features"]["total_intermediate_writes"],
                }
            )
    return rows, summaries


def _feature_rows(cells: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    vectors: dict[str, dict[str, list[float]]] = {}
    for cell_key, cell in cells.items():
        feasible = [
            path_id
            for path_id, item in cell["candidates"].items()
            if item["topology"].get("feasible") is True
        ]
        greedy_id = str(cell["greedy_path_id"])
        if greedy_id not in feasible:
            raise ValueError(f"greedy path is infeasible for {cell_key}")
        greedy = cell["candidates"][greedy_id]["features"]
        scope = f"{cell_key[0]}:{cell_key[1]}"
        vectors.setdefault(scope, {"raw": [], "normalized": []})
        for representation in ("raw", "normalized"):
            vectors[scope][representation] = []
        for path_id in sorted(feasible):
            raw = cell["candidates"][path_id]["features"]
            vectors[scope]["raw"].append([_finite(raw[name], name) for name in FEATURES])
            vectors[scope]["normalized"].append(
                [
                    math.log((_finite(raw[name], name) + EPSILON) / (_finite(greedy[name], name) + EPSILON))
                    for name in FEATURES
                ]
            )
    combined: dict[str, list[list[float]]] = {"raw": [], "normalized": []}
    for representation in combined:
        for values in vectors.values():
            combined[representation].extend(values[representation])
    vectors["all_feasible_candidates"] = combined
    rows: list[dict[str, Any]] = []
    for scope in sorted(vectors):
        for representation in ("raw", "normalized"):
            matrix = vectors[scope][representation]
            columns = list(zip(*matrix)) if matrix else []
            for index, feature in enumerate(FEATURES):
                values = list(columns[index])
                rows.append(
                    {
                        "row_kind": "feature",
                        "scope": scope,
                        "representation": representation,
                        "feature": feature,
                        "left_feature": None,
                        "right_feature": None,
                        "variance_population": statistics.pvariance(values) if values else None,
                        "distinct_count": len(set(values)),
                        "pair_tie_fraction": _tie_fraction(values),
                        "pair_count": len(values) * (len(values) - 1) // 2,
                        "pearson": None,
                        "spearman": None,
                    }
                )
            for left_index, left in enumerate(FEATURES):
                for right_index in range(left_index + 1, len(FEATURES)):
                    right = FEATURES[right_index]
                    left_values = list(columns[left_index])
                    right_values = list(columns[right_index])
                    rows.append(
                        {
                            "row_kind": "pair",
                            "scope": scope,
                            "representation": representation,
                            "feature": None,
                            "left_feature": left,
                            "right_feature": right,
                            "variance_population": None,
                            "distinct_count": None,
                            "pair_tie_fraction": None,
                            "pair_count": len(left_values) * (len(left_values) - 1) // 2,
                            "pearson": _pearson(left_values, right_values),
                            "spearman": _spearman(left_values, right_values),
                        }
                    )
    return rows


def _tie_fraction(values: Sequence[float]) -> float | None:
    pair_count = len(values) * (len(values) - 1) // 2
    if pair_count == 0:
        return None
    equal_pairs = sum(
        values[left] == values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return equal_pairs / pair_count


def _bootstrap_profiles(
    groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keys_by_cell: dict[tuple[str, str], list[str]] = {}
    values_by_key: dict[tuple[str, str, str], dict[int, float]] = {}
    for key, rows in groups.items():
        measurements = [row for row in rows if row["attempt_type"] == "measurement"]
        values_by_key[key] = {int(row["block"]): _finite(row["total_wall_s"], "total_wall_s", positive=True) for row in measurements}
        keys_by_cell.setdefault((key[0], key[1]), []).append(key[2])
    for cell_key in keys_by_cell:
        keys_by_cell[cell_key].sort()
    for profile in profiles:
        for cell_key, path_ids in keys_by_cell.items():
            selected = profile["selected_path_ids"].get(f"{cell_key[0]}:{cell_key[1]}")
            if selected not in path_ids:
                raise ValueError("a finite weight-search profile selects an unmeasured path")

    output: list[dict[str, Any]] = []
    selected_frequency: dict[str, dict[str, int]] = {}
    profile_frequency = {str(index): 0 for index in range(len(profiles))}
    for replicate in range(resamples):
        rng = random.Random(_stable_seed(seed, "paired_block", replicate))
        boot_medians: dict[tuple[str, str, str], float] = {}
        for cell_key in sorted(keys_by_cell):
            blocks = sorted(next(iter(values_by_key[(cell_key[0], cell_key[1], path_id)]) for path_id in keys_by_cell[cell_key]))
            draws = [blocks[rng.randrange(len(blocks))] for _ in blocks]
            for path_id in keys_by_cell[cell_key]:
                values = values_by_key[(cell_key[0], cell_key[1], path_id)]
                boot_medians[(cell_key[0], cell_key[1], path_id)] = _median([values[block] for block in draws])

        evaluated = []
        for profile in profiles:
            speedups = []
            selected_ids: dict[str, str] = {}
            for cell_key in sorted(keys_by_cell):
                cell_id = f"{cell_key[0]}:{cell_key[1]}"
                greedy_id = str(cells[cell_key]["greedy_path_id"])
                selected_id = str(profile["selected_path_ids"][cell_id])
                speedups.append(
                    boot_medians[(cell_key[0], cell_key[1], greedy_id)]
                    / boot_medians[(cell_key[0], cell_key[1], selected_id)]
                )
                selected_ids[cell_id] = selected_id
            geo = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
            minimum = min(speedups)
            evaluated.append(
                (
                    (round(geo, 12), round(minimum, 12), sum(value > 1.0 for value in speedups), tuple(-value for value in profile["weights"].values())),
                    profile,
                    geo,
                    minimum,
                    speedups,
                    selected_ids,
                )
            )
        _, best, geo, minimum, speedups, selected_ids = max(evaluated, key=lambda item: item[0])
        profile_index = int(best["profile_index"])
        profile_frequency[str(profile_index)] += 1
        for cell_id, path_id in selected_ids.items():
            selected_frequency.setdefault(cell_id, {})[path_id] = selected_frequency.setdefault(cell_id, {}).get(path_id, 0) + 1
        weights = best["weights"]
        grouped = _grouped_weights(weights)
        output.append(
            {
                "resample_index": replicate,
                "profile_index": profile_index,
                "geometric_mean_speedup": geo,
                "minimum_cell_speedup": minimum,
                "improved_cell_count": sum(value > 1.0 for value in speedups),
                "weights": dict(weights),
                "grouped_weights": grouped,
                "selected_path_ids": selected_ids,
            }
        )
    summary = {
        "resamples": resamples,
        "seed": seed,
        "method": "paired_measurement_block_bootstrap_over_finite_selection_equivalence_profiles_v1",
        "finite_profile_count": len(profiles),
        "finite_profile_source_semantics": "best_lexicographic_weight_vector_per_distinct_training_path_selection",
        "profile_frequency": profile_frequency,
        "selected_path_frequency": selected_frequency,
        "zero_weight_fraction": {
            feature: sum(
                math.isclose(
                    float(row["weights"][feature]), 0.0, abs_tol=1e-15
                )
                for row in output
            )
            / len(output)
            for feature in FEATURES
        },
        "quantiles": _stability_quantiles(output),
    }
    return output, summary


def _stability_quantiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    fields = [
        "geometric_mean_speedup",
        "minimum_cell_speedup",
        "movement_weight",
        "compute_weight",
        "coordination_weight",
    ]
    result = {}
    for field in fields:
        values = []
        for row in rows:
            if field in row:
                values.append(float(row[field]))
            else:
                values.append(float(row["grouped_weights"][field.removesuffix("_weight")]))
        result[field] = {
            "minimum": min(values),
            "q025": _percentile(values, 0.025),
            "median": _percentile(values, 0.5),
            "q975": _percentile(values, 0.975),
            "maximum": max(values),
        }
    return result


def _profile_evaluation(
    profile: Mapping[str, Any],
    cell_keys: Sequence[tuple[str, str]],
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
    stats: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[float, float, list[float]]:
    speedups = []
    for cell_key in sorted(cell_keys):
        cell_id = f"{cell_key[0]}:{cell_key[1]}"
        greedy_id = str(cells[cell_key]["greedy_path_id"])
        selected_id = str(profile["selected_path_ids"][cell_id])
        speedups.append(
            _median_runtime(stats, (*cell_key, greedy_id))
            / _median_runtime(stats, (*cell_key, selected_id))
        )
    return (
        math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        min(speedups),
        speedups,
    )


def _leave_one_out(
    cells: Mapping[tuple[str, str], Mapping[str, Any]],
    stats: Mapping[tuple[str, str, str], Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available_cells = {key[:2] for key in stats}
    circuit_ids = sorted({cell[0] for cell in available_cells})
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for omitted in circuit_ids:
        fit_cells = sorted(cell for cell in available_cells if cell[0] != omitted)
        # The calibration table is complete for every candidate, so select the
        # finite profile using all retained training cells and fixed medians.
        evaluated = []
        for profile in profiles:
            geo, minimum, speedups = _profile_evaluation(profile, fit_cells, cells, stats)
            evaluated.append(
                (
                    (round(geo, 12), round(minimum, 12), sum(value > 1.0 for value in speedups), tuple(-value for value in profile["weights"].values())),
                    profile,
                    geo,
                    minimum,
                )
            )
        _, best, fit_geo, fit_min = max(evaluated, key=lambda item: item[0])
        held_cells = sorted(cell for cell in available_cells if cell[0] == omitted)
        fold_speeds = []
        for evaluation_scope, eval_cells in (("fit", fit_cells), ("held_out", held_cells)):
            for cell_key in eval_cells:
                cell_id = f"{cell_key[0]}:{cell_key[1]}"
                greedy_id = str(cells[cell_key]["greedy_path_id"])
                selected_id = str(best["selected_path_ids"][cell_id])
                measured = [
                    _median_runtime(stats, (*cell_key, path_id))
                    for path_id in cells[cell_key]["candidates"]
                    if (cell_key[0], cell_key[1], path_id) in stats
                ]
                greedy = _median_runtime(stats, (*cell_key, greedy_id))
                selected = _median_runtime(stats, (*cell_key, selected_id))
                oracle = min(measured)
                ranking = sorted(
                    (
                        _median_runtime(stats, (*cell_key, path_id)),
                        path_id,
                    )
                    for path_id in cells[cell_key]["candidates"]
                    if (cell_key[0], cell_key[1], path_id) in stats
                )
                rank = next(index for index, (_, path_id) in enumerate(ranking, 1) if path_id == selected_id)
                speed = greedy / selected
                fold_speeds.append((evaluation_scope, speed))
                rows.append(
                    {
                        "omitted_training_circuit_id": omitted,
                        "evaluation_scope": evaluation_scope,
                        "cell_id": cell_id,
                        "circuit_id": cell_key[0],
                        "topology_id": cell_key[1],
                        "profile_index": best["profile_index"],
                        "selected_path_id": selected_id,
                        "greedy_path_id": greedy_id,
                        "selected_rank": rank,
                        "greedy_median_s": greedy,
                        "selected_median_s": selected,
                        "oracle_median_s": oracle,
                        "speedup_vs_greedy": speed,
                        "speedup_vs_oracle": oracle / selected,
                    }
                )
        held_values = [value for scope, value in fold_speeds if scope == "held_out"]
        summaries.append(
            {
                "omitted_training_circuit_id": omitted,
                "profile_index": best["profile_index"],
                "weights": best["weights"],
                "grouped_weights": _grouped_weights(best["weights"]),
                "fit_cell_count": len(fit_cells),
                "held_out_cell_count": len(held_cells),
                "fit_geometric_mean_speedup": fit_geo,
                "fit_minimum_cell_speedup": fit_min,
                "held_out_geometric_mean_speedup": math.exp(sum(math.log(value) for value in held_values) / len(held_values)),
                "held_out_minimum_cell_speedup": min(held_values),
                "held_out_speedups": held_values,
            }
        )
    return rows, summaries


def _historical_context(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": document.get("schema_version"),
        "split": document.get("split"),
        "sample_count": document.get("sample_count"),
        "session_count": document.get("session_count"),
        "all_successful": document.get("all_successful"),
        "geometric_mean_upmem_vs_greedy_speedup": document.get("geometric_mean_upmem_vs_greedy_speedup"),
        "minimum_cell_speedup": document.get("minimum_cell_speedup"),
        "maximum_cell_speedup": document.get("maximum_cell_speedup"),
        "improved_cell_count": document.get("improved_cell_count"),
        "unchanged_cell_count": document.get("unchanged_cell_count"),
        "regressed_cell_count": document.get("regressed_cell_count"),
        "comparisons": document.get("comparisons", []),
        "path_statistics": document.get("path_statistics", []),
        "raw_observations_available": False,
        "raw_uncertainty_available": False,
        "raw_artifact_recovered": False,
        "raw_artifact_sha256_recorded": document.get("raw_artifact_sha256"),
        "disposition": "historical_aggregate_only; no raw observations reconstructed",
    }


def _bv18_interpretation(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in document.get("path_statistics", [])
        if row.get("circuit_id") == "bv_18q"
    ]
    result = []
    for topology_id in sorted({str(row["topology_id"]) for row in rows}):
        cell = [row for row in rows if row["topology_id"] == topology_id]
        greedy = next(row for row in cell if "greedy" in row.get("roles", []))
        minimum_flops = next(
            row for row in cell if "minimum_flops" in row.get("roles", [])
        )
        selected = next(
            row for row in cell if "upmem_selected" in row.get("roles", [])
        )
        fastest = min(
            cell,
            key=lambda row: (
                float(row["median_total_wall_s"]),
                str(row["candidate_path_id"]),
            ),
        )
        greedy_time = float(greedy["median_total_wall_s"])
        selected_time = float(selected["median_total_wall_s"])
        fastest_time = float(fastest["median_total_wall_s"])
        uncertainty_scale = max(
            float(greedy["raw_mad_total_wall_s"]),
            float(selected["raw_mad_total_wall_s"]),
            float(fastest["raw_mad_total_wall_s"]),
        )
        candidate_range = max(
            float(row["median_total_wall_s"]) for row in cell
        ) - min(float(row["median_total_wall_s"]) for row in cell)
        if candidate_range <= uncertainty_scale:
            classification = "C_indistinguishable_within_reported_variability"
        elif greedy["candidate_path_id"] == fastest["candidate_path_id"]:
            classification = "A_greedy_is_fastest_measured"
        elif selected["candidate_path_id"] != fastest["candidate_path_id"]:
            classification = "B_faster_measured_candidate_missed"
        else:
            classification = "heuristic_captured_measured_headroom"
        result.append(
            {
                "circuit_id": "bv_18q",
                "topology_id": topology_id,
                "classification": classification,
                "greedy_path_id": greedy["candidate_path_id"],
                "minimum_flops_path_id": minimum_flops["candidate_path_id"],
                "heuristic_selected_path_id": selected["candidate_path_id"],
                "fastest_measured_path_id": fastest["candidate_path_id"],
                "paths_coincide": {
                    "minimum_flops_and_selected": minimum_flops["candidate_path_id"]
                    == selected["candidate_path_id"],
                    "greedy_and_fastest": greedy["candidate_path_id"]
                    == fastest["candidate_path_id"],
                },
                "greedy_median_s": greedy_time,
                "selected_median_s": selected_time,
                "fastest_median_s": fastest_time,
                "reported_raw_mad_scale_s": uncertainty_scale,
                "oracle_regret": selected_time / fastest_time,
                "available_headroom": greedy_time / fastest_time,
                "raw_observations_available": False,
            }
        )
    return result


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def analyze(
    calibration_path: Path,
    candidate_path: Path,
    profile_path: Path,
    weight_search_path: Path,
    validation_path: Path,
    test_path: Path,
    output_dir: Path,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    dataset = _load_object(candidate_path)
    calibration = _load_object(calibration_path)
    profile = _load_object(profile_path)
    validation = _load_object(validation_path)
    test = _load_object(test_path)
    _, cells = _candidate_index(dataset)
    groups, measured_candidates = _calibration_groups(calibration, cells)
    stats = _stats_map(groups, seed=bootstrap_seed)
    profiles = _read_profiles(weight_search_path)
    exact_weights = profile.get("weights")
    if not isinstance(exact_weights, dict) or set(exact_weights) != set(FEATURES):
        raise ValueError("fitted profile lacks exact six-term weights")
    frozen_profiles = [
        item for item in profiles if all(
            math.isclose(item["weights"][feature], float(exact_weights[feature]), abs_tol=1e-15)
            for feature in FEATURES
        )
    ]
    if len(frozen_profiles) != 1:
        raise ValueError("weight-search table does not contain exactly one frozen fitted profile")
    frozen_profile = frozen_profiles[0]
    if frozen_profile["selected_path_ids"] != {
        str(key): str(value) for key, value in profile["selected_path_ids"].items()
    }:
        raise ValueError("weight-search frozen selections do not match fitted profile")

    ranking_rows, ranking_summary = _ranking(cells, stats, frozen_profile, exact_weights)
    correlation_rows = _feature_rows(cells)
    stability_rows, stability_summary = _bootstrap_profiles(
        groups,
        cells,
        profiles,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    loo_rows, loo_summary = _leave_one_out(cells, stats, profiles)
    grouped = _grouped_weights(exact_weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "existing_v1_ranking.csv", RANKING_COLUMNS, ranking_rows)
    _write_csv(output_dir / "existing_v1_feature_correlations.csv", CORRELATION_COLUMNS, correlation_rows)
    stability_csv_rows = []
    for row in stability_rows:
        weights = row["weights"]
        grouped_row = row["grouped_weights"]
        stability_csv_rows.append(
            {
                "resample_index": row["resample_index"],
                "profile_index": row["profile_index"],
                "geometric_mean_speedup": row["geometric_mean_speedup"],
                "minimum_cell_speedup": row["minimum_cell_speedup"],
                "improved_cell_count": row["improved_cell_count"],
                "movement_weight": grouped_row["movement"],
                "compute_weight": grouped_row["compute"],
                "coordination_weight": grouped_row["coordination"],
                **{f"{feature}_weight": weights[feature] for feature in FEATURES},
                "selected_path_ids_json": row["selected_path_ids"],
            }
        )
    _write_csv(output_dir / "existing_v1_weight_stability.csv", STABILITY_COLUMNS, stability_csv_rows)
    _write_csv(output_dir / "existing_v1_leave_one_out.csv", LOO_COLUMNS, loo_rows)

    input_paths = {
        "calibration": calibration_path,
        "candidate_paths": candidate_path,
        "fitted_profile": profile_path,
        "weight_search_candidates": weight_search_path,
        "historical_validation": validation_path,
        "historical_test": test_path,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_source_head": _git_head(),
        "analysis_script_sha256": _sha256_file(Path(__file__)),
        "input_sha256": {name: _sha256_file(path) for name, path in input_paths.items()},
        "candidate_source_sha": dataset.get("source_sha"),
        "physical_execution_source_sha": calibration.get("physical_execution_source_sha"),
        "candidate_set_sha256": dataset.get("candidate_set_sha256", calibration.get("candidate_set_sha256")),
        "calibration_experiment_id": calibration.get("experiment_id"),
        "calibration_run_id": calibration.get("run_id"),
        "calibration_timing_scope": calibration.get("timing_scope"),
        "calibration_observation_status": {
            "compact_measurement_rows_available": True,
            "measurement_rows": len([row for row in calibration.get("observations", []) if row.get("attempt_type") == "measurement"]),
            "warmup_rows": len([row for row in calibration.get("observations", []) if row.get("attempt_type") == "warmup"]),
            "original_archive_available": False,
            "note": "analysis uses only surviving tracked calibration JSON rows",
        },
        "historical_evaluation_status": {
            "validation": _historical_context(validation),
            "test": _historical_context(test),
        },
        "historical_bv18_interpretation": _bv18_interpretation(test),
        "exact_weights": {feature: float(exact_weights[feature]) for feature in FEATURES},
        "grouped_weights": grouped,
        "fitted_profile_feature_model": profile.get("feature_model"),
        "normalization": {
            "formula": "log((candidate + epsilon) / (greedy + epsilon))",
            "epsilon_per_feature": {feature: EPSILON for feature in FEATURES},
        },
        "calibration_cells": [
            {
                "cell_id": f"{key[0]}:{key[1]}",
                "circuit_id": key[0],
                "topology_id": key[1],
                "candidate_count": len(path_ids),
                "candidate_path_ids": path_ids,
            }
            for key, path_ids in sorted(measured_candidates.items())
        ],
        "calibration_stats": [stats[key] for key in sorted(stats)],
        "ranking_metrics": ranking_summary,
        "feature_correlations": correlation_rows,
        "feature_dependency_note": (
            "B_host_dpu, B_mram_wram, I_dpu and N_sync are scored from the six-term profile; "
            "the fitted profile has zero E_num/P_wram weights, and grouped movement is the "
            "sum of the two movement weights. Correlations are descriptive and do not imply causality."
        ),
        "bootstrap_refits": stability_summary,
        "leave_one_training_circuit_out": loo_summary,
        "claim_boundary": (
            "Reporting-only pilot audit. Calibration uncertainty is computed from surviving "
            "compact measurement rows. Historical validation/test raw observations are unavailable; "
            "their aggregate reports are descriptive only and no raw uncertainty is reconstructed."
        ),
        "output_files": [
            "existing_v1_robustness.json",
            "existing_v1_ranking.csv",
            "existing_v1_feature_correlations.csv",
            "existing_v1_weight_stability.csv",
            "existing_v1_leave_one_out.csv",
        ],
    }
    (output_dir / "existing_v1_robustness.json").write_bytes(_canonical_bytes(result))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_BASE / "physical_calibration/path_runtime_calibration.json")
    parser.add_argument("--candidate-paths", type=Path, default=DEFAULT_BASE / "software/candidate_paths.json")
    parser.add_argument("--profile", type=Path, default=DEFAULT_BASE / "fit/physical_speedup_fit_v1.json")
    parser.add_argument("--weight-search", type=Path, default=DEFAULT_BASE / "fit/weight_search_candidates.csv")
    parser.add_argument("--validation", type=Path, default=DEFAULT_BASE / "validation/heuristic_validation.json")
    parser.add_argument("--test", type=Path, default=DEFAULT_BASE / "test/heuristic_test.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    try:
        result = analyze(
            args.calibration,
            args.candidate_paths,
            args.profile,
            args.weight_search,
            args.validation,
            args.test,
            args.output_dir,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (OSError, ValueError, json.JSONDecodeError, KeyError, IndexError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "written",
                "output_dir": str(args.output_dir.resolve()),
                "calibration_cells": len(result["calibration_cells"]),
                "bootstrap_refits": result["bootstrap_refits"]["resamples"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
