"""Pure offline analysis for measured UPMEM wave-path pools.

The caller supplies identity-verified candidate features, measured rows, an
explicit expected row set, and either fixed profiles or explicit training-fold
metadata.  This module performs no evidence I/O and executes no backend.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random
import statistics
from typing import Any

import numpy as np

from fit_upmem_wave_paths import (
    FIT_SPLITS,
    WaveFitResult,
    _checked_row,
    _coerce_cells,
    _row_key,
    _score,
    fit_upmem_wave_paths,
    select_wave_model_form,
)
from quantum_bench.upmem.path_heuristic import (
    FEATURE_NAMES,
    FeatureModelDecision,
    RawFeatureVector,
    WeightVector,
    normalize_features,
)


DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20260904
DEFAULT_FIT_SAMPLE_COUNT = 100_000
DEFAULT_FIT_SEED = 20260903
_NEAR_ZERO_HEADROOM = 1e-12
_MODEL_FORMS = ("six_term", "grouped")
_WAVE_TOPOLOGIES = ("1dpu_t8", "4dpu_t8")

_RowKey = tuple[str, str, str, str, str]
_CheckedRow = tuple[str, str, str, str, str, float]


@dataclass(frozen=True, slots=True)
class FixedWaveProfile:
    """A fixed profile accepted by the descriptive comparison."""

    profile_id: str
    weights: WeightVector
    model: FeatureModelDecision
    selected_path_ids: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _MeasurementPool:
    cells: dict[str, tuple[str, dict[str, RawFeatureVector]]]
    rows: dict[_RowKey, _CheckedRow]
    source_rows: dict[_RowKey, Mapping[str, object]]
    groups: dict[str, dict[str, dict[str, dict[str, float]]]]


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


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
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
    return tuple(ranks)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("correlation vectors must have equal nonzero length")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    left_centered = [value - left_mean for value in left_ranks]
    right_centered = [value - right_mean for value in right_ranks]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / denominator


def _competition_rank(values: Mapping[str, float], selected: str) -> int:
    value = values[selected]
    return 1 + sum(other < value for other in values.values())


def _median_mad(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty measured pool")
    center = float(statistics.median(values))
    return center, float(statistics.median(abs(value - center) for value in values))


def _coerce_profile(value: object, index: int) -> FixedWaveProfile:
    if isinstance(value, FixedWaveProfile):
        profile = value
    elif isinstance(value, WaveFitResult):
        profile = FixedWaveProfile(
            profile_id=f"profile_{index}",
            weights=value.weights,
            model=value.model,
            selected_path_ids=tuple(value.selected_path_ids),
        )
    else:
        raise TypeError("profiles must be FixedWaveProfile or WaveFitResult values")
    if not isinstance(profile.profile_id, str) or not profile.profile_id:
        raise ValueError("fixed profile ID must be a nonempty string")
    if not isinstance(profile.weights, WeightVector):
        raise TypeError("fixed profile weights must be a WeightVector")
    if not isinstance(profile.model, FeatureModelDecision) or profile.model.mode not in {
        "six_term",
        "grouped",
    }:
        raise ValueError("fixed profile model mode is invalid")
    if any(
        not isinstance(cell_id, str)
        or not cell_id
        or not isinstance(candidate_id, str)
        or not candidate_id
        for cell_id, candidate_id in profile.selected_path_ids
    ):
        raise ValueError("fixed profile selected path identities are invalid")
    if len(profile.selected_path_ids) != len(
        {cell_id for cell_id, _ in profile.selected_path_ids}
    ):
        raise ValueError("selected_path_ids contains duplicate cell IDs")
    if profile.weights.numeric != 0.0 or profile.weights.wram != 0.0:
        raise ValueError("E_num and P_wram must be inactive in a wave profile")
    if profile.model.mode == "grouped":
        if profile.model.active_features != ("movement", "compute", "coordination"):
            raise ValueError("grouped profile must declare all three grouped features")
        if not math.isclose(
            profile.weights.host_dpu,
            profile.weights.mram_wram,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("grouped profile must use equal-half movement weights")
    else:
        active = set(profile.model.active_features)
        if any(
            profile.weights.as_mapping()[feature] != 0.0
            for feature in FEATURE_NAMES
            if feature not in active
        ):
            raise ValueError("profile has nonzero weight outside declared active features")
    return profile


def _measurement_pool(
    cells: Mapping[str, Mapping[str, object]],
    measured_rows: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object] | Sequence[object]],
    split: str,
) -> _MeasurementPool:
    if split not in FIT_SPLITS:
        raise ValueError("wave analysis accepts the explicit training/development split only")
    cell_data = _coerce_cells(cells)
    declared: set[_RowKey] = set()
    for row in expected_rows:
        key = _row_key(row, split)
        if key[0] not in cell_data or key[1] not in cell_data[key[0]][1]:
            raise ValueError(f"expected row references unknown cell or candidate: {key[:2]!r}")
        if key in declared:
            raise ValueError("expected row set contains duplicate identities")
        declared.add(key)
    if not declared:
        raise ValueError("expected_rows must be a nonempty declared row set")

    observed: dict[_RowKey, _CheckedRow] = {}
    source_rows: dict[_RowKey, Mapping[str, object]] = {}
    for row in measured_rows:
        if not isinstance(row, Mapping):
            raise TypeError("measured_rows must contain mappings")
        checked = _checked_row(row, split)
        key = checked[:5]
        if key in observed:
            raise ValueError("measured rows contain duplicate identities")
        if key[0] not in cell_data or key[1] not in cell_data[key[0]][1]:
            raise ValueError(f"measured row references unknown cell or candidate: {key[:2]!r}")
        observed[key] = checked
        source_rows[key] = row
    if set(observed) != declared:
        missing = sorted(declared.difference(observed))
        extra = sorted(set(observed).difference(declared))
        raise ValueError(
            f"measured rows do not match declared expected row set: "
            f"missing={missing!r}, extra={extra!r}"
        )

    groups: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    measurement_counts = dict.fromkeys(cell_data, 0)
    for row in sorted(observed.values(), key=lambda item: item[:5]):
        if row[4] != "measurement":
            continue
        measurement_counts[row[0]] += 1
        groups.setdefault(row[0], {}).setdefault(row[2], {}).setdefault(row[3], {})[
            row[1]
        ] = row[5]
    missing_cells = sorted(cell_id for cell_id, count in measurement_counts.items() if count == 0)
    if missing_cells:
        raise ValueError(f"cells have zero measured observations: {missing_cells!r}")

    for cell_id, rounds in groups.items():
        greedy_id = cell_data[cell_id][0]
        for round_id, blocks in rounds.items():
            if not any(greedy_id in values for values in blocks.values()):
                raise ValueError(f"missing greedy pair for {cell_id!r}/{round_id!r}")
            greedy_blocks = {
                block for block, values in blocks.items() if greedy_id in values
            }
            for block, values in blocks.items():
                if greedy_id not in values:
                    raise ValueError(f"missing greedy pair for {cell_id!r}/{block!r}")
            candidate_blocks: dict[str, set[str]] = {}
            for block, values in blocks.items():
                for candidate_id in values:
                    candidate_blocks.setdefault(candidate_id, set()).add(block)
            for candidate_id, block_ids in candidate_blocks.items():
                if block_ids != greedy_blocks:
                    raise ValueError(
                        f"missing measurement block for {cell_id!r}/{candidate_id!r}"
                    )
    return _MeasurementPool(cell_data, observed, source_rows, groups)


def _cell_times(pool: _MeasurementPool, cell_id: str) -> dict[str, tuple[float, ...]]:
    times: dict[str, list[float]] = {}
    for key, row in pool.rows.items():
        if key[0] == cell_id and key[4] == "measurement":
            times.setdefault(key[1], []).append(row[5])
    return {candidate: tuple(values) for candidate, values in sorted(times.items())}


def _cell_logs(pool: _MeasurementPool, cell_id: str) -> dict[str, tuple[float, ...]]:
    greedy_id = pool.cells[cell_id][0]
    logs: dict[str, list[float]] = {greedy_id: []}
    for round_blocks in pool.groups[cell_id].values():
        for values in round_blocks.values():
            greedy_time = values[greedy_id]
            for candidate_id, candidate_time in values.items():
                logs.setdefault(candidate_id, []).append(
                    math.log(greedy_time) - math.log(candidate_time)
                )
    return {candidate: tuple(values) for candidate, values in sorted(logs.items())}


def _profile_cell_facts(
    pool: _MeasurementPool,
    profile: FixedWaveProfile,
) -> dict[str, Any]:
    selected = dict(profile.selected_path_ids)
    expected_cells = set(pool.cells)
    if set(selected) != expected_cells:
        raise ValueError(f"profile {profile.profile_id!r} does not select every analysis cell")
    result: list[dict[str, Any]] = []
    for cell_id, (greedy_id, raw_values) in pool.cells.items():
        times = _cell_times(pool, cell_id)
        measured_ids = tuple(sorted(times))
        selected_id = selected[cell_id]
        if selected_id not in times:
            raise ValueError(
                f"profile {profile.profile_id!r} selects an unmeasured path: "
                f"{cell_id}/{selected_id}"
            )
        greedy_raw = raw_values[greedy_id]
        normalized = {
            candidate_id: normalize_features(raw, greedy_raw)
            for candidate_id, raw in raw_values.items()
            if candidate_id in times
        }
        scores = {
            candidate_id: _score(normalized[candidate_id], profile.model, profile.weights)
            for candidate_id in measured_ids
        }
        medians: dict[str, float] = {}
        mads: dict[str, float] = {}
        for candidate_id, values in times.items():
            medians[candidate_id], mads[candidate_id] = _median_mad(values)
        logs = _cell_logs(pool, cell_id)
        paired_normalized_times = {
            candidate_id: math.exp(-statistics.fmean(logs[candidate_id]))
            for candidate_id in measured_ids
        }
        paired_log_time_mads = {
            candidate_id: _median_mad(tuple(-value for value in logs[candidate_id]))[1]
            for candidate_id in measured_ids
        }
        score_ranks = {
            candidate_id: _competition_rank(scores, candidate_id)
            for candidate_id in measured_ids
        }
        runtime_ranks = {
            candidate_id: _competition_rank(paired_normalized_times, candidate_id)
            for candidate_id in measured_ids
        }
        oracle_id = min(
            measured_ids,
            key=lambda candidate_id: (
                paired_normalized_times[candidate_id],
                candidate_id,
            ),
        )
        greedy_time = paired_normalized_times[greedy_id]
        selected_time = paired_normalized_times[selected_id]
        oracle_time = paired_normalized_times[oracle_id]
        log_headroom = -math.log(oracle_time)
        measurable = log_headroom > max(
            _NEAR_ZERO_HEADROOM, paired_log_time_mads[oracle_id]
        )
        denominator = 1.0 - oracle_time
        headroom = (
            (1.0 - selected_time) / denominator
            if measurable
            else None
        )
        candidate_facts = [
            {
                "candidate_path_id": candidate_id,
                "score": scores[candidate_id],
                "score_rank": score_ranks[candidate_id],
                "runtime_rank": runtime_ranks[candidate_id],
                "paired_normalized_time": paired_normalized_times[candidate_id],
                "paired_speedup_vs_greedy": 1.0 / paired_normalized_times[candidate_id],
                "raw_runtime_median_s": medians[candidate_id],
                "raw_runtime_mad_s": mads[candidate_id],
                "measurement_count": len(times[candidate_id]),
                "is_greedy": candidate_id == greedy_id,
                "is_profile_selected": candidate_id == selected_id,
                "is_oracle": candidate_id == oracle_id,
            }
            for candidate_id in measured_ids
        ]
        result.append(
            {
                "cell_id": cell_id,
                "measured_candidate_count": len(measured_ids),
                "greedy_path_id": greedy_id,
                "selected_path_id": selected_id,
                "selected_score_rank": score_ranks[selected_id],
                "selected_rank": runtime_ranks[selected_id],
                "top1": runtime_ranks[selected_id] == 1,
                "top3": runtime_ranks[selected_id] <= 3,
                "oracle_path_id": oracle_id,
                "greedy_paired_normalized_time": greedy_time,
                "selected_paired_normalized_time": selected_time,
                "oracle_paired_normalized_time": oracle_time,
                "greedy_raw_runtime_median_s": medians[greedy_id],
                "greedy_raw_runtime_mad_s": mads[greedy_id],
                "selected_raw_runtime_median_s": medians[selected_id],
                "selected_raw_runtime_mad_s": mads[selected_id],
                "oracle_raw_runtime_median_s": medians[oracle_id],
                "oracle_raw_runtime_mad_s": mads[oracle_id],
                "selected_speedup_vs_greedy": 1.0 / selected_time,
                "oracle_speedup_vs_greedy": 1.0 / oracle_time,
                "oracle_regret": selected_time / oracle_time,
                "primary_runtime_basis": "exp(-mean_paired_log_ratio)",
                "headroom": headroom,
                "headroom_reason": None
                if measurable
                else "no_positive_measurable_headroom",
                "score_runtime_spearman": _spearman(
                    [scores[candidate_id] for candidate_id in measured_ids],
                    [
                        paired_normalized_times[candidate_id]
                        for candidate_id in measured_ids
                    ],
                ),
                "candidates": candidate_facts,
            }
        )
    return {
        "profile_id": profile.profile_id,
        "model_form": profile.model.mode,
        "selected_path_ids": dict(profile.selected_path_ids),
        "cells": result,
    }


def _paired_log_speedups(
    pool: _MeasurementPool,
    selected: Mapping[str, str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for cell_id in sorted(pool.cells):
        logs = _cell_logs(pool, cell_id)
        candidate_id = selected[cell_id]
        if candidate_id not in logs or not logs[candidate_id]:
            raise ValueError(f"selected path has no paired observations: {cell_id}/{candidate_id}")
        result[cell_id] = statistics.fmean(logs[candidate_id])
    return result


def _shared_round_blocks(
    pool: _MeasurementPool,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for cell_id in sorted(pool.groups):
        for round_id, blocks in pool.groups[cell_id].items():
            current = tuple(sorted(blocks))
            previous = result.get(round_id)
            if previous is not None and previous != current:
                raise ValueError(
                    f"round {round_id!r} has inconsistent block sets across cells"
                )
            result[round_id] = current
    return dict(sorted(result.items()))


def _drawn_log_speedups(
    pool: _MeasurementPool,
    block_draws: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for cell_id in sorted(pool.groups):
        greedy_id = pool.cells[cell_id][0]
        candidate_logs: dict[str, list[float]] = {greedy_id: []}
        for round_id in sorted(pool.groups[cell_id]):
            blocks = pool.groups[cell_id][round_id]
            if round_id not in block_draws:
                raise ValueError(f"missing bootstrap block draw for round {round_id!r}")
            for block_id in block_draws[round_id]:
                values = blocks[block_id]
                greedy_time = values[greedy_id]
                for candidate_id, candidate_time in values.items():
                    candidate_logs.setdefault(candidate_id, []).append(
                        math.log(greedy_time) - math.log(candidate_time)
                    )
        result[cell_id] = {
            candidate_id: statistics.fmean(logs)
            for candidate_id, logs in sorted(candidate_logs.items())
        }
    return result


def _bootstrap_log_differences(
    pool: _MeasurementPool,
    selected_a: Mapping[str, str],
    selected_b: Mapping[str, str],
    *,
    resamples: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    shared_blocks = _shared_round_blocks(pool)
    geometric_differences: list[float] = []
    worst_differences: list[float] = []
    for _ in range(resamples):
        block_draws = {
            round_id: tuple(rng.choice(blocks) for _ in blocks)
            for round_id, blocks in shared_blocks.items()
        }
        logs_by_cell = _drawn_log_speedups(pool, block_draws)
        cell_a = {
            cell_id: logs_by_cell[cell_id][selected_a[cell_id]]
            for cell_id in sorted(pool.cells)
        }
        cell_b = {
            cell_id: logs_by_cell[cell_id][selected_b[cell_id]]
            for cell_id in sorted(pool.cells)
        }
        geometric_differences.append(
            statistics.fmean(cell_a.values()) - statistics.fmean(cell_b.values())
        )
        worst_differences.append(min(cell_a.values()) - min(cell_b.values()))
    return geometric_differences, worst_differences


def _fit_cells(
    cell_data: Mapping[str, tuple[str, dict[str, RawFeatureVector]]],
    cell_ids: Sequence[str],
) -> dict[str, dict[str, object]]:
    return {
        cell_id: {
            "greedy_path_id": cell_data[cell_id][0],
            "raw_features": cell_data[cell_id][1],
        }
        for cell_id in sorted(cell_ids)
    }


def _validate_fold_metadata(
    cell_data: Mapping[str, tuple[str, dict[str, RawFeatureVector]]],
    cell_to_circuit: Mapping[str, str],
    cell_to_topology: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    cell_ids = set(cell_data)
    if not isinstance(cell_to_circuit, Mapping) or not isinstance(cell_to_topology, Mapping):
        raise TypeError("cell_to_circuit and cell_to_topology must be mappings")
    if set(cell_to_circuit) != cell_ids or set(cell_to_topology) != cell_ids:
        raise ValueError("cell metadata mappings must cover exactly the analysis cells")
    metadata: dict[str, str] = {}
    by_circuit: dict[str, list[tuple[str, str]]] = {}
    for cell_id in sorted(cell_data):
        circuit_id = cell_to_circuit[cell_id]
        topology_id = cell_to_topology[cell_id]
        if not isinstance(circuit_id, str) or not circuit_id:
            raise ValueError(f"invalid circuit ID for cell {cell_id!r}")
        if not isinstance(topology_id, str) or not topology_id:
            raise ValueError(f"invalid topology ID for cell {cell_id!r}")
        metadata[cell_id] = circuit_id
        by_circuit.setdefault(circuit_id, []).append((topology_id, cell_id))
    circuit_cells: dict[str, tuple[str, ...]] = {}
    expected_topologies = set(_WAVE_TOPOLOGIES)
    for circuit_id, entries in sorted(by_circuit.items()):
        topology_ids = [topology_id for topology_id, _ in entries]
        if len(entries) != len(expected_topologies) or set(topology_ids) != expected_topologies:
            raise ValueError(
                f"circuit {circuit_id!r} must contain exactly both wave topologies"
            )
        circuit_cells[circuit_id] = tuple(sorted(cell_id for _, cell_id in entries))
    if len(circuit_cells) < 2:
        raise ValueError("leave-one-circuit-out requires at least two circuit instances")
    return metadata, circuit_cells


def _rows_for_cells(
    rows: Sequence[Mapping[str, object] | Sequence[object]],
    cell_ids: set[str],
    split: str,
) -> list[Mapping[str, object] | Sequence[object]]:
    return [row for row in rows if _row_key(row, split)[0] in cell_ids]


def _selected_paths_for_fit(
    pool: _MeasurementPool,
    fit: WaveFitResult,
) -> tuple[tuple[str, str], ...]:
    selected: list[tuple[str, str]] = []
    for cell_id, (greedy_id, raw_values) in pool.cells.items():
        measured_ids = tuple(sorted(_cell_times(pool, cell_id)))
        greedy_raw = raw_values[greedy_id]
        normalized = {
            candidate_id: normalize_features(raw, greedy_raw)
            for candidate_id, raw in raw_values.items()
            if candidate_id in measured_ids
        }
        selected_id = min(
            measured_ids,
            key=lambda candidate_id: (
                _score(normalized[candidate_id], fit.model, fit.weights),
                candidate_id,
            ),
        )
        selected.append((cell_id, selected_id))
    return tuple(selected)


def _profile_for_fit(
    pool: _MeasurementPool,
    fit: WaveFitResult,
    profile_id: str,
) -> FixedWaveProfile:
    return FixedWaveProfile(
        profile_id=profile_id,
        weights=fit.weights,
        model=fit.model,
        selected_path_ids=_selected_paths_for_fit(pool, fit),
    )


def _evaluation_facts(
    pool: _MeasurementPool,
    fit: WaveFitResult,
    profile_id: str,
) -> dict[str, Any]:
    profile = _profile_for_fit(pool, fit, profile_id)
    facts = _profile_cell_facts(pool, profile)
    selected = dict(profile.selected_path_ids)
    logs = _paired_log_speedups(pool, selected)
    for cell in facts["cells"]:
        cell["paired_log_speedup"] = logs[cell["cell_id"]]
    return {
        "profile_id": profile_id,
        "selected_path_ids": selected,
        "paired_log_speedups": logs,
        "paired_speedups": {
            cell_id: math.exp(value) for cell_id, value in logs.items()
        },
        "cells": facts["cells"],
    }


def _fit_facts(fit: WaveFitResult) -> dict[str, Any]:
    return {
        "model_form": fit.model.mode,
        "weights": fit.weights.as_mapping(),
        "training_selected_path_ids": dict(fit.selected_path_ids),
        "training_paired_speedups": dict(fit.cell_speedups),
        "training_geometric_mean_speedup": fit.geometric_mean_speedup,
        "training_worst_cell_speedup": fit.worst_cell_speedup,
        "seed": fit.seed,
        "sample_count": fit.sample_count,
        "evaluated_weight_vectors": fit.evaluated_weight_vectors,
        "declared_model": {
            "active_features": list(fit.model.active_features),
            "declared_rank": fit.model.matrix_rank,
        },
    }


def _feature_diagnostics(
    cell_data: Mapping[str, tuple[str, dict[str, RawFeatureVector]]],
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    vectors = []
    for cell_id in sorted(cell_ids):
        greedy_id, raw_values = cell_data[cell_id]
        greedy_raw = raw_values[greedy_id]
        vectors.extend(
            normalize_features(raw, greedy_raw).values
            for _, raw in sorted(raw_values.items())
        )
    if not vectors:
        raise ValueError("empirical feature diagnostics require candidate vectors")
    matrix = np.asarray(vectors, dtype=np.float64)
    means = [statistics.fmean(matrix[:, index]) for index in range(matrix.shape[1])]
    distinct_counts = []
    tie_group_counts = []
    tied_observation_counts = []
    for index in range(matrix.shape[1]):
        counts = Counter(float(value) for value in matrix[:, index])
        distinct_counts.append(len(counts))
        tie_groups = [count for count in counts.values() if count > 1]
        tie_group_counts.append(len(tie_groups))
        tied_observation_counts.append(sum(tie_groups))
    variances = [
        statistics.fmean(
            (float(value) - means[index]) ** 2 for value in matrix[:, index]
        )
        for index in range(matrix.shape[1])
    ]
    correlations: dict[str, float | None] = {}
    spearman_correlations: dict[str, float | None] = {}
    for left_index, left_name in enumerate(FEATURE_NAMES):
        for right_index in range(left_index + 1, len(FEATURE_NAMES)):
            right_name = FEATURE_NAMES[right_index]
            if variances[left_index] == 0.0 or variances[right_index] == 0.0:
                coefficient = None
            else:
                covariance = statistics.fmean(
                    (float(row[left_index]) - means[left_index])
                    * (float(row[right_index]) - means[right_index])
                    for row in matrix
                )
                coefficient = covariance / math.sqrt(
                    variances[left_index] * variances[right_index]
                )
            pair = f"{left_name}:{right_name}"
            correlations[pair] = coefficient
            spearman_correlations[pair] = _spearman(
                matrix[:, left_index].tolist(), matrix[:, right_index].tolist()
            )
    return {
        "normalized_feature_vector_count": len(vectors),
        "feature_order": list(FEATURE_NAMES),
        "variance": dict(zip(FEATURE_NAMES, variances, strict=True)),
        "distinct_value_count": dict(
            zip(FEATURE_NAMES, distinct_counts, strict=True)
        ),
        "tie_group_count": dict(zip(FEATURE_NAMES, tie_group_counts, strict=True)),
        "tied_observation_count": dict(
            zip(FEATURE_NAMES, tied_observation_counts, strict=True)
        ),
        "pairwise_pearson_correlation": correlations,
        "pairwise_spearman_correlation": spearman_correlations,
        "empirical_rank": int(np.linalg.matrix_rank(matrix)),
        "rank_interpretation": "descriptive normalized-feature-matrix rank, not model rank",
    }


def _bootstrap_cv_log_differences(
    fold_pools: Mapping[str, _MeasurementPool],
    selected_by_model: Mapping[str, Mapping[str, str]],
    *,
    resamples: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    shared_blocks: dict[str, tuple[str, ...]] = {}
    for pool in fold_pools.values():
        for round_id, blocks in _shared_round_blocks(pool).items():
            previous = shared_blocks.get(round_id)
            if previous is not None and previous != blocks:
                raise ValueError(
                    f"round {round_id!r} has inconsistent block sets across pooled cells"
                )
            shared_blocks[round_id] = blocks
    shared_blocks = dict(sorted(shared_blocks.items()))
    rng = random.Random(seed)
    geometric_differences: list[float] = []
    worst_differences: list[float] = []
    for _ in range(resamples):
        logs_by_model = {model: {} for model in _MODEL_FORMS}
        block_draws = {
            round_id: tuple(rng.choice(blocks) for _ in blocks)
            for round_id, blocks in shared_blocks.items()
        }
        for fold_id, pool in sorted(fold_pools.items()):
            drawn = _drawn_log_speedups(pool, block_draws)
            for cell_id in sorted(pool.cells):
                for model in _MODEL_FORMS:
                    candidate_id = selected_by_model[model][cell_id]
                    if candidate_id not in drawn[cell_id]:
                        raise ValueError(
                            f"selected path has no bootstrap observations: "
                            f"{cell_id}/{candidate_id}"
                        )
                    logs_by_model[model][cell_id] = drawn[cell_id][candidate_id]
        geometric_differences.append(
            statistics.fmean(logs_by_model["six_term"].values())
            - statistics.fmean(logs_by_model["grouped"].values())
        )
        worst_differences.append(
            min(logs_by_model["six_term"].values())
            - min(logs_by_model["grouped"].values())
        )
    return geometric_differences, worst_differences


def _resample_measurement_rows(
    pool: _MeasurementPool,
    rng: random.Random,
    split: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Resample complete measured blocks while copying actual source rows."""
    shared_blocks = _shared_round_blocks(pool)
    source_by_round_block: dict[tuple[str, str], list[tuple[_RowKey, Mapping[str, object]]]] = {}
    for key, row in sorted(pool.source_rows.items()):
        if key[4] == "measurement":
            source_by_round_block.setdefault((key[2], key[3]), []).append((key, row))

    sampled_rows: list[dict[str, object]] = []
    for round_id, block_ids in shared_blocks.items():
        for sample_index in range(len(block_ids)):
            source_block = rng.choice(block_ids)
            for _, source_row in source_by_round_block[(round_id, source_block)]:
                sampled_row = dict(source_row)
                sampled_row["block"] = f"stability_block_{sample_index}"
                sampled_rows.append(sampled_row)
    expected_rows = [
        {
            "cell_id": key[0],
            "candidate_path_id": key[1],
            "round_id": key[2],
            "block": key[3],
            "attempt_type": key[4],
            "split": split,
        }
        for row in sampled_rows
        for key in (_row_key(row, split),)
    ]
    return sampled_rows, expected_rows


def _bootstrap_refit_stability(
    cell_data: Mapping[str, tuple[str, dict[str, RawFeatureVector]]],
    circuit_cells: Mapping[str, Sequence[str]],
    full_pool: _MeasurementPool,
    *,
    split: str,
    resamples: int,
    fit_sample_count: int,
    fit_seed: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    selection_counts = {
        model: {
            cell_id: Counter(
                {candidate_id: 0 for candidate_id in _cell_times(full_pool, cell_id)}
            )
            for cell_id in sorted(cell_data)
        }
        for model in _MODEL_FORMS
    }
    weights_by_fold = {
        model: {fold_id: [] for fold_id in sorted(circuit_cells)}
        for model in _MODEL_FORMS
    }
    for resample_index in range(resamples):
        sampled_rows, sampled_expected = _resample_measurement_rows(
            full_pool, rng, split
        )
        for held_out_circuit, held_out_ids in sorted(circuit_cells.items()):
            held_out_set = set(held_out_ids)
            training_ids = tuple(
                cell_id for cell_id in sorted(cell_data) if cell_id not in held_out_set
            )
            training_set = set(training_ids)
            training_rows = _rows_for_cells(sampled_rows, training_set, split)
            training_expected = _rows_for_cells(
                sampled_expected, training_set, split
            )
            held_out_rows = _rows_for_cells(sampled_rows, held_out_set, split)
            held_out_expected = _rows_for_cells(
                sampled_expected, held_out_set, split
            )
            held_out_pool = _measurement_pool(
                _fit_cells(cell_data, held_out_ids),
                held_out_rows,
                held_out_expected,
                split,
            )
            for model in _MODEL_FORMS:
                fit = fit_upmem_wave_paths(
                    _fit_cells(cell_data, training_ids),
                    training_rows,
                    training_expected,
                    split=split,
                    model_form=model,
                    seed=fit_seed,
                    sample_count=fit_sample_count,
                )
                weights_by_fold[model][held_out_circuit].append(
                    {
                        "resample_index": resample_index,
                        "weights": fit.weights.as_mapping(),
                    }
                )
                evaluation = _evaluation_facts(
                    held_out_pool, fit, f"stability:{model}:{held_out_circuit}"
                )
                for cell_id, candidate_id in evaluation["selected_path_ids"].items():
                    selection_counts[model][cell_id][candidate_id] += 1

    selection_frequencies = {
        model: {
            cell_id: {
                candidate_id: {
                    "count": count,
                    "frequency": count / resamples,
                }
                for candidate_id, count in sorted(counter.items())
            }
            for cell_id, counter in sorted(cells.items())
        }
        for model, cells in selection_counts.items()
    }
    return {
        "status": "ran",
        "resamples": resamples,
        "fit_sample_count": fit_sample_count,
        "fit_seed": fit_seed,
        "bootstrap_seed": seed,
        "fit_count": resamples * len(circuit_cells) * len(_MODEL_FORMS),
        "block_resampling": (
            "complete measured blocks with one draw per round shared across all cells"
        ),
        "selection_frequencies": selection_frequencies,
        "weights_by_fold": weights_by_fold,
        "interpretation": (
            "bounded fold refits quantify resampled path and weight stability; "
            "they are not equivalent to full-search uncertainty"
        ),
    }


def compare_upmem_wave_models(
    cells: Mapping[str, Mapping[str, object]],
    measured_rows: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object] | Sequence[object]],
    cell_to_circuit: Mapping[str, str],
    cell_to_topology: Mapping[str, str],
    *,
    split: str = "training",
    fit_sample_count: int = DEFAULT_FIT_SAMPLE_COUNT,
    fit_seed: int = DEFAULT_FIT_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    stability_resamples: int = 0,
    stability_fit_sample_count: int | None = None,
    stability_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare the two preregistered models with training-only instance folds.

    Each fold fits on all other circuit instances, then predicts only measured
    candidates in the omitted circuit.  Fold weights remain separate; pooled
    metrics combine their omitted-cell paired log speedups and do not create a
    global profile.
    """
    if split != "training":
        raise ValueError("model comparison is training-only")
    if (
        isinstance(fit_sample_count, bool)
        or not isinstance(fit_sample_count, int)
        or fit_sample_count < 1
    ):
        raise ValueError("fit_sample_count must be a positive integer")
    if isinstance(fit_seed, bool) or not isinstance(fit_seed, int) or fit_seed < 0:
        raise ValueError("fit_seed must be a nonnegative integer")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if (
        isinstance(stability_resamples, bool)
        or not isinstance(stability_resamples, int)
        or stability_resamples < 0
    ):
        raise ValueError("stability_resamples must be a nonnegative integer")
    if stability_fit_sample_count is not None and (
        isinstance(stability_fit_sample_count, bool)
        or not isinstance(stability_fit_sample_count, int)
        or stability_fit_sample_count < 1
    ):
        raise ValueError("stability_fit_sample_count must be a positive integer")
    if stability_resamples and stability_fit_sample_count is None:
        raise ValueError(
            "stability_fit_sample_count is required when stability refits are enabled"
        )
    if (
        isinstance(stability_seed, bool)
        or not isinstance(stability_seed, int)
        or stability_seed < 0
    ):
        raise ValueError("stability_seed must be a nonnegative integer")

    cell_data = _coerce_cells(cells)
    cell_to_circuit, circuit_cells = _validate_fold_metadata(
        cell_data, cell_to_circuit, cell_to_topology
    )
    canonical_cells = _fit_cells(cell_data, tuple(cell_data))
    full_pool = _measurement_pool(canonical_cells, measured_rows, expected_rows, split)

    folds: list[dict[str, Any]] = []
    fold_pools: dict[str, _MeasurementPool] = {}
    selected_by_model = {model: {} for model in _MODEL_FORMS}
    all_evaluation_cells: dict[str, list[dict[str, Any]]] = {
        model: [] for model in _MODEL_FORMS
    }
    all_logs: dict[str, dict[str, float]] = {model: {} for model in _MODEL_FORMS}
    for held_out_circuit in sorted(circuit_cells):
        held_out_ids = circuit_cells[held_out_circuit]
        held_out_set = set(held_out_ids)
        training_ids = tuple(
            cell_id for cell_id in sorted(cell_data) if cell_id not in held_out_set
        )
        training_circuits = tuple(
            circuit_id
            for circuit_id in sorted(circuit_cells)
            if circuit_id != held_out_circuit
        )
        training_set = set(training_ids)
        training_rows = _rows_for_cells(measured_rows, training_set, split)
        training_expected = _rows_for_cells(expected_rows, training_set, split)
        held_out_rows = _rows_for_cells(measured_rows, held_out_set, split)
        held_out_expected = _rows_for_cells(expected_rows, held_out_set, split)
        held_out_pool = _measurement_pool(
            _fit_cells(cell_data, held_out_ids),
            held_out_rows,
            held_out_expected,
            split,
        )
        fold_pools[held_out_circuit] = held_out_pool
        fit_results: dict[str, WaveFitResult] = {}
        evaluations: dict[str, dict[str, Any]] = {}
        for model in _MODEL_FORMS:
            fit = fit_upmem_wave_paths(
                _fit_cells(cell_data, training_ids),
                training_rows,
                training_expected,
                split=split,
                model_form=model,
                seed=fit_seed,
                sample_count=fit_sample_count,
            )
            fit_results[model] = fit
            evaluation = _evaluation_facts(
                held_out_pool, fit, f"{model}:{held_out_circuit}"
            )
            evaluations[model] = evaluation
            selected_by_model[model].update(evaluation["selected_path_ids"])
            all_evaluation_cells[model].extend(evaluation["cells"])
            all_logs[model].update(evaluation["paired_log_speedups"])
        agreement = {
            cell_id: evaluations["six_term"]["selected_path_ids"][cell_id]
            == evaluations["grouped"]["selected_path_ids"][cell_id]
            for cell_id in held_out_ids
        }
        folds.append(
            {
                "fold_id": held_out_circuit,
                "held_out_circuit_id": held_out_circuit,
                "training_circuit_ids": list(training_circuits),
                "held_out_cell_ids": list(held_out_ids),
                "held_out_topology_ids": [
                    cell_to_topology[cell_id] for cell_id in held_out_ids
                ],
                "training_cell_ids": list(training_ids),
                "training_feature_diagnostics": _feature_diagnostics(
                    cell_data, training_ids
                ),
                "fits": {model: _fit_facts(fit_results[model]) for model in _MODEL_FORMS},
                "fold_selection": {
                    model: evaluations[model]["selected_path_ids"]
                    for model in _MODEL_FORMS
                },
                "evaluations": evaluations,
                "selection_agreement": agreement,
            }
        )

    pooled_metrics: dict[str, dict[str, Any]] = {}
    for model in _MODEL_FORMS:
        logs = all_logs[model]
        facts = all_evaluation_cells[model]
        pooled_metrics[model] = {
            "cell_count": len(logs),
            "paired_log_speedups": dict(sorted(logs.items())),
            "paired_speedups": {
                cell_id: math.exp(value) for cell_id, value in sorted(logs.items())
            },
            "geometric_mean_log_speedup": statistics.fmean(logs.values()),
            "geometric_mean_speedup": math.exp(statistics.fmean(logs.values())),
            "worst_cell_log_speedup": min(logs.values()),
            "worst_cell_speedup": math.exp(min(logs.values())),
            "selected_rank_by_cell": {
                cell["cell_id"]: cell["selected_rank"] for cell in facts
            },
            "top1_count": sum(bool(cell["top1"]) for cell in facts),
            "top3_count": sum(bool(cell["top3"]) for cell in facts),
            "oracle_regret_by_cell": {
                cell["cell_id"]: cell["oracle_regret"] for cell in facts
            },
        }

    agreement_by_cell = {
        cell_id: selected_by_model["six_term"][cell_id]
        == selected_by_model["grouped"][cell_id]
        for cell_id in sorted(cell_data)
    }
    geo_differences, worst_differences = _bootstrap_cv_log_differences(
        fold_pools,
        selected_by_model,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    pooled_geo_difference = (
        pooled_metrics["six_term"]["geometric_mean_log_speedup"]
        - pooled_metrics["grouped"]["geometric_mean_log_speedup"]
    )
    pooled_worst_difference = (
        pooled_metrics["six_term"]["worst_cell_log_speedup"]
        - pooled_metrics["grouped"]["worst_cell_log_speedup"]
    )
    geo_interval = (
        _percentile(geo_differences, 0.025),
        _percentile(geo_differences, 0.975),
    )
    chosen_model = select_wave_model_form(
        geo_interval,
        six_term_worst_speedup=pooled_metrics["six_term"]["worst_cell_speedup"],
        grouped_worst_speedup=pooled_metrics["grouped"]["worst_cell_speedup"],
    )
    stability = (
        _bootstrap_refit_stability(
            cell_data,
            circuit_cells,
            full_pool,
            split=split,
            resamples=stability_resamples,
            fit_sample_count=stability_fit_sample_count,
            fit_seed=fit_seed,
            seed=stability_seed,
        )
        if stability_resamples
        else {
            "status": "disabled",
            "resamples": 0,
            "fit_sample_count": stability_fit_sample_count,
            "fit_seed": fit_seed,
            "bootstrap_seed": stability_seed,
            "fit_count": 0,
            "interpretation": (
                "not run; current CV bootstrap conditions on original fold "
                "selections and is not full-refit uncertainty"
            ),
        }
    )
    return {
        "schema_version": "upmem_wave_model_comparison_v1",
        "split": split,
        "fold_unit": "leave_one_circuit_instance_out_with_both_topologies_together",
        "cell_to_circuit": dict(sorted(cell_to_circuit.items())),
        "cell_to_topology": dict(sorted(cell_to_topology.items())),
        "fit_search": {
            "model_forms": list(_MODEL_FORMS),
            "sample_count": fit_sample_count,
            "seed": fit_seed,
            "same_sample_count_and_seed_for_both_models": True,
        },
        "folds": folds,
        "pooled_cv": {
            "fold_count": len(folds),
            "cell_count": len(cell_data),
            "fold_specific_model_metrics": pooled_metrics,
            "selection_agreement": {
                "matching_cell_count": sum(agreement_by_cell.values()),
                "cell_count": len(agreement_by_cell),
                "fraction": sum(agreement_by_cell.values()) / len(agreement_by_cell),
                "per_cell": agreement_by_cell,
            },
            "bootstrap": {
                "method": (
                    "paired_whole_measurement_block_within_round_shared_across_"
                    "all_pooled_cells_percentile_v1"
                ),
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "geometric_mean_log_difference_six_minus_grouped": pooled_geo_difference,
                "geometric_mean_log_difference_95pct": list(geo_interval),
                "worst_cell_log_difference_six_minus_grouped": pooled_worst_difference,
                "worst_cell_log_difference_95pct": [
                    _percentile(worst_differences, 0.025),
                    _percentile(worst_differences, 0.975),
                ],
                "interpretation": (
                    "conditions on original fold selections; not full-refit "
                    "uncertainty"
                ),
            },
        },
        "model_selection": {
            "helper": "select_wave_model_form",
            "selected_model_form": chosen_model,
            "six_minus_grouped_log_geometric_speedup_interval_95pct": list(
                geo_interval
            ),
            "six_term_worst_cell_speedup": pooled_metrics["six_term"][
                "worst_cell_speedup"
            ],
            "grouped_worst_cell_speedup": pooled_metrics["grouped"][
                "worst_cell_speedup"
            ],
            "interpretation": (
                "preregistered complexity choice; an interval containing zero is "
                "not proof of equivalence"
            ),
        },
        "refit_stability": stability,
        "cv_interpretation": (
            "adaptive measured pools were selected using full training data, "
            "including omitted-instance timings; this is conditional "
            "development/model-choice evidence, not unbiased generalization of "
            "adaptive search"
        ),
    }


def analyze_upmem_wave_paths(
    cells: Mapping[str, Mapping[str, object]],
    measured_rows: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object] | Sequence[object]],
    profiles: Sequence[FixedWaveProfile | WaveFitResult],
    *,
    split: str = "training",
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return measured-pool rankings and a descriptive fixed-profile comparison.

    Exactly two fixed profiles are compared.  Bootstrap draws reuse one block
    draw for every candidate in a cell/round, preserving greedy pairings and
    allowing candidates to have different round sets.  No bootstrap refit or
    model decision is performed.
    """
    if len(profiles) != 2:
        raise ValueError("analysis requires exactly two fixed profiles")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    pool = _measurement_pool(cells, measured_rows, expected_rows, split)
    fixed = tuple(_coerce_profile(value, index) for index, value in enumerate(profiles))
    if fixed[0].profile_id == fixed[1].profile_id:
        raise ValueError("fixed profile IDs must be distinct")
    profile_facts = tuple(_profile_cell_facts(pool, profile) for profile in fixed)
    selected = tuple(dict(profile.selected_path_ids) for profile in fixed)
    log_speedups = tuple(_paired_log_speedups(pool, paths) for paths in selected)
    agreement_by_cell = {
        cell_id: selected[0][cell_id] == selected[1][cell_id]
        for cell_id in sorted(pool.cells)
    }
    agreement_count = sum(agreement_by_cell.values())
    geo_logs = tuple(statistics.fmean(values.values()) for values in log_speedups)
    worst_logs = tuple(min(values.values()) for values in log_speedups)
    geo_differences, worst_differences = _bootstrap_log_differences(
        pool,
        selected[0],
        selected[1],
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    comparison = {
        "profile_a": fixed[0].profile_id,
        "profile_b": fixed[1].profile_id,
        "selection_agreement": {
            "matching_cell_count": agreement_count,
            "cell_count": len(agreement_by_cell),
            "fraction": agreement_count / len(agreement_by_cell),
            "per_cell": agreement_by_cell,
        },
        "geometric_mean_log_speedup": {
            fixed[0].profile_id: geo_logs[0],
            fixed[1].profile_id: geo_logs[1],
        },
        "worst_cell_log_speedup": {
            fixed[0].profile_id: worst_logs[0],
            fixed[1].profile_id: worst_logs[1],
        },
        "bootstrap": {
            "method": "paired_whole_measurement_block_within_round_percentile_v1",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "geometric_mean_log_difference": geo_logs[0] - geo_logs[1],
            "geometric_mean_log_difference_95pct": [
                _percentile(geo_differences, 0.025),
                _percentile(geo_differences, 0.975),
            ],
            "worst_cell_log_difference": worst_logs[0] - worst_logs[1],
            "worst_cell_log_difference_95pct": [
                _percentile(worst_differences, 0.025),
                _percentile(worst_differences, 0.975),
            ],
            "selection_agreement": agreement_by_cell,
            "interpretation": (
                "conditions on supplied fixed selections; not full-refit uncertainty"
            ),
        },
    }
    for facts, logs in zip(profile_facts, log_speedups, strict=True):
        for cell in facts["cells"]:
            cell["paired_log_speedup"] = logs[cell["cell_id"]]
    return {
        "schema_version": "upmem_wave_path_analysis_v1",
        "split": split,
        "profile_results": list(profile_facts),
        "comparison": comparison,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_FIT_SAMPLE_COUNT",
    "DEFAULT_FIT_SEED",
    "FixedWaveProfile",
    "analyze_upmem_wave_paths",
    "compare_upmem_wave_models",
]
