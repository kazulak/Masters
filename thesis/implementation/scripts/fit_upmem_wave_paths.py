"""Pure offline fitting for the P6 wave-path primary objective.

The caller supplies identity-verified per-cell raw features and an explicit
expected row set from stage manifests. This module does not read evidence,
generate paths, construct runtime rows, or execute anything. It preserves the
existing ``scope_id=steady_execution_v1`` and derives the primary quantity as
``session_open_s + total_wall_s + session_close_s``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Literal

import numpy as np

from quantum_bench.upmem.path_heuristic import (
    FEATURE_NAMES,
    FeatureModelDecision,
    NormalizedFeatureVector,
    RawFeatureVector,
    WeightVector,
    explicit_feature_model,
    normalize_features,
)


FIT_SPLITS = frozenset({"training", "development"})
_FAILURE_STATUSES = frozenset({"failed", "failure", "unsupported", "error"})
_SIX_ACTIVE = FEATURE_NAMES[:4]
_RowKey = tuple[str, str, str, str, str]
_CheckedRow = tuple[str, str, str, str, str, float]


@dataclass(frozen=True, slots=True)
class WaveFitResult:
    """Transparent result for one fixed model form and offline search."""

    weights: WeightVector
    model: FeatureModelDecision
    selected_path_ids: tuple[tuple[str, str], ...]
    cell_speedups: tuple[tuple[str, float], ...]
    candidate_speedups: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    paired_observation_counts: tuple[tuple[str, str, int], ...]
    geometric_mean_speedup: float
    worst_cell_speedup: float
    objective: tuple[float, float, tuple[float, ...]]
    seed: int
    sample_count: int
    evaluated_weight_vectors: int


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _token(name: str, value: object) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a nonempty scalar identifier")
    if not isinstance(value, (str, int)):
        raise TypeError(f"{name} must be a string or integer")
    return _text(name, str(value))


def _nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _check_split(row: Mapping[str, object], expected: str) -> None:
    split = row.get("split", expected)
    if split not in FIT_SPLITS or split != expected:
        raise ValueError("wave fitting accepts the explicit training/development split only")


def _attempt(value: object) -> str:
    attempt = _text("attempt_type", value).lower()
    if attempt not in {"warmup", "measurement"}:
        raise ValueError("attempt_type must be warmup or measurement")
    return attempt


def _check_status(value: object) -> None:
    if value is None:
        return
    status = _text("status", value).lower()
    if status in _FAILURE_STATUSES or status == "fallback":
        raise ValueError("measured row explicitly reports failure or fallback")
    if status != "success":
        raise ValueError("measured row status must be success")


def _check_fallback(value: object) -> None:
    if value is None:
        return
    if value is not False and value != "false":
        raise ValueError("measured row explicitly reports fallback")


def _raw(value: object) -> RawFeatureVector:
    if isinstance(value, RawFeatureVector):
        return value
    if isinstance(value, Mapping):
        return RawFeatureVector.from_mapping(value)
    raise TypeError("candidate raw features must be RawFeatureVector records")


def _coerce_cells(cells: Mapping[str, Mapping[str, object]]) -> dict[str, tuple[str, dict[str, RawFeatureVector]]]:
    if not isinstance(cells, Mapping) or not cells:
        raise ValueError("cells must be a nonempty cell mapping")
    result: dict[str, tuple[str, dict[str, RawFeatureVector]]] = {}
    for outer_id, spec in cells.items():
        if not isinstance(spec, Mapping):
            raise TypeError("each cell must be a mapping")
        cell_id = _text("cell_id", spec.get("cell_id", outer_id))
        if cell_id in result:
            raise ValueError(f"duplicate cell ID: {cell_id!r}")
        greedy = _text("greedy_path_id", spec.get("greedy_path_id"))
        raw_values = spec.get("raw_features")
        if not isinstance(raw_values, Mapping) or not raw_values:
            raise ValueError(f"cell {cell_id!r} requires raw_features")
        raw = {_text("candidate_path_id", path_id): _raw(value)
               for path_id, value in raw_values.items()}
        if greedy not in raw:
            raise ValueError(f"cell {cell_id!r} is missing its greedy candidate")
        result[cell_id] = (greedy, raw)
    return dict(sorted(result.items()))


def _row_key(row: Mapping[str, object] | Sequence[object], split: str) -> _RowKey:
    if isinstance(row, Mapping):
        values = row
    elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) == 5:
        values = dict(zip(
            ("cell_id", "candidate_path_id", "round_id", "block", "attempt_type"),
            row,
            strict=True,
        ))
    else:
        raise TypeError("expected_rows must contain mappings or five-tuples")
    _check_split(values, split)
    return (
        _token("cell_id", values.get("cell_id")),
        _token("candidate_path_id", values.get("candidate_path_id")),
        _token("round_id", values.get("round_id")),
        _token("block", values.get("block")),
        _attempt(values.get("attempt_type")),
    )


def _checked_row(row: Mapping[str, object], split: str) -> _CheckedRow:
    _check_split(row, split)
    for field in ("scope_id", "timing_scope"):
        if field in row and row[field] not in (None, "steady_execution_v1"):
            raise ValueError("wave fitting requires scope_id steady_execution_v1")
    _check_status(row.get("status"))
    _check_fallback(row.get("fallback"))
    key = _row_key(row, split)
    open_s = _nonnegative("session_open_s", row.get("session_open_s"))
    steady_s = _nonnegative("total_wall_s", row.get("total_wall_s"))
    close_s = _nonnegative("session_close_s", row.get("session_close_s"))
    inclusive = open_s + steady_s + close_s
    if not math.isfinite(inclusive) or inclusive <= 0.0:
        raise ValueError("session-inclusive time must be finite and strictly positive")
    return (*key, inclusive)


def _model(form: Literal["six_term", "grouped"]) -> FeatureModelDecision:
    if form == "grouped":
        return explicit_feature_model("grouped")
    if form == "six_term":
        return replace(
            explicit_feature_model("six_term"),
            active_features=_SIX_ACTIVE,
            zero_range_features=("E_num", "P_wram"),
            matrix_rank=len(_SIX_ACTIVE),
            reason="explicit six-term model with E_num and P_wram inactive",
        )
    raise ValueError("model_form must be six_term or grouped")


def _weights(model: FeatureModelDecision, values: Sequence[float]) -> WeightVector:
    if len(values) != len(model.active_features):
        raise ValueError("simplex dimension does not match model")
    if model.mode == "six_term":
        mapping = dict.fromkeys(FEATURE_NAMES, 0.0)
        mapping.update(zip(model.active_features, values, strict=True))
    else:
        grouped = dict(zip(model.active_features, values, strict=True))
        movement = float(grouped.get("movement", 0.0))
        mapping = {
            "B_host_dpu": movement / 2.0,
            "B_mram_wram": movement / 2.0,
            "I_dpu": float(grouped.get("compute", 0.0)),
            "N_sync": float(grouped.get("coordination", 0.0)),
            "E_num": 0.0,
            "P_wram": 0.0,
        }
    return WeightVector.from_values(mapping, inactive=("E_num", "P_wram"))


def _score(normalized: NormalizedFeatureVector, model: FeatureModelDecision, weights: WeightVector) -> float:
    projected = model.project(normalized)
    if model.mode == "six_term":
        projected_weights = tuple(weights[feature] for feature in model.active_features)
    else:
        projected_weights = (weights.host_dpu + weights.mram_wram, weights.dpu_work, weights.sync)
    return float(sum(value * weight for value, weight in zip(projected, projected_weights, strict=True)))


def _simplex(dimension: int, seed: int, sample_count: int) -> tuple[tuple[float, ...], ...]:
    rng = np.random.default_rng(seed)
    rows = rng.dirichlet(np.ones(dimension), size=sample_count)
    vectors = [tuple(float(value) for value in row) for row in rows]
    vectors.append((1.0 / dimension,) * dimension)
    vectors.extend(
        tuple(1.0 if index == vertex else 0.0 for index in range(dimension))
        for vertex in range(dimension)
    )
    return tuple(vectors)


def _objective_key(result: WaveFitResult) -> tuple[float, float, tuple[float, ...]]:
    return (-result.geometric_mean_speedup, -result.worst_cell_speedup, result.weights.as_tuple())


def fit_upmem_wave_paths(
    cells: Mapping[str, Mapping[str, object]],
    measured_rows: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object] | Sequence[object]],
    *,
    split: Literal["training", "development"] = "training",
    model_form: Literal["six_term", "grouped"] = "six_term",
    seed: int = 20260903,
    sample_count: int = 100_000,
) -> WaveFitResult:
    """Fit equal-cell geometric paired speedups from an explicit row set.

    Warmups are validated against ``expected_rows`` but excluded from ratios.
    Measurement rows pair only within the same cell, round, and block. Thus
    adaptive candidates may have different round sets without fabricated
    equal-count observations. Only candidates with retained pairs can win.
    """
    if split not in FIT_SPLITS:
        raise ValueError("split must be training or development")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
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
    if set(observed) != declared:
        missing = sorted(declared.difference(observed))
        extra = sorted(set(observed).difference(declared))
        raise ValueError(f"measured rows do not match declared expected row set: missing={missing!r}, extra={extra!r}")

    by_group: dict[tuple[str, str, str], list[_CheckedRow]] = {}
    for row in sorted(observed.values(), key=lambda item: item[:5]):
        if row[4] == "measurement":
            by_group.setdefault((row[0], row[2], row[3]), []).append(row)
    log_pairs: dict[tuple[str, str], list[float]] = {}
    for (cell_id, _round, _block), group in by_group.items():
        greedy_id = cell_data[cell_id][0]
        greedy_rows = [row for row in group if row[1] == greedy_id]
        for row in group:
            if row[1] == greedy_id:
                continue
            if len(greedy_rows) != 1:
                raise ValueError(f"missing or duplicate greedy pair for {cell_id!r}/{row[1]!r}")
            log_pairs.setdefault((cell_id, row[1]), []).append(
                math.log(greedy_rows[0][5]) - math.log(row[5])
            )
    pair_counts = tuple(
        (cell_id, candidate_id, len(logs))
        for (cell_id, candidate_id), logs in sorted(log_pairs.items())
    )
    greedy_measurement_counts = {
        cell_id: sum(
            row[0] == cell_id and row[4] == "measurement" and row[1] == greedy_id
            for row in observed.values()
        )
        for cell_id, (greedy_id, _) in cell_data.items()
    }
    missing_measurements = sorted(
        cell_id for cell_id, count in greedy_measurement_counts.items() if count == 0
    )
    if missing_measurements:
        raise ValueError(f"cells have zero measured observations: {missing_measurements!r}")
    pair_count_rows = list(pair_counts)
    pair_count_rows.extend(
        (cell_id, greedy_id, 0)
        for cell_id, (greedy_id, _) in cell_data.items()
        if not any(candidate_cell == cell_id for candidate_cell, _ in log_pairs)
    )
    pair_counts = tuple(sorted(pair_count_rows))

    normalized: dict[str, dict[str, NormalizedFeatureVector]] = {}
    for cell_id, (greedy_id, raw) in cell_data.items():
        greedy = raw[greedy_id]
        normalized[cell_id] = {
            candidate_id: normalize_features(features, greedy)
            for candidate_id, features in raw.items()
        }
    candidate_speedups: dict[tuple[str, str], float] = {
        (cell_id, greedy_id): 1.0 for cell_id, (greedy_id, _) in cell_data.items()
    }
    for key, logs in log_pairs.items():
        value = math.exp(sum(logs) / len(logs))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("paired speedup is non-finite")
        candidate_speedups[key] = value
    available_by_cell = {
        cell_id: tuple(sorted(
            {greedy_id} | {
                candidate_id
                for candidate_cell, candidate_id in log_pairs
                if candidate_cell == cell_id
            }
        ))
        for cell_id, (greedy_id, _) in cell_data.items()
    }
    candidate_speedups_by_cell = {
        cell_id: tuple(
            (candidate_id, candidate_speedups[(cell_id, candidate_id)])
            for candidate_id in available
        )
        for cell_id, available in available_by_cell.items()
    }
    all_candidates = tuple(
        (cell_id, candidate_speedups_by_cell[cell_id]) for cell_id in cell_data
    )

    model = _model(model_form)
    vectors = _simplex(len(model.active_features), seed, sample_count)
    best: WaveFitResult | None = None
    for vector in vectors:
        weights = _weights(model, vector)
        selected: list[tuple[str, str]] = []
        cell_speedups: list[tuple[str, float]] = []
        for cell_id, (greedy_id, _) in cell_data.items():
            available = available_by_cell[cell_id]
            selected_id = min(
                available,
                key=lambda candidate_id: (
                    _score(normalized[cell_id][candidate_id], model, weights),
                    candidate_id,
                ),
            )
            selected.append((cell_id, selected_id))
            cell_speedups.append((cell_id, candidate_speedups[(cell_id, selected_id)]))
        values = tuple(value for _, value in cell_speedups)
        geometric = math.exp(sum(math.log(value) for value in values) / len(values))
        worst = min(values)
        result = WaveFitResult(
            weights=weights,
            model=model,
            selected_path_ids=tuple(selected),
            cell_speedups=tuple(cell_speedups),
            candidate_speedups=tuple(all_candidates),
            paired_observation_counts=pair_counts,
            geometric_mean_speedup=geometric,
            worst_cell_speedup=worst,
            objective=(geometric, worst, weights.as_tuple()),
            seed=seed,
            sample_count=sample_count,
            evaluated_weight_vectors=len(vectors),
        )
        if best is None or _objective_key(result) < _objective_key(best):
            best = result
    assert best is not None
    return best


def select_wave_model_form(
    cross_validated_log_advantage_interval: tuple[float, float],
    *,
    six_term_worst_speedup: float,
    grouped_worst_speedup: float,
) -> Literal["six_term", "grouped"]:
    """Apply the preregistered complexity rule to training-only CV results.

    The interval is six-term minus grouped log geometric speedup. Choosing
    grouped when the interval includes zero does not establish equivalence.
    """
    if len(cross_validated_log_advantage_interval) != 2:
        raise ValueError("advantage interval requires two ordered finite bounds")
    lower, upper = cross_validated_log_advantage_interval
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in (lower, upper)
    ) or lower > upper:
        raise ValueError("advantage interval requires two ordered finite bounds")
    six_worst = _nonnegative("six_term_worst_speedup", six_term_worst_speedup)
    grouped_worst = _nonnegative("grouped_worst_speedup", grouped_worst_speedup)
    if six_worst == 0.0 or grouped_worst == 0.0:
        raise ValueError("worst-cell speedups must be strictly positive")
    return "six_term" if lower > 0.0 and six_worst >= grouped_worst else "grouped"


__all__ = ["FIT_SPLITS", "WaveFitResult", "fit_upmem_wave_paths", "select_wave_model_form"]
