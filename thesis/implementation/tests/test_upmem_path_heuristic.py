from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import math

import pytest

from quantum_bench.upmem.path_heuristic import (
    COST_MODEL_ID,
    FEATURE_NAMES,
    canonicalize_path,
    ConventionalPathFeatures,
    NormalizedFeatureVector,
    PathCandidate,
    RawFeatureVector,
    RuntimeMeasurement,
    TrainingCell,
    WeightVector,
    choose_feature_model,
    equal_model_weights,
    extract_conventional_features,
    extract_plan_features,
    explain_score,
    fit_weights,
    geometric_mean,
    normalize_features,
    path_id,
    score_features,
    select_best_candidate,
    select_calibration_candidates,
)
from quantum_bench.model import ContractNode, ContractionDAG, TensorSpec, TensorView
from quantum_bench.upmem.plan import (
    UpmemPlan,
    UpmemStage,
    UpmemTopology,
    UpmemWorkUnit,
)


def _id(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _raw(
    host: float,
    mram: float = 100.0,
    work: float = 100.0,
    sync: float = 100.0,
    numeric: float = 0.0,
    wram: float = 1.0,
) -> RawFeatureVector:
    return RawFeatureVector(
        host_dpu_bytes=host,
        mram_wram_bytes=mram,
        dpu_work=work,
        sync_events=sync,
        numeric_overhead=numeric,
        wram_pressure=wram,
    )


def _candidate(
    label: str,
    features: RawFeatureVector,
    *,
    topology: str = "topology",
    flops: float = 10.0,
    peak: float = 10.0,
    writes: float = 10.0,
    greedy: bool = False,
    feasible_topologies: tuple[str, ...] | None = None,
) -> PathCandidate:
    return PathCandidate(
        path_id=_id(label),
        conventional=ConventionalPathFeatures(
            flops=flops,
            macs=flops / 2.0,
            peak_intermediate_elements=peak,
            peak_intermediate_bytes=peak * 16.0,
            total_intermediate_writes=writes,
            maximum_intermediate_rank=2,
            contraction_count=2,
        ),
        features_by_topology=((topology, features),),
        feasible_topologies=feasible_topologies,
        is_greedy=greedy,
    )


def test_feature_vectors_and_weights_are_immutable_simplex_records() -> None:
    raw = _raw(4.0)
    with pytest.raises(FrozenInstanceError):
        raw.host_dpu_bytes = 8.0  # type: ignore[misc]

    first = WeightVector.from_values((2.0, 4.0, 0.0, 0.0, 0.0, 0.0))
    second = WeightVector.from_values((1.0, 2.0, 0.0, 0.0, 0.0, 0.0))
    assert first == second
    assert math.isclose(sum(first.as_tuple()), 1.0)
    assert first.host_dpu == pytest.approx(1.0 / 3.0)
    assert first.mram_wram == pytest.approx(2.0 / 3.0)

    inactive = WeightVector.from_values(
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        inactive=("B_mram_wram", "I_dpu", "N_sync", "E_num", "P_wram"),
    )
    assert inactive.as_tuple() == pytest.approx((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_greedy_relative_log_normalization_uses_unit_epsilons() -> None:
    candidate = _raw(3.0, mram=0.0, work=9.0, sync=4.0)
    greedy = _raw(1.0, mram=3.0, work=3.0, sync=1.0)
    normalized = normalize_features(candidate, greedy)
    assert normalized.values[0] == pytest.approx(math.log(4.0 / 2.0))
    assert normalized.values[1] == pytest.approx(math.log(1.0 / 4.0))
    assert normalized.values[2] == pytest.approx(math.log(10.0 / 4.0))
    assert normalized.values[3] == pytest.approx(math.log(5.0 / 2.0))
    assert COST_MODEL_ID == "upmem_slr_cost_v1"


def test_path_pair_order_is_canonical_but_step_order_is_preserved() -> None:
    forward = ((2, 0), (3, 1))
    swapped_pairs = ((0, 2), (1, 3))
    swapped_steps = ((3, 1), (2, 0))

    assert canonicalize_path(forward) == ((0, 2), (1, 3))
    assert canonicalize_path(forward) == canonicalize_path(swapped_pairs)
    assert canonicalize_path(forward) != canonicalize_path(swapped_steps)
    assert path_id(forward, circuit_id="fixture") == path_id(
        swapped_pairs,
        circuit_id="fixture",
    )


def test_feature_dependencies_and_inactive_numeric_terms_are_explicit() -> None:
    from quantum_bench.upmem.path_heuristic import feature_dependency_metadata

    metadata = feature_dependency_metadata()
    assert tuple(item.feature for item in metadata) == FEATURE_NAMES
    numeric = next(item for item in metadata if item.feature == "E_num")
    assert numeric.independently_identifiable is False
    assert "four-real-product arithmetic" in numeric.excludes
    sync = next(item for item in metadata if item.feature == "N_sync")
    assert "one event per wave (also the packed-operation alias)" in sync.includes
    assert "packed-operation count as a second wave event" in sync.excludes


def test_movement_heavy_score_can_choose_higher_flop_path() -> None:
    greedy = _candidate("greedy", _raw(100.0), flops=100.0, greedy=True)
    low_flop = _candidate("low-flop", _raw(1_000.0), flops=1.0)
    low_movement = _candidate("low-movement", _raw(10.0), flops=200.0)
    movement_weights = WeightVector.from_values((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    selected = select_best_candidate(
        (greedy, low_flop, low_movement),
        "topology",
        movement_weights,
    )
    assert selected.path_id == low_movement.path_id
    assert min(
        (greedy, low_flop, low_movement),
        key=lambda item: (item.conventional.flops, item.path_id),
    ).path_id == low_flop.path_id
    assert score_features(
        low_movement.raw_for("topology"),
        greedy.raw_for("topology"),
        movement_weights,
    ) < 0.0


def test_same_weight_vector_can_select_different_paths_by_topology() -> None:
    greedy_raw = _raw(100.0, work=100.0)
    candidate_a = PathCandidate(
        path_id=_id("a"),
        conventional=ConventionalPathFeatures(
            flops=10.0,
            macs=5.0,
            peak_intermediate_elements=10.0,
            peak_intermediate_bytes=160.0,
            total_intermediate_writes=10.0,
            maximum_intermediate_rank=2,
            contraction_count=2,
        ),
        features_by_topology=(
            ("1d", _raw(10.0, work=100.0)),
            ("4d", _raw(100.0, work=10.0)),
        ),
        feasible_topologies=("1d", "4d"),
    )
    candidate_b = PathCandidate(
        path_id=_id("b"),
        conventional=candidate_a.conventional,
        features_by_topology=(
            ("1d", _raw(100.0, work=10.0)),
            ("4d", _raw(10.0, work=100.0)),
        ),
        feasible_topologies=("1d", "4d"),
    )
    greedy = PathCandidate(
        path_id=_id("greedy"),
        conventional=candidate_a.conventional,
        features_by_topology=(("1d", greedy_raw), ("4d", greedy_raw)),
        feasible_topologies=("1d", "4d"),
        is_greedy=True,
    )
    weights = WeightVector.from_values((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert select_best_candidate((greedy, candidate_a, candidate_b), "1d", weights).path_id == candidate_a.path_id
    assert select_best_candidate((greedy, candidate_a, candidate_b), "4d", weights).path_id == candidate_b.path_id


def test_infeasible_candidate_never_wins() -> None:
    greedy = _candidate("greedy", _raw(100.0), greedy=True)
    infeasible = _candidate(
        "infeasible",
        _raw(1.0),
        feasible_topologies=(),
    )
    weights = WeightVector.from_values((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    selected = select_best_candidate((greedy, infeasible), "topology", weights)
    assert selected.path_id == greedy.path_id


def test_ties_are_broken_by_full_path_id_and_calibration_is_unique() -> None:
    greedy = _candidate("greedy", _raw(100.0), greedy=True)
    lower_id = _candidate("lower-id", _raw(100.0), flops=1.0)
    higher_id = _candidate("higher-id", _raw(100.0), flops=1.0)
    selected = select_best_candidate(
        (greedy, higher_id, lower_id),
        "topology",
        WeightVector.equal(),
    )
    assert selected.path_id == min(lower_id.path_id, higher_id.path_id, greedy.path_id)

    candidates = [
        greedy,
        lower_id,
        higher_id,
        _candidate("peak", _raw(100.0), peak=1.0),
        _candidate("writes", _raw(100.0), writes=1.0),
        _candidate("far-a", _raw(1.0, mram=500.0, work=2.0), flops=20.0),
        _candidate("far-b", _raw(500.0, mram=1.0, work=2.0), flops=30.0),
    ]
    calibration = select_calibration_candidates(candidates, "topology", limit=6)
    assert len(calibration) == 6
    assert len({candidate.path_id for candidate in calibration}) == 6
    assert calibration[0].path_id == greedy.path_id


def test_identifiability_removes_constants_and_rejects_correlated_terms() -> None:
    correlated = tuple(
        NormalizedFeatureVector((x, 2.0 * x, float(index), float(index + 1), 0.0, 0.0))
        for index, x in enumerate((0.0, 0.2, 0.5, 1.0))
    )
    decision = choose_feature_model(correlated)
    assert decision.mode == "grouped"
    assert ("B_host_dpu", "B_mram_wram") in decision.correlated_pairs
    assert decision.zero_range_features == ("E_num", "P_wram")
    assert decision.active_features == FEATURE_NAMES[:0] + (
        "movement",
        "compute",
        "coordination",
    )
    weights = equal_model_weights(decision)
    assert weights.host_dpu == pytest.approx(weights.mram_wram)
    assert weights.numeric == 0.0
    assert weights.wram == 0.0


def test_plan_feature_extraction_uses_existing_four_lane_facts() -> None:
    plan = UpmemPlan(
        logical_plan_id="0" * 64,
        numeric_policy="split_complex_float32_v1",
        topology=UpmemTopology(dpu_count=1, tasklets_per_dpu=2),
        stages=(
            UpmemStage(
                stage_id="contract_batch:0",
                kind="contract_batch",
                node_ids=("contract_0",),
                work_units=(
                    UpmemWorkUnit(
                        node_id="contract_0",
                        stable_tile_id="tile_0",
                        wave=0,
                        logical_rank=0,
                        logical_dpu=0,
                        batch_start=0,
                        batch_size=1,
                        m_start=0,
                        m_size=2,
                        n_start=0,
                        n_size=3,
                        k_start=0,
                        k_size=4,
                        estimated_input_bytes=40,
                        estimated_output_bytes=24,
                        aligned_mram_bytes=96,
                        estimated_arithmetic_work=24,
                    ),
                ),
            ),
        ),
    )
    facts = extract_plan_features(plan)
    assert facts.h2d_bytes == 160
    assert facts.d2h_bytes == 96
    assert facts.raw.host_dpu_bytes == 256.0
    assert facts.raw.dpu_work == 4 * 2 * 3 * 4
    assert facts.raw.numeric_overhead == 0.0
    assert facts.wave_count == facts.packed_operation_count == 1
    assert facts.dpu_launch_count == 4
    assert facts.raw.sync_events == pytest.approx(
        facts.wave_count
        + facts.dpu_launch_count
        + facts.host_reduce_count
        + facts.barrier_events
    )


def test_conventional_intermediate_features_exclude_final_output() -> None:
    first_left = TensorSpec(
        id="a",
        labels=(0, 1),
        shape=(2, 2),
        structure="input",
    )
    first_right = TensorSpec(
        id="b",
        labels=(1, 2),
        shape=(2, 2),
        structure="input",
    )
    final_right = TensorSpec(
        id="c",
        labels=(2, 3),
        shape=(2, 3),
        structure="input",
    )
    intermediate = TensorSpec(
        id="intermediate",
        labels=(0, 2),
        shape=(2, 2),
        structure="contraction",
        produced_by="contract-0",
    )
    final = TensorSpec(
        id="final",
        labels=(0, 3),
        shape=(2, 3),
        structure="contraction",
        produced_by="contract-1",
    )
    first = ContractNode(
        node_id="contract-0",
        left=TensorView(tensor_id="a", labels=(0, 1), shape=(2, 2)),
        right=TensorView(tensor_id="b", labels=(1, 2), shape=(2, 2)),
        output=intermediate,
        contracted_labels=(1,),
        output_labels=(0, 2),
    )
    second = ContractNode(
        node_id="contract-1",
        left=TensorView(tensor_id="intermediate", labels=(0, 2), shape=(2, 2)),
        right=TensorView(tensor_id="c", labels=(2, 3), shape=(2, 3)),
        output=final,
        contracted_labels=(2,),
        output_labels=(0, 3),
        dependencies=("contract-0",),
    )
    dag = ContractionDAG(
        tensors=(first_left, first_right, final_right),
        nodes=(first, second),
        output=TensorView(tensor_id="final", labels=(0, 3), shape=(2, 3)),
    )

    features = extract_conventional_features(dag)

    assert features.flops == pytest.approx(40.0)
    assert features.contraction_count == 2
    assert features.peak_intermediate_elements == pytest.approx(4.0)
    assert features.peak_intermediate_bytes == pytest.approx(64.0)
    assert features.total_intermediate_writes == pytest.approx(4.0)
    assert features.maximum_intermediate_rank == 2


def test_explanation_contains_raw_normalized_weight_and_contribution_rows() -> None:
    raw = _raw(10.0, mram=50.0)
    greedy = _raw(20.0, mram=25.0)
    weights = WeightVector.from_values((1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    rows = explain_score(raw, greedy, weights)
    assert tuple(row.feature for row in rows) == FEATURE_NAMES
    assert rows[0].raw == 10.0
    assert rows[0].weight == pytest.approx(0.5)
    assert rows[0].contribution == pytest.approx(rows[0].normalized * rows[0].weight)


def test_fit_requires_complete_candidate_sets_and_returns_geometric_mean_objective() -> None:
    greedy = _candidate("greedy", _raw(100.0), greedy=True)
    measured_but_slower = _candidate("measured", _raw(90.0))
    unmeasured_better = _candidate("unmeasured", _raw(200.0))
    cell = TrainingCell(
        cell_id="train-1d",
        topology="topology",
        candidates=(greedy, measured_but_slower, unmeasured_better),
    )
    rows = (
        RuntimeMeasurement(cell_id="train-1d", candidate_id=greedy.path_id, runtime_s=10.0),
        RuntimeMeasurement(
            cell_id="train-1d",
            candidate_id=measured_but_slower.path_id,
            runtime_s=9.0,
        ),
    )
    with pytest.raises(ValueError, match="exact measured candidate set"):
        fit_weights((cell,), rows, seed=7, random_sample_count=16)

    complete_rows = rows + (
        RuntimeMeasurement(
            cell_id="train-1d",
            candidate_id=unmeasured_better.path_id,
            runtime_s=20.0,
        ),
    )
    result = fit_weights((cell,), complete_rows, seed=7, random_sample_count=16)
    assert result.selected_path_ids == (("train-1d", measured_but_slower.path_id),)
    assert result.geometric_mean_speedup == pytest.approx(10.0 / 9.0)
    assert result.minimum_cell_speedup == pytest.approx(10.0 / 9.0)
    # One active feature yields one simplex vertex and one equal-weight vector
    # in addition to the requested deterministic samples.
    assert result.evaluated_weight_vectors == 18

    with pytest.raises(ValueError, match="training rows only"):
        fit_weights(
            (cell,),
            complete_rows
            + (
                RuntimeMeasurement(
                    cell_id="train-1d",
                    candidate_id=greedy.path_id,
                    runtime_s=10.0,
                    split="test",
                ),
            ),
            seed=7,
            random_sample_count=2,
        )


def test_fit_rejects_duplicate_failed_and_mixed_runtime_evidence() -> None:
    greedy = _candidate("greedy-evidence", _raw(100.0), greedy=True)
    candidate = _candidate("candidate-evidence", _raw(90.0))
    cell = TrainingCell(
        cell_id="train-evidence",
        topology="topology",
        candidates=(greedy, candidate),
    )
    common = {
        "split": "train",
        "source_sha": "a" * 40,
        "timing_scope": "steady_execution_v1",
    }
    rows = (
        RuntimeMeasurement(
            cell_id=cell.cell_id,
            candidate_id=greedy.path_id,
            runtime_s=10.0,
            observation_id="block-0",
            **common,
        ),
        RuntimeMeasurement(
            cell_id=cell.cell_id,
            candidate_id=candidate.path_id,
            runtime_s=9.0,
            observation_id="block-0",
            **common,
        ),
    )

    with pytest.raises(ValueError, match="duplicate runtime evidence"):
        fit_weights((cell,), rows + (rows[0],), random_sample_count=2)
    with pytest.raises(ValueError, match="failed or unsupported"):
        fit_weights(
            (cell,),
            (
                rows[0],
                RuntimeMeasurement(
                    cell_id=cell.cell_id,
                    candidate_id=candidate.path_id,
                    runtime_s=9.0,
                    status="failed",
                    observation_id="block-0",
                    **common,
                ),
            ),
            random_sample_count=2,
        )
    with pytest.raises(ValueError, match="mixed source_sha"):
        fit_weights(
            (cell,),
            rows
            + (
                RuntimeMeasurement(
                    cell_id=cell.cell_id,
                    candidate_id=candidate.path_id,
                    runtime_s=9.1,
                    source_sha="b" * 40,
                    timing_scope="steady_execution_v1",
                    observation_id="block-1",
                ),
            ),
            random_sample_count=2,
        )
    with pytest.raises(ValueError, match="mixed timing_scope"):
        fit_weights(
            (cell,),
            rows
            + (
                RuntimeMeasurement(
                    cell_id=cell.cell_id,
                    candidate_id=candidate.path_id,
                    runtime_s=9.1,
                    source_sha="a" * 40,
                    timing_scope="session_inclusive_v1",
                    observation_id="block-1",
                ),
            ),
            random_sample_count=2,
        )


def test_geometric_mean_and_fit_are_deterministic() -> None:
    assert geometric_mean((2.0, 8.0)) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        geometric_mean((1.0, 0.0))

    greedy = _candidate("greedy", _raw(100.0), greedy=True)
    candidate = _candidate("candidate", _raw(50.0))
    cell = TrainingCell(
        cell_id="train-1d",
        topology="topology",
        candidates=(greedy, candidate),
    )
    rows = (
        RuntimeMeasurement(cell_id="train-1d", candidate_id=greedy.path_id, runtime_s=10.0),
        RuntimeMeasurement(cell_id="train-1d", candidate_id=candidate.path_id, runtime_s=5.0),
    )
    first = fit_weights((cell,), rows, seed=9, random_sample_count=8)
    second = fit_weights((cell,), rows, seed=9, random_sample_count=8)
    assert first == second
