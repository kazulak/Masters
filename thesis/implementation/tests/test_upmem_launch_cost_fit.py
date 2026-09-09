from copy import deepcopy
import math

import pytest

from quantum_bench.upmem.path_heuristic import (
    LaunchCostFacts,
    LaunchCostScales,
    fit_launch_cost_weights,
    validate_launch_cost_observations,
)


IDENTITY = {
    "source_sha": "939a016" + "0" * 33,
    "execution_source": "459935f586fdd16c82013838e6d27a12604c3093",
    "policy_id": "split_complex_float32_v1",
    "timing_scope": "steady_execution_v1",
}
SCALES = LaunchCostScales(20, 20, 1, 1, 1)


def cell(family="A", facts=None):
    return {
        "family": family, "split": "training", "greedy_path_id": "g",
        "path_facts": facts if facts is not None else {
            "g": LaunchCostFacts(20, 20, 0, ()),
            "p": LaunchCostFacts(0, 0, 0, ()),
        },
    }


def round_rows(cells, times, round_id="r0"):
    """times gives explicit measurement sequences; warmups reset block to zero."""
    lengths = {len(values) for paths in times.values() for values in paths.values()}
    assert len(lengths) == 1
    declared = {
        "round_id": round_id, "accepted": True,
        "identity": {**IDENTITY, "candidate_set_sha256": "a" * 64},
        "cell_paths": {key: tuple(paths) for key, paths in times.items()},
        "warmup_blocks": (0,), "measurement_blocks": tuple(range(lengths.pop())),
    }
    rows = []
    for cell_id, paths in times.items():
        for path_id, values in paths.items():
            for attempt, observations in (("warmup", [12345.0]), ("measurement", values)):
                for block, total in enumerate(observations):
                    rows.append({
                        "cell_id": cell_id, "path_id": path_id, "round_id": round_id,
                        "block": block, "attempt_type": attempt,
                        "split": cells[cell_id]["split"],
                        "identity": dict(declared["identity"]), "status": "success",
                        "validation": "pass", "fallback": False,
                        "full_precision_passed": True, "policy_reference_passed": True,
                        "execution_resource_admission_passed": True,
                        "startup_resource_admission_passed": True,
                        "session_open_s": 0.0, "total_wall_s": total, "session_close_s": 0.0,
                    })
    return declared, rows


def observations(cells, rounds, rows):
    return validate_launch_cost_observations(cells, rounds, rows, expected_identity=IDENTITY)


def fit(cells, rounds, rows):
    return fit_launch_cost_weights(cells, rounds, rows, scales=SCALES, expected_identity=IDENTITY)


def test_paired_log_median_differs_from_ratio_of_medians_and_ignores_warmups():
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [1, 10, 100], "p": [1, 100, 10]}})
    rows[0]["total_wall_s"] = 1e200
    before = deepcopy((cells, declared, rows))
    result = observations(cells, [declared], rows)
    assert result["c"]["p"] == {"median_log_speedup": 0.0, "paired_count": 3}
    assert (cells, declared, rows) == before
    # A second pairing has median ratio 2 but ratio of medians 100/50.5.
    declared, rows = round_rows(cells, {"c": {"g": [1, 100, 101], "p": [0.5, 99, 50.5]}})
    result = observations(cells, [declared], rows)
    assert result["c"]["p"]["median_log_speedup"] == pytest.approx(math.log(2))
    assert result["c"]["p"]["median_log_speedup"] != pytest.approx(math.log(100 / 50.5))
    assert cells == before[0]


def test_round_pairing_unequal_coverage_inclusive_arithmetic_and_no_mutation():
    cells = {"c": cell(facts={
        "g": LaunchCostFacts(20, 20, 0, ()),
        "a": LaunchCostFacts(0, 0, 0, ()), "b": LaunchCostFacts(1, 1, 0, ()),
        "unmeasured": LaunchCostFacts(0, 0, 0, ()),
    })}
    first, rows1 = round_rows(cells, {"c": {"g": [10, 20], "a": [5, 10]}})
    second, rows2 = round_rows(cells, {"c": {"g": [1000, 2000], "b": [250, 500]}}, "r1")
    rows = rows1 + rows2
    # Redistribute the same inclusive time across open/steady/close.
    for row in rows:
        total = row["total_wall_s"]
        row.update(session_open_s=total / 4, total_wall_s=total / 2, session_close_s=total / 4)
    before = deepcopy((cells, [first, second], rows))
    profile, table = fit(cells, [first, second], rows)
    paired = profile["candidate_observations"]["c"]
    assert paired["a"]["median_log_speedup"] == pytest.approx(math.log(2))
    assert paired["b"]["median_log_speedup"] == pytest.approx(math.log(4))
    assert paired["g"] == {"median_log_speedup": 0.0, "paired_count": 4}
    assert paired["b"]["paired_count"] == 2
    assert all(item["cells"]["c"]["path_id"] != "unmeasured" for item in table)
    assert (cells, [first, second], rows) == before
    reversed_profile, reversed_table = fit(dict(reversed(tuple(cells.items()))), [second, first], reversed(rows))
    assert (reversed_profile, reversed_table) == (profile, table)


def test_family_balancing_uses_frozen_cells_not_row_counts():
    cells = {"a2": cell(), "b1": cell("B"), "a1": cell()}
    declared, rows = round_rows(cells, {
        "a1": {"g": [8, 8, 8], "p": [2, 2, 2]},
        "a2": {"g": [8, 8, 8], "p": [2, 2, 2]},
        "b1": {"g": [8, 8, 8], "p": [16, 16, 16]},
    })
    profile, table = fit(cells, [declared], rows)
    assert profile["cell_weights"] == {"a1": 0.25, "a2": 0.25, "b1": 0.5}
    assert profile["J"] == pytest.approx(math.log(2) / 2)
    assert profile["worst_cell_log_speedup"] == pytest.approx(-math.log(2))
    assert profile["integer_weights"] == (2, 2, 2, 2, 2)
    assert len(table) == profile["evaluated_weight_vectors"] == 1001
    assert len({item["integer_weights"] for item in table}) == 1001
    assert all(sum(item["integer_weights"]) == 10 for item in table)
    assert all(tuple(item["cells"]) == ("a1", "a2", "b1") for item in table)


def test_worst_cell_then_distance_then_lexicographic_weight_tie():
    facts = {
        "g": LaunchCostFacts(20, 20, 0, ()),
        "a": LaunchCostFacts(0, 10, 0, ()),
        "b": LaunchCostFacts(10, 0, 0, ()),
    }
    cells = {"c1": cell(facts=facts), "c2": cell(facts=facts)}
    declared, rows = round_rows(cells, {
        "c1": {"g": [8], "a": [1], "b": [4]},
        "c2": {"g": [8], "a": [16], "b": [4]},
    })
    profile, table = fit(cells, [declared], rows)
    assert profile["J"] == pytest.approx(math.log(2))
    assert profile["worst_cell_log_speedup"] == pytest.approx(math.log(2))
    assert profile["integer_weights"] == (1, 2, 2, 2, 3)
    assert all(item["path_id"] == "b" for item in profile["cells"].values())
    uniform = next(item for item in table if item["integer_weights"] == (2, 2, 2, 2, 2))
    assert uniform["rounded_J"] == round(profile["J"], 12)
    assert uniform["cells"]["c1"]["path_id"] == "a"  # equal cost, lexical path ID


def test_greedy_only_cell_is_valid_and_uniform_wins():
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [1, 2, 3]}})
    profile, table = fit(cells, [declared], rows)
    assert profile["integer_weights"] == (2, 2, 2, 2, 2)
    assert profile["J"] == profile["worst_cell_log_speedup"] == 0.0
    assert all(item["cells"]["c"]["path_id"] == "g" for item in table)


def test_rounded_objective_precedes_raw_objective_in_ties():
    facts = {
        "g": LaunchCostFacts(20, 20, 0, ()),
        "a": LaunchCostFacts(0, 10, 0, ()),
        "b": LaunchCostFacts(10, 0, 0, ()),
    }
    cells = {"c1": cell(facts=facts), "c2": cell(facts=facts)}
    declared, rows = round_rows(cells, {
        "c1": {"g": [1], "a": [math.exp(-0.3)], "b": [math.exp(-0.2)]},
        "c2": {"g": [1], "a": [math.exp(-(0.1 + 2e-13))], "b": [math.exp(-0.2)]},
    })
    profile, table = fit(cells, [declared], rows)
    uniform = next(item for item in table if item["integer_weights"] == (2, 2, 2, 2, 2))
    assert uniform["J"] > profile["J"]
    assert uniform["rounded_J"] == round(profile["J"], 12)
    assert profile["integer_weights"] == (1, 2, 2, 2, 3)


@pytest.mark.parametrize("field", ["identity", "status", "validation", "fallback", "full_precision_passed", "policy_reference_passed", "split"])
def test_required_row_metadata_cannot_be_omitted(field):
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [1]}})
    del rows[0][field]
    with pytest.raises(ValueError):
        observations(cells, [declared], rows)


@pytest.mark.parametrize("field,value", [
    ("split", "test"), ("split", "validation"), ("status", "failed"),
    ("status", "unsupported"), ("validation", "fail"), ("fallback", True),
    ("fallback", "false"), ("full_precision_passed", 1),
    ("policy_reference_passed", False), ("session_open_s", -1),
    ("total_wall_s", 0), ("session_close_s", float("inf")),
    ("total_wall_s", float("nan")), ("block", True), ("block", 99),
    ("path_id", "unlisted"), ("round_id", "wrong"), ("cell_id", "wrong"),
    ("attempt_type", "unsupported"),
])
def test_invalid_rows_fail_even_for_warmup(field, value):
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [1], "p": [1]}})
    rows[0][field] = value
    with pytest.raises((ValueError, TypeError)):
        observations(cells, [declared], rows)


@pytest.mark.parametrize("field", ["source_sha", "execution_source", "policy_id", "timing_scope", "candidate_set_sha256"])
def test_row_identity_drift_rejected(field):
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [1], "p": [1]}})
    rows[0]["identity"][field] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        observations(cells, [declared], rows)


@pytest.mark.parametrize("change", ["missing", "duplicate", "no_control", "missing_cell", "unaccepted", "duplicate_round", "no_warmup", "no_measurement"])
def test_exact_round_completeness(change):
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [1, 2, 3], "p": [1, 2, 3]}})
    rounds = [declared]
    if change == "missing":
        rows.pop(0)
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    elif change == "no_control":
        declared["cell_paths"]["c"] = ("p",)
    elif change == "missing_cell":
        cells["absent"] = cell("B")
    elif change == "unaccepted":
        declared["accepted"] = 1
    elif change == "duplicate_round":
        rounds.append(deepcopy(declared))
    elif change == "no_warmup":
        declared["warmup_blocks"] = ()
    else:
        declared["measurement_blocks"] = ()
    with pytest.raises(ValueError):
        observations(cells, rounds, rows)


def test_development_membership_and_common_identity_cannot_be_retargeted():
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [1], "p": [1]}})
    declared["identity"]["source_sha"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        observations(cells, [declared], rows)
    cells["c"]["split"] = "test"
    with pytest.raises(ValueError, match="training"):
        observations(cells, [declared], rows)


@pytest.mark.parametrize("field,value", [
    ("policy_id", "complex_int8_shared_scale_v1"),
    ("policy_id", "float32"),
    ("policy_id", "split_complex_float32_v1 "),
    ("source_sha", "939a016"),
    ("source_sha", "a" * 39),
    ("source_sha", "a" * 41),
    ("source_sha", "A" * 40),
    ("source_sha", "g" * 40),
    ("source_sha", "a" * 39 + " "),
])
def test_self_consistent_invalid_identity_is_rejected(field, value):
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [1], "p": [1]}})
    identity = {**IDENTITY, field: value}
    declared["identity"][field] = value
    for row in rows:
        row["identity"][field] = value
    with pytest.raises(ValueError, match=field):
        fit_launch_cost_weights(cells, [declared], rows, scales=SCALES, expected_identity=identity)


@pytest.mark.parametrize("field", [
    "execution_resource_admission_passed", "startup_resource_admission_passed",
])
@pytest.mark.parametrize("attempt", ["warmup", "measurement"])
@pytest.mark.parametrize("value", [False, None, 1, 1.0, "true", "missing"])
def test_hard_admission_requires_explicit_boolean_true(field, attempt, value):
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [1], "p": [1]}})
    row = next(row for row in rows if row["attempt_type"] == attempt)
    if value == "missing":
        del row[field]
    else:
        row[field] = value
    with pytest.raises(ValueError, match=field):
        fit(cells, [declared], rows)


def test_diagnostic_collection_ineligibility_preserves_passing_hard_admission():
    cells = {"c": cell()}
    declared, rows = round_rows(cells, {"c": {"g": [2], "p": [1]}})
    for row in rows:
        row["collection_resource_admission_passed"] = False
    profile, _ = fit(cells, [declared], rows)
    assert profile["J"] == pytest.approx(math.log(2))
