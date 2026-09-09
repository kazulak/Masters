from dataclasses import FrozenInstanceError
import math

import pytest

from quantum_bench.upmem.path_heuristic import (
    LAUNCH_COST_MODEL_ID,
    LaunchCostFacts,
    LaunchCostLaunch,
    LaunchCostScales,
    development_greedy_scales,
    launch_cost_weight_grid,
    upmem_launch_cost_v1,
    validate_launch_cost_weights,
)


def _launch(m: tuple[float, ...], w: tuple[float, ...]) -> LaunchCostLaunch:
    return LaunchCostLaunch(m_by_dpu=m, w_by_dpu=w)


def _facts(
    h: float = 0.0,
    p: float = 0.0,
    n: float = 0.0,
    launches: tuple[LaunchCostLaunch, ...] = (),
) -> LaunchCostFacts:
    return LaunchCostFacts(h=h, p=p, n=n, launches=launches)


def _unit_scales() -> LaunchCostScales:
    return LaunchCostScales(1.0, 1.0, 1.0, 1.0, 1.0)


def test_serial_launches_add_after_per_launch_maximum() -> None:
    one_launch = _facts(launches=(_launch((0.0, 0.0), (4.0, 4.0)),))
    two_launches = _facts(
        launches=(
            _launch((0.0, 0.0), (4.0, 0.0)),
            _launch((0.0, 0.0), (4.0, 0.0)),
        )
    )
    weights = (0, 0, 0, 0, 10)

    assert one_launch.total_w() == two_launches.total_w() == 8.0
    assert upmem_launch_cost_v1(one_launch, weights, _unit_scales()) == pytest.approx(4.0)
    assert upmem_launch_cost_v1(two_launches, weights, _unit_scales()) == pytest.approx(8.0)


def test_joint_max_is_taken_after_weighted_movement_and_work() -> None:
    facts = _facts(launches=(_launch((10.0, 0.0), (0.0, 10.0)),))

    assert upmem_launch_cost_v1(facts, (0, 0, 0, 5, 5), _unit_scales()) == pytest.approx(5.0)


def test_four_product_and_fused_work_totals_are_invariant() -> None:
    four_products = _facts(
        launches=tuple(_launch((1.0,), (10.0,)) for _ in range(4))
    )
    fused = _facts(launches=(_launch((4.0,), (40.0,)),))
    weights = (0, 0, 0, 0, 10)

    assert four_products.total_w() == pytest.approx(fused.total_w())
    assert upmem_launch_cost_v1(four_products, weights, _unit_scales()) == pytest.approx(40.0)
    assert upmem_launch_cost_v1(fused, weights, _unit_scales()) == pytest.approx(40.0)


def test_h_p_n_are_serial_terms_and_not_launch_maxima() -> None:
    facts = _facts(
        h=4.0,
        p=8.0,
        n=2.0,
        launches=(_launch((5.0,), (10.0,)),),
    )
    scales = LaunchCostScales(2.0, 4.0, 1.0, 5.0, 10.0)

    assert upmem_launch_cost_v1(facts, (2, 2, 2, 2, 2), scales) == pytest.approx(1.6)


def test_development_scales_use_global_training_greedy_medians_only() -> None:
    training_a = _facts(
        h=2.0,
        p=4.0,
        n=6.0,
        launches=(_launch((8.0,), (10.0,)),),
    )
    training_b = _facts(
        h=4.0,
        p=2.0,
        n=8.0,
        launches=(_launch((12.0,), (14.0,)),),
    )
    evaluation = _facts(
        h=1000.0,
        p=1000.0,
        n=1000.0,
        launches=(_launch((1000.0,), (1000.0,)),),
    )
    cells = {
        "training-a": ("training", training_a),
        "training-b": ("training", training_b),
        "test": ("test", evaluation),
    }

    scales = development_greedy_scales(cells)

    assert scales.as_tuple() == pytest.approx((3.0, 3.0, 7.0, 10.0, 12.0))
    assert scales == development_greedy_scales(
        {key: value for key, value in cells.items() if key != "test"}
    )


def test_weight_grid_has_all_integer_simplex_points() -> None:
    grid = launch_cost_weight_grid()

    assert LAUNCH_COST_MODEL_ID == "upmem_launch_cost_v1"
    assert len(grid) == 1001
    assert len(set(grid)) == 1001
    assert grid == tuple(sorted(grid))
    assert grid[0] == (0, 0, 0, 0, 10)
    assert grid[-1] == (10, 0, 0, 0, 0)
    assert all(validate_launch_cost_weights(weights) == weights for weights in grid)


def test_fact_records_are_immutable_and_validate_numeric_inputs() -> None:
    facts = _facts(launches=(_launch((1.0,), (2.0,)),))
    with pytest.raises(FrozenInstanceError):
        facts.h = 3.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        facts.launches[0].m_by_dpu = (3.0,)  # type: ignore[misc]

    with pytest.raises(ValueError):
        LaunchCostLaunch((math.nan,), (1.0,))
    with pytest.raises(ValueError):
        LaunchCostFacts(-1.0, 0.0, 0.0, ())
    with pytest.raises(TypeError):
        LaunchCostFacts(0.0, 0.0, 0.0, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LaunchCostScales(0.0, 1.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "weights",
    [
        (0, 0, 0, 0),
        (0, 0, 0, 0, 11),
        (0, 0, 0, 0, -1),
        (0, 0, 0, 0, 10.0),
        (True, 0, 0, 0, 9),
    ],
)
def test_weights_require_five_nonnegative_integers_summing_to_ten(weights: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_launch_cost_weights(weights)  # type: ignore[arg-type]


def test_scales_require_a_training_cell() -> None:
    with pytest.raises(ValueError, match="training greedy cell"):
        development_greedy_scales(
            {"test": ("test", _facts(launches=(_launch((1.0,), (1.0,)),)))}
        )
