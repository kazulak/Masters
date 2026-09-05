from __future__ import annotations

from dataclasses import replace

import pytest

from quantum_bench.model import (
    ContractNode,
    ContractionDAG,
    ReduceNode,
    TensorSpec,
    TensorView,
)
from quantum_bench.upmem.plan import UpmemPlan, UpmemTopology, plan_upmem
from quantum_bench.upmem.scheduling import schedule_dag_waves


_POLICY = "split_complex_float32_v1"


def _view(tensor_id: str, labels: tuple[int, ...], shape: tuple[int, ...]) -> TensorView:
    return TensorView(tensor_id=tensor_id, labels=labels, shape=shape)


def _contract(
    node_id: str,
    left: TensorView,
    right: TensorView,
    output_id: str,
    output_labels: tuple[int, ...],
    output_shape: tuple[int, ...],
    *,
    dependencies: tuple[str, ...] = (),
) -> ContractNode:
    contracted = tuple(
        label
        for label in left.labels
        if label in right.labels and label not in output_labels
    )
    return ContractNode(
        node_id=node_id,
        left=left,
        right=right,
        output=TensorSpec(
            output_id,
            output_labels,
            output_shape,
            "dense",
            produced_by=node_id,
        ),
        contracted_labels=contracted,
        output_labels=output_labels,
        dependencies=dependencies,
    )


def _independent_node(
    node_id: str, *, m: int = 2, k: int = 2, n: int = 2
) -> tuple[ContractNode, tuple[TensorSpec, TensorSpec]]:
    left = TensorSpec(f"{node_id}:left", (0, 1), (m, k), "dense")
    right = TensorSpec(f"{node_id}:right", (1, 2), (k, n), "dense")
    node = _contract(
        node_id,
        _view(left.id, left.labels, left.shape),
        _view(right.id, right.labels, right.shape),
        f"{node_id}:out",
        (0, 2),
        (m, n),
    )
    return node, (left, right)


def _chain_dag() -> ContractionDAG:
    first, first_inputs = _independent_node("a")
    second_right = TensorSpec("b:right", (2, 3), (2, 2), "dense")
    second = _contract(
        "b",
        _view("a:out", (0, 2), (2, 2)),
        _view(second_right.id, second_right.labels, second_right.shape),
        "b:out",
        (0, 3),
        (2, 2),
        dependencies=("a",),
    )
    return ContractionDAG(
        tensors=(*first_inputs, second_right),
        nodes=(first, second),
        output=_view("b:out", (0, 3), (2, 2)),
    )


def _d4_split_k_chain_dag() -> ContractionDAG:
    first_left = TensorSpec("a:left", (0, 1), (300, 300), "dense")
    first_right = TensorSpec("a:right", (1, 2), (300, 300), "dense")
    first = _contract(
        "a",
        _view(first_left.id, first_left.labels, first_left.shape),
        _view(first_right.id, first_right.labels, first_right.shape),
        "a:out",
        (0, 2),
        (300, 300),
    )
    second_right = TensorSpec("b:right", (2, 3), (300, 2), "dense")
    second = _contract(
        "b",
        _view("a:out", (0, 2), (300, 300)),
        _view(second_right.id, second_right.labels, second_right.shape),
        "b:out",
        (0, 3),
        (300, 2),
        dependencies=("a",),
    )
    return ContractionDAG(
        tensors=(first_left, first_right, second_right),
        nodes=(first, second),
        output=_view("b:out", (0, 3), (300, 2)),
    )


def _fork_join_dag() -> ContractionDAG:
    first, first_inputs = _independent_node("a")
    second, second_inputs = _independent_node("b")
    reduction = ReduceNode(
        node_id="reduce",
        inputs=(
            _view("a:out", (0, 2), (2, 2)),
            _view("b:out", (0, 2), (2, 2)),
        ),
        output=TensorSpec(
            "reduce:out", (0, 2), (2, 2), "dense", produced_by="reduce"
        ),
        dependencies=("a", "b"),
    )
    final_right = TensorSpec("final:right", (2, 3), (2, 2), "dense")
    final = _contract(
        "final",
        _view("reduce:out", (0, 2), (2, 2)),
        _view(final_right.id, final_right.labels, final_right.shape),
        "final:out",
        (0, 3),
        (2, 2),
        dependencies=("reduce",),
    )
    return ContractionDAG(
        tensors=(*first_inputs, *second_inputs, final_right),
        nodes=(first, second, reduction, final),
        output=_view("final:out", (0, 3), (2, 2)),
    )


def _independent_dag(
    *specifications: tuple[str, int, int, int]
) -> ContractionDAG:
    nodes: list[ContractNode] = []
    tensors: list[TensorSpec] = []
    for node_id, m, k, n in specifications:
        node, inputs = _independent_node(node_id, m=m, k=k, n=n)
        nodes.append(node)
        tensors.extend(inputs)
    output_node = nodes[-1]
    return ContractionDAG(
        tensors=tuple(tensors),
        nodes=tuple(nodes),
        output=_view(
            output_node.output.id,
            output_node.output.labels,
            output_node.output.shape,
        ),
    )


def _plan(
    dag: ContractionDAG, *, dpu_count: int = 2, rank_count: int = 1
) -> UpmemPlan:
    return plan_upmem(
        dag,
        numeric_policy=_POLICY,
        topology=UpmemTopology(
            dpu_count=dpu_count,
            tasklets_per_dpu=1,
            rank_count=rank_count,
        ),
    )


def _contract_stages(
    stages: tuple[object, ...],
) -> tuple[object, ...]:
    return tuple(stage for stage in stages if stage.kind == "contract_batch")


def test_chain_is_serial_by_dependency_and_preserves_unit_order() -> None:
    dag = _chain_dag()
    plan = _plan(dag, dpu_count=2)
    scheduled = schedule_dag_waves(dag, plan)

    assert [(stage.kind, stage.node_ids) for stage in scheduled] == [
        ("contract_batch", ("a",)),
        ("contract_batch", ("b",)),
    ]
    for node_id in ("a", "b"):
        original = tuple(
            unit.stable_tile_id
            for stage in plan.stages
            for unit in stage.work_units
            if unit.node_id == node_id
        )
        actual = tuple(
            unit.stable_tile_id
            for stage in scheduled
            for unit in stage.work_units
            if unit.node_id == node_id
        )
        assert actual == original


def test_fork_join_waits_for_host_reduce_completion() -> None:
    dag = _fork_join_dag()
    scheduled = schedule_dag_waves(dag, _plan(dag, dpu_count=2))

    assert [(stage.kind, stage.node_ids) for stage in scheduled] == [
        ("contract_batch", ("a", "b")),
        ("host_reduce", ("reduce",)),
        ("contract_batch", ("final",)),
    ]
    assert scheduled[1].work_units == ()


def test_more_ready_nodes_than_dpus_are_selected_by_work_then_node_id() -> None:
    dag = _independent_dag(("a", 2, 2, 2), ("b", 2, 2, 2), ("c", 2, 2, 2))
    scheduled = schedule_dag_waves(dag, _plan(dag, dpu_count=2))

    assert [stage.node_ids for stage in _contract_stages(scheduled)] == [
        ("a", "b"),
        ("c",),
    ]


def test_critical_path_prioritizes_a_small_node_with_a_large_tail() -> None:
    first, first_inputs = _independent_node("a")
    tail_right = TensorSpec("tail:right", (2, 3), (2, 300), "dense")
    tail = _contract(
        "tail",
        _view("a:out", (0, 2), (2, 2)),
        _view(tail_right.id, tail_right.labels, tail_right.shape),
        "tail:out",
        (0, 3),
        (2, 300),
        dependencies=("a",),
    )
    other, other_inputs = _independent_node("b", m=2, k=2, n=4)
    dag = ContractionDAG(
        tensors=(*first_inputs, tail_right, *other_inputs),
        nodes=(first, tail, other),
        output=_view("tail:out", (0, 3), (2, 300)),
    )

    scheduled = schedule_dag_waves(dag, _plan(dag, dpu_count=2))

    assert scheduled[0].node_ids == ("a", "b")
    assert scheduled[1].node_ids == ("tail",)


def test_d4_chain_preserves_original_split_k_wave_assignments() -> None:
    dag = _d4_split_k_chain_dag()
    plan = _plan(dag, dpu_count=4)
    original = {
        node_id: tuple(
            (unit.stable_tile_id, unit.wave, unit.logical_dpu)
            for stage in plan.stages
            for unit in stage.work_units
            if unit.node_id == node_id
        )
        for node_id in ("a", "b")
    }
    assert all(
        len({wave for _, wave, _ in units}) > 1
        for units in original.values()
    )

    scheduled = schedule_dag_waves(dag, plan)

    assert [stage.node_ids for stage in _contract_stages(scheduled)] == [
        ("a",),
        ("b",),
    ]
    for node_id in ("a", "b"):
        actual = tuple(
            (unit.stable_tile_id, unit.wave, unit.logical_dpu)
            for stage in scheduled
            for unit in stage.work_units
            if unit.node_id == node_id
        )
        assert actual == original[node_id]


def test_reduced_groups_split_each_original_wave_without_merging_boundaries() -> None:
    dag = _independent_dag(("a", 300, 300, 300), ("b", 300, 300, 300))
    plan = _plan(dag, dpu_count=4)
    scheduled = schedule_dag_waves(dag, plan)
    stage = scheduled[0]

    assert stage.node_ids == ("a", "b")
    for node_id, group in (("a", {0, 1}), ("b", {2, 3})):
        original_units = tuple(
            unit
            for source_stage in plan.stages
            for unit in source_stage.work_units
            if unit.node_id == node_id
        )
        actual_by_tile = {
            unit.stable_tile_id: unit
            for unit in stage.work_units
            if unit.node_id == node_id
        }
        original_waves = sorted({unit.wave for unit in original_units})
        assert all(
            len(
                {
                    actual_by_tile[unit.stable_tile_id].wave
                    for unit in original_units
                    if unit.wave == original_wave
                }
            ) == 2
            for original_wave in original_waves
        )
        mapped_waves = [
            {
                actual_by_tile[unit.stable_tile_id].wave
                for unit in original_units
                if unit.wave == original_wave
            }
            for original_wave in original_waves
        ]
        flattened_waves = [wave for waves in mapped_waves for wave in waves]
        assert len(flattened_waves) == len(set(flattened_waves))
        assert all(
            {
                unit.logical_dpu
                for unit in stage.work_units
                if unit.node_id == node_id and unit.wave == wave
            }
            == group
            for wave in sorted({unit.wave for unit in stage.work_units})
            if any(
                unit.node_id == node_id and unit.wave == wave
                for unit in stage.work_units
            )
        )


def test_group_cap_uses_widest_original_wave_not_total_unit_count() -> None:
    dag = _independent_dag(
        ("narrow", 128, 300, 128),
        ("wide", 300, 1, 300),
    )
    scheduled = schedule_dag_waves(dag, _plan(dag, dpu_count=4))
    stage = scheduled[0]

    narrow = tuple(unit for unit in stage.work_units if unit.node_id == "narrow")
    wide = tuple(unit for unit in stage.work_units if unit.node_id == "wide")
    assert len(narrow) > 1
    assert {unit.logical_dpu for unit in narrow} == {0}
    assert {unit.logical_dpu for unit in wide} == {1, 2, 3}


def test_uneven_work_gets_spare_dpus_with_a_fixed_group_and_unit_cap() -> None:
    dag = _independent_dag(("heavy", 600, 2, 1), ("light", 2, 2, 2))
    plan = _plan(dag, dpu_count=4)
    scheduled = schedule_dag_waves(dag, plan)
    stage = scheduled[0]

    assert stage.node_ids == ("heavy", "light")
    heavy = tuple(unit for unit in stage.work_units if unit.node_id == "heavy")
    light = tuple(unit for unit in stage.work_units if unit.node_id == "light")
    assert tuple(unit.logical_dpu for unit in heavy) == (0, 1, 2)
    assert tuple(unit.wave for unit in heavy) == (0, 0, 0)
    assert tuple(unit.logical_dpu for unit in light) == (3,)
    assert tuple(unit.wave for unit in light) == (0,)
    assert {
        (unit.wave, unit.logical_dpu) for unit in stage.work_units
    } == {(0, 0), (0, 1), (0, 2), (0, 3)}


def test_fewer_ready_nodes_leave_unused_dpus_idle() -> None:
    dag = _independent_dag(("a", 2, 2, 2), ("b", 2, 2, 2))
    scheduled = schedule_dag_waves(dag, _plan(dag, dpu_count=4))
    units = scheduled[0].work_units

    assert scheduled[0].node_ids == ("a", "b")
    assert {unit.logical_dpu for unit in units} == {0, 1}
    assert {unit.wave for unit in units} == {0}


def test_one_dpu_degenerates_to_serial_upmem_without_fallback() -> None:
    dag = _independent_dag(("a", 2, 2, 2), ("b", 2, 2, 2))
    scheduled = schedule_dag_waves(dag, _plan(dag, dpu_count=1))

    assert [stage.node_ids for stage in _contract_stages(scheduled)] == [
        ("a",),
        ("b",),
    ]
    assert all(
        unit.logical_rank == 0 and unit.logical_dpu == 0
        for stage in scheduled
        for unit in stage.work_units
    )


def test_multitile_split_k_preserves_stable_tile_and_partial_order() -> None:
    dag = _independent_dag(("split", 300, 300, 2))
    plan = _plan(dag, dpu_count=1)
    original = tuple(
        unit
        for stage in plan.stages
        for unit in stage.work_units
        if unit.node_id == "split"
    )
    scheduled = schedule_dag_waves(dag, plan)
    actual = scheduled[0].work_units

    assert len(original) > 1
    assert len({unit.k_start for unit in original}) > 1
    assert len({unit.m_start for unit in original}) > 1
    assert tuple(unit.stable_tile_id for unit in actual) == tuple(
        unit.stable_tile_id for unit in original
    )
    assert tuple(unit.wave for unit in actual) == tuple(range(len(actual)))
    assert all(
        replace(unit, wave=original_index, logical_rank=0, logical_dpu=0)
        == actual[original_index]
        for original_index, unit in enumerate(original)
    )


def test_repeated_planning_is_equal_and_does_not_mutate_the_input_plan() -> None:
    dag = _fork_join_dag()
    plan = _plan(dag, dpu_count=2)
    before = plan

    first = schedule_dag_waves(dag, plan)
    second = schedule_dag_waves(dag, plan)

    assert first == second
    assert plan == before


def test_scheduler_rejects_identity_coverage_rank_and_dependency_errors() -> None:
    dag = _chain_dag()
    plan = _plan(dag)

    with pytest.raises(ValueError, match="logical_plan_id"):
        schedule_dag_waves(dag, replace(plan, logical_plan_id="a" * 64))

    with pytest.raises(ValueError, match="node coverage"):
        schedule_dag_waves(dag, replace(plan, stages=plan.stages[:-1]))

    with pytest.raises(ValueError, match="unknown dependency"):
        schedule_dag_waves(
            replace(
                dag,
                nodes=(
                    dag.nodes[0],
                    replace(dag.nodes[1], dependencies=("missing",)),
                ),
            ),
            plan,
        )

    with pytest.raises(ValueError, match="exactly one rank"):
        schedule_dag_waves(dag, _plan(dag, dpu_count=2, rank_count=2))


def test_scheduler_rejects_unsupported_resources_and_kernel_policy() -> None:
    dag = _chain_dag()
    plan = _plan(dag)

    with pytest.raises(ValueError, match="64"):
        schedule_dag_waves(
            dag,
            replace(
                plan,
                topology=UpmemTopology(
                    dpu_count=65,
                    tasklets_per_dpu=1,
                    rank_count=1,
                ),
            ),
        )
    with pytest.raises(ValueError, match="24"):
        schedule_dag_waves(
            dag,
            replace(
                plan,
                topology=UpmemTopology(
                    dpu_count=2,
                    tasklets_per_dpu=25,
                    rank_count=1,
                ),
            ),
        )
    with pytest.raises(ValueError, match="unsupported kernel policy"):
        schedule_dag_waves(dag, replace(plan, kernel_policy="mixed_kernel_v1"))


def test_scheduler_rejects_duplicate_dag_node_ids() -> None:
    dag = _chain_dag()
    duplicate = replace(dag, nodes=(dag.nodes[0], dag.nodes[0]))
    plan = _plan(dag)

    with pytest.raises(ValueError, match="duplicate node IDs"):
        schedule_dag_waves(duplicate, plan)
