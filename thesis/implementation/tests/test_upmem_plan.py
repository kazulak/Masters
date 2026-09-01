from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from quantum_bench.lowering import contraction_dag_hash, slice_contraction
from quantum_bench.model import (
    ContractNode,
    ContractionDAG,
    ReduceNode,
    TensorSpec,
    TensorView,
)
from quantum_bench.results import UnsupportedExecution
from quantum_bench.upmem.plan import (
    PLAN_SCHEMA_VERSION,
    UpmemPlan,
    UpmemResources,
    UpmemStage,
    UpmemTopology,
    UpmemWorkUnit,
    _INT8_PRODUCT,
    _INT64_MAX,
    _validate_final_int8_bounds,
    collection_resource_admission,
    physical_plan_id,
    plan_upmem,
    validate_upmem_plan,
)
from quantum_bench.upmem import plan as upmem_plan_module
from quantum_bench.upmem.tiling import TileLoweringError
from quantum_bench.upmem.protocol import INT32_MAX, MAX_CONTRACTED


def _view(
    tensor_id: str, labels: tuple[int, ...], shape: tuple[int, ...]
) -> TensorView:
    return TensorView(tensor_id=tensor_id, labels=labels, shape=shape)


def _contract(
    node_id: str,
    left_id: str,
    right_id: str,
    output_id: str,
    left_labels: tuple[int, ...],
    left_shape: tuple[int, ...],
    right_labels: tuple[int, ...],
    right_shape: tuple[int, ...],
    output_labels: tuple[int, ...],
    output_shape: tuple[int, ...],
    dependencies: tuple[str, ...] = (),
) -> ContractNode:
    contracted = tuple(
        label
        for label in left_labels
        if label in right_labels and label not in output_labels
    )
    return ContractNode(
        node_id=node_id,
        left=_view(left_id, left_labels, left_shape),
        right=_view(right_id, right_labels, right_shape),
        output=TensorSpec(output_id, output_labels, output_shape, "dense"),
        contracted_labels=contracted,
        output_labels=output_labels,
        dependencies=dependencies,
    )


def _dag() -> ContractionDAG:
    node = _contract(
        "contract_0",
        "a",
        "b",
        "out",
        (0, 1),
        (2, 3),
        (1, 2),
        (3, 4),
        (0, 2),
        (2, 4),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("a", (0, 1), (2, 3), "dense"),
            TensorSpec("b", (1, 2), (3, 4), "dense"),
        ),
        nodes=(node,),
        output=_view("out", (0, 2), (2, 4)),
    )


def _diamond_with_unrelated_node() -> ContractionDAG:
    first = _contract(
        "a_node", "a0", "a1", "p", (0, 1), (2, 3), (1, 4), (3, 2), (0, 4), (2, 2)
    )
    second = _contract(
        "b_node", "b0", "b1", "q", (0, 3), (2, 3), (3, 4), (3, 2), (0, 4), (2, 2)
    )
    middle = _contract(
        "middle", "u0", "u1", "u", (4, 5), (2, 3), (5, 3), (3, 2), (4, 3), (2, 2)
    )
    reduced = ReduceNode(
        node_id="reduce_node",
        inputs=(_view("p", (0, 4), (2, 2)), _view("q", (0, 4), (2, 2))),
        output=TensorSpec("result", (0, 4), (2, 2), "dense"),
        dependencies=("a_node", "b_node"),
    )
    final = _contract(
        "final_node",
        "result",
        "u",
        "final_result",
        (0, 4),
        (2, 2),
        (4, 3),
        (2, 2),
        (0, 3),
        (2, 2),
        dependencies=("reduce_node", "middle"),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("a0", (0, 1), (2, 3), "dense"),
            TensorSpec("a1", (1, 4), (3, 2), "dense"),
            TensorSpec("b0", (0, 3), (2, 3), "dense"),
            TensorSpec("b1", (3, 4), (3, 2), "dense"),
            TensorSpec("u0", (4, 5), (2, 3), "dense"),
            TensorSpec("u1", (5, 3), (3, 2), "dense"),
        ),
        nodes=(first, second, middle, reduced, final),
        output=_view("final_result", (0, 3), (2, 2)),
    )


def _topology(**kwargs: int) -> UpmemTopology:
    return UpmemTopology(
        dpu_count=kwargs.get("dpu_count", 2),
        tasklets_per_dpu=kwargs.get("tasklets_per_dpu", 1),
        rank_count=kwargs.get("rank_count", 1),
    )


def _sliced_dag(*, k_sizes: tuple[int, int, int, int] = (3, 3, 3, 3)) -> ContractionDAG:
    tensors: list[TensorSpec] = []
    nodes: list[ContractNode] = []
    inputs: list[TensorView] = []
    for index, k_size in enumerate(k_sizes):
        left = TensorSpec(
            f"left_{index}", (0, 1, 2), (2, 4, k_size), "dense", dtype="complex128"
        )
        right = TensorSpec(
            f"right_{index}", (2, 1, 3), (k_size, 4, 2), "dense", dtype="complex128"
        )
        output = TensorSpec(
            f"partial_{index}",
            (0, 3),
            (2, 2),
            "dense",
            dtype="complex128",
            produced_by=f"branch_{index}",
        )
        tensors.extend((left, right))
        nodes.append(
            ContractNode(
                node_id=f"branch_{index}",
                left=TensorView(
                    tensor_id=left.id,
                    labels=(0, 2),
                    shape=(2, k_size),
                    slice_spec=((1, index),),
                ),
                right=TensorView(
                    tensor_id=right.id,
                    labels=(2, 3),
                    shape=(k_size, 2),
                    slice_spec=((1, index),),
                ),
                output=output,
                contracted_labels=(2,),
                output_labels=(0, 3),
            )
        )
        inputs.append(TensorView(tensor_id=output.id, labels=(0, 3), shape=(2, 2)))
    reduce = ReduceNode(
        node_id="reduce",
        inputs=tuple(inputs),
        output=TensorSpec("out", (0, 3), (2, 2), "dense", dtype="complex128"),
        reduced_labels=(1,),
        dependencies=tuple(node.node_id for node in nodes),
    )
    return ContractionDAG(
        tensors=tuple(tensors),
        nodes=tuple(nodes) + (reduce,),
        output=TensorView(tensor_id="out", labels=(0, 3), shape=(2, 2)),
    )


def test_mapping_is_deterministic_and_uses_dag_hash() -> None:
    dag = _dag()
    first = plan_upmem(
        dag, numeric_policy="split_complex_float32_v1", topology=_topology()
    )
    second = plan_upmem(
        dag, numeric_policy="split_complex_float32_v1", topology=_topology()
    )

    assert PLAN_SCHEMA_VERSION == 1
    assert first == second
    assert first.logical_plan_id == contraction_dag_hash(dag)
    assert physical_plan_id(first) == physical_plan_id(second)


def test_collection_admission_uses_one_arithmetic_dominant_wave() -> None:
    def unit(
        *, stage: str, wave: int, dpu: int, m_size: int, work: int
    ) -> UpmemWorkUnit:
        return UpmemWorkUnit(
            node_id=stage,
            stable_tile_id=f"{stage}:{wave}:{dpu}",
            wave=wave,
            logical_rank=0,
            logical_dpu=dpu,
            batch_start=0,
            batch_size=1,
            m_start=dpu * m_size,
            m_size=m_size,
            n_start=0,
            n_size=1,
            k_start=0,
            k_size=1,
            estimated_input_bytes=8,
            estimated_output_bytes=8,
            aligned_mram_bytes=24,
            estimated_arithmetic_work=work,
        )

    plan = UpmemPlan(
        logical_plan_id="a" * 64,
        numeric_policy="split_complex_float32_v1",
        topology=UpmemTopology(dpu_count=2, rank_count=1, tasklets_per_dpu=8),
        stages=(
            UpmemStage(
                stage_id="contract_batch:full-but-small",
                kind="contract_batch",
                node_ids=("full-but-small",),
                work_units=(
                    unit(stage="full-but-small", wave=0, dpu=0, m_size=8, work=4),
                    unit(stage="full-but-small", wave=0, dpu=1, m_size=8, work=4),
                ),
            ),
            UpmemStage(
                stage_id="contract_batch:dominant",
                kind="contract_batch",
                node_ids=("dominant",),
                work_units=(
                    unit(stage="dominant", wave=0, dpu=0, m_size=4, work=16),
                ),
            ),
        ),
    )

    facts = collection_resource_admission(plan)

    assert facts["dominant_work_stage_id"] == "contract_batch:dominant"
    assert facts["dominant_work_wave"] == 0
    assert facts["dominant_work_wave_arithmetic_work"] == 16
    assert facts["dominant_work_wave_populated_dpu_slots"] == 1
    assert facts["dominant_work_wave_tasklet_row_sufficiency_passed"] is False
    assert facts["fully_populated_wave_count"] == 1
    assert facts["total_wave_count"] == 2
    assert facts["arithmetic_weighted_dpu_slot_utilization"] == pytest.approx(2 / 3)
    assert facts["arithmetic_weighted_tasklet_utilization"] == pytest.approx(2 / 3)
    assert facts["collection_resource_admission_passed"] is False


def test_four_way_logical_slice_maps_to_one_sorted_batch_and_reduce() -> None:
    dag = _sliced_dag()
    plan = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=_topology(dpu_count=2),
    )
    assert [(stage.kind, stage.node_ids) for stage in plan.stages] == [
        (
            "contract_batch",
            ("branch_0", "branch_1", "branch_2", "branch_3"),
        ),
        ("host_reduce", ("reduce",)),
    ]
    batch = plan.stages[0]
    assert {unit.node_id for unit in batch.work_units} == set(batch.node_ids)
    tile_ids = tuple(unit.stable_tile_id for unit in batch.work_units)
    assert len(set(tile_ids)) == len(tile_ids)
    assert all(
        unit.stable_tile_id.startswith(f"{unit.node_id}:") for unit in batch.work_units
    )
    assert plan == plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=_topology(dpu_count=2),
    )


def test_two_label_cartesian_slice_maps_to_one_batch_and_reduce() -> None:
    left = TensorSpec("left", (0, 1, 2), (2, 2, 2), "dense", dtype="complex128")
    right = TensorSpec("right", (1, 2, 3), (2, 2, 2), "dense", dtype="complex128")
    node = ContractNode(
        node_id="cartesian",
        left=_view(left.id, left.labels, left.shape),
        right=_view(right.id, right.labels, right.shape),
        output=TensorSpec(
            "out",
            (0, 3),
            (2, 2),
            "dense",
            dtype="complex128",
            produced_by="cartesian",
        ),
        contracted_labels=(1, 2),
        output_labels=(0, 3),
    )
    dag = slice_contraction(
        ContractionDAG(
            tensors=(left, right),
            nodes=(node,),
            output=_view("out", (0, 3), (2, 2)),
        ),
        node_id=node.node_id,
        labels=(2, 1),
    )

    plan = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=_topology(dpu_count=1),
    )
    reduction = next(item for item in dag.nodes if isinstance(item, ReduceNode))
    branch_ids = tuple(sorted(reduction.dependencies))

    assert reduction.reduced_labels == (1, 2)
    assert len(branch_ids) == 4
    assert [(stage.kind, stage.node_ids) for stage in plan.stages] == [
        ("contract_batch", branch_ids),
        ("host_reduce", (reduction.node_id,)),
    ]


def test_incompatible_logical_slice_geometry_fails_closed() -> None:
    with pytest.raises(UnsupportedExecution) as caught:
        plan_upmem(
            _sliced_dag(k_sizes=(3, 3, 3, 2)),
            numeric_policy="split_complex_float32_v1",
            topology=_topology(dpu_count=2),
        )
    assert caught.value.stage == "mapping"
    assert caught.value.capability == "upmem_slice_batch_compatibility"


def test_logical_slice_with_extra_fixed_label_fails_closed() -> None:
    dag = _sliced_dag()
    first = dag.nodes[0]
    assert isinstance(first, ContractNode)
    extra_left = replace(dag.tensors[0], labels=(0, 1, 2, 4), shape=(2, 4, 3, 2))
    extra_branch = replace(
        first,
        left=TensorView(
            tensor_id=extra_left.id,
            labels=(0, 2),
            shape=(2, 3),
            slice_spec=((1, 0), (3, 0)),
        ),
    )
    extra_dag = replace(
        dag,
        tensors=(extra_left, *dag.tensors[1:]),
        nodes=(extra_branch, *dag.nodes[1:]),
    )
    with pytest.raises(UnsupportedExecution) as caught:
        plan_upmem(
            extra_dag,
            numeric_policy="split_complex_float32_v1",
            topology=_topology(dpu_count=2),
        )
    assert caught.value.stage == "mapping"
    assert caught.value.capability == "upmem_slice_batch_compatibility"


def test_validator_rejects_grouped_work_unit_membership_tampering() -> None:
    dag = _sliced_dag()
    plan = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=_topology(dpu_count=2),
    )
    batch = plan.stages[0]
    tampered = replace(
        plan,
        stages=(replace(batch, work_units=batch.work_units[:-1]), *plan.stages[1:]),
    )
    with pytest.raises(ValueError, match="differs from pure recomputation"):
        validate_upmem_plan(dag, tampered)


def test_real_tile_byte_and_mac_semantics_are_explicit() -> None:
    dag = _dag()
    float_plan = plan_upmem(
        dag, numeric_policy="split_complex_float32_v1", topology=_topology()
    )
    int_plan = plan_upmem(
        dag, numeric_policy="split_complex_int8_shared_scale_v1", topology=_topology()
    )

    float_unit = float_plan.stages[0].work_units[0]
    int_unit = int_plan.stages[0].work_units[0]
    assert (float_unit.estimated_input_bytes, float_unit.estimated_output_bytes) == (
        6 * 4 + 12 * 4,
        8 * 4,
    )
    assert (int_unit.estimated_input_bytes, int_unit.estimated_output_bytes) == (
        6 + 12,
        8 * 4,
    )
    assert float_unit.aligned_mram_bytes == 6 * 4 + 12 * 4 + 8 * 4
    assert int_unit.aligned_mram_bytes == 8 + 16 + 32
    assert float_unit.estimated_arithmetic_work == 2 * 4 * 3
    assert int_unit.estimated_arithmetic_work == 2 * 4 * 3


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            "split_complex_float32_v1",
            (64, 144, 88, 296),
        ),
        (
            "split_complex_int8_shared_scale_v1",
            (16, 40, 88, 144),
        ),
    ],
)
def test_remainder_tile_alignment_and_byte_sum(
    policy: str, expected: tuple[int, int, int, int]
) -> None:
    unit = (
        plan_upmem(_remainder_dag(), numeric_policy=policy, topology=_topology())
        .stages[0]
        .work_units[0]
    )
    left, right, output, total = expected
    assert unit.aligned_mram_bytes == total
    assert unit.aligned_mram_bytes == left + right + output
    assert (unit.estimated_input_bytes, unit.estimated_output_bytes) == (
        (15 + 35) * (1 if policy.endswith("int8_shared_scale_v1") else 4),
        21 * 4,
    )


def test_reduce_stage_allows_unrelated_ready_node_between_producers() -> None:
    plan = plan_upmem(
        _diamond_with_unrelated_node(),
        numeric_policy="split_complex_float32_v1",
        topology=_topology(),
    )
    assert [stage.stage_id for stage in plan.stages] == [
        "contract_batch:a_node",
        "contract_batch:b_node",
        "contract_batch:middle",
        "host_reduce:reduce_node",
        "contract_batch:final_node",
    ]
    assert plan.stages[-2].work_units == ()
    units = tuple(unit for stage in plan.stages for unit in stage.work_units)
    assert len({unit.stable_tile_id for unit in units}) == len(units)
    assert all(unit.stable_tile_id.startswith(f"{unit.node_id}:") for unit in units)


def test_work_units_are_sorted_by_required_final_key() -> None:
    dag = _contract_dag_with_large_tiles()
    plan = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=UpmemTopology(dpu_count=4, tasklets_per_dpu=2, rank_count=2),
    )
    units = plan.stages[0].work_units
    keys = [
        (
            u.logical_rank,
            u.logical_dpu,
            u.wave,
            u.batch_start,
            u.m_start,
            u.n_start,
            u.k_start,
            u.stable_tile_id,
        )
        for u in units
    ]
    assert keys == sorted(keys)
    assert [
        (unit.wave, unit.logical_rank, unit.logical_dpu, unit.stable_tile_id)
        for unit in units
    ] == [
        (0, 0, 0, "large:b_0:out_0_0:k_0"),
        (1, 0, 0, "large:b_0:out_0_0:k_1"),
        (2, 0, 0, "large:b_0:out_0_0:k_2"),
        (0, 0, 1, "large:b_0:out_0_1:k_0"),
        (1, 0, 1, "large:b_0:out_0_1:k_1"),
        (2, 0, 1, "large:b_0:out_0_1:k_2"),
        (0, 1, 0, "large:b_0:out_1_0:k_0"),
        (1, 1, 0, "large:b_0:out_1_0:k_1"),
        (2, 1, 0, "large:b_0:out_1_0:k_2"),
        (0, 1, 1, "large:b_0:out_1_1:k_0"),
        (1, 1, 1, "large:b_0:out_1_1:k_1"),
        (2, 1, 1, "large:b_0:out_1_1:k_2"),
    ]
    geometry = [
        (u.batch_start, u.m_start, u.m_size, u.n_start, u.n_size, u.k_start, u.k_size)
        for u in units
    ]
    assert len(geometry) == len(set(geometry)) == 12


@pytest.mark.parametrize("dpu_count", (1, 2, 4))
def test_one_rank_t8_dispatch_is_deterministic_and_fully_occupied(
    dpu_count: int,
) -> None:
    dag = _one_rank_t8_dag()
    topology = UpmemTopology(
        dpu_count=dpu_count,
        tasklets_per_dpu=8,
        rank_count=1,
    )
    first = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=topology,
    )
    second = plan_upmem(
        dag,
        numeric_policy="split_complex_float32_v1",
        topology=topology,
    )
    units = first.stages[0].work_units
    wave_ids = tuple(sorted({unit.wave for unit in units}))
    slots = {(unit.wave, unit.logical_rank, unit.logical_dpu) for unit in units}
    facts = collection_resource_admission(first)

    assert first == second
    assert wave_ids == tuple(range(len(units) // dpu_count))
    assert all(unit.logical_rank == 0 for unit in units)
    assert len(slots) == len(units)
    assert all(
        sum(unit.wave == wave for unit in units) <= dpu_count for wave in wave_ids
    )
    assert len(
        {
            physical_plan_id(
                plan_upmem(
                    dag,
                    numeric_policy="split_complex_float32_v1",
                    topology=UpmemTopology(
                        dpu_count=count,
                        tasklets_per_dpu=8,
                        rank_count=1,
                    ),
                )
            )
            for count in (1, 2, 4)
        }
    ) == 3
    assert facts["dominant_work_wave_populated_dpu_slots"] == dpu_count
    assert facts["fully_populated_wave_count"] == len(wave_ids)
    assert facts["arithmetic_weighted_dpu_slot_utilization"] == pytest.approx(1.0)
    assert facts["collection_resource_admission_passed"] is True


def _one_rank_t8_dag() -> ContractionDAG:
    node = _contract(
        "one_rank_t8",
        "left",
        "right",
        "out",
        (0, 1),
        (300, 8),
        (1, 2),
        (8, 300),
        (0, 2),
        (300, 300),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("left", (0, 1), (300, 8), "dense"),
            TensorSpec("right", (1, 2), (8, 300), "dense"),
        ),
        nodes=(node,),
        output=_view("out", (0, 2), (300, 300)),
    )


def _general_work_dag(work_count: int) -> ContractionDAG:
    """Create exactly ``work_count`` output tiles for pure mapping tests."""

    node = _contract(
        f"general_work_{work_count}",
        "left",
        "right",
        "out",
        (0, 1),
        (256 * work_count, 8),
        (1, 2),
        (8, 8),
        (0, 2),
        (256 * work_count, 8),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("left", (0, 1), (256 * work_count, 8), "dense"),
            TensorSpec("right", (1, 2), (8, 8), "dense"),
        ),
        nodes=(node,),
        output=_view("out", (0, 2), (256 * work_count, 8)),
    )


def _general_work_units(
    *, dpu_count: int, tasklets_per_dpu: int, work_count: int
) -> tuple[UpmemWorkUnit, ...]:
    plan = plan_upmem(
        _general_work_dag(work_count),
        numeric_policy="split_complex_float32_v1",
        topology=UpmemTopology(
            dpu_count=dpu_count,
            rank_count=1,
            tasklets_per_dpu=tasklets_per_dpu,
        ),
    )
    return plan.stages[0].work_units


@pytest.mark.parametrize("tasklets_per_dpu", range(1, 25))
def test_pure_mapping_accepts_every_supported_tasklet_count(
    tasklets_per_dpu: int,
) -> None:
    units = _general_work_units(
        dpu_count=1, tasklets_per_dpu=tasklets_per_dpu, work_count=1
    )

    assert len(units) == 1


@pytest.mark.parametrize(
    ("tasklets_per_dpu", "error"),
    [
        (0, UnsupportedExecution),
        (25, UnsupportedExecution),
        (True, TypeError),
        (False, TypeError),
        (1.0, TypeError),
        ("3", TypeError),
    ],
)
def test_pure_mapping_rejects_invalid_tasklet_counts(
    tasklets_per_dpu: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _general_work_units(
            dpu_count=1,
            tasklets_per_dpu=tasklets_per_dpu,  # type: ignore[arg-type]
            work_count=1,
        )


@pytest.mark.parametrize("dpu_count", range(1, 65))
def test_pure_mapping_accepts_every_one_rank_dpu_count(dpu_count: int) -> None:
    units = _general_work_units(
        dpu_count=dpu_count, tasklets_per_dpu=1, work_count=dpu_count
    )

    assert len(units) == dpu_count
    assert {(unit.logical_rank, unit.logical_dpu) for unit in units} == {
        (0, index) for index in range(dpu_count)
    }


@pytest.mark.parametrize(
    ("dpu_count", "error"),
    [
        (0, UnsupportedExecution),
        (65, UnsupportedExecution),
        (True, TypeError),
        (False, TypeError),
        (1.0, TypeError),
        ("3", TypeError),
    ],
)
def test_pure_mapping_rejects_invalid_one_rank_dpu_counts(
    dpu_count: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _general_work_units(
            dpu_count=dpu_count,  # type: ignore[arg-type]
            tasklets_per_dpu=1,
            work_count=1,
        )


def test_general_one_rank_waves_cover_every_requested_work_count_exactly_once() -> None:
    for dpu_count in range(1, 65):
        for work_count in (
            1,
            max(1, dpu_count - 1),
            dpu_count,
            dpu_count + 1,
            2 * dpu_count + 1,
        ):
            dag = _general_work_dag(work_count)
            topology = UpmemTopology(
                dpu_count=dpu_count, rank_count=1, tasklets_per_dpu=1
            )
            first = plan_upmem(
                dag, numeric_policy="split_complex_float32_v1", topology=topology
            )
            second = plan_upmem(
                dag, numeric_policy="split_complex_float32_v1", topology=topology
            )
            units = first.stages[0].work_units
            flattened = tuple(
                sorted(
                    units,
                    key=lambda unit: (unit.wave, unit.logical_rank, unit.logical_dpu),
                )
            )

            assert first == second
            validate_upmem_plan(dag, first)
            assert len(units) == work_count
            assert [unit.m_start for unit in flattened] == list(
                range(0, 256 * work_count, 256)
            )
            assert [
                (unit.wave, unit.logical_rank, unit.logical_dpu) for unit in flattened
            ] == [(index // dpu_count, 0, index % dpu_count) for index in range(work_count)]
            assert len(
                {(unit.wave, unit.logical_rank, unit.logical_dpu) for unit in units}
            ) == work_count
            assert tuple(sorted({unit.wave for unit in units})) == tuple(
                range((work_count + dpu_count - 1) // dpu_count)
            )


@pytest.mark.parametrize(
    ("work_count", "dpu_count", "expected_slots"),
    [
        (1, 4, ((0, 0),)),
        (3, 2, ((0, 0), (0, 1), (1, 0))),
        (
            10,
            3,
            (
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
                (3, 0),
            ),
        ),
        (
            17,
            4,
            (
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
                (4, 0),
            ),
        ),
    ],
)
def test_general_resource_adversarial_work_dpu_pairs_retain_exact_wave_slots(
    work_count: int, dpu_count: int, expected_slots: tuple[tuple[int, int], ...]
) -> None:
    units = _general_work_units(
        dpu_count=dpu_count,
        tasklets_per_dpu=1,
        work_count=work_count,
    )

    flattened = tuple(sorted(units, key=lambda unit: (unit.wave, unit.logical_dpu)))
    assert tuple((unit.wave, unit.logical_dpu) for unit in flattened) == expected_slots
    assert [unit.m_start for unit in flattened] == list(range(0, 256 * work_count, 256))
    tail_wave, tail_dpu = expected_slots[-1]
    assert (flattened[-1].wave, flattened[-1].logical_dpu) == (tail_wave, tail_dpu)


def _contract_dag_with_large_tiles() -> ContractionDAG:
    node = _contract(
        "large",
        "left",
        "right",
        "large_out",
        (0, 1),
        (300, 300),
        (1, 2),
        (300, 300),
        (0, 2),
        (300, 300),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("left", (0, 1), (300, 300), "dense"),
            TensorSpec("right", (1, 2), (300, 300), "dense"),
        ),
        nodes=(node,),
        output=_view("large_out", (0, 2), (300, 300)),
    )


def _remainder_dag() -> ContractionDAG:
    node = _contract(
        "remainder",
        "left",
        "right",
        "out",
        (0, 1),
        (3, 5),
        (1, 2),
        (5, 7),
        (0, 2),
        (3, 7),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("left", (0, 1), (3, 5), "dense"),
            TensorSpec("right", (1, 2), (5, 7), "dense"),
        ),
        nodes=(node,),
        output=_view("out", (0, 2), (3, 7)),
    )


def _oversized_k_dag() -> ContractionDAG:
    node = _contract(
        "oversized_k",
        "left",
        "right",
        "out",
        (0, 1),
        (1, MAX_CONTRACTED + 1),
        (1, 2),
        (MAX_CONTRACTED + 1, 1),
        (0, 2),
        (1, 1),
    )
    return ContractionDAG(
        tensors=(
            TensorSpec("left", (0, 1), (1, MAX_CONTRACTED + 1), "dense"),
            TensorSpec("right", (1, 2), (MAX_CONTRACTED + 1, 1), "dense"),
        ),
        nodes=(node,),
        output=_view("out", (0, 2), (1, 1)),
    )


def test_identity_changes_for_policy_topology_stage_and_work_order() -> None:
    dag = _dag()
    base = plan_upmem(
        dag, numeric_policy="split_complex_float32_v1", topology=_topology()
    )
    assert physical_plan_id(base) != physical_plan_id(
        plan_upmem(
            dag,
            numeric_policy="split_complex_int8_shared_scale_v1",
            topology=_topology(),
        )
    )
    assert physical_plan_id(base) != physical_plan_id(
        plan_upmem(
            dag,
            numeric_policy="split_complex_float32_v1",
            topology=_topology(dpu_count=4),
        )
    )
    changed_stage_order = replace(base.stages[0], stage_id="contract_batch:changed")
    changed_stage_plan = replace(base, stages=(changed_stage_order,))
    assert physical_plan_id(base) != physical_plan_id(changed_stage_plan)
    large = plan_upmem(
        _contract_dag_with_large_tiles(),
        numeric_policy="split_complex_float32_v1",
        topology=_topology(dpu_count=4, rank_count=2),
    )
    reversed_units = replace(
        large.stages[0], work_units=tuple(reversed(large.stages[0].work_units))
    )
    reordered_plan = replace(large, stages=(reversed_units,))
    assert physical_plan_id(large) != physical_plan_id(reordered_plan)
    unit = base.stages[0].work_units[0]
    changed_unit = replace(unit, stable_tile_id=unit.stable_tile_id + ":changed")
    changed_stage = replace(base.stages[0], work_units=(changed_unit,))
    changed_plan = replace(base, stages=(changed_stage,))
    assert physical_plan_id(base) != physical_plan_id(changed_plan)


def test_validator_rejects_tampering() -> None:
    dag = _dag()
    plan = plan_upmem(
        dag, numeric_policy="split_complex_float32_v1", topology=_topology()
    )
    validate_upmem_plan(dag, plan)
    assert plan.kernel_policy == "dpu_real_tile_v4_wram_panel_v1"
    tampered = replace(plan, kernel_policy="different")
    with pytest.raises(ValueError, match="differs from pure recomputation"):
        validate_upmem_plan(dag, tampered)


@pytest.mark.parametrize(
    "tamper", ["stage_order", "node_id", "tile", "bytes", "topology"]
)
def test_validator_rejects_structurally_valid_tampering(tamper: str) -> None:
    dag = _contract_dag_with_large_tiles()
    if tamper == "stage_order":
        dag = _diamond_with_unrelated_node()
    plan = plan_upmem(
        dag, numeric_policy="split_complex_float32_v1", topology=_topology()
    )
    if tamper == "stage_order":
        tampered = replace(plan, stages=tuple(reversed(plan.stages)))
    elif tamper == "node_id":
        stage = plan.stages[0]
        units = tuple(replace(unit, node_id="wrong_node") for unit in stage.work_units)
        changed = replace(stage, node_ids=("wrong_node",), work_units=units)
        tampered = replace(plan, stages=(changed, *plan.stages[1:]))
    elif tamper == "tile":
        stage = plan.stages[0]
        changed = replace(stage.work_units[0], m_start=stage.work_units[0].m_start + 1)
        tampered = replace(
            plan,
            stages=(
                replace(stage, work_units=(changed, *stage.work_units[1:])),
                *plan.stages[1:],
            ),
        )
    elif tamper == "bytes":
        stage = plan.stages[0]
        changed = replace(
            stage.work_units[0],
            estimated_input_bytes=stage.work_units[0].estimated_input_bytes + 1,
        )
        tampered = replace(
            plan,
            stages=(
                replace(stage, work_units=(changed, *stage.work_units[1:])),
                *plan.stages[1:],
            ),
        )
    else:
        tampered = replace(
            plan,
            topology=UpmemTopology(dpu_count=4, tasklets_per_dpu=1, rank_count=2),
        )
    with pytest.raises(ValueError, match="differs from pure recomputation"):
        validate_upmem_plan(dag, tampered)


def test_structurally_invalid_stage_records_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="host_reduce"):
        UpmemStage(
            stage_id="reduce", kind="host_reduce", node_ids=("a", "b"), work_units=()
        )
    with pytest.raises(ValueError, match="host_reduce"):
        UpmemStage(
            stage_id="reduce",
            kind="host_reduce",
            node_ids=("a",),
            work_units=(
                UpmemWorkUnit(
                    node_id="a",
                    stable_tile_id="tile",
                    wave=0,
                    logical_rank=0,
                    logical_dpu=0,
                    batch_start=0,
                    batch_size=1,
                    m_start=0,
                    m_size=1,
                    n_start=0,
                    n_size=1,
                    k_start=0,
                    k_size=1,
                    estimated_input_bytes=0,
                    estimated_output_bytes=0,
                    aligned_mram_bytes=0,
                    estimated_arithmetic_work=1,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("wave", True), ("m_size", 0), ("node_id", ""), ("k_start", -1)],
)
def test_work_unit_intrinsic_validation(field: str, value: object) -> None:
    kwargs = dict(
        node_id="node",
        stable_tile_id="tile",
        wave=0,
        logical_rank=0,
        logical_dpu=0,
        batch_start=0,
        batch_size=1,
        m_start=0,
        m_size=1,
        n_start=0,
        n_size=1,
        k_start=0,
        k_size=1,
        estimated_input_bytes=0,
        estimated_output_bytes=0,
        aligned_mram_bytes=0,
        estimated_arithmetic_work=1,
    )
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        UpmemWorkUnit(**kwargs)


@pytest.mark.parametrize(
    ("dpu_count", "rank_count", "tasklets", "message"),
    [
        (0, 1, 1, "positive"),
        (1, 0, 1, "positive"),
        (3, 2, 1, "divisible"),
        (65, 1, 1, "at most 64"),
        (1, 1, 0, r"\[1, 24\]"),
        (1, 1, 25, r"\[1, 24\]"),
    ],
)
def test_mapping_rejects_topology_boundaries(
    dpu_count: int, rank_count: int, tasklets: int, message: str
) -> None:
    with pytest.raises(UnsupportedExecution, match=message):
        plan_upmem(
            _dag(),
            numeric_policy="split_complex_float32_v1",
            topology=UpmemTopology(
                dpu_count=dpu_count,
                tasklets_per_dpu=tasklets,
                rank_count=rank_count,
            ),
        )


def test_mapping_rejects_abi_max_contracted_shape_only() -> None:
    with pytest.raises(UnsupportedExecution, match="max_contracted"):
        plan_upmem(
            _oversized_k_dag(),
            numeric_policy="split_complex_float32_v1",
            topology=_topology(),
        )


@pytest.mark.parametrize(
    ("contracted_size", "tile_k", "should_fail"),
    [
        (INT32_MAX // _INT8_PRODUCT, INT32_MAX // _INT8_PRODUCT, False),
        (INT32_MAX // _INT8_PRODUCT, INT32_MAX // _INT8_PRODUCT + 1, True),
    ],
)
def test_final_int8_int32_boundary(
    contracted_size: int, tile_k: int, should_fail: bool
) -> None:
    assert _INT8_PRODUCT == 127 * 127
    assert INT32_MAX // _INT8_PRODUCT == 133144
    tile = SimpleNamespace(id="tile", k_size=tile_k)
    if should_fail:
        with pytest.raises(UnsupportedExecution, match="int32"):
            _validate_final_int8_bounds("node", contracted_size, (tile,))
    else:
        _validate_final_int8_bounds("node", contracted_size, (tile,))


def test_final_int8_int64_boundary() -> None:
    assert _INT8_PRODUCT == 127 * 127
    boundary = _INT64_MAX // (2 * _INT8_PRODUCT)
    _validate_final_int8_bounds("node", boundary, ())
    with pytest.raises(UnsupportedExecution, match="int64"):
        _validate_final_int8_bounds("node", boundary + 1, ())


def test_mapping_rejects_unknown_numeric_policy() -> None:
    with pytest.raises(UnsupportedExecution, match="numeric policy"):
        plan_upmem(_dag(), numeric_policy="unknown", topology=_topology())  # type: ignore[arg-type]


def test_mapping_rejects_valid_dag_without_contract_work() -> None:
    input_view = _view("input", (0,), (2,))
    dag = ContractionDAG(
        tensors=(TensorSpec("input", (0,), (2,), "dense"),),
        nodes=(),
        output=input_view,
    )

    with pytest.raises(UnsupportedExecution) as exc_info:
        plan_upmem(
            dag,
            numeric_policy="split_complex_float32_v1",
            topology=_topology(),
        )

    assert exc_info.value.stage == "mapping"
    assert exc_info.value.capability == "upmem_no_contract_work"


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (ValueError("injected mapper defect"), "injected mapper defect"),
        (TileLoweringError("injected unsupported tile"), "not representable"),
    ],
)
def test_final_mapper_preserves_expected_exception_boundary(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_message: str,
) -> None:
    def raise_error(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(upmem_plan_module, "plan_tile_shapes", raise_error)
    expected_exception = (
        ValueError
        if isinstance(error, ValueError) and not isinstance(error, TileLoweringError)
        else UnsupportedExecution
    )

    with pytest.raises(expected_exception, match=expected_message) as exc_info:
        plan_upmem(
            _dag(),
            numeric_policy="split_complex_float32_v1",
            topology=_topology(),
        )

    if isinstance(error, TileLoweringError):
        assert exc_info.value.stage == "mapping"
        assert exc_info.value.capability == "upmem_v4_geometry"


def test_resources_are_immutable_and_callback_is_not_identity() -> None:
    def callback() -> None:
        return None

    first = UpmemResources(
        session_root="session",
        host_binary="host",
        dpu_binary="dpu",
        initialization_binary="init",
        rank_paths=("rank0",),
        session_opener=callback,
    )
    second = replace(first, session_opener=lambda: None)
    assert first == second
    assert "session_opener" not in repr(first)
    with pytest.raises((AttributeError, TypeError)):
        first.rank_paths = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="tuple"):
        UpmemResources(
            session_root="session",
            host_binary="host",
            dpu_binary="dpu",
            initialization_binary="init",
            rank_paths=["rank0"],  # type: ignore[arg-type]
        )


def test_resources_fix_request_transport_to_packed_operation() -> None:
    resources = UpmemResources(
        session_root="session",
        host_binary="host",
        dpu_binary="dpu",
        initialization_binary="init",
    )
    assert resources.request_transport == "packed_operation_v1"

    with pytest.raises(ValueError, match="fixed to packed_operation_v1"):
        UpmemResources(
            session_root="session",
            host_binary="host",
            dpu_binary="dpu",
            initialization_binary="init",
            request_transport="directory_v1",  # type: ignore[arg-type]
        )
